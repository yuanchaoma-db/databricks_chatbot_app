from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime, timedelta
import json
import httpx
from databricks.sdk.service.serving import EndpointStateReady
from fastapi import Query
import requests
import backoff
import time  # Add this import at the top
load_dotenv(override=True)

app = FastAPI(title="Databricks Chat API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Databricks workspace client
client = WorkspaceClient(host=f"https://{os.getenv("DATABRICKS_HOST")}", token=os.getenv("DATABRICKS_TOKEN"))

# Constants
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")

# Cache to track streaming support for endpoints
streaming_support_cache = {
    'last_updated': datetime.now(),
    'endpoints': {}  # Format: {'endpoint_name': {'supports_streaming': bool, 'last_checked': datetime}}
}

# Models
class MessageRequest(BaseModel):
    content: str

class MessageResponse(BaseModel):
    content: str
    role: str

class ChatHistoryItem(BaseModel):
    messages: List[MessageResponse]
    id: str
    title: str

class ChatHistoryResponse(BaseModel):
    chats: List[ChatHistoryItem]

class CreateChatRequest(BaseModel):
    title: str

class ServingEndpoint(BaseModel):
    names: List[str]

# In-memory storage for chats (in a production app, use a database)
chats_db = {}

# Routes
@app.get("/")
async def root():
    return {"message": "Databricks Chat API is running"}

# Add this function to check and update cache
async def check_endpoint_streaming(model: str) -> bool:
    current_time = datetime.now()
    cache_entry = streaming_support_cache['endpoints'].get(model)
    
    # If cache entry exists and is less than 24 hours old, use cached value
    if cache_entry and (current_time - cache_entry['last_checked']) < timedelta(days=1):
        return cache_entry['supports_streaming']
    
    # Update cache entry
    streaming_support_cache['endpoints'][model] = {
        'supports_streaming': True,  # Default to False
        'last_checked': current_time
    }
    return True

# Add this function before the chat endpoint
@backoff.on_exception(
    backoff.expo,
    (httpx.HTTPError, Exception),
    max_tries=3,
    max_time=30
)
async def make_databricks_request(client: httpx.AsyncClient, url: str, headers: dict, data: dict):
    return await client.post(url, headers=headers, json=data)

async def extract_sources_from_trace(data: dict) -> list:
    """Extract sources from the Databricks API trace data."""
    sources = []
    if "databricks_output" in data and "trace" in data["databricks_output"]:
        trace = data["databricks_output"]["trace"]
        if "data" in trace and "spans" in trace["data"]:
            for span in trace["data"]["spans"]:
                if span.get("name") == "VectorStoreRetriever":
                    retriever_output = json.loads(span["attributes"].get("mlflow.spanOutputs", "[]"))
                    sources = [
                        {
                            "page_content": doc["page_content"],
                            "metadata": doc["metadata"]
                        } 
                        for doc in retriever_output
                    ]
    return sources

@app.post("/api/chat")
async def chat(message: MessageRequest, model: str = Query(...)):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
            "Content-Type": "application/json"
        }

        async def generate():
            async with httpx.AsyncClient() as client:
                supports_streaming = await check_endpoint_streaming(model)
                
                if not supports_streaming:
                    # Direct non-streaming request
                    print("non Streaming is running")
                    response = await make_databricks_request(
                        client,
                        f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                        headers=headers,
                        data={
                            "messages": [{"role": "user", "content": message.content}],
                            "databricks_options": {"return_trace": True}
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        sources = await extract_sources_from_trace(data)
                        
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['message']['content']
                            yield f"data: {json.dumps({'content': content, 'sources': sources})}\n\n"
                            yield "event: done\ndata: {}\n\n"
                else:
                    try:
                        print("streaming is running")
                        start_time = time.time()
                        first_token_time = None
                        
                        async with client.stream('POST', 
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                            headers=headers,
                            json={
                                "messages": [{"role": "user", "content": message.content}],
                                "stream": True,
                                "databricks_options": {"return_trace": True}
                            },
                            timeout=30.0
                        ) as response:
                            if response.status_code == 200:
                                has_content = False
                                sources = []
                                
                                async for line in response.aiter_lines():
                                    if line.startswith('data: '):
                                        try:
                                            json_str = line[6:]
                                            data = json.loads(json_str)
                                            
                                            # Record time of first token
                                            if first_token_time is None and 'choices' in data and len(data['choices']) > 0:
                                                first_token_time = time.time()
                                                ttft = first_token_time - start_time
                                                
                                            sources = await extract_sources_from_trace(data)
                                            
                                            if 'choices' in data and len(data['choices']) > 0:
                                                delta = data['choices'][0].get('delta', {})
                                                content = delta.get('content', '')
                                                
                                                response_data = {
                                                    'content': content if content else None,
                                                    'sources': sources if sources else None,
                                                    'metrics': {
                                                        'timeToFirstToken': ttft if first_token_time is not None else None,
                                                        'totalTime': time.time() - start_time
                                                    }
                                                }
                                                if content or sources:
                                                    has_content = True
                                                    yield f"data: {json.dumps(response_data)}\n\n"
                                        except json.JSONDecodeError:
                                            continue

                                if has_content:
                                    # Update cache to indicate streaming support
                                    streaming_support_cache['endpoints'][model] = {
                                        'supports_streaming': True,
                                        'last_checked': datetime.now()
                                    }
                                    total_time = time.time() - start_time
                                    yield f"data: {json.dumps({'metrics': {'timeToFirstToken': ttft, 'totalTime': total_time}})}\n\n"
                                else:
                                    raise Exception("No streaming content received")
                            else:
                                raise Exception("Streaming not supported")

                    except (Exception, httpx.ReadTimeout, httpx.HTTPError) as e:  # Explicitly catch timeout
                        print(f"Streaming failed with error: {str(e)}, falling back to non-streaming")
                        # Update cache to indicate no streaming support
                        streaming_support_cache['endpoints'][model] = {
                            'supports_streaming': False,
                            'last_checked': datetime.now()
                        }
                        # Fall back to non-streaming request
                        response = await make_databricks_request(
                            client,
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                            headers=headers,
                            data={
                                "messages": [{"role": "user", "content": message.content}],
                                "databricks_options": {"return_trace": True}
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            sources = await extract_sources_from_trace(data)
                        
                            if 'choices' in data and len(data['choices']) > 0:
                                content = data['choices'][0]['message']['content']
                                yield f"data: {json.dumps({'content': content, 'sources': sources})}\n\n"
                                yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Databricks API: {str(e)}")

@app.get("/api/chats", response_model=ChatHistoryResponse)
async def get_chat_history():
    # Convert the in-memory database to the response model
    chats = [
        ChatHistoryItem(
            id=chat_id,
            title=chat_data["title"],
            messages=[
                MessageResponse(content=msg["content"], role=msg["role"]) 
                for msg in chat_data.get("messages", [])
            ]
        )
        for chat_id, chat_data in chats_db.items()
    ]
    
    # Sort by most recent
    chats.sort(key=lambda x: chats_db[x.id].get("created_at", ""), reverse=True)
    
    return ChatHistoryResponse(chats=chats)


@app.get("/api/chats/{chat_id}", response_model=ChatHistoryItem)
async def get_chat(chat_id: str):
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat_data = chats_db[chat_id]
    
    return ChatHistoryItem(
        id=chat_id,
        title=chat_data["title"],
        messages=[
            MessageResponse(content=msg["content"], role=msg["role"]) 
            for msg in chat_data.get("messages", [])
        ]
    )

@app.post("/api/chats/{chat_id}/messages", response_model=MessageResponse)
async def add_message_to_chat(chat_id: str, message: MessageRequest):
    if chat_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Add user message to chat history
    user_message = {
        "content": message.content,
        "role": "user",
        "timestamp": datetime.now().isoformat()
    }
    
    if "messages" not in chats_db[chat_id]:
        chats_db[chat_id]["messages"] = []
    
    chats_db[chat_id]["messages"].append(user_message)
    
    try:
        # Make the API call to Databricks
        response = client.serving_endpoints.query(
            SERVING_ENDPOINT_NAME,
            temperature=0.7,
            messages=[ChatMessage(content=message.content, role=ChatMessageRole.USER)],
        )
        
        bot_response_text = response.choices[0].message.content
        
        # Add bot response to chat history
        bot_message = {
            "content": bot_response_text,
            "role": "assistant",
            "timestamp": datetime.now().isoformat()
        }
        
        chats_db[chat_id]["messages"].append(bot_message)
        
        return MessageResponse(
            content=bot_response_text,
            role="assistant"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Databricks API: {str(e)}")

@app.get("/api/serving-endpoints")
async def list_endpoints():
    try:
        serving_endpoints = []
        endpoints = client.serving_endpoints.list()
        for endpoint in endpoints:
            if endpoint.state.ready == EndpointStateReady.READY:
                serving_endpoints.append(endpoint.name)
        return ServingEndpoint(names=sorted(serving_endpoints))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching endpoints: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
