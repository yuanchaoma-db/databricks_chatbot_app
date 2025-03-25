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
from datetime import datetime
import json
import httpx
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

# In-memory storage for chats (in a production app, use a database)
chats_db = {}

# Routes
@app.get("/")
async def root():
    return {"message": "Databricks Chat API is running"}

@app.post("/api/chat")
async def chat(message: MessageRequest):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        async def generate():
            async with httpx.AsyncClient() as client:
                try:
                    # First try with streaming
                    async with client.stream('POST', 
                        f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                        headers=headers,
                        json={
                            "messages": [{"role": "user", "content": message.content}],
                            # "stream": True
                        }
                    ) as response:
                        if response.status_code == 200:
                            has_content = False
                            async for line in response.aiter_lines():
                                if line.startswith('data: '):
                                    try:
                                        json_str = line[6:]
                                        data = json.loads(json_str)
                                        if 'choices' in data and len(data['choices']) > 0:
                                            delta = data['choices'][0].get('delta', {})
                                            content = delta.get('content', '')
                                            if content:
                                                has_content = True
                                                yield f"data: {json.dumps({'content': content})}\n\n"
                                    except json.JSONDecodeError:
                                        continue
                            
                            # If we reached the end without getting any content, fall back to non-streaming
                            if not has_content:
                                raise Exception("No streaming content received")

                        else:
                            raise Exception("Streaming not supported")

                except Exception as e:
                    print(f"Falling back to non-streaming: {str(e)}")
                    # Fallback to non-streaming request
                    response = await client.post(
                        f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                        headers=headers,
                        json={
                            "messages": [{"role": "user", "content": message.content}]
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0]['message']['content']
                            # Send the full content
                            yield f"data: {json.dumps({'content': content})}\n\n"
                            # Send done event
                            yield "event: done\ndata: {}\n\n"
                    else:
                        raise HTTPException(status_code=response.status_code, 
                                         detail="Error calling Databricks API")

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

@app.post("/api/chats", response_model=ChatHistoryItem)
async def create_chat(chat_request: CreateChatRequest):
    chat_id = str(uuid.uuid4())
    
    # Create a new chat
    chats_db[chat_id] = {
        "id": chat_id,
        "title": chat_request.title,
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    
    return ChatHistoryItem(
        id=chat_id,
        title=chat_request.title,
        messages=[]
    )

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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)