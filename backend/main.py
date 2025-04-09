from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from databricks.sdk import WorkspaceClient
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
import logging
import asyncio
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
    'endpoints': {}  # Format: {'endpoint_name': {'supports_streaming': bool, 'supports_trace': bool, 'last_checked': datetime}}
}

# Models
class MessageRequest(BaseModel):
    content: str

class MessageResponse(BaseModel):
    message_id: str
    content: str
    role: str
    model: str
    timestamp: datetime
    sources: Optional[List[dict]] = None
    metrics: Optional[dict] = None
    isThinking: Optional[bool] = None

class ChatHistoryItem(BaseModel):
    sessionId: str  
    firstQuery: str  
    messages: List[MessageResponse]
    timestamp: datetime
    isActive: bool = True 

class ChatHistoryResponse(BaseModel):
    sessions: List[ChatHistoryItem]

class CreateChatRequest(BaseModel):
    title: str

class ServingEndpoint(BaseModel):
    names: List[str]

class ErrorRequest(BaseModel):
    message_id: str
    content: str
    role: str = "assistant"
    model: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[dict]] = None
    metrics: Optional[dict] = None

class RegenerateRequest(BaseModel):
    message_id: str
    original_content: str

# In-memory storage for chats (in a production app, use a database)
chats_db = {}

# Routes
@app.get("/")
async def root():
    return {"message": "Databricks Chat API is running"}

# Add this function to check and update cache
async def check_endpoint_capabilities(model: str) -> tuple[bool, bool]:
    """
    Check if endpoint supports streaming and trace data.
    Returns (supports_streaming, supports_trace)
    """
    current_time = datetime.now()
    cache_entry = streaming_support_cache['endpoints'].get(model)
    
    # If cache entry exists and is less than 24 hours old, use cached value
    if cache_entry and (current_time - cache_entry['last_checked']) < timedelta(days=1):
        return cache_entry['supports_streaming'], cache_entry['supports_trace']
    
    # Cache expired or doesn't exist - fetch fresh data
    try:
        endpoint = client.serving_endpoints.get(model)
        supports_trace = any(
            entity.name == 'feedback'
            for entity in endpoint.config.served_entities
        )
        
        # Update cache with fresh data
        streaming_support_cache['endpoints'][model] = {
            'supports_streaming': True,
            'supports_trace': supports_trace,
            'last_checked': current_time
        }
        return True, supports_trace
        
    except Exception as e:
        logger.error(f"Error checking endpoint capabilities: {str(e)}")
        # If error occurs, return default values
        return True, False

# First, modify the make_databricks_request function to include timeout
@backoff.on_exception(
    backoff.expo,
    (httpx.HTTPError, httpx.ReadTimeout, httpx.HTTPStatusError),  # Add HTTPStatusError to handle 429
    max_tries=3,
    max_time=30
)
async def make_databricks_request(client: httpx.AsyncClient, url: str, headers: dict, data: dict):
    logger.info(f"Making Databricks request to {url}")
    response = await client.post(url, headers=headers, json=data, timeout=30.0)
    
    # Handle rate limit error specifically
    if response.status_code == 429:
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
            logger.info(f"Rate limited. Waiting {wait_time} seconds before retry.")
            await asyncio.sleep(wait_time)
        raise httpx.HTTPStatusError("Rate limit exceeded", request=response.request, response=response)
    
    return response

async def extract_sources_from_trace(data: dict) -> list:
    """Extract sources from the Databricks API trace data."""
    sources = []
    if data is not None and "databricks_output" in data and "trace" in data["databricks_output"]:
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

async def handle_databricks_response(
    response: httpx.Response,
    start_time: float
) -> tuple[bool, Optional[str]]:
    """
    Handle Databricks API response.
    Returns (success, response_string)
    """
    total_time = time.time() - start_time
    logger.info(f"Response status: {response.status_code}")
    logger.info(f"Response content: {response.json()}")

    if response.status_code == 200:
        data = response.json()
        sources = await extract_sources_from_trace(data)
        logger.info(f"Sources: {sources}")

        if not data:
            return False, None, None
        
        if 'choices' in data and len(data['choices']) > 0:
            content = data['choices'][0]['message']['content']
            response_data = {
                'content': content,
                'sources': sources,
                'metrics': {'totalTime': total_time}
            }
            return True, response_data
        elif 'messages' in data and len(data['messages']) > 0:
            message = data['messages'][0]
            if message.get('role') == 'assistant' and 'content' in message:
                response_data = {
                    'content': message['content'],
                    'sources': sources,
                    'metrics': {'totalTime': total_time}
                }
                return True, response_data
        
             
    else:
        # Handle specific known cases
        error_data = response.json()
        response_data = {
            'content': error_data.get('error_code', 'Encountered an error') + ". " + error_data.get('message', 'Error processing response.'),
            'sources': [],
            'metrics': {'totalTime': total_time}
        }
        return True, response_data
    
    # For other non-200 responses or invalid responses, return False to trigger fallback
    return False, None

# Add these new functions to handle chat sessions
def save_message_to_session(
    session_id: str,
    message: MessageResponse,
    is_first_message: bool = False
) -> None:
    """Save a message to a chat session, creating the session if it doesn't exist"""
    if session_id not in chats_db:
        chats_db[session_id] = {
            "session_id": session_id,
            "first_query": message.content if is_first_message else "",
            "messages": [],
            "timestamp": message.timestamp,
            "is_active": True
        }
    
    chats_db[session_id]["messages"].append(
        message.model_dump(exclude_none=True)
    )
    if is_first_message:
        chats_db[session_id]["first_query"] = message.content

@app.post("/api/error")
async def error(
    error: ErrorRequest,
    model: str = Query(...),
    session_id: str = Query(...)
):
    # Create error message response
    error_message = MessageResponse(
        message_id=error.message_id,
        content=error.content,
        role=error.role,
        model=error.model,
        timestamp=error.timestamp,
        sources=error.sources,
        metrics=error.metrics
    )
    
    # Save to session
    save_message_to_session(session_id, error_message)
    return {"status": "error saved"}

# Modify the chat endpoint to handle sessions
@app.post("/api/chat")
async def chat(
    message: MessageRequest, 
    model: str = Query(...),
    session_id: str = Query(...)
):
    try:
        headers = {
            "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
            "Content-Type": "application/json"
        }

        # Generate assistant message ID once at the start
        assistant_message_id = str(uuid.uuid4())

        # Save user message to session
        user_message = MessageResponse(
            message_id=str(uuid.uuid4()),
            content=message.content,
            role="user",
            model=model,
            timestamp=datetime.now()
        )
        save_message_to_session(
            session_id, 
            user_message,
            is_first_message=len(chats_db.get(session_id, {}).get("messages", [])) == 0
        )

        async def generate():
            streaming_timeout = httpx.Timeout(
                connect=8.0,  # More time to establish connection
                read=30.0,    # More time to read streaming responses
                write=8.0,
                pool=8.0
            )
            regular_timeout = httpx.Timeout(
                connect=5.0,
                read=30.0,
                write=5.0,
                pool=5.0
            )
            supports_streaming, supports_trace = await check_endpoint_capabilities(model)
            request_data = {
                "messages": [{"role": "user", "content": message.content}],
            }
            if supports_trace:
                request_data["databricks_options"] = {"return_trace": True}

            if not supports_streaming:
                async with httpx.AsyncClient(timeout=regular_timeout) as regular_client:
                    try:
                        logger.info("non Streaming is running")
                        start_time = time.time()
                        response = await make_databricks_request(
                            regular_client,
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                            headers=headers,
                            data=request_data
                        )
                        logger.info(f"Response: {response.json()}")
                        success, response_data = await handle_databricks_response(response, start_time)
                        if success and response_data:
                            assistant_message = MessageResponse(
                                message_id=assistant_message_id,
                                content=response_data["content"],
                                role="assistant",
                                model=model,
                                timestamp=datetime.now(),
                                sources=response_data.get("sources"),
                                metrics=response_data.get("metrics")
                            )
                            save_message_to_session(session_id, assistant_message)
                            
                            yield f"data: {json.dumps(response_data)}\n\n"
                            yield "event: done\ndata: {}\n\n"
                
                        else:
                            logger.error("Failed to process Databricks response")
                            error_response = {
                                'message_id': assistant_message_id,
                                'content': 'Failed to process response from model',
                                'sources': [],
                                'metrics': {'totalTime': time.time() - start_time}
                            }
                            
                            assistant_message = MessageResponse(
                                message_id=assistant_message_id,
                                content=error_response["content"],
                                role="assistant",
                                model=model,
                                timestamp=datetime.now(),
                                sources=error_response.get("sources"),
                                metrics=error_response.get("metrics")
                            )
                            save_message_to_session(session_id, assistant_message)
                            yield f"data: {json.dumps(error_response)}\n\n"
                            yield "event: done\ndata: {}\n\n"
                    except Exception as e:
                        logger.error(f"Error in non-streaming response: {str(e)}")
                        assistant_message = MessageResponse(
                            message_id=assistant_message_id,
                            content="Request timed out. Please try again later.",
                            role="assistant",
                            model=model,
                            timestamp=datetime.now(),
                            sources=[],
                            metrics={"totalTime": time.time() - start_time}
                        )
                        save_message_to_session(session_id, assistant_message)
                        yield f"data: {json.dumps({'message_id': assistant_message_id,'content': 'Failed to process response from model', 'metrics': {'totalTime': time.time() - start_time}})}\n\n"
                        yield "event: done\ndata: {}\n\n"
            else:
                async with httpx.AsyncClient(timeout=streaming_timeout) as streaming_client:
                    try:
                        logger.info("streaming is running")
                        request_data["stream"] = True
                        start_time = time.time()
                        first_token_time = None
                        accumulated_content = ""
                        sources = []

                        async with streaming_client.stream('POST', 
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                            headers=headers,
                            json=request_data,
                            timeout=streaming_timeout
                        ) as response:
                            if response.status_code == 200:
                                has_content = False
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
                                                accumulated_content += content
                                                
                                                # Include the same assistant_message_id in each chunk
                                                response_data = {
                                                    'message_id': assistant_message_id,  # Use the pre-generated ID
                                                    'content': content if content else None,
                                                    'sources': sources if sources else None,
                                                    'metrics': {
                                                        'timeToFirstToken': ttft if first_token_time is not None else None,
                                                        'totalTime': time.time() - start_time
                                                    }
                                                }
                                                if content:
                                                    has_content = True
                                                    yield f"data: {json.dumps(response_data)}\n\n"
                                        except json.JSONDecodeError:
                                            continue

                                if has_content:
                                    # Save complete message to session after streaming is done
                                    assistant_message = MessageResponse(
                                        message_id=assistant_message_id,  # Use the same ID here
                                        content=accumulated_content,
                                        role="assistant",
                                        model=model,
                                        timestamp=datetime.now(),
                                        sources=sources,
                                        metrics={'timeToFirstToken': ttft, 'totalTime': time.time() - start_time}
                                    )
                                    save_message_to_session(session_id, assistant_message)
                                    
                                    # Update cache to indicate streaming support
                                    streaming_support_cache['endpoints'][model] = {
                                        'supports_streaming': True,
                                        'supports_trace': supports_trace,
                                        'last_checked': datetime.now()
                                    }
                                    total_time = time.time() - start_time
                                    yield f"data: {json.dumps({'message_id': assistant_message_id,'metrics': {'timeToFirstToken': ttft, 'totalTime': total_time}})}\n\n"
                            else:
                                raise Exception("Streaming not supported")

                    except (httpx.ReadTimeout, httpx.HTTPError, Exception) as e:
                        logger.error(f"Streaming failed with error: {str(e)}, falling back to non-streaming")
                        streaming_support_cache['endpoints'][model].update({
                            'supports_streaming': False,
                            'last_checked': datetime.now()
                        })
                        
                        # Force a delay before fallback to ensure previous connection is terminated
                        await asyncio.sleep(1)
                        
                        # Use a fresh timeout for the fallback
                        fallback_timeout = httpx.Timeout(
                            connect=10.0,
                            read=60.0,
                            write=10.0,
                            pool=10.0
                        )
                        
                        # Only for this fallback request, create a transport with connection pooling disabled
                        # This won't affect your other normal connections
                        fallback_transport = httpx.AsyncHTTPTransport(
                            retries=3,
                            limits=httpx.Limits(max_keepalive_connections=0, max_connections=1)
                        )
                        
                        async with httpx.AsyncClient(
                            timeout=fallback_timeout,
                            transport=fallback_transport  # Use the special transport only for fallback
                        ) as fallback_client:
                            try:
                                # Reset start time for fallback
                                start_time = time.time()
                                # Ensure stream is set to False
                                request_data["stream"] = False
                                # Add a random query parameter to avoid any caching
                                url = f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations?nocache={uuid.uuid4()}"
                                logger.info(f"Making fallback request with fresh connection to {url}")
                                
                                # Make direct request instead of using make_databricks_request
                                response = await fallback_client.post(
                                    url,
                                    headers=headers,
                                    json=request_data,
                                    timeout=fallback_timeout
                                )
                                
                                # Process the response
                                success, response_data = await handle_databricks_response(response, start_time)
                                if success and response_data:
                                    # Save assistant message to session
                                    assistant_message = MessageResponse(
                                        message_id=assistant_message_id,
                                        content=response_data["content"],
                                        role="assistant",
                                        model=model,
                                        timestamp=datetime.now(),
                                        sources=response_data.get("sources"),
                                        metrics=response_data.get("metrics")
                                    )
                                    save_message_to_session(session_id, assistant_message)
                                    
                                    yield f"data: {json.dumps(response_data)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                                else:
                                    logger.error("Failed to process Databricks response")
                                    error_response = {
                                        'message_id': assistant_message_id,
                                        'content': 'Failed to process response from model',
                                        'sources': [],
                                        'metrics': {'totalTime': time.time() - start_time}
                                    }
                                    assistant_message = MessageResponse(
                                        message_id=assistant_message_id,
                                        content=error_response["content"],
                                        role="assistant",
                                        model=model,
                                        timestamp=datetime.now(),
                                        sources=error_response.get("sources"),
                                        metrics=error_response.get("metrics")
                                    )
                                    save_message_to_session(session_id, assistant_message)
                                    yield f"data: {json.dumps(error_response)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                            except httpx.ReadTimeout as timeout_error:
                                logger.error(f"Fallback request failed with timeout: {str(timeout_error)}")
                                assistant_message = MessageResponse(
                                        message_id=assistant_message_id,
                                        content="Request timed out. Please try again later.",
                                        role="assistant",
                                        model=model,
                                        timestamp=datetime.now(),
                                        sources=[],
                                        metrics={"totalTime": time.time() - start_time}
                                    )
                                save_message_to_session(session_id, assistant_message)
                                yield f"data: {json.dumps({'message_id': assistant_message_id,'content': 'Request timed out. Please try again later.'})}\n\n"
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
        logger.error(f"Error calling Databricks API: {str(e)}")
        error_message = "An error occurred while processing your request."
        
        # Handle rate limit errors specifically
        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
            error_message = "The service is currently experiencing high demand. Please wait a moment and try again."
        start_time = time.time()
        error_response = {
            'message_id': assistant_message_id,
            'content': error_message,
            'sources': [],
            'metrics': {'totalTime': time.time() - start_time},
            'timestamp': datetime.now().isoformat()
        }
        
        assistant_message = MessageResponse(
            message_id=assistant_message_id,
            content=error_response["content"],
            role="assistant",
            model=model,
            timestamp=datetime.now(),
            sources=error_response.get("sources"),
            metrics=error_response.get("metrics")
        )
        save_message_to_session(session_id, assistant_message)
        
        async def error_generate():
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "event: done\ndata: {}\n\n"
            
        return StreamingResponse(
            error_generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

@app.get("/api/chats", response_model=ChatHistoryResponse)
async def get_chat_history():
    sessions = [
        ChatHistoryItem(
            sessionId=session_id,
            firstQuery=session_data.get("first_query", "New Chat"),
            messages=[
                MessageResponse(
                    message_id=msg["message_id"],
                    content=msg["content"],
                    role=msg["role"],
                    model=msg.get("model", ""),
                    timestamp=msg.get("timestamp", datetime.now()),
                    sources=msg.get("sources"),
                    metrics=msg.get("metrics"),
                    isThinking=False
                ) 
                for msg in session_data.get("messages", [])
            ],
            timestamp=session_data.get("timestamp", datetime.now()),
            isActive=session_data.get("is_active", True)
        )
        for session_id, session_data in chats_db.items()
    ]
    
    # Sort sessions by timestamp in descending order (newest first)
    sessions.sort(key=lambda x: x.timestamp, reverse=True)
    
    return ChatHistoryResponse(sessions=sessions)


@app.get("/api/chats/{session_id}", response_model=ChatHistoryItem)
async def get_chat(session_id: str):
    if session_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    chat_data = chats_db[session_id]
    
    return ChatHistoryItem(
        sessionId=session_id,
        firstQuery=chat_data["first_query"],
        messages=[
            MessageResponse(
                message_id=msg["message_id"],
                content=msg["content"],
                role=msg["role"],
                model=msg.get("model", ""),
                timestamp=msg.get("timestamp", datetime.now()),
                sources=msg.get("sources"),
                metrics=msg.get("metrics"),
                isThinking=False
            ) 
            for msg in chat_data.get("messages", [])
        ],
        timestamp=chat_data.get("timestamp", datetime.now()),
        isActive=chat_data.get("is_active", True)
    )

        
@app.get("/api/serving-endpoints")
async def list_endpoints():
    try:
        serving_endpoints = []
        endpoints = client.serving_endpoints.list()
        for endpoint in endpoints:
            if endpoint.state.ready == EndpointStateReady.READY:
                # Check if endpoint has a feedback entity
                supports_trace = any(
                    entity.name == 'feedback'
                    for entity in endpoint.config.served_entities
                )
                
                # Update the cache with the endpoint's capabilities
                streaming_support_cache['endpoints'][endpoint.name] = {
                    'supports_streaming': True,
                    'supports_trace': supports_trace,
                    'last_checked': datetime.now()
                }
                
                serving_endpoints.append(endpoint.name)
                
        return ServingEndpoint(names=sorted(serving_endpoints))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching endpoints: {str(e)}")

# Add endpoint to get session messages
@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    if session_id not in chats_db:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": chats_db[session_id]["messages"]}

@app.post("/api/regenerate")
async def regenerate_message(
    request: RegenerateRequest,
    model: str = Query(...),
    session_id: str = Query(...)
):
    headers = {
        "Authorization": f"Bearer {os.getenv('DATABRICKS_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    if session_id not in chats_db:
        logger.error(f"Session {session_id} not found in chats_db")
        raise HTTPException(
            status_code=404, 
            detail=f"Chat session {session_id} not found. Please ensure you're using a valid session ID."
        )
    
    chat_data = chats_db[session_id]
    messages = chat_data["messages"]
    
    message_index = next(
        (i for i, msg in enumerate(messages) 
         if msg["message_id"] == request.message_id), 
        None
    )
    
    if message_index is None:
        logger.error(f"Message {request.message_id} not found in session {session_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"Message {request.message_id} not found in chat session {session_id}."
        )

    async def generate():
        try:
            timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                supports_streaming, supports_trace = await check_endpoint_capabilities(model)
                request_data = {
                    "messages": [{"role": "user", "content": request.original_content}],
                }
                if supports_trace:
                    request_data["databricks_options"] = {"return_trace": True}

                start_time = time.time()
                first_token_time = None
                accumulated_content = ""
                sources = []

                if supports_streaming:
                    request_data["stream"] = True
                    async with client.stream(
                        'POST',
                        f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                        headers=headers,
                        json=request_data,
                        timeout=timeout
                    ) as response:
                        if response.status_code == 200:
                            async for line in response.aiter_lines():
                                if line.startswith('data: '):
                                    try:
                                        data = json.loads(line[6:])
                                        if 'choices' in data and data['choices']:
                                            if first_token_time is None:
                                                first_token_time = time.time()
                                                ttft = first_token_time - start_time

                                            content = data['choices'][0].get('delta', {}).get('content', '')
                                            if content:
                                                accumulated_content += content
                                                current_time = time.time()
                                                response_data = {
                                                    'message_id': request.message_id,
                                                    'content': content,
                                                    'metrics': {
                                                        'timeToFirstToken': ttft,
                                                        'totalTime': current_time - start_time
                                                    }
                                                }
                                                yield f"data: {json.dumps(response_data)}\n\n"
                                    except json.JSONDecodeError:
                                        continue

                            # Final update with complete message
                            total_time = time.time() - start_time
                            messages[message_index] = {
                                "message_id": request.message_id,
                                "content": accumulated_content,
                                "role": "assistant",
                                "model": model,
                                "timestamp": datetime.now().isoformat(),
                                "sources": sources,
                                "metrics": {
                                    "timeToFirstToken": ttft,
                                    "totalTime": total_time
                                }
                            }
                            
                            chats_db[session_id]["messages"] = messages
                            
                            # Send final metrics
                            yield f"data: {json.dumps({'metrics': {'timeToFirstToken': ttft, 'totalTime': total_time}})}\n\n"
                            yield "event: done\ndata: {}\n\n"
                else:
                    # Non-streaming case
                    regen_transport = httpx.AsyncHTTPTransport(
                        retries=3,
                        limits=httpx.Limits(max_keepalive_connections=0, max_connections=1)
                    )
                    async with httpx.AsyncClient(
                        timeout=timeout,
                        transport=regen_transport
                    ) as client:
                        response = await client.post(
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{model}/invocations",
                            headers=headers,
                            json=request_data,
                            timeout=timeout
                        )
                    
                        success, response_data = await handle_databricks_response(response, start_time)
                        if success and response_data:
                            total_time = time.time() - start_time
                            messages[message_index] = {
                                "message_id": request.message_id,
                                "content": response_data["content"],
                                "role": "assistant",
                                "model": model,
                                "timestamp": datetime.now().isoformat(),
                                "sources": response_data.get("sources", []),
                                "metrics": {
                                    "totalTime": total_time
                                }
                            }
                            
                            chats_db[session_id]["messages"] = messages
                            yield f"data: {json.dumps({**response_data, 'message_id': request.message_id})}\n\n"
                            yield "event: done\ndata: {}\n\n"
                    
        except Exception as e:
            logger.error(f"Error in regeneration: {str(e)}")
            total_time = time.time() - start_time
            error_response = {
                "message_id": request.message_id,
                "content": "Failed to regenerate response. Please try again.",
                "sources": [],
                "metrics": {"totalTime": total_time},
                "timestamp": datetime.now().isoformat()
            }
            
            messages[message_index] = {
                "message_id": request.message_id,
                "content": error_response["content"],
                "role": "assistant",
                "model": model,
                "timestamp": error_response["timestamp"],
                "sources": [],
                "metrics": error_response["metrics"]
            }
            
            chats_db[session_id]["messages"] = messages
            yield f"data: {json.dumps(error_response)}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )

@app.post("/api/regenerate/error")
async def regenerate_error(
    error: ErrorRequest,
    model: str = Query(...),
    session_id: str = Query(...)
):
    # Find and update the message in the chat history
    if session_id not in chats_db:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    chat_data = chats_db[session_id]
    messages = chat_data["messages"]
    
    # Find the message to update
    message_index = next(
        (i for i, msg in enumerate(messages) 
         if msg["message_id"] == error.message_id), 
        None
    )
    
    if message_index is None:
        raise HTTPException(status_code=404, detail="Message not found")
    
    # Update the message with error content
    messages[message_index] = {
        "message_id": error.message_id,
        "content": error.content,
        "role": error.role,
        "model": model,
        "timestamp": error.timestamp.isoformat(),
        "sources": error.sources,
        "metrics": error.metrics
    }
    
    # Save updated chat data
    chats_db[session_id]["messages"] = messages
    
    return {"status": "error saved"}

# Example endpoint for user login or session initialization
@app.post("/api/login")
async def login(request: Request):
    # Extract user information from headers
    headers = request.headers
    email = headers.get("X-Forwarded-Email")
    username = headers.get("X-Forwarded-Preferred-Username")
    user = headers.get("X-Forwarded-User")
    ip = headers.get("X-Real-Ip")
    user_access_token = headers.get("X-Forwarded-Access-Token")

    # Store user information in session or database
    user_info = {
        "email": email,
        "username": username,
        "user": user,
        "ip": ip,
        "user_access_token": user_access_token
    }

    # Example response
    return {"message": "User logged in", "user_info": user_info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
