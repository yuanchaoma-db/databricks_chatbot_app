from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse
from typing import Dict, List, Optional
from databricks.sdk import WorkspaceClient
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime, timedelta
import json
import httpx
import backoff
import time  
import logging
import copy
import asyncio
import threading
from chat_database import ChatDatabase
from token_minter import TokenMinter
from collections import defaultdict
from contextlib import asynccontextmanager
from models import MessageRequest, MessageResponse, ChatHistoryItem, ChatHistoryResponse, CreateChatRequest, ErrorRequest, RegenerateRequest
from utils.config import SERVING_ENDPOINT_NAME, DATABRICKS_HOST, CLIENT_ID, CLIENT_SECRET
from utils.chat_history_cache import ChatHistoryCache
from utils.request_handler import RequestHandler
from utils.error_handler import ErrorHandler
from utils.message_handler import MessageHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(request_handler.request_worker())
    yield

app = FastAPI(lifespan=lifespan)
#ui_app = StaticFiles(directory="frontend/build", html=True)
api_app = FastAPI()
app.mount("/chat-api", api_app)
#app.mount("/", ui_app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize token minter
token_minter = TokenMinter(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    host=DATABRICKS_HOST
)


# Cache to track streaming support for endpoints
streaming_support_cache = {
    'last_updated': datetime.now(),
    'endpoints': {}  # Format: {'endpoint_name': {'supports_streaming': bool, 'supports_trace': bool, 'last_checked': datetime}}
}
chat_history_cache = ChatHistoryCache()
chat_db = ChatDatabase()
message_handler = MessageHandler(chat_db, chat_history_cache)
error_handler = ErrorHandler(message_handler)

request_handler = RequestHandler(SERVING_ENDPOINT_NAME)
streaming_semaphore = request_handler.streaming_semaphore

async def get_user_info(request: Request = None) -> dict:
    """Get user information from request headers"""
    if not request:
        # For testing purposes, return test user info
        return {
            "email": "test@databricks.com",
            "user_id": "test_user1",
            "username": "test_user1"
        }
    
    # user_info = {
    #     "email": request.headers.get("X-Forwarded-Email"),
    #     "user_id": request.headers.get("X-Forwarded-User"),
    #     "username": request.headers.get("X-Forwarded-Preferred-Username", "").split("@")[0]
    # }
    user_info = {
        "email": "test@databricks.com",
        "user_id": "test_user1",
        "username": "test_user1"
    }
    if not user_info["user_id"]:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return user_info

# Dependency to get auth headers
async def get_auth_headers() -> dict:
    token = token_minter.get_token()
    return {"Authorization": f"Bearer {token}"}
    

# Routes
@api_app.get("/")
async def root():
    return {"message": "Databricks Chat API is running"}

# Add this function to check and update cache
async def check_endpoint_capabilities(model: str) -> tuple[bool, bool]:
    """
    Check if endpoint supports streaming and trace data.
    Returns (supports_streaming, supports_trace)
    """
    client = WorkspaceClient()
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

# Global request queue (maxsize can be adjusted)
request_queue = request_handler.request_queue

@api_app.post("/error")
async def error(
    error: ErrorRequest,
    user_info: dict = Depends(get_user_info)
):
    await error_handler.handle_error_endpoint(error, user_info)

@api_app.get("/model")
async def get_model():
    return {"model": SERVING_ENDPOINT_NAME}

# Modify the chat endpoint to handle sessions
@api_app.post("/chat")
async def chat(
    message: MessageRequest,
    request: Request,
    user_info: dict = Depends(get_user_info),
    headers: dict = Depends(get_auth_headers)
):
    try:
        user_id = user_info["user_id"]
        is_first_message = chat_db.is_first_message(message.session_id, user_id)
        
        # Load chat history with caching
        chat_history = await load_chat_history(message.session_id, user_id, is_first_message)
        
        # Save user message to session with user info
        user_message = message_handler.create_message(
            message_id=str(uuid.uuid4()),
            content=message.content,
            role="user",
            session_id=message.session_id,
            user_id=user_id,
            user_info=user_info,
            is_first_message=is_first_message
        )

        async def generate():
            streaming_timeout = httpx.Timeout(
                connect=8.0,
                read=30.0,
                write=8.0,
                pool=8.0
            )
            supports_streaming, supports_trace = await check_endpoint_capabilities(SERVING_ENDPOINT_NAME)
            if message.include_history:
                request_data = {
                    "messages": [{"role": msg["role"], "content": msg["content"]} for msg in chat_history] + 
                            [{"role": "user", "content": message.content}],
                }
            else:
                request_data = {
                    "messages": [{"role": "user", "content": message.content}]
                }
            if supports_trace:
                request_data["databricks_options"] = {"return_trace": True}

            if not supports_streaming:
                try:
                    logger.info("non Streaming is running")
                    start_time = time.time()
                    response = await request_handler.enqueue_request(
                        f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                        headers=headers,
                        data=request_data
                    )
                    response_data = await request_handler.handle_databricks_response(response, start_time)
                    
                    assistant_message = message_handler.create_message(
                        message_id=str(uuid.uuid4()),
                        content=response_data["content"],
                        role="assistant",
                        session_id=message.session_id,
                        user_id=user_id,
                        user_info=user_info,
                        sources=response_data.get("sources"),
                        metrics=response_data.get("metrics")
                    )
                    
                    # Include message_id in response_data
                    yield f"data: {json.dumps(assistant_message.model_dump_json())}\n\n"
                    yield "event: done\ndata: {}\n\n"
                except Exception as e:
                    logger.error(f"Error in non-streaming response: {str(e)}")
                    
                    error_message = message_handler.create_error_message(
                        session_id=message.session_id,
                        user_id=user_id,
                        error_content="Request timed out. Please try again later."

                    )
                    yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
                    yield "event: done\ndata: {}\n\n"
            else:
                async with streaming_semaphore:
                    async with httpx.AsyncClient(timeout=streaming_timeout) as streaming_client:
                        try:
                            logger.info("streaming is running")
                            request_data["stream"] = True
                            assistant_message_id = str(uuid.uuid4())
                            start_time = time.time()
                            first_token_time = None
                            accumulated_content = ""
                            sources = None
                            ttft = None

                            async with streaming_client.stream('POST', 
                                f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                                headers=headers,
                                json=request_data,
                                timeout=streaming_timeout
                            ) as response:
                                if response.status_code == 200:
                                    async for line in response.aiter_lines():
                                        if line.startswith('data: '):
                                            try:
                                                json_str = line[6:]
                                                data = json.loads(json_str)
                                                
                                                # Record time of first token
                                                if first_token_time is None and 'choices' in data and len(data['choices']) > 0:
                                                    first_token_time = time.time()
                                                    ttft = first_token_time - start_time
                                                    
                                                if 'choices' in data and len(data['choices']) > 0:
                                                    delta = data['choices'][0].get('delta', {})
                                                    content = delta.get('content', '')
                                                    accumulated_content += content
                                                    # Extract sources if this is the final message containing databricks_options
                                                    if 'databricks_output' in data:
                                                        sources = await request_handler.extract_sources_from_trace(data)
                                                    # Include the same assistant_message_id in each chunk
                                                    response_data = create_response_data(
                                                        assistant_message_id,
                                                        content if content else None,
                                                        sources,
                                                        ttft if first_token_time is not None else None,
                                                        time.time() - start_time
                                                    )
                                                    
                                                    yield f"data: {json.dumps(response_data)}\n\n"
                                                if "delta" in data:
                                                    delta = data["delta"]
                                                    if delta["role"] == "assistant" and delta.get("content"):
                                                        content = delta['content']
                                                        accumulated_content += content
                                                        response_data = create_response_data(
                                                            assistant_message_id,
                                                            content+"\n\n",
                                                            sources,
                                                            ttft if first_token_time is not None else None,
                                                            time.time() - start_time
                                                        )
                                                        yield f"data: {json.dumps(response_data)}\n\n"    
                                            except json.JSONDecodeError:
                                                continue
                                    assistant_message = message_handler.create_message(
                                        message_id=assistant_message_id,
                                        content=accumulated_content,
                                        role="assistant",
                                        session_id=message.session_id,
                                        user_id=user_id,
                                        user_info=user_info,
                                        sources=sources,
                                        metrics={'timeToFirstToken': ttft, 'totalTime': time.time() - start_time}
                                    )
                                    # Update cache to indicate streaming support
                                    streaming_support_cache['endpoints'][SERVING_ENDPOINT_NAME] = {
                                        'supports_streaming': True,
                                        'supports_trace': supports_trace,
                                        'last_checked': datetime.now()
                                    }
                                    # Send final message with complete content and sources
                                    final_response = {
                                        'message_id': assistant_message_id,
                                        'sources': sources,
                                    }
                                    yield f"data: {json.dumps(final_response)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                                    
                                        
                                else:
                                    print("streaming not supported")
                                    raise Exception("Streaming not supported")

                        except (httpx.ReadTimeout, httpx.HTTPError, Exception) as e:
                            logger.error(f"Streaming failed with error: {str(e)}, falling back to non-streaming")
                            if SERVING_ENDPOINT_NAME in streaming_support_cache['endpoints']:
                                streaming_support_cache['endpoints'][SERVING_ENDPOINT_NAME].update({
                                    'supports_streaming': False,
                                    'last_checked': datetime.now()
                                })
                            try:
                                # Reset start time for fallback
                                start_time = time.time()
                                # Ensure stream is set to False
                                request_data["stream"] = False
                                # Add a random query parameter to avoid any caching
                                url = f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations?nocache={uuid.uuid4()}"
                                logger.info(f"Making fallback request with fresh connection to {url}")
                                
                                # Make direct request instead of using make_databricks_request
                                response = await request_handler.enqueue_request(url, headers, request_data)
                                
                                # Process the response
                                response_data = await request_handler.handle_databricks_response(response, start_time)
                                assistant_message = message_handler.create_message(
                                    message_id=str(uuid.uuid4()),
                                    content=response_data["content"],
                                    role="assistant",
                                    session_id=message.session_id,
                                    user_id=user_id,
                                    user_info=user_info,
                                    sources=response_data.get("sources"),
                                    metrics=response_data.get("metrics")
                                )

                                yield f"data: {json.dumps(assistant_message.model_dump_json())}\n\n"
                                yield "event: done\ndata: {}\n\n"
                                
                            except httpx.ReadTimeout as timeout_error:
                                logger.error(f"Fallback request failed with timeout: {str(timeout_error)}")
                                error_message = message_handler.create_error_message(
                                    session_id=message.session_id,
                                    user_id=user_id,
                                    error_content="Request timed out. Please try again later."
                                )

                                yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
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
        
        error_message = message_handler.create_error_message(
            session_id=message.session_id,
            user_id=user_id,
            error_content=error_message
        )
        
        async def error_generate():
            yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
            yield "event: done\ndata: {}\n\n"
            
        return StreamingResponse(
            error_generate(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

@api_app.get("/chats", response_model=ChatHistoryResponse)
async def get_chat_history(user_info: dict = Depends(get_user_info)):
    user_id = user_info["user_id"]
    return chat_db.get_chat_history(user_id)

@api_app.get("/chats/{session_id}", response_model=ChatHistoryItem)
async def get_chat(session_id: str, user_info: dict = Depends(get_user_info)):
    user_id = user_info["user_id"]
    return chat_db.get_chat(session_id, user_id)

@api_app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user_info: dict = Depends(get_user_info)):
    try:
        user_id = user_info["user_id"]
        chat_data = chat_db.get_chat(session_id, user_id)
        if not chat_data:
            logger.error(f"Session {session_id} not found in database")
            raise HTTPException(
                status_code=404, 
                detail=f"Chat session {session_id} not found. Please ensure you're using a valid session ID."
            )
        
        return {"messages": chat_data.messages}
        
    except Exception as e:
        logger.error(f"Error getting session messages: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving session messages."
        )

@api_app.post("/regenerate")
async def regenerate_message(
    request: RegenerateRequest,
    user_info: dict = Depends(get_user_info),
    headers: dict = Depends(get_auth_headers)
):
    try:
        user_id = user_info["user_id"]
        chat_history = await load_chat_history(request.session_id, user_id, False)
        message_index = next(
            (i for i, msg in enumerate(chat_history) 
             if msg.get('message_id') == request.message_id), 
            None
        )
        if message_index is None:
            logger.error(f"Message {request.message_id} not found in session {request.session_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Message {request.message_id} not found in chat session {request.session_id}."
            )

        original_message = chat_history[message_index]
        original_timestamp = original_message.get('timestamp')
        if not original_timestamp:
            original_timestamp = datetime.now().isoformat()

        history_up_to_message = chat_history[:message_index]

        async def generate():
            try:
                timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    supports_streaming, supports_trace = await check_endpoint_capabilities(SERVING_ENDPOINT_NAME)
                    if request.include_history:
                        request_data = {
                            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in history_up_to_message]
                        }
                    else:
                        request_data = {
                            "messages": [{"role": "user", "content": history_up_to_message[-1]["content"]}]
                        }
                    print("request_data====>", request_data)
                    if supports_trace:
                        request_data["databricks_options"] = {"return_trace": True}

                    start_time = time.time()
                    first_token_time = None
                    accumulated_content = ""
                    sources = None
                    ttft = None
                    if supports_streaming:
                        request_data["stream"] = True
                        async with streaming_semaphore:
                            async with client.stream(
                                'POST',
                                f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
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
                                                    # Extract sources if this is the final message containing databricks_output
                                                    if 'databricks_output' in data:
                                                        sources = await request_handler.extract_sources_from_trace(data)
                                                    
                                                    accumulated_content += content
                                                    current_time = time.time()
                                                    response_data = create_response_data(
                                                        request.message_id,
                                                        content,
                                                        sources,
                                                        ttft,
                                                        current_time - start_time,
                                                        original_timestamp
                                                    )
                                                    yield f"data: {json.dumps(response_data)}\n\n"

                                                if "delta" in data:
                                                    delta = data["delta"]
                                                    if delta["role"] == "assistant" and delta.get("content"):
                                                        content = delta['content']
                                                        accumulated_content += content
                                                        response_data = create_response_data(
                                                            request.message_id,
                                                            content,
                                                            sources,
                                                            ttft if first_token_time is not None else None,
                                                            time.time() - start_time,
                                                            original_timestamp
                                                        )
                                                        yield f"data: {json.dumps(response_data)}\n\n"    
                                            except json.JSONDecodeError:
                                                continue

                                    # Final update with complete message
                                    updated_message = message_handler.update_message(
                                        session_id=request.session_id,
                                        message_id=request.message_id,
                                        user_id=user_id,
                                        content=accumulated_content,
                                        sources=sources,
                                        timestamp=original_timestamp,
                                        metrics={
                                            "timeToFirstToken": ttft,
                                            "totalTime": time.time() - start_time
                                        }
                                    )
                                    
                                    final_response = {
                                            'message_id': request.message_id,
                                            'sources': sources
                                        }
                                    yield f"data: {json.dumps(final_response)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                    else:
                        response = await request_handler.enqueue_request(f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations", headers, request_data)
                        response_data = await request_handler.handle_databricks_response(response, start_time)
                        updated_message = message_handler.update_message(
                            session_id=request.session_id,
                            message_id=request.message_id,
                            user_id=user_id,
                            content=response_data["content"],
                            sources=response_data.get("sources", []),
                            timestamp=original_timestamp,
                            metrics={
                                "totalTime": time.time() - start_time
                            }
                        )
                        
                        yield f"data: {updated_message.model_dump_json()}\n\n"
                        yield "event: done\ndata: {}\n\n"
                    
            except Exception as e:
                logger.error(f"Error in regeneration: {str(e)}")
                error_message = message_handler.update_message(
                    session_id=request.session_id,
                    message_id=request.message_id,
                    user_id=user_id,
                    content="Failed to regenerate response. Please try again.",
                    sources=[],
                    timestamp=original_timestamp,
                    metrics=None)
                
                
                yield f"data: {error_message.model_dump_json()}\n\n"
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
        logger.error(f"Error in regenerate_message endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while regenerating the message."
        )

@api_app.post("/regenerate/error")
async def regenerate_error(
    error: ErrorRequest,
    user_info: dict = Depends(get_user_info)
):
    return await error_handler.handle_error_endpoint(error, user_info)

# Add new endpoint for rating messages
@api_app.post("/messages/{message_id}/rate")
async def rate_message(
    message_id: str,
    rating: str | None = Query(..., regex="^(up|down)$"),
    user_info: dict = Depends(get_user_info)
):
    try:
        user_id = user_info["user_id"]
        success = chat_db.update_message_rating(message_id, user_id, rating)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Message {message_id} not found"
            )
        return {"status": "success", "rating": rating}
    except Exception as e:
        logger.error(f"Error rating message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while rating the message"
        )

# Add logout endpoint
@api_app.get("/logout")
async def logout():
    return RedirectResponse(url=f"https://{os.getenv('DATABRICKS_HOST')}/login.html", status_code=303)

@api_app.get("/login")
async def login(user_info: dict = Depends(get_user_info)):
    return {"user_info": user_info}

async def load_chat_history(session_id: str, user_id: str, is_first_message: bool) -> List[Dict]:
    """
    Load chat history with caching mechanism.
    Returns chat history in cache format.
    """
    # Try to get from cache first
    chat_history = copy.deepcopy(chat_history_cache.get_history(session_id))
    if chat_history:
        chat_history = convert_messages_to_cache_format(chat_history.messages)
    # If cache is empty and not first message, load from database
    elif not chat_history and not is_first_message:
        chat_data = chat_db.get_chat(session_id, user_id)
        if chat_data and chat_data.messages:
            # Convert to cache format
            chat_history = convert_messages_to_cache_format(chat_data.messages)
            # Store in cache
            for msg in chat_history:
                chat_history_cache.add_message(session_id, msg)
    
    return chat_history or []

def convert_messages_to_cache_format(messages: List) -> List[Dict]:
    """
    Convert database messages to cache format.
    Returns last 20 messages in cache format.
    """
    if not messages:
        return []
    
    return [
        {
            "role": msg.role,
            "content": msg.content,
            "message_id": msg.message_id,
            "timestamp": msg.timestamp.isoformat() if isinstance(msg.timestamp, datetime) else msg.timestamp
        } 
        for msg in messages[-20:]
    ]

def create_response_data(
    message_id: str,
    content: str,
    sources: Optional[List],
    ttft: Optional[float],
    total_time: float,
    timestamp: Optional[str] = None
) -> Dict:
    """Create standardized response data for both streaming and non-streaming responses."""
    # Convert content to string if it's a dictionary
    if isinstance(content, dict):
        content = content.get('content', '')
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    # Create response data
    response_data = {
        'message_id': message_id,
        'content': content,
        'sources': sources if sources else None,
        'metrics': {
            'timeToFirstToken': ttft,
            'totalTime': total_time
        }
    }
    
    # Add timestamp if provided
    if timestamp:
        response_data['timestamp'] = timestamp
        
    # Convert any datetime objects in the response to strings
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
