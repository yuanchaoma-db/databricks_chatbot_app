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
from databricks.sdk.service.serving import EndpointStateReady
import backoff
import time  
import logging
import asyncio
import threading
from chat_database import ChatDatabase
from token_minter import TokenMinter
from collections import defaultdict
from contextlib import asynccontextmanager
from models import MessageRequest, MessageResponse, ChatHistoryItem, ChatHistoryResponse, CreateChatRequest, ErrorRequest, RegenerateRequest
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(request_worker())
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

# Constants
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")
assert SERVING_ENDPOINT_NAME, "SERVING_ENDPOINT_NAME is not set"

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")
CLIENT_ID = os.environ.get("DATABRICKS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("DATABRICKS_CLIENT_SECRET")

# Initialize token minter
token_minter = TokenMinter(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    host=DATABRICKS_HOST
)

# Initialize Databricks workspace client
client = WorkspaceClient()
# Cache to track streaming support for endpoints
streaming_support_cache = {
    'last_updated': datetime.now(),
    'endpoints': {}  # Format: {'endpoint_name': {'supports_streaming': bool, 'supports_trace': bool, 'last_checked': datetime}}
}

chat_db = ChatDatabase()

class ChatHistoryCache:
    """In-memory cache for chat history"""
    def __init__(self):
        self.cache: Dict[str, List[Dict]] = defaultdict(list)
        self.lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Dict]:
        """Get chat history from cache"""
        with self.lock:
            return self.cache[session_id]

    def add_message(self, session_id: str, message: Dict):
        """Add a message to the cache"""
        with self.lock:
            # Add created_at if not present
            if 'created_at' not in message:
                message['created_at'] = datetime.now().isoformat()
            if 'timestamp' not in message:
                message['timestamp'] = message['created_at']
            
            self.cache[session_id].append(message)
            # Keep only last 10 messages
            if len(self.cache[session_id]) > 10:
                self.cache[session_id] = self.cache[session_id][-10:]

    def clear_session(self, session_id: str):
        """Clear a session from cache"""
        with self.lock:
            if session_id in self.cache:
                del self.cache[session_id]

    def update_message(self, session_id: str, message_id: str, new_content: str):
        """Update a message in the cache while preserving order"""
        with self.lock:
            messages = self.cache[session_id]
            for msg in messages:
                if msg.get('message_id') == message_id:
                    # Preserve the original created_at and position
                    msg['content'] = new_content
                    # Update timestamp but keep created_at
                    msg['timestamp'] = datetime.now().isoformat()
                    break

# Initialize the cache
chat_history_cache = ChatHistoryCache()

# Limit to 3 concurrent streaming requests
streaming_semaphore = asyncio.Semaphore(10)

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
request_queue = asyncio.Queue(maxsize=100)

# Worker to process requests from the queue
async def request_worker():
    while True:
        try:
            fut, args, kwargs = await request_queue.get()
            try:
                result = await make_databricks_request(*args, **kwargs)
                if not fut.done():
                    fut.set_result(result)
            except Exception as e:
                logger.error(f"Error in request worker: {str(e)}")
                if not fut.done():
                    fut.set_exception(e)
            finally:
                request_queue.task_done()
        except Exception as e:
            logger.error(f"Critical error in request worker: {str(e)}")
            await asyncio.sleep(1)  # Add delay before retrying

async def enqueue_request(*args, **kwargs):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put((fut, args, kwargs))
    try:
        return await fut
    except Exception as e:
        logger.error(f"Error in enqueue_request: {str(e)}")
        raise

# First, modify the make_databricks_request function to include timeout
@backoff.on_exception(
    backoff.expo,
    (httpx.HTTPError, httpx.ReadTimeout, httpx.HTTPStatusError, RuntimeError),  # Add RuntimeError for client closed
    max_tries=3,
    max_time=30
)
async def make_databricks_request(client: httpx.AsyncClient, url: str, headers: dict, data: dict):
    try:
        logger.info(f"Making Databricks request to {url}")
        if client.is_closed:
            raise RuntimeError("Client is closed, creating new client")
        
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
    except RuntimeError as e:
        logger.error(f"Client error: {str(e)}")
        # Create a new client and retry
        async with httpx.AsyncClient(timeout=30.0) as new_client:
            response = await new_client.post(url, headers=headers, json=data, timeout=30.0)
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

    if response.status_code == 200:
        data = response.json()
        sources = await extract_sources_from_trace(data)

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
            'metrics': None
        }
        return True, response_data
    

@api_app.post("/error")
async def error(
    error: ErrorRequest,
    user_info: dict = Depends(get_user_info)
):
    user_id = user_info["user_id"]
    
    # Get the chat session from database
    chat_data = chat_db.get_chat(error.session_id, user_id)
    if not chat_data:
        logger.error(f"Session {error.session_id} not found in database")
        raise HTTPException(
            status_code=404, 
            detail=f"Chat session {error.session_id} not found. Please ensure you're using a valid session ID."
        )
    
    # Check if this is a new error message or updating an existing one
    is_new_error = not any(msg.message_id == error.message_id for msg in chat_data.messages)
    
    if is_new_error:
        # Create new error message
        error_message = MessageResponse(
            message_id=str(uuid.uuid4()),  # Generate new ID for new error
            content=error.content,
            role=error.role,
            model=SERVING_ENDPOINT_NAME,
            timestamp=error.timestamp,
            sources=error.sources,
            metrics=error.metrics
        )
        # Save new error message to database
        chat_db.save_message_to_session(error.session_id, user_id, error_message)
        # Add to cache
        chat_history_cache.add_message(error.session_id, {
            "role": error.role,
            "content": error.content,
            "message_id": error_message.message_id,
            "timestamp": error.timestamp
        })
    else:
        # Update existing message
        error_message = MessageResponse(
            message_id=error.message_id,  # Use existing message ID
            content=error.content,
            role=error.role,
            model=SERVING_ENDPOINT_NAME,
            timestamp=error.timestamp,
            sources=error.sources,
            metrics=error.metrics
        )
        # Update message in database
        chat_db.update_message(error.session_id, user_id, error_message)
        # Update in cache
        chat_history_cache.update_message(error.session_id, error.message_id, error.content)
    
    return {"status": "error saved", "message_id": error_message.message_id}

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
        chat_history = chat_history_cache.get_history(message.session_id)

        # If cache is empty and not first message, load from database
        if not chat_history and not is_first_message:
            chat_data = chat_db.get_chat(message.session_id, user_id)
            if chat_data and chat_data.messages:
                # Convert to cache format and store
                chat_history = [
                    {"role": msg.role, "content": msg.content, "message_id": msg.message_id}
                    for msg in chat_data.messages[-10:]
                ]
                for msg in chat_history:
                    chat_history_cache.add_message(message.session_id, msg)

        # Save user message to session with user info
        user_message = MessageResponse(
            message_id=str(uuid.uuid4()),
            content=message.content,
            role="user",
            model=SERVING_ENDPOINT_NAME,
            timestamp=datetime.now(),
            created_at=datetime.now()
        )
        chat_db.save_message_to_session(
            message.session_id,
            user_id,
            user_message,
            user_info=user_info,
            is_first_message=is_first_message
        )

        # Add current message to cache
        chat_history_cache.add_message(message.session_id, {
            "role": "user",
            "content": message.content,
            "message_id": user_message.message_id,
            "created_at": user_message.timestamp
        })

        async def generate():
            streaming_timeout = httpx.Timeout(
                connect=8.0,
                read=30.0,
                write=8.0,
                pool=8.0
            )
            regular_timeout = httpx.Timeout(
                connect=5.0,
                read=30.0,
                write=5.0,
                pool=5.0
            )
            supports_streaming, supports_trace = await check_endpoint_capabilities(SERVING_ENDPOINT_NAME)
            request_data = {
                "messages": [{"role": msg["role"], "content": msg["content"]} for msg in chat_history] + 
                           [{"role": "user", "content": message.content}],
            }
            if supports_trace:
                request_data["databricks_options"] = {"return_trace": True}

            if not supports_streaming:
                async with httpx.AsyncClient(timeout=regular_timeout) as regular_client:
                    try:
                        logger.info("non Streaming is running")
                        start_time = time.time()
                        response = await enqueue_request(
                            regular_client,
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                            headers=headers,
                            data=request_data
                        )
                        success, response_data = await handle_databricks_response(response, start_time)
                        if success and response_data:
                            assistant_message_id = str(uuid.uuid4())
                            assistant_message = MessageResponse(
                                message_id=assistant_message_id,
                                content=response_data["content"],
                                role="assistant",
                                model=SERVING_ENDPOINT_NAME,
                                timestamp=datetime.now(),
                                sources=response_data.get("sources"),
                                metrics=response_data.get("metrics")
                            )
                            chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                            
                            # Add assistant message to cache
                            chat_history_cache.add_message(message.session_id, {
                                "role": "assistant",
                                "content": response_data["content"],
                                "message_id": assistant_message_id,
                                "created_at": assistant_message.timestamp
                            })
                            
                            # Include message_id in response_data
                            yield f"data: {json.dumps(assistant_message.model_dump_json())}\n\n"
                            yield "event: done\ndata: {}\n\n"
                
                        else:
                            logger.error("Failed to process Databricks response")
                            error_message_id = str(uuid.uuid4())
                            error_response = {
                                'message_id': error_message_id,
                                'content': 'Failed to process response from model',
                                'sources': [],
                                'metrics': None
                            }
                            
                            error_message = MessageResponse(
                                message_id=error_message_id,
                                content=error_response["content"],
                                role="assistant",
                                model=SERVING_ENDPOINT_NAME,
                                timestamp=datetime.now(),
                                sources=error_response.get("sources"),
                                metrics=error_response.get("metrics")
                            )
                            chat_history_cache.add_message(message.session_id, {
                                "role": "assistant",
                                "content": error_response["content"],
                                "message_id": error_message_id,
                                "created_at": datetime.now()
                            })
                            chat_db.save_message_to_session(message.session_id, user_id, error_message)
                            yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
                            yield "event: done\ndata: {}\n\n"
                    except Exception as e:
                        logger.error(f"Error in non-streaming response: {str(e)}")
                        error_message_id = str(uuid.uuid4())
                        error_message = MessageResponse(
                            message_id=error_message_id,
                            content="Request timed out. Please try again later.",
                            role="assistant",
                            model=SERVING_ENDPOINT_NAME,
                            timestamp=datetime.now(),
                            sources=[],
                            metrics=None
                        )
                        chat_db.save_message_to_session(message.session_id, user_id, error_message)
                        
                        # Add error message to cache
                        chat_history_cache.add_message(message.session_id, {
                            "role": "assistant",
                            "content": "Request timed out. Please try again later.",
                            "message_id": error_message_id,
                            "created_at": datetime.now()
                        })
                        
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

                            async with streaming_client.stream('POST', 
                                f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
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
                                                    
                                                if 'choices' in data and len(data['choices']) > 0:
                                                    delta = data['choices'][0].get('delta', {})
                                                    content = delta.get('content', '')
                                                    accumulated_content += content
                                                    # Extract sources if this is the final message containing databricks_options
                                                    if 'databricks_output' in data:
                                                        sources = await extract_sources_from_trace(data)
                                                    # Include the same assistant_message_id in each chunk
                                                    response_data = {
                                                        'message_id': assistant_message_id,
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
                                            message_id=assistant_message_id,
                                            content=accumulated_content,
                                            role="assistant",
                                            model=SERVING_ENDPOINT_NAME,
                                            timestamp=datetime.now(),
                                            sources=sources,
                                            metrics={'timeToFirstToken': ttft, 'totalTime': time.time() - start_time}
                                        )
                                        chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                                        
                                        # Add assistant message to cache
                                        chat_history_cache.add_message(message.session_id, {
                                            "role": "assistant",
                                            "content": accumulated_content,
                                            "message_id": assistant_message_id
                                        })
                                        
                                        # Update cache to indicate streaming support
                                        streaming_support_cache['endpoints'][SERVING_ENDPOINT_NAME] = {
                                            'supports_streaming': True,
                                            'supports_trace': supports_trace,
                                            'last_checked': datetime.now()
                                        }
                                        yield "event: done\ndata: {}\n\n"
                                else:
                                    raise Exception("Streaming not supported")

                        except (httpx.ReadTimeout, httpx.HTTPError, Exception) as e:
                            logger.error(f"Streaming failed with error: {str(e)}, falling back to non-streaming")
                            if SERVING_ENDPOINT_NAME in streaming_support_cache['endpoints']:
                                streaming_support_cache['endpoints'][SERVING_ENDPOINT_NAME].update({
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
                            fallback_transport = httpx.AsyncHTTPTransport(
                                retries=3,
                                limits=httpx.Limits(max_keepalive_connections=0, max_connections=1)
                            )
                            
                            async with httpx.AsyncClient(
                                timeout=fallback_timeout,
                                transport=fallback_transport
                            ) as fallback_client:
                                try:
                                    # Reset start time for fallback
                                    start_time = time.time()
                                    # Ensure stream is set to False
                                    request_data["stream"] = False
                                    # Add a random query parameter to avoid any caching
                                    url = f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations?nocache={uuid.uuid4()}"
                                    logger.info(f"Making fallback request with fresh connection to {url}")
                                    
                                    # Make direct request instead of using make_databricks_request
                                    response = await enqueue_request(fallback_client, url, headers, request_data)
                                    
                                    # Process the response
                                    success, response_data = await handle_databricks_response(response, start_time)
                                    if success and response_data:
                                        # Save assistant message to session
                                        assistant_message = MessageResponse(
                                            message_id=str(uuid.uuid4()),
                                            content=response_data["content"],
                                            role="assistant",
                                            model=SERVING_ENDPOINT_NAME,
                                            timestamp=datetime.now(),
                                            sources=response_data.get("sources"),
                                            metrics=response_data.get("metrics")
                                        )
                                        chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                                        
                                        # Add assistant message to cache
                                        chat_history_cache.add_message(message.session_id, {
                                            "role": "assistant",
                                            "content": response_data["content"],
                                            "message_id": assistant_message.message_id
                                        })
                                        
                                        yield f"data: {json.dumps(assistant_message.model_dump_json())}\n\n"
                                        yield "event: done\ndata: {}\n\n"
                                    else:
                                        logger.error("Failed to process Databricks response")
                                        error_message_id = str(uuid.uuid4())
                                        error_response = {
                                            'message_id': error_message_id,
                                            'content': 'Failed to process response from model',
                                            'sources': [],
                                            'metrics': None
                                        }
                                        error_message = MessageResponse(
                                            message_id=error_message_id,
                                            content=error_response["content"],
                                            role="assistant",
                                            model=SERVING_ENDPOINT_NAME,
                                            timestamp=datetime.now(),
                                            sources=error_response.get("sources"),
                                            metrics=error_response.get("metrics")
                                        )
                                        chat_db.save_message_to_session(message.session_id, user_id, error_message)
                                        
                                        # Add error message to cache with the original message ID
                                        chat_history_cache.add_message(message.session_id, {
                                            "role": "assistant",
                                            "content": error_response["content"],
                                            "message_id": error_message_id
                                        })
                                        
                                        yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
                                        yield "event: done\ndata: {}\n\n"
                                except httpx.ReadTimeout as timeout_error:
                                    logger.error(f"Fallback request failed with timeout: {str(timeout_error)}")
                                    error_message_id = str(uuid.uuid4())
                                    error_message = MessageResponse(
                                        message_id=error_message_id,
                                        content="Request timed out. Please try again later.",
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=[],
                                        metrics={"totalTime": time.time() - start_time}
                                    )
                                    chat_db.save_message_to_session(message.session_id, user_id, error_message)
                                    
                                    # Add error message to cache with the original message ID
                                    chat_history_cache.add_message(message.session_id, {
                                        "role": "assistant",
                                        "content": "Request timed out. Please try again later.",
                                        "message_id": error_message_id
                                    })
                                    
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
        start_time = time.time()
        error_message_id = str(uuid.uuid4())
        error_response = {
            'message_id': error_message_id,
            'content': error_message,
            'sources': [],
            'metrics': None,
            'timestamp': datetime.now().isoformat()
        }
        
        error_message = MessageResponse(
            message_id=error_message_id,
            content=error_response["content"],
            role="assistant",
            model=SERVING_ENDPOINT_NAME,
            timestamp=datetime.now(),
            sources=error_response.get("sources"),
            metrics=error_response.get("metrics")
        )
        chat_history_cache.add_message(message.session_id, {
            "role": "assistant",
            "content": error_response["content"],
            "message_id": error_message_id,
            "created_at": datetime.now()
        })
        chat_db.save_message_to_session(message.session_id, user_id, error_message)
        
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
        # Get chat history from cache
        chat_history = chat_history_cache.get_history(request.session_id)
        logger.info(f"Initial cache state for session {request.session_id}: {chat_history}")
        
        # If cache is empty, load from database
        if not chat_history:
            chat_data = chat_db.get_chat(request.session_id, user_id)
            if chat_data and chat_data.messages:
                chat_history = [
                    {"role": msg.role, "content": msg.content, "message_id": msg.message_id}
                    for msg in chat_data.messages[-10:]
                ]
                for msg in chat_history:
                    chat_history_cache.add_message(request.session_id, msg)
                logger.info(f"Loaded history from database into cache: {chat_history}")

        # Find the message to regenerate in the history
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

        # Get the original message's timestamp
        original_message = chat_history[message_index]
        logger.info(f"Found message to regenerate: {original_message}")
        original_timestamp = original_message.get('timestamp')
        print("original_timestamp=", original_timestamp)
        if not original_timestamp:
            original_timestamp = datetime.now().isoformat()

        # Get history up to the message being regenerated
        history_up_to_message = chat_history[:message_index]
        # The message after the one being regenerated should be the user's message
        user_message = chat_history[message_index + 1] if message_index + 1 < len(chat_history) else None

        async def generate():
            try:
                timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    supports_streaming, supports_trace = await check_endpoint_capabilities(SERVING_ENDPOINT_NAME)
                    request_data = {
                        "messages": [{"role": msg["role"], "content": msg["content"]} for msg in history_up_to_message] + 
                                   [{"role": "user", "content": request.original_content}],
                    }
                    if supports_trace:
                        request_data["databricks_options"] = {"return_trace": True}

                    start_time = time.time()
                    first_token_time = None
                    accumulated_content = ""
                    sources = None

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
                                                        sources = await extract_sources_from_trace(data)
                                                    
                                                    if content:
                                                        accumulated_content += content
                                                        current_time = time.time()
                                                        response_data = {
                                                            'message_id': request.message_id,
                                                            'content': content,
                                                            'sources': sources if sources else None,
                                                            'metrics': {
                                                                'timeToFirstToken': ttft,
                                                                'totalTime': current_time - start_time
                                                            },
                                                            'timestamp': original_timestamp
                                                        }
                                                        yield f"data: {json.dumps(response_data)}\n\n"
                                            except json.JSONDecodeError:
                                                continue

                                    # Final update with complete message
                                    total_time = time.time() - start_time
                                    updated_message = MessageResponse(
                                        message_id=request.message_id,
                                        content=accumulated_content,
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=original_timestamp,
                                        sources=sources,
                                        metrics={
                                            "timeToFirstToken": ttft,
                                            "totalTime": total_time
                                        }
                                    )
                                    # Update message in cache
                                    chat_history_cache.update_message(
                                        request.session_id,
                                        request.message_id,
                                        accumulated_content
                                    )
                                    
                                    # Update message in database
                                    chat_db.update_message(request.session_id, user_id, updated_message)
                                    
                                    yield "event: done\ndata: {}\n\n"
                    else:
                        # Non-streaming case
                        regen_transport = httpx.AsyncHTTPTransport(
                            retries=3,
                            limits=httpx.Limits(max_keepalive_connections=0, max_connections=1)
                        )
                        async with streaming_semaphore:
                            async with httpx.AsyncClient(
                                timeout=timeout,
                                transport=regen_transport
                            ) as client:
                                response = await enqueue_request(client, f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations", headers, request_data)
                            
                                success, response_data = await handle_databricks_response(response, start_time)
                                if success and response_data:
                                    total_time = time.time() - start_time
                                    updated_message = MessageResponse(
                                        message_id=request.message_id,
                                        content=response_data["content"],
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=original_timestamp,
                                        sources=response_data.get("sources", []),
                                        metrics={
                                            "totalTime": total_time
                                        }
                                    )
                                    # Update message in cache
                                    chat_history_cache.update_message(
                                        request.session_id,
                                        request.message_id,
                                        response_data["content"]
                                    )
                                    # Update message in database
                                    chat_db.update_message(request.session_id, user_id, updated_message)
                                    
                                    # Include original timestamp in response
                                    response_data["timestamp"] = original_timestamp
                                    response_data["message_id"] = request.message_id
                                    yield f"data: {updated_message.model_dump_json()}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                    
            except Exception as e:
                logger.error(f"Error in regeneration: {str(e)}")
                total_time = time.time() - start_time
                error_response = {
                    "message_id": request.message_id,
                    "content": "Failed to regenerate response. Please try again.",
                    "sources": [],
                    "metrics": None,
                    "timestamp": original_timestamp
                }
                
                error_message = MessageResponse(
                    message_id=request.message_id,
                    content=error_response["content"],
                    role="assistant",
                    model=SERVING_ENDPOINT_NAME,
                    timestamp=original_timestamp,
                    sources=[],
                    metrics=None
                )
                # Update error message in cache
                chat_history_cache.update_message(request.session_id, request.message_id, error_response["content"])
                
                # Update error message in database
                chat_db.update_message(request.session_id, user_id, error_message)
                
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
    try:
        user_id = user_info["user_id"]
        # Get the chat session from database
        chat_data = chat_db.get_chat(error.session_id, user_id)
        if not chat_data:
            logger.error(f"Session {error.session_id} not found in database")
            raise HTTPException(
                status_code=404, 
                detail=f"Chat session {error.session_id} not found. Please ensure you're using a valid session ID."
            )
        
        # Check if this is a new error message or updating an existing one
        is_new_error = not any(msg.message_id == error.message_id for msg in chat_data.messages)
        
        if is_new_error:
            # Create new error message
            error_message = MessageResponse(
                message_id=str(uuid.uuid4()),  # Generate new ID for new error
                content=error.content,
                role=error.role,
                model=SERVING_ENDPOINT_NAME,
                timestamp=error.timestamp,
                sources=error.sources,
                metrics=error.metrics
            )
            # Save new error message to database
            chat_db.save_message_to_session(error.session_id, user_id, error_message)
            # Add to cache
            chat_history_cache.add_message(error.session_id, {
                "role": error.role,
                "content": error.content,
                "message_id": error_message.message_id,
                "timestamp": error.timestamp
            })
            
        else:
            # Update existing message
            error_message = MessageResponse(
                message_id=error.message_id,  # Use existing message ID
                content=error.content,
                role=error.role,
                model=SERVING_ENDPOINT_NAME,
                timestamp=error.timestamp,
                sources=error.sources,
                metrics=error.metrics
            )
            # Update message in database
            chat_db.update_message(error.session_id, user_id, error_message)
            # Update in cache
            chat_history_cache.update_message(error.session_id, error.message_id, error.content)
        
        return {"status": "error saved", "message_id": error_message.message_id}
        
    except Exception as e:
        logger.error(f"Error in regenerate_error endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while saving the error message."
        )

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
