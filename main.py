from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response, Request, Query
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
import requests
import backoff
import time  
import logging
import asyncio
import threading
import sqlite3
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv(override=True)

app = FastAPI()
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
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")

class TokenMinter:
    """
    A class to handle OAuth token generation and renewal for Databricks.
    Automatically refreshes the token before it expires.
    """
    def __init__(self, client_id: str, client_secret: str, host: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.host = host
        self.token = None
        self.expiry_time = None
        self.lock = threading.Lock()
        self._refresh_token()
        
    def _refresh_token(self) -> None:
        """Internal method to refresh the OAuth token"""
        url = f"https://{self.host}/oidc/v1/token"
        auth = (self.client_id, self.client_secret)
        data = {'grant_type': 'client_credentials', 'scope': 'all-apis'}
        
        try:
            response = requests.post(url, auth=auth, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            with self.lock:
                self.token = token_data.get('access_token')
                # Set expiry time to 55 minutes (slightly less than the 60-minute expiry)
                self.expiry_time = datetime.now() + timedelta(minutes=55)
                
            logger.info("Successfully refreshed Databricks OAuth token")
        except Exception as e:
            logger.error(f"Failed to refresh Databricks OAuth token: {str(e)}")
            raise
    
    def get_token(self) -> str:
        """
        Get a valid token, refreshing if necessary.
        
        Returns:
            str: The current valid OAuth token
        """
        with self.lock:
            # Check if token is expired or about to expire (within 5 minutes)
            if not self.token or not self.expiry_time or datetime.now() + timedelta(minutes=5) >= self.expiry_time:
                self._refresh_token()
            return self.token


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

# Models
class MessageRequest(BaseModel):
    content: str
    session_id: str

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


class ErrorRequest(BaseModel):
    message_id: str
    content: str
    role: str = "assistant"
    model: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[dict]] = None
    metrics: Optional[dict] = None
    session_id: str

class RegenerateRequest(BaseModel):
    message_id: str
    original_content: str
    session_id: str

class ChatDatabase:
    def __init__(self, db_file='chat_history.db'):
        self.db_file = db_file
        self.db_lock = threading.Lock()
        self.connection_pool = {}
        self.init_db()
        # Cache for first message status
        self.first_message_cache = {}
    
    def get_connection(self):
        """Get a database connection from the pool or create a new one"""
        thread_id = threading.get_ident()
        if thread_id not in self.connection_pool:
            conn = sqlite3.connect(self.db_file)
            conn.row_factory = sqlite3.Row
            self.connection_pool[thread_id] = conn
        return self.connection_pool[thread_id]
    
    def close_connection(self):
        """Close the database connection for the current thread"""
        thread_id = threading.get_ident()
        if thread_id in self.connection_pool:
            self.connection_pool[thread_id].close()
            del self.connection_pool[thread_id]
    
    def init_db(self):
        """Initialize the database with required tables and indexes"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                # Enable foreign key constraints
                cursor.execute('PRAGMA foreign_keys = ON')
                
                # Create sessions table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    first_query TEXT,
                    timestamp TEXT NOT NULL,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                
                # Create messages table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    model TEXT,
                    timestamp TEXT NOT NULL,
                    sources TEXT,
                    metrics TEXT,
                    rating TEXT CHECK(rating IN ('up', 'down', NULL)),
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)')
                
                # Add rating column if it doesn't exist
                try:
                    cursor.execute('SELECT rating FROM messages LIMIT 1')
                except sqlite3.OperationalError:
                    logger.info("Adding rating column to messages table")
                    cursor.execute('ALTER TABLE messages ADD COLUMN rating TEXT CHECK(rating IN ("up", "down", NULL))')
                
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error initializing database: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def save_message_to_session(self, session_id: str, message: MessageResponse, is_first_message: bool = False):
        """Save a message to a chat session, creating the session if it doesn't exist"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                # Check if session exists
                cursor.execute('SELECT session_id FROM sessions WHERE session_id = ?', (session_id,))
                if not cursor.fetchone():
                    # Create new session
                    cursor.execute('''
                    INSERT INTO sessions (session_id, first_query, timestamp, is_active)
                    VALUES (?, ?, ?, ?)
                    ''', (
                        session_id,
                        message.content if is_first_message else "",
                        message.timestamp.isoformat(),
                        1
                    ))
                
                # Save message
                cursor.execute('''
                INSERT INTO messages (
                    message_id, session_id, content, role, model, 
                    timestamp, sources, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.message_id,
                    session_id,
                    message.content,
                    message.role,
                    message.model,
                    message.timestamp.isoformat(),
                    json.dumps(message.sources) if message.sources else None,
                    json.dumps(message.metrics) if message.metrics else None
                ))
                
                # Update cache after saving message
                self.first_message_cache[session_id] = False
                
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Error saving message to session: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def update_message(self, session_id: str, message: MessageResponse):
        """Update an existing message in the database"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                cursor.execute('''
                UPDATE messages 
                SET content = ?, 
                    role = ?, 
                    model = ?, 
                    timestamp = ?, 
                    sources = ?, 
                    metrics = ?
                WHERE message_id = ? AND session_id = ?
                ''', (
                    message.content,
                    message.role,
                    message.model,
                    message.timestamp.isoformat(),
                    json.dumps(message.sources) if message.sources else None,
                    json.dumps(message.metrics) if message.metrics else None,
                    message.message_id,
                    session_id
                ))
                
                if cursor.rowcount == 0:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Message {message.message_id} not found in session {session_id}"
                    )
                
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Error updating message: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def get_chat_history(self) -> ChatHistoryResponse:
        """Retrieve all chat sessions with their messages"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                SELECT s.session_id, s.first_query, s.timestamp, s.is_active,
                       m.message_id, m.content, m.role, m.model, m.timestamp as message_timestamp,
                       m.sources, m.metrics
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                ORDER BY s.timestamp DESC, m.timestamp ASC
                ''')
                
                sessions = {}
                for row in cursor.fetchall():
                    session_id = row['session_id']
                    if session_id not in sessions:
                        sessions[session_id] = ChatHistoryItem(
                            sessionId=session_id,
                            firstQuery=row['first_query'],
                            messages=[],
                            timestamp=datetime.fromisoformat(row['timestamp']),
                            isActive=bool(row['is_active'])
                        )
                    
                    if row['message_id']:  # message_id exists
                        sessions[session_id].messages.append(MessageResponse(
                            message_id=row['message_id'],
                            content=row['content'],
                            role=row['role'],
                            model=row['model'],
                            timestamp=datetime.fromisoformat(row['message_timestamp']),
                            sources=json.loads(row['sources']) if row['sources'] else None,
                            metrics=json.loads(row['metrics']) if row['metrics'] else None
                        ))
                
                return ChatHistoryResponse(sessions=list(sessions.values()))
            except sqlite3.Error as e:
                logger.error(f"Error getting chat history: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def get_chat(self, session_id: str) -> ChatHistoryItem:
        """Retrieve a specific chat session"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                # Get session info
                cursor.execute('''
                SELECT first_query, timestamp, is_active
                FROM sessions
                WHERE session_id = ?
                ''', (session_id,))
                
                session_data = cursor.fetchone()
                if not session_data:
                    raise HTTPException(status_code=404, detail="Chat not found")
                
                # Get messages
                cursor.execute('''
                SELECT message_id, content, role, model, timestamp, sources, metrics
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC
                ''', (session_id,))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append(MessageResponse(
                        message_id=row['message_id'],
                        content=row['content'],
                        role=row['role'],
                        model=row['model'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        sources=json.loads(row['sources']) if row['sources'] else None,
                        metrics=json.loads(row['metrics']) if row['metrics'] else None
                    ))
                
                return ChatHistoryItem(
                    sessionId=session_id,
                    firstQuery=session_data['first_query'],
                    messages=messages,
                    timestamp=datetime.fromisoformat(session_data['timestamp']),
                    isActive=bool(session_data['is_active'])
                )
            except sqlite3.Error as e:
                logger.error(f"Error getting chat: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def clear_session(self, session_id: str):
        """Clear a session and its messages"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                # Delete messages
                cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
                # Delete session
                cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
                # Clear cache
                if session_id in self.first_message_cache:
                    del self.first_message_cache[session_id]
                
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Error clearing session: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def is_first_message(self, session_id: str) -> bool:
        """Check if this is the first message in a session"""
        # Check cache first
        if session_id in self.first_message_cache:
            return self.first_message_cache[session_id]
            
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                SELECT COUNT(*) FROM messages 
                WHERE session_id = ?
                ''', (session_id,))
                
                count = cursor.fetchone()[0]
                is_first = count == 0
                
                # Update cache
                self.first_message_cache[session_id] = is_first
                return is_first
            except sqlite3.Error as e:
                logger.error(f"Error checking first message: {str(e)}")
                raise
            finally:
                cursor.close()

    def update_message_rating(self, message_id: str, rating: str | None) -> bool:
        """Update the rating of a message"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                cursor.execute('''
                UPDATE messages 
                SET rating = ?
                WHERE message_id = ?
                ''', (rating, message_id))
                
                if cursor.rowcount == 0:
                    conn.rollback()
                    return False
                
                conn.commit()
                return True
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Error updating message rating: {str(e)}")
                return False
            finally:
                cursor.close()

    def get_message_rating(self, message_id: str) -> str | None:
        """Get the rating of a message"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                SELECT rating
                FROM messages
                WHERE message_id = ?
                ''', (message_id,))
                
                result = cursor.fetchone()
                return result['rating'] if result else None
            except sqlite3.Error as e:
                logger.error(f"Error getting message rating: {str(e)}")
                return None
            finally:
                cursor.close()

# Initialize the database
chat_db = ChatDatabase()

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
            'metrics': None
        }
        return True, response_data
    

@api_app.post("/error")
async def error(
    error: ErrorRequest,
):
    # Create error message response
    error_message = MessageResponse(
        message_id=error.message_id,
        content=error.content,
        role=error.role,
        model=SERVING_ENDPOINT_NAME,
        timestamp=error.timestamp,
        sources=error.sources,
        metrics=error.metrics
    )
    
    # Save to session
    chat_db.save_message_to_session(error.session_id, error_message)
    return {"status": "error saved"}

@api_app.get("/model")
async def get_model():
    return {"model": SERVING_ENDPOINT_NAME}

# Modify the chat endpoint to handle sessions
@api_app.post("/chat")
async def chat(
    message: MessageRequest,
    headers: dict = Depends(get_auth_headers)
):
    try:
        # Generate assistant message ID once at the start
        assistant_message_id = str(uuid.uuid4())

        # Check if this is the first message in the session
        is_first_message = chat_db.is_first_message(message.session_id)

        # Save user message to session
        user_message = MessageResponse(
            message_id=str(uuid.uuid4()),
            content=message.content,
            role="user",
            model=SERVING_ENDPOINT_NAME,
            timestamp=datetime.now()
        )
        chat_db.save_message_to_session(
            message.session_id, 
            user_message,
            is_first_message=is_first_message
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
            supports_streaming, supports_trace = await check_endpoint_capabilities(SERVING_ENDPOINT_NAME)
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
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
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
                                model=SERVING_ENDPOINT_NAME,
                                timestamp=datetime.now(),
                                sources=response_data.get("sources"),
                                metrics=response_data.get("metrics")
                            )
                            chat_db.save_message_to_session(message.session_id, assistant_message)
                            
                            yield f"data: {json.dumps(response_data)}\n\n"
                            yield "event: done\ndata: {}\n\n"
                
                        else:
                            logger.error("Failed to process Databricks response")
                            error_response = {
                                'message_id': assistant_message_id,
                                'content': 'Failed to process response from model',
                                'sources': [],
                                'metrics': None
                            }
                            
                            assistant_message = MessageResponse(
                                message_id=assistant_message_id,
                                content=error_response["content"],
                                role="assistant",
                                model=SERVING_ENDPOINT_NAME,
                                timestamp=datetime.now(),
                                sources=error_response.get("sources"),
                                metrics=error_response.get("metrics")
                            )
                            chat_db.save_message_to_session(message.session_id, assistant_message)
                            yield f"data: {json.dumps(error_response)}\n\n"
                            yield "event: done\ndata: {}\n\n"
                    except Exception as e:
                        logger.error(f"Error in non-streaming response: {str(e)}")
                        assistant_message = MessageResponse(
                            message_id=assistant_message_id,
                            content="Request timed out. Please try again later.",
                            role="assistant",
                            model=SERVING_ENDPOINT_NAME,
                            timestamp=datetime.now(),
                            sources=[],
                            metrics=None
                        )
                        chat_db.save_message_to_session(message.session_id, assistant_message)
                        yield f"data: {json.dumps({'message_id': assistant_message_id,'content': 'Failed to process response from model', 'metrics': None})}\n\n"
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
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=sources,
                                        metrics={'timeToFirstToken': ttft, 'totalTime': time.time() - start_time}
                                    )
                                    chat_db.save_message_to_session(message.session_id, assistant_message)
                                    
                                    # Update cache to indicate streaming support
                                    streaming_support_cache['endpoints'][SERVING_ENDPOINT_NAME] = {
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
                                url = f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations?nocache={uuid.uuid4()}"
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
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=response_data.get("sources"),
                                        metrics=response_data.get("metrics")
                                    )
                                    chat_db.save_message_to_session(message.session_id, assistant_message)
                                    
                                    yield f"data: {json.dumps(response_data)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                                else:
                                    logger.error("Failed to process Databricks response")
                                    error_response = {
                                        'message_id': assistant_message_id,
                                        'content': 'Failed to process response from model',
                                        'sources': [],
                                        'metrics': None
                                    }
                                    assistant_message = MessageResponse(
                                        message_id=assistant_message_id,
                                        content=error_response["content"],
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=error_response.get("sources"),
                                        metrics=error_response.get("metrics")
                                    )
                                    chat_db.save_message_to_session(message.session_id, assistant_message)
                                    yield f"data: {json.dumps(error_response)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                            except httpx.ReadTimeout as timeout_error:
                                logger.error(f"Fallback request failed with timeout: {str(timeout_error)}")
                                assistant_message = MessageResponse(
                                        message_id=assistant_message_id,
                                        content="Request timed out. Please try again later.",
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=[],
                                        metrics={"totalTime": time.time() - start_time}
                                    )
                                chat_db.save_message_to_session(message.session_id, assistant_message)
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
            'metrics': None,
            'timestamp': datetime.now().isoformat()
        }
        
        assistant_message = MessageResponse(
            message_id=assistant_message_id,
            content=error_response["content"],
            role="assistant",
            model=SERVING_ENDPOINT_NAME,
            timestamp=datetime.now(),
            sources=error_response.get("sources"),
            metrics=error_response.get("metrics")
        )
        chat_db.save_message_to_session(message.session_id, assistant_message)
        
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

@api_app.get("/chats", response_model=ChatHistoryResponse)
async def get_chat_history():
    return chat_db.get_chat_history()

@api_app.get("/chats/{session_id}", response_model=ChatHistoryItem)
async def get_chat(session_id: str):
    return chat_db.get_chat(session_id)

# Add endpoint to get session messages
@api_app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    try:
        # Get the chat session from database
        chat_data = chat_db.get_chat(session_id)
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
    headers: dict = Depends(get_auth_headers)
):
    try:
        # Get the chat session from database
        chat_data = chat_db.get_chat(request.session_id)
        if not chat_data:
            logger.error(f"Session {request.session_id} not found in database")
            raise HTTPException(
                status_code=404, 
                detail=f"Chat session {request.session_id} not found. Please ensure you're using a valid session ID."
            )
        
        messages = chat_data.messages
        message_index = next(
            (i for i, msg in enumerate(messages) 
             if msg.message_id == request.message_id), 
            None
        )
        
        if message_index is None:
            logger.error(f"Message {request.message_id} not found in session {request.session_id}")
            raise HTTPException(
                status_code=404, 
                detail=f"Message {request.message_id} not found in chat session {request.session_id}."
            )

        async def generate():
            try:
                timeout = httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    supports_streaming, supports_trace = await check_endpoint_capabilities(SERVING_ENDPOINT_NAME)
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
                                updated_message = MessageResponse(
                                    message_id=request.message_id,
                                    content=accumulated_content,
                                    role="assistant",
                                    model=SERVING_ENDPOINT_NAME,
                                    timestamp=datetime.now(),
                                    sources=sources,
                                    metrics={
                                        "timeToFirstToken": ttft,
                                        "totalTime": total_time
                                    }
                                )
                                
                                # Update message in database
                                chat_db.update_message(request.session_id, updated_message)
                                
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
                                f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                                headers=headers,
                                json=request_data,
                                timeout=timeout
                            )
                        
                            success, response_data = await handle_databricks_response(response, start_time)
                            if success and response_data:
                                total_time = time.time() - start_time
                                updated_message = MessageResponse(
                                    message_id=request.message_id,
                                    content=response_data["content"],
                                    role="assistant",
                                    model=SERVING_ENDPOINT_NAME,
                                    timestamp=datetime.now(),
                                    sources=response_data.get("sources", []),
                                    metrics={
                                        "totalTime": total_time
                                    }
                                )
                                
                                # Update message in database
                                chat_db.update_message(request.session_id, updated_message)
                                
                                yield f"data: {json.dumps({**response_data, 'message_id': request.message_id})}\n\n"
                                yield "event: done\ndata: {}\n\n"
                    
            except Exception as e:
                logger.error(f"Error in regeneration: {str(e)}")
                total_time = time.time() - start_time
                error_response = {
                    "message_id": request.message_id,
                    "content": "Failed to regenerate response. Please try again.",
                    "sources": [],
                    "metrics": None,
                    "timestamp": datetime.now().isoformat()
                }
                
                error_message = MessageResponse(
                    message_id=request.message_id,
                    content=error_response["content"],
                    role="assistant",
                    model=SERVING_ENDPOINT_NAME,
                    timestamp=datetime.now(),
                    sources=[],
                    metrics=None
                )
                
                # Update error message in database
                chat_db.update_message(request.session_id, error_message)
                
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

    except Exception as e:
        logger.error(f"Error in regenerate_message endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while regenerating the message."
        )

@api_app.post("/regenerate/error")
async def regenerate_error(
    error: ErrorRequest,
    headers: dict = Depends(get_auth_headers)
):
    try:
        # Get the chat session from database
        chat_data = chat_db.get_chat(error.session_id)
        if not chat_data:
            logger.error(f"Session {error.session_id} not found in database")
            raise HTTPException(
                status_code=404, 
                detail=f"Chat session {error.session_id} not found. Please ensure you're using a valid session ID."
            )
        
        # Create error message response
        error_message = MessageResponse(
            message_id=error.message_id,
            content=error.content,
            role=error.role,
            model=SERVING_ENDPOINT_NAME,
            timestamp=error.timestamp,
            sources=error.sources,
            metrics=error.metrics
        )
        
        # Update error message in database
        chat_db.update_message(error.session_id, error_message)
        
        return {"status": "error saved"}
        
    except Exception as e:
        logger.error(f"Error in regenerate_error endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while saving the error message."
        )

# Example endpoint for user login or session initialization
@api_app.get("/login")
async def login(request: Request):
    # Extract user information from headers
    headers = request.headers
    email = headers.get("X-Forwarded-Email")
    username = headers.get("X-Forwarded-Preferred-Username").split("@")[0]
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

# Add new endpoint for rating messages
@api_app.post("/messages/{message_id}/rate")
async def rate_message(
    message_id: str,
    rating: str | None = Query(..., regex="^(up|down)$")
):
    try:
        success = chat_db.update_message_rating(message_id, rating)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
