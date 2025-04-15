from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Response, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
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
from collections import defaultdict
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
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")
assert SERVING_ENDPOINT_NAME, "SERVING_ENDPOINT_NAME is not set"

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
        self.first_message_cache = {}
        self.init_db()
    
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
                
                # Create sessions table with user information
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    user_email TEXT,
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
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    role TEXT NOT NULL,
                    model TEXT,
                    timestamp TEXT NOT NULL,
                    sources TEXT,
                    metrics TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                ''')
                
                # Create ratings table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS message_ratings (
                    message_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    rating TEXT CHECK(rating IN ('up', 'down')),
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (message_id, user_id),
                    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
                ''')
                
                # Create indexes for better query performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_message_id ON message_ratings(message_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON message_ratings(user_id)')
                
                conn.commit()
            except sqlite3.Error as e:
                logger.error(f"Error initializing database: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def save_message_to_session(self, session_id: str, user_id: str, message: MessageResponse, user_info: dict = None, is_first_message: bool = False):
        """Save a message to a chat session, creating the session if it doesn't exist"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                logger.info(f"Saving message: session_id={session_id}, user_id={user_id}, message_id={message.message_id}")
                
                # Check if session exists
                cursor.execute('SELECT session_id FROM sessions WHERE session_id = ? and user_id = ?', (session_id, user_id))
                if not cursor.fetchone():
                    logger.info(f"Creating new session: session_id={session_id}, user_id={user_id}")
                    # Create new session with user info
                    cursor.execute('''
                    INSERT INTO sessions (session_id, user_id, user_email, first_query, timestamp, is_active)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        session_id,
                        user_id,  # Use the provided user_id directly
                        user_info.get('email') if user_info else None,
                        message.content if is_first_message else "",
                        message.timestamp.isoformat(),
                        1
                    ))
                
                # Save message with user_id
                cursor.execute('''
                INSERT INTO messages (
                    message_id, session_id, user_id, content, role, model, 
                    timestamp, sources, metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    message.message_id,
                    session_id,
                    user_id,  # Use the provided user_id directly
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
    
    def update_message(self, session_id: str, user_id: str, message: MessageResponse):
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
                WHERE message_id = ? AND session_id = ? AND user_id = ?
                ''', (
                    message.content,
                    message.role,
                    message.model,
                    message.timestamp.isoformat(),
                    json.dumps(message.sources) if message.sources else None,
                    json.dumps(message.metrics) if message.metrics else None,
                    message.message_id,
                    session_id,
                    user_id
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
    
    def get_chat_history(self, user_id: str = None) -> ChatHistoryResponse:
        """Retrieve chat sessions with their messages for a specific user"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                if user_id:
                    cursor.execute('''
                    SELECT s.session_id, s.first_query, s.timestamp, s.is_active,
                           m.message_id, m.content, m.role, m.model, m.timestamp as message_timestamp,
                           m.sources, m.metrics
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id and m.user_id = s.user_id
                    WHERE s.user_id = ?
                    ORDER BY s.timestamp DESC, m.timestamp ASC
                    ''', (user_id,))
                else:
                    cursor.execute('''
                    SELECT s.session_id, s.first_query, s.timestamp, s.is_active,
                           m.message_id, m.content, m.role, m.model, m.timestamp as message_timestamp,
                           m.sources, m.metrics
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id and m.user_id = s.user_id
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
                
                # Sort messages by timestamp for each session
                for session in sessions.values():
                    session.messages.sort(key=lambda x: x.timestamp)
                
                return ChatHistoryResponse(sessions=list(sessions.values()))
            except sqlite3.Error as e:
                logger.error(f"Error getting chat history: {str(e)}")
                raise
            finally:
                cursor.close()
    
    def get_chat(self, session_id: str, user_id: str = None) -> ChatHistoryItem:
        """Retrieve a specific chat session"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                logger.info(f"Getting chat for session_id: {session_id}, user_id: {user_id}")
                
                # Get session info with user check
                if user_id:
                    cursor.execute('''
                    SELECT first_query, timestamp, is_active
                    FROM sessions
                    WHERE session_id = ? AND user_id = ?
                    ''', (session_id, user_id))
                else:
                    cursor.execute('''
                    SELECT first_query, timestamp, is_active
                    FROM sessions
                    WHERE session_id = ?
                    ''', (session_id,))
                
                session_data = cursor.fetchone()
                if not session_data:
                    logger.error(f"Session not found: session_id={session_id}, user_id={user_id}")
                    raise HTTPException(status_code=404, detail="Chat not found")
                
                # Get messages
                cursor.execute('''
                SELECT message_id, content, role, model, timestamp, sources, metrics, user_id
                FROM messages
                WHERE session_id = ? and user_id = ?
                ORDER BY timestamp ASC
                ''', (session_id, user_id))
                
                messages = []
                for row in cursor.fetchall():
                    logger.info(f"Found message: message_id={row['message_id']}, user_id={row['user_id']}")
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
    
    def clear_session(self, session_id: str, user_id: str):
        """Clear a session and its messages"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                # Delete messages
                cursor.execute('DELETE FROM messages WHERE session_id = ? and user_id = ?', (session_id, user_id))
                # Delete session
                cursor.execute('DELETE FROM sessions WHERE session_id = ? and user_id = ?', (session_id, user_id))
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
    
    def is_first_message(self, session_id: str, user_id: str) -> bool:
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
                WHERE session_id = ? and user_id = ?
                ''', (session_id, user_id))
                
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

    def update_message_rating(self, message_id: str, user_id: str, rating: str | None) -> bool:
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                conn.execute('BEGIN TRANSACTION')
                
                # First verify the message exists and belongs to the user
                cursor.execute('''
                SELECT message_id, session_id FROM messages 
                WHERE message_id = ? AND user_id = ?
                ''', (message_id, user_id))
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"Message {message_id} not found for user {user_id}")
                    conn.rollback()
                    return False
                
                session_id = result['session_id']
                
                if rating is None:
                    # Remove the rating
                    cursor.execute('''
                    DELETE FROM message_ratings 
                    WHERE message_id = ? AND user_id = ?
                    ''', (message_id, user_id))
                else:
                    # Insert or update the rating
                    cursor.execute('''
                    INSERT INTO message_ratings (message_id, user_id, session_id, rating)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(message_id, user_id) DO UPDATE SET rating = excluded.rating
                    ''', (message_id, user_id, session_id, rating))
                
                conn.commit()
                return True
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"Error updating message rating: {str(e)}")
                return False
            finally:
                cursor.close()

    def get_message_rating(self, message_id: str, user_id: str) -> str | None:
        """Get the rating of a message"""
        with self.db_lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute('''
                SELECT rating
                FROM message_ratings
                WHERE message_id = ? AND user_id = ?
                ''', (message_id, user_id))
                
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

    if response.status_code == 200:
        data = response.json()
        print(f"Data: {data}")
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
    request: Request
):
    # user_info = {
    #     "email": request.headers.get("X-Forwarded-Email"),
    #     "user_id": request.headers.get("X-Forwarded-User"),
    #     "username": request.headers.get("X-Forwarded-Preferred-Username", "").split("@")[0]
    # }
    # Get user info from headers
    user_info = {
        "email": "test@databricks.com",
        "user_id": "test_user1",
        "username": "test_user1"
    }
    user_id = user_info["user_id"]
    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")

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
    chat_db.save_message_to_session(error.session_id, user_id, error_message)
    return {"status": "error saved"}

@api_app.get("/model")
async def get_model():
    return {"model": SERVING_ENDPOINT_NAME}

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
        """Update a message in the cache"""
        with self.lock:
            for msg in self.cache[session_id]:
                if msg.get('message_id') == message_id:
                    msg['content'] = new_content
                    break

# Initialize the cache
chat_history_cache = ChatHistoryCache()

# Modify the chat endpoint to handle sessions
@api_app.post("/chat")
async def chat(
    message: MessageRequest,
    request: Request,
    headers: dict = Depends(get_auth_headers)
):
    try:
        # Get user info from headers
        user_info = {
            "email": "test@databricks.com",
            "user_id": "test_user1",
            "username": "test_user1"
        }
        user_id = user_info["user_id"]
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")

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
            timestamp=datetime.now()
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
            "message_id": user_message.message_id
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
                        response = await make_databricks_request(
                            regular_client,
                            f"https://{os.getenv('DATABRICKS_HOST')}/serving-endpoints/{SERVING_ENDPOINT_NAME}/invocations",
                            headers=headers,
                            data=request_data
                        )
                        success, response_data = await handle_databricks_response(response, start_time)
                        if success and response_data:
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
                            
                            yield f"data: {json.dumps(response_data)}\n\n"
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
                            
                            assistant_message = MessageResponse(
                                message_id=error_message_id,
                                content=error_response["content"],
                                role="assistant",
                                model=SERVING_ENDPOINT_NAME,
                                timestamp=datetime.now(),
                                sources=error_response.get("sources"),
                                metrics=error_response.get("metrics")
                            )
                            chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                            yield f"data: {json.dumps(error_response)}\n\n"
                            yield "event: done\ndata: {}\n\n"
                    except Exception as e:
                        logger.error(f"Error in non-streaming response: {str(e)}")
                        error_message_id = str(uuid.uuid4())
                        assistant_message = MessageResponse(
                            message_id=error_message_id,
                            content="Request timed out. Please try again later.",
                            role="assistant",
                            model=SERVING_ENDPOINT_NAME,
                            timestamp=datetime.now(),
                            sources=[],
                            metrics=None
                        )
                        chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                        yield f"data: {json.dumps({'message_id': error_message_id,'content': 'Failed to process response from model', 'metrics': None})}\n\n"
                        yield "event: done\ndata: {}\n\n"
            else:
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
                                    yield f"data: {assistant_message.model_dump_json()}\n\n"
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
                                    
                                    yield f"data: {json.dumps(response_data)}\n\n"
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
                                    assistant_message = MessageResponse(
                                        message_id=error_message_id,
                                        content=error_response["content"],
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=error_response.get("sources"),
                                        metrics=error_response.get("metrics")
                                    )
                                    chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                                    yield f"data: {json.dumps(error_response)}\n\n"
                                    yield "event: done\ndata: {}\n\n"
                            except httpx.ReadTimeout as timeout_error:
                                logger.error(f"Fallback request failed with timeout: {str(timeout_error)}")
                                error_message_id = str(uuid.uuid4())
                                assistant_message = MessageResponse(
                                        message_id=error_message_id,
                                        content="Request timed out. Please try again later.",
                                        role="assistant",
                                        model=SERVING_ENDPOINT_NAME,
                                        timestamp=datetime.now(),
                                        sources=[],
                                        metrics={"totalTime": time.time() - start_time}
                                    )
                                chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
                                yield f"data: {json.dumps({'message_id': error_message_id,'content': 'Request timed out. Please try again later.'})}\n\n"
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
        
        assistant_message = MessageResponse(
            message_id=error_message_id,
            content=error_response["content"],
            role="assistant",
            model=SERVING_ENDPOINT_NAME,
            timestamp=datetime.now(),
            sources=error_response.get("sources"),
            metrics=error_response.get("metrics")
        )
        chat_db.save_message_to_session(message.session_id, user_id, assistant_message)
        
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
async def get_chat_history(request: Request):
    # Get user info from headers
    # user_id = request.headers.get("X-Forwarded-User")
    user_id = "test_user1"
    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return chat_db.get_chat_history(user_id)

@api_app.get("/chats/{session_id}", response_model=ChatHistoryItem)
async def get_chat(session_id: str, request: Request):
    # Get user info from headers
    # user_id = request.headers.get("X-Forwarded-User")
    user_id = "test_user1"
    if not user_id:
        raise HTTPException(status_code=401, detail="User not authenticated")
    return chat_db.get_chat(session_id, user_id)

# Add endpoint to get session messages
@api_app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    try:
        # Get the chat session from database
        # user_id = request.headers.get("X-Forwarded-User")
        user_id = "test_user1"
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
    request_obj: Request,
    headers: dict = Depends(get_auth_headers)
):
    try:
        # Get user info from headers
        user_info = {
            "email": "test@databricks.com",
            "user_id": "test_user1",
            "username": "test_user1"
        }
        user_id = user_info["user_id"]
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Get chat history from cache
        chat_history = chat_history_cache.get_history(request.session_id)
        
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
                                chat_db.update_message(request.session_id, user_id, updated_message)
                                
                                # Update message in cache
                                chat_history_cache.update_message(
                                    request.session_id,
                                    request.message_id,
                                    accumulated_content
                                )
                                
                                # Send final metrics
                                yield f"data: {updated_message.model_dump_json()}\n\n"
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
                                chat_db.update_message(request.session_id, user_id, updated_message)
                                
                                # Update message in cache
                                chat_history_cache.update_message(
                                    request.session_id,
                                    request.message_id,
                                    response_data["content"]
                                )
                                
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
                chat_db.update_message(request.session_id, user_id, error_message)
                
                
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
    request: Request,
    headers: dict = Depends(get_auth_headers)
):
    try:
        # user_info = {
    #     "email": request.headers.get("X-Forwarded-Email"),
    #     "user_id": request.headers.get("X-Forwarded-User"),
    #     "username": request.headers.get("X-Forwarded-Preferred-Username", "").split("@")[0]
    # }
        # Get user info from headers
        user_info = {
            "email": "test@databricks.com",
            "user_id": "test_user1",
            "username": "test_user1"
        }
        user_id = user_info["user_id"]
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")

        # Get the chat session from database
        chat_data = chat_db.get_chat(error.session_id, user_id)
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
        chat_db.update_message(error.session_id, user_id, error_message)
        
        return {"status": "error saved"}
        
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
    headers: dict = Depends(get_auth_headers)
):
    try:
        # Get user info from headers
        #user_id = headers.get("X-Forwarded-User")
        user_id = "test_user1"
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
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
async def login(request: Request):
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
    return {"user_info": user_info}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
