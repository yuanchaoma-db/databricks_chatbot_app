from typing import Dict, List, Optional
from datetime import datetime
import copy
from chat_database import ChatDatabase
from utils.chat_history_cache import ChatHistoryCache
from utils.error_handler import ErrorHandler
from fastapi import Request
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from models import MessageResponse
async def check_endpoint_capabilities(model: str, streaming_support_cache: dict) -> tuple[bool, bool]:
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
        # If error occurs, return default values
        return True, False
    
async def get_user_info(request: Request = None) -> dict:
        """Get user information from request headers"""
        if not request:
            # For testing purposes, return test user info
            return {
                "email": "test@databricks.com",
                "user_id": "test_user1",
                "username": "test_user1"
            }
        
        user_info = {
            "email": request.headers.get("X-Forwarded-Email"),
            "user_id": request.headers.get("X-Forwarded-User"),
            "username": request.headers.get("X-Forwarded-Preferred-Username", "").split("@")[0]
        }
        # user_info = {
        #     "email": "test@databricks.com",
        #     "user_id": "test_user1",
        #     "username": "test_user1"
        # }
        if not user_info["user_id"]:
            raise ErrorHandler.handle_error(status_code=401, detail="User not authenticated")
        return user_info

async def load_chat_history(session_id: str, user_id: str, is_first_message: bool, chat_history_cache: ChatHistoryCache, chat_db: ChatDatabase) -> List[Dict]:
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
                message_response = MessageResponse(
                    message_id=msg["message_id"],
                    content=msg["content"],
                    role=msg["role"],
                    timestamp=msg["timestamp"],
                    created_at=msg["created_at"]
                )
                chat_history_cache.add_message(session_id, message_response)
    
    return chat_history or []

def convert_messages_to_cache_format(messages: List) -> List[Dict]:
    """
    Convert database messages to cache format.
    Returns last 20 messages in cache format.
    """
    if not messages:
        return []
    formatted_messages = []
    for msg in messages[-20:]:
        formatted_messages.append({
            "role": msg.role,
            "content": msg.content,
            "message_id": msg.message_id,
            "timestamp": msg.timestamp.isoformat() if isinstance(msg.timestamp, datetime) else msg.timestamp,   
            "created_at": msg.created_at.isoformat() if isinstance(msg.created_at, datetime) else msg.created_at
        })
    return formatted_messages
    
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
