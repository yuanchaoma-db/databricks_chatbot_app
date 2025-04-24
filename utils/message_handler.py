from datetime import datetime
import uuid
from typing import Dict, Optional
from models import MessageResponse
from chat_database import ChatDatabase
from .chat_history_cache import ChatHistoryCache
from .config import SERVING_ENDPOINT_NAME


class MessageHandler:
    def __init__(self, chat_db: ChatDatabase, chat_history_cache: ChatHistoryCache):
        self.chat_db = chat_db
        self.chat_history_cache = chat_history_cache

    def create_message(self, content: str, role: str, session_id: str, user_id: str, 
                      sources: Optional[list] = None, metrics: Optional[dict] = None) -> MessageResponse:
        """Create a new message and save it to both database and cache"""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        message = MessageResponse(
            message_id=message_id,
            content=content,
            role=role,
            model=SERVING_ENDPOINT_NAME,
            timestamp=timestamp,
            sources=sources,
            metrics=metrics
        )
        
        # Save to database
        self.chat_db.save_message_to_session(session_id, user_id, message)
        
        # Add to cache
        self.chat_history_cache.add_message(session_id, {
            "role": role,
            "content": content,
            "message_id": message_id,
            "created_at": timestamp,
            "sources": sources,
            "metrics": metrics,
            "model": SERVING_ENDPOINT_NAME
        })
        
        return message

    def update_message(self, session_id: str, message_id: str, user_id: str, 
                      new_content: str, sources: Optional[list] = None, 
                      metrics: Optional[dict] = None) -> MessageResponse:
        """Update an existing message in both database and cache"""
        timestamp = datetime.now()
        
        message = MessageResponse(
            message_id=message_id,
            content=new_content,
            role="assistant",
            model=SERVING_ENDPOINT_NAME,
            timestamp=timestamp,
            sources=sources,
            metrics=metrics
        )
        
        # Update in database
        self.chat_db.update_message(session_id, user_id, message)
        
        # Update in cache with all fields
        self.chat_history_cache.update_message(session_id, message_id, {
            "content": new_content,
            "sources": sources,
            "metrics": metrics,
            "timestamp": timestamp,
            "model": SERVING_ENDPOINT_NAME
        })
        
        return message

    def create_error_message(self, session_id: str, user_id: str, error_content: str) -> MessageResponse:
        """Create an error message and save it"""
        return self.create_message(
            content=error_content,
            role="assistant",
            session_id=session_id,
            user_id=user_id,
            sources=[],
            metrics=None
        ) 