from fastapi import HTTPException, Depends
import logging
from typing import Dict, Any
from .config import ERROR_MESSAGES, SERVING_ENDPOINT_NAME
from models import ErrorRequest, MessageResponse
from .message_handler import MessageHandler
import uuid

logger = logging.getLogger(__name__)

class ErrorHandler:
    def __init__(self, message_handler: MessageHandler):
        self.message_handler = message_handler

    @staticmethod
    def handle_error(status_code: int = 500, detail: str = None) -> None:
        """Handle and log errors, raising appropriate HTTP exceptions"""
        if not detail:
            detail = ERROR_MESSAGES["general"]
            
        raise HTTPException(status_code=status_code, detail=detail)

    @staticmethod
    def handle_rate_limit_error() -> Dict[str, Any]:
        """Handle rate limit errors specifically"""
        return {
            "error": ERROR_MESSAGES["rate_limit"],
            "status_code": 429
        }

    @staticmethod
    def handle_timeout_error() -> Dict[str, Any]:
        """Handle timeout errors"""
        return {
            "error": ERROR_MESSAGES["timeout"],
            "status_code": 408
        }

    @staticmethod
    def handle_not_found_error(resource_id: str) -> None:
        """Handle not found errors"""
        raise HTTPException(
            status_code=404,
            detail=ERROR_MESSAGES["not_found"].format(
                resource_id=resource_id
            )
        )

    async def handle_error_endpoint(
        self,
        error: ErrorRequest,
        user_info: dict
    ) -> Dict[str, str]:
        """Handle the error endpoint"""
        user_id = user_info["user_id"]
    
        # Get the chat session from database
        chat_data = self.message_handler.chat_db.get_chat(error.session_id, user_id)
        if not chat_data:
            logger.error(f"Session {error.session_id} not found in database")
            self.handle_not_found_error(error.session_id)
        
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
            self.message_handler.chat_db.save_message_to_session(error.session_id, user_id, error_message)
            # Add to cache
            self.message_handler.chat_history_cache.add_message(error.session_id, {
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
            self.message_handler.chat_db.update_message(error.session_id, user_id, error_message)
            # Update in cache
            self.message_handler.chat_history_cache.update_message(error.session_id, error.message_id, error.content)
        
        return {"status": "error saved", "message_id": error_message.message_id}
