from fastapi import HTTPException, Depends
import logging
from typing import Dict, Any
from .config import ERROR_MESSAGES
from models import ErrorRequest
from .message_handler import MessageHandler

logger = logging.getLogger(__name__)

class ErrorHandler:
    def __init__(self, message_handler: MessageHandler):
        self.message_handler = message_handler

    @staticmethod
    def handle_error(error: Exception, status_code: int = 500, detail: str = None) -> None:
        """Handle and log errors, raising appropriate HTTP exceptions"""
        logger.error(f"Error occurred: {str(error)}")
        
        if isinstance(error, HTTPException):
            raise error
            
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
    def handle_not_found_error(resource_type: str, resource_id: str) -> None:
        """Handle not found errors"""
        raise HTTPException(
            status_code=404,
            detail=ERROR_MESSAGES["not_found"].format(
                resource_type=resource_type,
                resource_id=resource_id
            )
        )

    async def handle_error_endpoint(
        self,
        error: ErrorRequest,
        user_info: dict
    ) -> Dict[str, str]:
        """Handle the error endpoint"""
        try:
            user_id = user_info["user_id"]
            
            # Get the chat session from database
            chat_data = self.message_handler.chat_db.get_chat(error.session_id, user_id)
            if not chat_data:
                self.handle_not_found_error("Chat session", error.session_id)
            
            # Check if this is a new error message or updating an existing one
            is_new_error = not any(msg.message_id == error.message_id for msg in chat_data.messages)
            
            if is_new_error:
                # Create new error message
                error_message = self.message_handler.create_message(
                    content=error.content,
                    role=error.role,
                    session_id=error.session_id,
                    user_id=user_id,
                    sources=error.sources,
                    metrics=error.metrics
                )
            else:
                # Update existing message
                error_message = self.message_handler.update_message(
                    session_id=error.session_id,
                    message_id=error.message_id,
                    user_id=user_id,
                    new_content=error.content,
                    sources=error.sources,
                    metrics=error.metrics
                )
            
            return {"status": "error saved", "message_id": error_message.message_id}
            
        except Exception as e:
            self.handle_error(e) 