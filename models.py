from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

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