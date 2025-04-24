import json
import time
from typing import AsyncGenerator, Optional
from fastapi.responses import StreamingResponse
from models import MessageResponse

class StreamingHandler:
    @staticmethod
    def create_streaming_response(generator: AsyncGenerator) -> StreamingResponse:
        """Create a streaming response with proper headers"""
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )

    @staticmethod
    async def stream_message(message: MessageResponse) -> AsyncGenerator[str, None]:
        """Stream a single message"""
        yield f"data: {json.dumps(message.model_dump_json())}\n\n"
        yield "event: done\ndata: {}\n\n"

    @staticmethod
    async def stream_chunk(
        message_id: str,
        content: str,
        sources: Optional[list] = None,
        metrics: Optional[dict] = None,
        timestamp: Optional[str] = None
    ) -> str:
        """Stream a chunk of content"""
        response_data = {
            'message_id': message_id,
            'content': content,
            'sources': sources,
            'metrics': metrics,
            'timestamp': timestamp
        }
        return f"data: {json.dumps(response_data)}\n\n"

    @staticmethod
    async def stream_done(sources: Optional[list] = None) -> AsyncGenerator[str, None]:
        """Stream the done event"""
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield "event: done\ndata: {}\n\n" 