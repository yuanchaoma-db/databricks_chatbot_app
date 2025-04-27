import json
import time
from typing import AsyncGenerator, Optional
from fastapi.responses import StreamingResponse
from models import MessageResponse
from utils.request_handler import RequestHandler

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
    async def stream_message(line, assistant_message_id: str, start_time: float, first_token_time: float, accumulated_content: str, sources: list, request_handler: RequestHandler) -> AsyncGenerator[str, None]:
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
            response_data = {
                'message_id': assistant_message_id,
                'content': content if content else None,
                'sources': sources if sources else None,
                'metrics': {
                    'timeToFirstToken': ttft if first_token_time is not None else None,
                    'totalTime': time.time() - start_time
                }
            }
            
            yield f"data: {json.dumps(response_data)}\n\n"
        if "delta" in data:
            delta = data["delta"]
            if delta["role"] == "assistant" and delta.get("content"):
                content = delta['content']
                accumulated_content += content
                response_data = {
                    'message_id': assistant_message_id,
                    'content': content+"\n\n",
                    'sources': sources if sources else None,
                    'metrics': {
                        'timeToFirstToken': ttft if first_token_time is not None else None,
                        'totalTime': time.time() - start_time
                    }
                }
                yield f"data: {json.dumps(response_data)}\n\n"   

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