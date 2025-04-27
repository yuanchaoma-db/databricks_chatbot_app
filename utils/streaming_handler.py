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

    async def handle_streaming_response(
        client: httpx.AsyncClient,
        request_data: Dict,
        headers: Dict,
        session_id: str,
        message_id: str,
        user_id: str,
        original_timestamp: str,
        start_time: float,
        first_token_time: Optional[float],
        accumulated_content: str,
        sources: Optional[Dict],
        ttft: Optional[float]
    ) -> AsyncGenerator[str, None]:
        """Handle streaming response from the model."""
        try:
            async for line in client.aiter_lines():
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
                            # Include the same message_id in each chunk
                            response_data = create_response_data(
                                message_id,
                                content if content else None,
                                sources,
                                ttft if first_token_time is not None else None,
                                time.time() - start_time,
                                original_timestamp
                            )
                            
                            yield f"data: {json.dumps(response_data)}\n\n"
                        if "delta" in data:
                            delta = data["delta"]
                            if delta["role"] == "assistant" and delta.get("content"):
                                content = delta['content']
                                accumulated_content += content
                                response_data = create_response_data(
                                    message_id,
                                    content+"\n\n",
                                    sources,
                                    ttft if first_token_time is not None else None,
                                    time.time() - start_time,
                                    original_timestamp
                                )
                                yield f"data: {json.dumps(response_data)}\n\n"    
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            raise

    async def handle_non_streaming_response(
        request_handler,
        url: str,
        headers: Dict,
        request_data: Dict,
        session_id: str,
        user_id: str,
        message_handler
    ) -> AsyncGenerator[str, None]:
        """Handle non-streaming response from the model."""
        try:
            start_time = time.time()
            response = await request_handler.enqueue_request(url, headers, request_data)
            response_data = await request_handler.handle_databricks_response(response, start_time)
            
            assistant_message = message_handler.create_message(
                message_id=str(uuid.uuid4()),
                content=response_data["content"],
                role="assistant",
                session_id=session_id,
                user_id=user_id,
                user_info=user_info,
                sources=response_data.get("sources"),
                metrics=response_data.get("metrics")
            )
            
            yield f"data: {json.dumps(assistant_message.model_dump_json())}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.error(f"Error in non-streaming response: {str(e)}")
            error_message = message_handler.create_error_message(
                session_id=session_id,
                user_id=user_id,
                error_content="Request timed out. Please try again later."
            )
            yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
            yield "event: done\ndata: {}\n\n"

    async def handle_streaming_regeneration(
        client: httpx.AsyncClient,
        request_data: Dict,
        headers: Dict,
        session_id: str,
        message_id: str,
        user_id: str,
        original_timestamp: str,
        start_time: float,
        first_token_time: Optional[float],
        accumulated_content: str,
        sources: Optional[Dict],
        ttft: Optional[float]
    ) -> AsyncGenerator[str, None]:
        """Handle streaming message regeneration."""
        try:
            async for response_chunk in handle_streaming_response(
                client, request_data, headers, session_id, message_id, user_id,
                original_timestamp, start_time, first_token_time, accumulated_content,
                sources, ttft
            ):
                yield response_chunk
        except Exception as e:
            logger.error(f"Error in streaming regeneration: {str(e)}")
            error_message = message_handler.create_error_message(
                session_id=session_id,
                user_id=user_id,
                error_content="Failed to regenerate response. Please try again."
            )
            yield f"data: {json.dumps(error_message.model_dump_json())}\n\n"
            yield "event: done\ndata: {}\n\n" 