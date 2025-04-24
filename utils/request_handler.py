import os
import httpx
import backoff
import asyncio
import logging
import json
import time
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime
from .config import (
    DATABRICKS_HOST,
    API_TIMEOUT,
    STREAMING_TIMEOUT,
    MAX_CONCURRENT_STREAMS,
    MAX_QUEUE_SIZE
)

logger = logging.getLogger(__name__)

class RequestHandler:
    def __init__(self, endpoint_name: str):
        self.host = DATABRICKS_HOST
        self.endpoint_name = endpoint_name
        self.request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.streaming_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)

    async def request_worker(self):
        """Worker to process requests from the queue"""
        while True:
            try:
                fut, args, kwargs = await self.request_queue.get()
                try:
                    result = await self.make_databricks_request(*args, **kwargs)
                    if not fut.done():
                        fut.set_result(result)
                except Exception as e:
                    logger.error(f"Error in request worker: {str(e)}")
                    if not fut.done():
                        fut.set_exception(e)
                finally:
                    self.request_queue.task_done()
            except Exception as e:
                logger.error(f"Critical error in request worker: {str(e)}")
                await asyncio.sleep(1)

    async def enqueue_request(self, *args, **kwargs):
        """Enqueue a request to be processed by the worker"""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self.request_queue.put((fut, args, kwargs))
        try:
            return await fut
        except Exception as e:
            logger.error(f"Error in enqueue_request: {str(e)}")
            raise

    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, httpx.ReadTimeout, httpx.HTTPStatusError, RuntimeError),
        max_tries=3,
        max_time=30
    )
    async def make_databricks_request(client: httpx.AsyncClient, url: str, headers: dict, data: dict):
        try:
            if client.is_closed:
                raise RuntimeError("Client is closed, creating new client")
            
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
        except RuntimeError as e:
            logger.error(f"Client error: {str(e)}")
            # Create a new client and retry
            async with httpx.AsyncClient(timeout=30.0) as new_client:
                response = await new_client.post(url, headers=headers, json=data, timeout=30.0)
                return response


    async def extract_sources_from_trace(self, data: dict) -> list:
        """Extract sources from the Databricks API trace data."""
        sources = []
        try:
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
                            break  # Found what we need
            logger.debug(f"Extracted sources: {sources}")
        except Exception as e:
            logger.error(f"Error extracting sources: {str(e)}")
            sources = []
        return sources

    async def handle_databricks_response(
        self,
        response: httpx.Response,
        start_time: float
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Handle Databricks API response.
        Returns (success, response_data)
        """
        total_time = time.time() - start_time

        if response.status_code == 200:
            data = response.json()
            sources = await self.extract_sources_from_trace(data)

            if not data:
                return False, None
            
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