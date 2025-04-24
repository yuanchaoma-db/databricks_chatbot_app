from typing import Dict, List
from collections import defaultdict
import threading
from datetime import datetime

class ChatHistoryCache:
    """In-memory cache for chat history"""
    def __init__(self):
        self.cache: Dict[str, List[Dict]] = defaultdict(list)
        self.first_queries: Dict[str, str] = {}  # Store first query for each session
        self.lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Dict]:
        """Get chat history from cache"""
        with self.lock:
            return self.cache[session_id]

    def get_first_query(self, session_id: str) -> str:
        """Get first query for a session"""
        with self.lock:
            return self.first_queries.get(session_id, "")

    def set_first_query(self, session_id: str, first_query: str):
        """Set first query for a session"""
        with self.lock:
            self.first_queries[session_id] = first_query

    def add_message(self, session_id: str, message: Dict):
        """Add a message to the cache"""
        with self.lock:
            if 'created_at' not in message:
                message['created_at'] = datetime.now().isoformat()
            if 'timestamp' not in message:
                message['timestamp'] = message['created_at']
            
            # If this is the first message and it's from the user, set it as first_query
            if not self.cache[session_id] and message.get('role') == 'user':
                self.first_queries[session_id] = message['content']
            
            self.cache[session_id].append(message)
            # Keep only last 10 messages
            if len(self.cache[session_id]) > 10:
                self.cache[session_id] = self.cache[session_id][-10:]

    def clear_session(self, session_id: str):
        """Clear a session from cache"""
        with self.lock:
            if session_id in self.cache:
                del self.cache[session_id]
            if session_id in self.first_queries:
                del self.first_queries[session_id]

    def update_message(self, session_id: str, message_id: str, message_data: dict):
        """Update a message in the cache while preserving order"""
        with self.lock:
            messages = self.cache[session_id]
            for msg in messages:
                if msg.get('message_id') == message_id:
                    msg.update(message_data)
                    if 'timestamp' in message_data:
                        msg['timestamp'] = message_data['timestamp'].isoformat()
                    break 