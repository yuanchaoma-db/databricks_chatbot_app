import axios from 'axios';
import { Message, Chat } from '../types';

const API_URL = 'http://localhost:8000/api';

export const sendMessage = async (content: string, onChunk: (chunk: string) => void): Promise<void> => {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({ content })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No reader available');

    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6);
          if (jsonStr) {
            try {
              const data = JSON.parse(jsonStr);
              if (data.content) {
                onChunk(data.content);
              }
            } catch (e) {
              console.error('Error parsing JSON:', e);
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

export const getChatHistory = async (): Promise<Chat[]> => {
  try {
    //const response = await axios.get(`${API_URL}/chats`);
    //return response.data.chats;
    
    return [];
  } catch (error) {
    console.error('Error fetching chat history:', error);
    throw error;
  }
}; 

export const createNewChat = async (): Promise<Chat> => {
  try {
    // You might want to make an actual API call here
    // For now, we'll create a mock chat
    const newChat: Chat = {
      id: `chat-${Date.now()}`,
      sessionId: 'session-1',
      firstQuery: 'Kids activities',
      messages: [],
      timestamp: new Date(),
    };
    
    // If you have an actual API endpoint:
    // const response = await fetch('/api/chats', {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    // });
    // const newChat = await response.json();
    
    return newChat;
  } catch (error) {
    console.error('Error creating new chat:', error);
    throw error;
  }
};