import axios from 'axios';
import { Message, Chat, ServingEndpoint } from '../types';

const API_URL = 'http://localhost:8000/api';

export const sendMessage = async (
  content: string, 
  model: string,
  onChunk: (chunk: { content?: string, sources?: any[] }) => void
): Promise<void> => {
  try {
    const response = await fetch(`${API_URL}/chat?model=${model}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      },
      body: JSON.stringify({ content, model })
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
              onChunk(data);
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

export const fetchServingEndpoints = async (): Promise<ServingEndpoint> => {
  try {
    const response = await fetch(`${API_URL}/serving-endpoints`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching endpoints:', error);
    throw error;
  }
};