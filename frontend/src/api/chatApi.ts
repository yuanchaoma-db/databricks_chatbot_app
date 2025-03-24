import axios from 'axios';
import { Message, Chat } from '../types';

const API_URL = 'http://localhost:8000/api';

export const sendMessage = async (content: string): Promise<Message> => {
  try {
    const response = await axios.post(`${API_URL}/chat`, { content });
    return response.data;
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