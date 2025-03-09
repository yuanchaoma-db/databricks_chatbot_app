import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Message, Chat } from '../types';
import { sendMessage as apiSendMessage, getChatHistory, createNewChat } from '../api/chatApi';

interface ChatContextType {
  currentChat: Chat | null;
  chats: Chat[];
  messages: Message[];
  loading: boolean;
  sendMessage: (content: string) => Promise<void>;
  selectChat: (chatId: string) => void;
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  createChat: () => Promise<Chat>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);

  useEffect(() => {
    const fetchChats = async () => {
      try {
        const chatHistory = await getChatHistory();
        setChats(chatHistory);
        if (chatHistory.length > 0) {
          setCurrentChat(chatHistory[0]);
        }
      } catch (error) {
        console.error('Failed to fetch chat history:', error);
      }
    };

    fetchChats();
  }, []);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage: Message = { content, role: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    
    // Set loading state
    setLoading(true);
    
    try {
      // Send to API
      const response = await apiSendMessage(content);
      
      // Add bot response
      setMessages((prev) => [...prev, response]);
    } catch (error) {
      // Handle error
      const errorMessage: Message = { 
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant' 
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const selectChat = (chatId: string) => {
    const selected = chats.find(chat => chat.id === chatId) || null;
    setCurrentChat(selected);
    setMessages(selected?.messages || []);
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const createChat = async () => {
    try {
      // Call API to create a new chat
      const newChat = await createNewChat();
      
      // Update chats list
      setChats((prev) => [newChat, ...prev]);
      
      // Set as current chat
      setCurrentChat(newChat);
      
      // Clear messages
      setMessages([]);
      
      // Ensure sidebar is open to show the new chat
      if (!isSidebarOpen) {
        setIsSidebarOpen(true);
      }
      
      return newChat;
    } catch (error) {
      console.error('Failed to create new chat:', error);
      throw error;
    }
  };

  return (
    <ChatContext.Provider value={{
      currentChat,
      chats,
      messages,
      loading,
      sendMessage,
      selectChat,
      isSidebarOpen,
      toggleSidebar,
      createChat
    }}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}; 