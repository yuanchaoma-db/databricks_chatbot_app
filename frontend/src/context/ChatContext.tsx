import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Message, Chat } from '../types';
import { sendMessage as apiSendMessage, getChatHistory, createNewChat } from '../api/chatApi';
import { v4 as uuid } from 'uuid';

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
  startNewSession: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>(uuid());

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

    const userMessage: Message = { 
      content, 
      role: 'user',
      timestamp: new Date()
    };

    let chatToUse: Chat;

    if (!currentChat) {
      chatToUse = {
        id: uuid(),
        sessionId: currentSessionId,
        firstQuery: content,
        messages: [userMessage],
        timestamp: new Date(),
        isActive: true
      };
      
      setCurrentChat(chatToUse);
      setChats(prev => {
        const updatedChats = [chatToUse, ...prev];
        return updatedChats;
      });
      setMessages([userMessage]);
    } else {
      chatToUse = currentChat; 
      const updatedMessages = [...messages, userMessage];
      
      setMessages(updatedMessages);
      setChats(prev => prev.map(chat => 
        chat.sessionId === currentChat.sessionId
          ? { ...chat, messages: updatedMessages }
          : chat
      ));
    }

    setLoading(true);
    
    try {
      const response = await apiSendMessage(content);
      
      const botMessage = { ...response, timestamp: new Date() };
      const updatedMessages = [...messages, userMessage, botMessage];
      console.log('Final updated messages:', updatedMessages);
      
      setMessages(updatedMessages);
      setChats(prev => {
        const newChats = prev.map(chat => 
          chat.sessionId === chatToUse.sessionId
            ? { ...chat, messages: updatedMessages }
            : chat
        );
        return newChats;
      });
    } catch (error) {
      const errorMessage: Message = { 
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant',
        timestamp: new Date()
      };
      const updatedMessages = [...messages, userMessage, errorMessage];
      
      setMessages(updatedMessages);
      setChats(prev => {
        const newChats = prev.map(chat => 
          chat.sessionId === chatToUse.sessionId // Use chatToUse instead of currentChat
            ? { ...chat, messages: updatedMessages }
            : chat
        );
        return newChats;
      });
    } finally {
      setLoading(false);
    }
  };

  const selectChat = (chatId: string) => {
    const selected = chats.find(chat => chat.id === chatId);
    if (selected) {
      setCurrentChat(selected);
      
      const sessionMessages = chats
        .filter(chat => chat.sessionId === selected.sessionId)
        .flatMap(chat => chat.messages);
      
      setMessages(sessionMessages);
    }
  };

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const createChat = async () => {
    try {
      const newChat = await createNewChat();
      setChats((prev) => [newChat, ...prev]);
      setCurrentChat(newChat);
      setMessages([]);
      
      return newChat;
    } catch (error) {
      console.error('Failed to create new chat:', error);
      throw error;
    }
  };

  const startNewSession = () => {
    const newSessionId = uuid();
    setCurrentSessionId(newSessionId);
    setCurrentChat(null);
    setMessages([]);
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
      createChat,
      startNewSession
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