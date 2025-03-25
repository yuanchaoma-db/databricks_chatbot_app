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
  startNewSession: () => void;
  copyMessage: (content: string) => void;
  regenerateMessage: (messageId: string) => Promise<void>;
  rateMessage: (messageId: string, rating: 'up' | 'down') => void;
  messageRatings: {[messageId: string]: 'up' | 'down'};
}
 
const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string>(uuid());
  const [messageRatings, setMessageRatings] = useState<{[messageId: string]: 'up' | 'down'}>({});

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
      id: uuid(),
      content, 
      role: 'user',
      timestamp: new Date()
    };

    const thinkingMessage: Message = {
      id: uuid(),
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      isThinking: true
    };

    let chatToUse: Chat;
    let updatedMessages: Message[];

    if (!currentChat) {
      chatToUse = {
        id: uuid(),
        sessionId: currentSessionId,
        firstQuery: content,
        messages: [userMessage, thinkingMessage],
        timestamp: new Date(),
        isActive: true
      };
      
      setCurrentChat(chatToUse);
      setChats(prev => [chatToUse, ...prev]);
      updatedMessages = [userMessage, thinkingMessage];
    } else {
      chatToUse = currentChat;
      updatedMessages = [...messages, userMessage, thinkingMessage];
    }
    
    setMessages(updatedMessages);
    setLoading(true);
    
    try {
      const response = await apiSendMessage(content);
      
      const botMessage = { 
        ...response, 
        id: thinkingMessage.id,
        timestamp: new Date() 
      };

      // Use updatedMessages instead of messages to include the latest state
      const finalMessages = updatedMessages.map(msg => 
        msg.id === thinkingMessage.id ? botMessage : msg
      );
      
      setMessages(finalMessages);
      setChats(prev => prev.map(chat => 
        chat.id === chatToUse.id
          ? { ...chat, messages: finalMessages }
          : chat
      ));
    } catch (error) {
      const errorMessage: Message = { 
        id: thinkingMessage.id,
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant',
        timestamp: new Date()
      };

      // Use updatedMessages instead of messages to include the latest state
      const finalMessages = updatedMessages.map(msg => 
        msg.id === thinkingMessage.id ? errorMessage : msg
      );
      
      setMessages(finalMessages);
      setChats(prev => prev.map(chat => 
        chat.id === chatToUse.id
          ? { ...chat, messages: finalMessages }
          : chat
      ));
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


  const startNewSession = () => {
    const newSessionId = uuid();
    setCurrentSessionId(newSessionId);
    setCurrentChat(null);
    setMessages([]);
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
      .then(() => {
        console.log('Message copied to clipboard');
      })
      .catch(err => {
        console.error('Failed to copy message:', err);
      });
  };

  const regenerateMessage = async (messageId: string) => {
    const messageIndex = messages.findIndex(msg => msg.id === messageId);
    if (messageIndex === -1) return;

    const previousUserMessage = [...messages]
      .slice(0, messageIndex)
      .reverse()
      .find(msg => msg.role === 'user');
    
    if (!previousUserMessage) return;

    setLoading(true);

    // Replace with thinking indicator
    const thinkingMessage: Message = {
      id: messageId,
      content: '',  // Empty content since we'll show the indicator
      role: 'assistant',
      timestamp: new Date(),
      isThinking: true
    };
    
    const messagesWithThinking = [...messages];
    messagesWithThinking[messageIndex] = thinkingMessage;
    setMessages(messagesWithThinking);

    try {
      const response = await apiSendMessage(previousUserMessage.content);
      const botMessage = { 
        ...response, 
        id: uuid(),
        timestamp: new Date() 
      };

      // Replace thinking message with the new bot message
      const updatedMessages = [...messages];
      updatedMessages[messageIndex] = botMessage;
      
      setMessages(updatedMessages);
      setChats(prev => {
        const newChats = prev.map(chat => 
          chat.sessionId === currentChat?.sessionId
            ? { ...chat, messages: updatedMessages }
            : chat
        );
        return newChats;
      });
    } catch (error) {
      const errorMessage: Message = { 
        id: uuid(),
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant',
        timestamp: new Date()
      };
      // Replace the error message at the same index
      const updatedMessages = [...messages];
      updatedMessages[messageIndex] = errorMessage;
      setMessages(updatedMessages);
    } finally {
      setLoading(false);
    }
  };

  const rateMessage = (messageId: string, rating: 'up' | 'down') => {
    setMessageRatings(prev => {
      // If same rating is clicked again, remove the rating
      if (prev[messageId] === rating) {
        const { [messageId]: _, ...rest } = prev;
        return rest;
      }
      // Otherwise set the new rating
      return { ...prev, [messageId]: rating };
    });
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
      startNewSession,
      copyMessage,
      regenerateMessage,
      rateMessage,
      messageRatings,
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