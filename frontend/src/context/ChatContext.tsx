import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Message, Chat } from '../types';
import { sendMessage as apiSendMessage, getChatHistory, API_URL, postError, regenerateMessage as apiRegenerateMessage, postRegenerateError, getModel } from '../api/chatApi';
import { v4 as uuid } from 'uuid';

interface ChatContextType {
  currentChat: Chat | null;
  chats: Chat[];
  messages: Message[];
  loading: boolean;
  model: string;
  sendMessage: (content: string) => Promise<void>;
  selectChat: (chatId: string) => void;
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  startNewSession: () => void;
  copyMessage: (content: string) => void;
  regenerateMessage: (messageId: string) => Promise<void>;
  rateMessage: (messageId: string, rating: 'up' | 'down') => void;
  messageRatings: {[messageId: string]: 'up' | 'down'};
  logout: () => void;
}
 
const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(uuid());
  const [messageRatings, setMessageRatings] = useState<{[messageId: string]: 'up' | 'down'}>({});
  const [model, setModel] = useState<string>('');
  useEffect(() => {
    const fetchChats = async () => {
      try {
        const chatHistory = await getChatHistory();
        console.log('Fetched chat history:', chatHistory); // Debug log
        setChats(chatHistory.sessions || []);
        if (chatHistory.sessions?.length > 0) {
          setCurrentChat(chatHistory.sessions[0]);
        }
      } catch (error) {
        console.error('Failed to fetch chat history:', error);
      }
    };
    const fetchModel = async () => {
      try {
        const model = await getModel();
        setModel(model);
      } catch (error) {
        console.error('Failed to fetch model:', error);
      }
    };
    fetchChats();
    fetchModel();
  }, []);

  const sendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Create new session if needed
    if (!currentSessionId) {
      startNewSession();
    }

    const userMessage: Message = { 
      message_id: uuid(),
      content, 
      role: 'user',
      timestamp: new Date()
    };

    const thinkingMessage: Message = {
      message_id: uuid(),
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      isThinking: true
    };

    // Update local state for immediate feedback
    setMessages(prev => [...prev, userMessage, thinkingMessage]);
    setLoading(true);
    const startTime = Date.now();
    try {
      let accumulatedContent = '';
      let messageSources: any[] | null = null;
      let messageMetrics: { timeToFirstToken?: number; totalTime?: number } | null = null;
      let messageId = '';
      
      if (!currentSessionId) {
        throw new Error('No active session ID');
      }
      
      // Send message to backend with session ID
      await apiSendMessage(content, currentSessionId, (chunk) => {
        if (chunk.content) {
          accumulatedContent = chunk.content;
        }
        if (chunk.sources) {
          messageSources = chunk.sources;
        }
        if (chunk.metrics) {
          messageMetrics = chunk.metrics;
        }
        if (chunk.message_id) {
          messageId = chunk.message_id;
        }
    
        // Update only the display messages
        setMessages(prev => prev.map(msg => 
          msg.message_id === thinkingMessage.message_id 
            ? { 
                ...msg, 
                content: chunk.content || '',
                sources: chunk.sources,
                metrics: chunk.metrics,
                isThinking: false,
                model: model
              }
            : msg
        ));
      });

      // Final message update when stream is complete
      const botMessage: Message = {
        message_id: messageId,
        content: accumulatedContent,
        role: 'assistant',
        timestamp: new Date(),
        isThinking: false,
        model: model,
        sources: messageSources,
        metrics: messageMetrics
      };

      // Update display messages
      setMessages(prev => prev.map(msg => 
        msg.message_id === messageId ? botMessage : msg
      ));
      
    } catch (error) {
      const errorMessage: Message = { 
        message_id: thinkingMessage.message_id,
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant',
        timestamp: new Date(),
        model: '',
        metrics: null
      };

      setMessages(prev => prev.map(msg => 
        msg.message_id === thinkingMessage.message_id ? errorMessage : msg
      ));

      // Post error to backend
      if (currentSessionId) {
        await postError(currentSessionId, errorMessage);
      }

    } finally {
      // After completion, fetch updated chat history
      const historyResponse = await fetch(`${API_URL}/chats`);
      const historyData = await historyResponse.json();
      console.log('Fetched chat history:', historyData);
      setChats(historyData.sessions || []);

      setLoading(false);
    }
  };

  const selectChat = (sessionId: string) => {
    
    const selected = chats.find(chat => chat.sessionId === sessionId);
    
    if (selected) {
      setCurrentChat(selected);
      setCurrentSessionId(sessionId);
      
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
    const messageIndex = messages.findIndex(msg => msg.message_id === messageId);
    if (messageIndex === -1) return;

    const previousUserMessage = [...messages]
      .slice(0, messageIndex)
      .reverse()
      .find(msg => msg.role === 'user');
    
    if (!previousUserMessage || !currentSessionId) return;

    setLoading(true);
    const startTime = Date.now();

    // Replace with thinking indicator
    const thinkingMessage: Message = {
      message_id: messageId,
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      isThinking: true,
      model: model
    };
    
    setMessages(prev => {
      const updatedMessages = [...prev];
      updatedMessages[messageIndex] = thinkingMessage;
      return updatedMessages;
    });

    try {
      let messageSources: any[] | null = null;
      let messageMetrics: {
        timeToFirstToken?: number;
        totalTime?: number;
      } | null = null;
      let accumulatedContent = '';
      
      await apiRegenerateMessage(
        previousUserMessage.content,
        currentSessionId,
        messageId,
        (chunk) => {
          if (chunk.content) {
            accumulatedContent = chunk.content;
            setMessages(prev => {
              const updatedMessages = [...prev];
              const currentMessage = updatedMessages[messageIndex];
              updatedMessages[messageIndex] = {
                ...currentMessage,
                message_id: messageId,
                content: accumulatedContent,
                sources: chunk.sources || messageSources,
                metrics: chunk.metrics || messageMetrics,
                isThinking: false,
                model: model
              };
              return updatedMessages;
            });
          }
          if (chunk.sources) {
            messageSources = chunk.sources;
          }
          if (chunk.metrics) {
            messageMetrics = chunk.metrics;
          }
        }
      );

      // Final message update when stream is complete
      const finalMessage: Message = {
        message_id: messageId,
        content: accumulatedContent,
        role: 'assistant',
        timestamp: new Date(),
        isThinking: false,
        model: model,
        sources: messageSources,
        metrics: messageMetrics
      };

      setMessages(prev => prev.map(msg => 
        msg.message_id === messageId ? finalMessage : msg
      ));

    } catch (error) {
      console.error('Error regenerating message:', error);
      const errorMessage: Message = { 
        message_id: messageId,
        content: error instanceof Error && error.message === 'HTTP error! status: 429' 
          ? 'The service is currently experiencing high demand. Please wait a moment and try again.'
          : 'Sorry, I encountered an error while regenerating the message. Please try again.', 
        role: 'assistant',
        timestamp: new Date(),
        model: model,
        isThinking: false,
        metrics: null
      };
      
      setMessages(prev => {
        const updatedMessages = [...prev];
        const messageIndex = updatedMessages.findIndex(msg => msg.message_id === messageId);
        if (messageIndex !== -1) {
          updatedMessages[messageIndex] = errorMessage;
        }
        return updatedMessages;
      });

      if (currentSessionId && messageId) {
        await postRegenerateError(currentSessionId, messageId, errorMessage);
      }
    } finally {

      const historyResponse = await fetch(`${API_URL}/chats`);
      const historyData = await historyResponse.json();
      console.log('Fetched chat history:', historyData);
      setChats(historyData.sessions || []);
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

  const logout = () => {
    // Clear all chat data
    setMessages([]);
    setChats([]);
    setCurrentChat(null);
    setCurrentSessionId(null);
    
    // Add any additional logout logic here (e.g., API calls, clearing tokens)
    // You might want to redirect to a login page or show the welcome screen
  };

  return (
    <ChatContext.Provider value={{
      currentChat,
      chats,
      messages,
      loading,
      model,
      sendMessage,
      selectChat,
      isSidebarOpen,
      toggleSidebar,
      startNewSession,
      copyMessage,
      regenerateMessage,
      rateMessage,
      messageRatings,
      logout,
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