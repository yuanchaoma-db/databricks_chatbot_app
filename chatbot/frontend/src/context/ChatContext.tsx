import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Message, Chat } from '../types';
import { sendMessage as apiSendMessage, getChatHistory, API_URL, postError, regenerateMessage as apiRegenerateMessage, postRegenerateError, getModel, rateMessage as apiRateMessage, logout as apiLogout } from '../api/chatApi';
import { v4 as uuid } from 'uuid';

interface ChatContextType {
  currentChat: Chat | null;
  chats: Chat[];
  messages: Message[];
  loading: boolean;
  model: string;
  sendMessage: (content: string, includeHistory: boolean) => Promise<void>;
  selectChat: (chatId: string) => void;
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  startNewSession: () => void;
  copyMessage: (content: string) => void;
  regenerateMessage: (messageId: string, includeHistory: boolean) => Promise<void>;
  rateMessage: (messageId: string, rating: 'up' | 'down') => void;
  messageRatings: {[messageId: string]: 'up' | 'down'};
  logout: () => void;
  error: string | null;
  clearError: () => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider = ({ children }: { children: ReactNode }) => {
  const [currentChat, setCurrentChat] = useState<Chat | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState<boolean>(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(uuid());
  const [messageRatings, setMessageRatings] = useState<{[messageId: string]: 'up' | 'down'}>({});
  const [model, setModel] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const clearError = () => setError(null);

  useEffect(() => {
    const fetchChats = async () => {
      try {
        const chatHistory = await getChatHistory();
        console.log('Fetched chat history:', chatHistory);
        setChats(chatHistory.sessions || []);
        if (chatHistory.sessions?.length > 0) {
          setCurrentChat(chatHistory.sessions[0]);
        }
      } catch (error) {
        console.error('Failed to fetch chat history:', error);
        setError('Failed to load chat history. Please try again.');
      }
    };

    const fetchModel = async () => {
      try {
        const model = await getModel();
        setModel(model);
      } catch (error) {
        console.error('Failed to fetch model:', error);
        setError('Failed to load model information.');
      }
    };

    fetchChats();
    fetchModel();
  }, []);

  const sendMessage = async (content: string, includeHistory: boolean = true) => {
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

    setMessages(prev => [...prev, userMessage, thinkingMessage]);
    setLoading(true);
    setError(null);

    try {
      let accumulatedContent = '';
      let messageSources: any[] | null = null;
      let messageMetrics: { timeToFirstToken?: number; totalTime?: number } | null = null;
      let messageId = '';
      
      if (!currentSessionId) {
        throw new Error('No active session ID');
      }
      
      await apiSendMessage(content, currentSessionId, includeHistory, (chunk) => {
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

      setMessages(prev => prev.filter(msg => 
        msg.message_id !== thinkingMessage.message_id 
      ).concat(botMessage));
      
    } catch (error) {
      console.error('Error sending message:', error);
      setError('Failed to send message. Please try again.');
      
      // Create error message with proper message_id
      const errorMessageId = uuid();
      const errorMessage: Message = { 
        message_id: errorMessageId,
        content: 'Sorry, I encountered an error. Please try again.',
        role: 'assistant',
        timestamp: new Date(),
        isThinking: false,
        model: model
      };
      
      // Keep the user message but update the thinking message to show error
      setMessages(prev => prev.map(msg => 
        msg.message_id === thinkingMessage.message_id 
          ? errorMessage
          : msg
      ));
      
      // Sync error message with backend
      if (currentSessionId) {
        const response = await postError(currentSessionId, errorMessage);
        if (response.message_id) {
          // Update the message with the new message_id from backend
          setMessages(prev => prev.map(msg => 
            msg.message_id === errorMessageId 
              ? { ...msg, message_id: response.message_id }
              : msg
          ));
        }
      }
    } finally {
      try {
        const historyResponse = await fetch(`${API_URL}/chats`);
        const historyData = await historyResponse.json();
        console.log('Fetched chat history:', historyData);
        if (historyData.sessions) {
          setChats(historyData.sessions);
        }
      } catch (error) {
        console.error('Error fetching chat history:', error);
        setError('Failed to update chat history.');
      }
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
        setError('Failed to copy message to clipboard.');
      });
  };

  const regenerateMessage = async (messageId: string, includeHistory: boolean = true) => {
    console.log('Regenerating message:', messages);
    console.log('Current session ID:', currentSessionId);
    
    const messageIndex = messages.findIndex(msg => msg.message_id === messageId);
    if (messageIndex === -1) {
      console.error('Message not found:', messageId);
      setError('Cannot regenerate message: message not found.');
      return;
    }

    const previousUserMessage = [...messages]
      .slice(0, messageIndex)
      .reverse()
      .find(msg => msg.role === 'user');
    
    if (!previousUserMessage || !currentSessionId) {
      console.error('Cannot regenerate: missing user message or session ID');
      setError('Cannot regenerate message: missing context or session ID.');
      return;
    }
    
    setLoading(true);
    setError(null);

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
        includeHistory,
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
      setError('Failed to regenerate message. Please try again.');
      
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
      console.log('error message:', errorMessage);
      if (currentSessionId) {
        const response = await postRegenerateError(currentSessionId, messageId, errorMessage);
        if (response.message_id && response.message_id !== messageId) {
          // Update the message with the new message_id from backend if it changed
          setMessages(prev => prev.map(msg => 
            msg.message_id === messageId 
              ? { ...msg, message_id: response.message_id }
              : msg
          ));
        }
      }
    } finally {
      try {
        const historyResponse = await fetch(`${API_URL}/chats`);
        const historyData = await historyResponse.json();
        console.log('Fetched chat history:', historyData);
        setChats(historyData.sessions || []);
      } catch (error) {
        console.error('Error fetching chat history:', error);
        setError('Failed to update chat history.');
      }
      setLoading(false);
    }
  };

  const rateMessage = async (messageId: string, rating: 'up' | 'down') => {
    try {
      // If clicking the same rating again, remove it
      const newRating = messageRatings[messageId] === rating ? null : rating;
      
      // Update UI state immediately
      setMessageRatings(prev => {
        if (newRating === null) {
          const { [messageId]: _, ...rest } = prev;
          return rest;
        }
        return { ...prev, [messageId]: newRating };
      });

      // Send to backend
      await apiRateMessage(messageId, newRating);
    } catch (error) {
      console.error('Failed to rate message:', error);
      // Revert UI state on error
      setMessageRatings(prev => {
        const { [messageId]: _, ...rest } = prev;
        return rest;
      });
      setError('Failed to rate message. Please try again.');
    }
  };

  const logout = () => {
    // Clear local state
    setCurrentChat(null);
    setChats([]);
    setMessages([]);
    setCurrentSessionId(null);
    setMessageRatings({});
    
    // Call the logout API endpoint which will handle the redirect
    apiLogout();
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
      error,
      clearError
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