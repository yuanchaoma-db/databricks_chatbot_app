import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Message, Chat } from '../types';
import { sendMessage as apiSendMessage, getChatHistory } from '../api/chatApi';
import { v4 as uuid } from 'uuid';

interface ChatContextType {
  currentChat: Chat | null;
  chats: Chat[];
  messages: Message[];
  loading: boolean;
  sendMessage: (content: string, model: string) => Promise<void>;
  selectChat: (chatId: string) => void;
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  startNewSession: () => void;
  copyMessage: (content: string) => void;
  regenerateMessage: (messageId: string, model: string) => Promise<void>;
  rateMessage: (messageId: string, rating: 'up' | 'down') => void;
  messageRatings: {[messageId: string]: 'up' | 'down'};
  logout: () => void;
  selectedModel: string;
  setSelectedModel: (model: string) => void;
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
  const [selectedModel, setSelectedModel] = useState<string>('');

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

  const sendMessage = async (content: string, model: string) => {
    if (!content.trim()) return;
    setSelectedModel(model);

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
        sessionId: currentSessionId || uuid(),
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
      let accumulatedContent = '';
      
      await apiSendMessage(content, model, (chunk) => {
        accumulatedContent += chunk;
        
        setMessages(prev => prev.map(msg => 
          msg.id === thinkingMessage.id 
            ? { 
                ...msg, 
                content: accumulatedContent,
                isThinking: false,
              }
            : msg
        ));
      });

      // Final message update when stream is complete
      const botMessage: Message = {
        id: thinkingMessage.id,
        content: accumulatedContent,
        role: 'assistant',
        timestamp: new Date(),
        isThinking: false,
        model: model
      };

      setMessages(prev => prev.map(msg => 
        msg.id === thinkingMessage.id ? botMessage : msg
      ));

      setChats(prev => prev.map(chat => 
        chat.id === chatToUse.id
          ? { ...chat, messages: messages.map(msg => 
              msg.id === thinkingMessage.id ? botMessage : msg
            ) }
          : chat
      ));

    } catch (error) {
      const errorMessage: Message = { 
        id: thinkingMessage.id,
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => prev.map(msg => 
        msg.id === thinkingMessage.id ? errorMessage : msg
      ));

      setChats(prev => prev.map(chat => 
        chat.id === chatToUse.id
          ? { ...chat, messages: messages.map(msg => 
              msg.id === thinkingMessage.id ? errorMessage : msg
            ) }
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

  const regenerateMessage = async (messageId: string, model: string) => {
    setSelectedModel(model);
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
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      isThinking: true,
      model: model
    };
    
    const messagesWithThinking = [...messages];
    messagesWithThinking[messageIndex] = thinkingMessage;
    setMessages(messagesWithThinking);

    try {
      let accumulatedContent = '';
      
      await apiSendMessage(previousUserMessage.content, model, (chunk) => {
        accumulatedContent += chunk;
        
        // Update the thinking message with accumulated content
        setMessages(prev => {
          const updatedMessages = [...prev];
          updatedMessages[messageIndex] = {
            ...thinkingMessage,
            content: accumulatedContent,
            isThinking: false,
            model: model
          };
          return updatedMessages;
        });
      });

      // Final message update when stream is complete
      const botMessage: Message = {
        id: messageId,
        content: accumulatedContent,
        role: 'assistant',
        timestamp: new Date(),
        isThinking: false,
        model: model
      };

      setMessages(prev => {
        const updatedMessages = [...prev];
        updatedMessages[messageIndex] = botMessage;
        return updatedMessages;
      });

      setChats(prev => prev.map(chat => 
        chat.sessionId === currentChat?.sessionId
          ? { ...chat, messages: messages.map(msg => 
              msg.id === messageId ? botMessage : msg
            ) }
          : chat
      ));

    } catch (error) {
      const errorMessage: Message = { 
        id: messageId,
        content: 'Sorry, I encountered an error. Please try again.', 
        role: 'assistant',
        timestamp: new Date(),
        model: model
      };
      
      setMessages(prev => {
        const updatedMessages = [...prev];
        updatedMessages[messageIndex] = errorMessage;
        return updatedMessages;
      });
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
      sendMessage,
      selectChat,
      isSidebarOpen,
      toggleSidebar,
      startNewSession,
      copyMessage,
      selectedModel,
      setSelectedModel,
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