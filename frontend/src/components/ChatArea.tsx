import React, { useRef, useEffect, useState } from 'react';
import styled from 'styled-components';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useChat } from '../context/ChatContext';
import buttonIconUrl from '../assets/images/buttonIcon.svg';
import ChatTopNav from './ChatTopNav';

interface ChatContainerProps {
  'data-testid'?: string;
  sidebarOpen: boolean;
}

const ChatContainer = styled.div<ChatContainerProps>`
  display: flex;
  flex-direction: column;
  flex: 1;
  height: 100vh;
  margin-left: ${props => props.sidebarOpen ? '300px' : '100px'};
  width: ${props => props.sidebarOpen ? 'calc(100% - 300px)' : 'calc(100% - 100px)'};
  transition: margin-left 0.3s ease, width 0.3s ease;
  overflow: hidden;
`;

const ChatContent = styled.div`
  display: flex;
  flex-direction: column;
  flex: 1;
  padding: 0 16px;
  overflow-y: auto;
  height: calc(100vh - 48px); 
`;

const WelcomeContainer = styled.div<{ visible: boolean }>`
  display: ${props => props.visible ? 'flex' : 'none'};
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  max-width: 660px;
  margin: auto;
  padding: 24px 16px;
`;

const WelcomeMessage = styled.h1`
  font-size: 24px;
  font-weight: 600;
  color: #333;
  margin-bottom: 24px;
  text-align: center;
`;

const SuggestionButtons = styled.div`
  display: flex;
  flex-wrap: nowrap;
  justify-content: center;
  gap: 8px;
  margin-top: 24px;
  width: 100%;
`;

const SuggestionButton = styled.button`
  display: flex;
  align-items: center;
  text-wrap-mode: nowrap;
  text-overflow: ellipsis;
  gap: 8px;
  padding: 8px 8px;
  background-color: #FFFFFF;
  border: 1px solid #C0CDD8;
  border-radius: 8px;
  color: #11171C;
  font-size: 13px;
  cursor: pointer;
  box-shadow: 0px 1px 0px rgba(0, 0, 0, 0.05);
  
  &:hover {
    background-color:rgb(252, 240, 252);
  }
`;

const SuggestionIcon = styled.div`
  width: 16px;
  height: 16px;
  background-image: url(${buttonIconUrl});
  background-size: contain;
  background-repeat: no-repeat;
`;

const Disclaimer = styled.div`
  font-size: 12px;
  color: #767676;
  text-align: center;
  margin-top: 24px;
`;

const FixedInputWrapper = styled.div<{ visible: boolean }>`
  display: ${props => props.visible ? 'flex' : 'none'};
  flex-direction: column;
  align-items: center;
  width: 100%;
  max-width: 660px;
  margin: 2px auto;
  position: sticky;
  bottom: 20px;
  background-color: white;
  z-index: 10;
  box-shadow: 0 -10px 20px rgba(255, 255, 255, 0.9);
`;

const DisclaimerFixed = styled.div`
  font-size: 12px;
  color: #767676;
  text-align: center;
  margin-top: 8px;
  width: 100%;
`;

const MessagesContainer = styled.div`
  display: flex;
  flex-direction: column;
  width: 650px;
  margin: 0 auto;
  max-width: 100%;
`;

const ChatArea: React.FC = () => {
  const { messages, loading, isSidebarOpen, sendMessage, regenerateMessage } = useChat();
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);  
  const [hasStartedChat, setHasStartedChat] = useState(false);

  useEffect(() => {
    if (messages?.length > 0 && !hasStartedChat) {
      setHasStartedChat(true);
    } else if (messages?.length === 0 && hasStartedChat) {
      // Reset hasStartedChat when messages is cleared (new session)
      setHasStartedChat(false);
    }
  }, [messages, hasStartedChat]);
  
  console.log(`ChatArea Messages====>: ${messages}`);
  useEffect(() => {
    if (messagesEndRef.current && !isRegenerating) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isRegenerating]);
  
  const hasMessages = messages?.length > 0 || hasStartedChat;
  
  const handleSuggestionClick = (suggestion: string) => {
    setHasStartedChat(true);
    sendMessage(suggestion);
  };
  
  const handleRegenerate = async (messageId: string) => {
    setIsRegenerating(true);
    await regenerateMessage(messageId);
  };
  
  return (
    <ChatContainer data-testid="chat-area" sidebarOpen={isSidebarOpen}>
      <ChatTopNav />
      <ChatContent data-testid="chat-content">
        <WelcomeContainer visible={!hasMessages} data-testid="welcome-container">
          <WelcomeMessage data-testid="welcome-message">What can I help with?</WelcomeMessage>
          <ChatInput setIsRegenerating={setIsRegenerating} data-testid="chat-input" />
          <SuggestionButtons data-testid="suggestion-buttons">
            <SuggestionButton data-testid="suggestion-button" onClick={() => handleSuggestionClick("Find tables to query")}>
              <SuggestionIcon />
              <span>Find tables to query</span>
            </SuggestionButton>
            <SuggestionButton data-testid="suggestion-button" onClick={() => handleSuggestionClick("Debug my notebook")}>
              <SuggestionIcon />
              <span>Debug my notebook</span>
            </SuggestionButton>
            <SuggestionButton data-testid="suggestion-button" onClick={() => handleSuggestionClick("Fix my code")}>
              <SuggestionIcon />
              <span>Fix my code</span>
            </SuggestionButton>
            <SuggestionButton onClick={() => handleSuggestionClick("What is Unity Catalog?")}>
              <SuggestionIcon />
              <span>What is Unity Catalog?</span>
            </SuggestionButton>
          </SuggestionButtons>
          <Disclaimer>Chatbot may make mistakes. Check important info.</Disclaimer>
        </WelcomeContainer>
        
        {hasMessages && (
          <MessagesContainer data-testid="messages-container">
            {messages.map((message, index) => (
              <ChatMessage    
                key={index} 
                message={message}
                onRegenerate={handleRegenerate}
                data-testid={`message-${index}`}
              />
            ))}
            {!isRegenerating && <div ref={messagesEndRef} />}
          </MessagesContainer>
        )}
      </ChatContent>
      
      <FixedInputWrapper visible={hasMessages} data-testid="fixed-input-wrapper">
        <ChatInput fixed={true} setIsRegenerating={setIsRegenerating} />
        <DisclaimerFixed>Chatbot may make mistakes. Check important info.</DisclaimerFixed>
      </FixedInputWrapper>
    </ChatContainer>
  );
};

export default ChatArea; 