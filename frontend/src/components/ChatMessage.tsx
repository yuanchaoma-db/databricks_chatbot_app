import React from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import copyIconUrl from '../assets/images/copy_icon.svg';
import refreshIconUrl from '../assets/images/sync_icon.svg';
import thumbsUpIconUrl from '../assets/images/thumbs_up_icon.svg';
import thumbsDownIconUrl from '../assets/images/thumbs_down_icon.svg';
import sourceIconUrl from '../assets/images/source_icon.svg';
import buttonIconUrl from '../assets/images/buttonIcon.svg';
import downIconUrl from '../assets/images/down_icon.svg';
import { Message } from '../types';
import { useChat } from '../context/ChatContext';

const MessageContainer = styled.div<{ isUser: boolean }>`
  display: flex;
  flex-direction: column;
  width: 100%;
  align-items: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  margin-top: ${props => props.isUser ? '8px' : '16px'};
  max-width: ${props => props.isUser ? '80%' : '100%'};
  margin-bottom: 10px;
`;

const UserMessageContent = styled.div`
  background-color: #F5F5F5;
  color: #11171C;
  padding: 8px 16px;
  border-radius: 12px;
  font-size: 13px;
  line-height: 1.5;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
`;

const BotMessageContent = styled.div`
  border-radius: 12px;
  width: 100%;
  padding: 6px;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
  text-align: left;
  font-size: 13px;
  margin: 2px 0;
`;

const ModelInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 0;
`;

const ModelIcon = styled.div`
  width: 20px;
  height: 20px;
  background-image: url(${buttonIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
`;

const ModelName = styled.span`
  font-size: 13px;
  color: #11171C;
  font-weight: 600;
`;

const MessageFooter = styled.div`
  display: flex;
  flex-direction: column;
  margin-top: 8px;
  gap: 8px;
  align-items: flex-start;
`;

const MessageActions = styled.div`
  display: flex;
  gap: 8px;
  margin-top: 4px;
  margin-bottom: 24px;
`;

const ActionButton = styled.button`
  width: 24px;
  height: 24px;
  border: none;
  background-color: transparent;
  cursor: pointer;
  display: flex;
  justify-content: center;
  align-items: center;
  
  &:hover {
    background-color: #F0F0F0;
    border-radius: 4px;
  }
`;

const CopyButton = styled(ActionButton)`
  background-image: url(${copyIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  }
`;

const RefreshButton = styled(ActionButton)`
  background-image: url(${refreshIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  }
`;

const ThumbsUpButton = styled(ActionButton)<{ active: boolean }>`
  background-image: url(${thumbsUpIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  ${props => props.active && `
    background-color: rgba(34, 114, 180, 0.08);
  `}

`;

const ThumbsDownButton = styled(ActionButton)<{ active: boolean }>`
  background-image: url(${thumbsDownIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  ${props => props.active && `
    background-color: rgba(34, 114, 180, 0.08);
  `}
  
`;

const SourcesButton = styled.button`
  background: none;
  border: none;
  color: #11171C;
  font-size: 13px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border: 1px solid #E0E0E0;
  border-radius: 4px;
  background-image: url(${downIconUrl});
  background-size: 14px;
  background-repeat: no-repeat;
  background-position: right 4px center;
  padding-right: 24px;
  
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    border: 1px solid #2272B4;
    color: #0E538B;
  }
`;

const SourcesRow = styled.div`
  display: flex;
  gap: 8px;
`;

const ThinkingIndicator = styled.div`
  font-size: 13px;
  color: #5F7281;
  margin-bottom: 8px;
  align-self: flex-start;
  text-align: left;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const Spinner = styled.div`
  display: inline-block;
  width: 16px;
  height: 16px;
  border: 1px solid transparent;
  border-top: 1px solid #5F7281;
  border-right: 1px solid #5F7281;
  border-radius: 50%;
  animation: spin 0.5s linear infinite;
  margin-right: 8px;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

interface ChatMessageProps {
  message: Message;
  onRegenerate: (messageId: string) => Promise<void>;
  'data-testid'?: string;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, onRegenerate }) => {
  const { copyMessage, regenerateMessage, rateMessage, messageRatings } = useChat();
  const isUser = message.role === 'user';
  
  const currentRating = messageRatings[message.id];

  const handleCopy = () => {
    copyMessage(message.content);
  };

  const handleRegenerate = async () => {
    await onRegenerate(message.id);
  };

  const handleThumbsUp = () => {
    rateMessage(message.id, 'up');
  };

  const handleThumbsDown = () => {
    rateMessage(message.id, 'down');
  };
  
  if (isUser) {
    return (
      <MessageContainer isUser={true} data-testid="user-message-container">
        <UserMessageContent data-testid="user-message-content">
          {message.content}
        </UserMessageContent>
      </MessageContainer>
    );
  }

  if (message.isThinking) {
    return (
      <MessageContainer isUser={false} data-testid="bot-message-container">
        <ModelInfo data-testid="model-info">
          <ModelIcon data-testid="model-icon" />
          <ModelName data-testid="model-name">{message.model || 'Databricks LLM'}</ModelName>
        </ModelInfo>
        <BotMessageContent>
          <ThinkingIndicator>
            <Spinner />
            Thinking...
          </ThinkingIndicator>
        </BotMessageContent>
      </MessageContainer>
    );
  }
  
  return (
    <MessageContainer isUser={false} data-testid="bot-message-container">
      <ModelInfo data-testid="model-info">
        <ModelIcon data-testid="model-icon" />
        <ModelName data-testid="model-name">
          {message.model || 'Databricks LLM'}
        </ModelName>
      </ModelInfo>
      
      <BotMessageContent data-testid="bot-message-content">
        <ReactMarkdown>{message.content}</ReactMarkdown>
        <MessageFooter>
          <SourcesRow>
            <SourcesButton>Sources</SourcesButton>
          </SourcesRow>
          <MessageActions data-testid="message-actions">
            <CopyButton 
              onClick={handleCopy} 
              title="Copy" 
              data-testid={`copy-button-${message.id}`}
            />
            <RefreshButton 
              onClick={handleRegenerate} 
              title="Regenerate" 
              data-testid={`refresh-button-${message.id}`}
            />
            <ThumbsUpButton 
              onClick={handleThumbsUp} 
              title="Thumbs Up" 
              active={currentRating === 'up'}
              data-testid={`thumbs-up-button-${message.id}`}
            />
            <ThumbsDownButton 
              onClick={handleThumbsDown} 
              title="Thumbs Down" 
              active={currentRating === 'down'}
              data-testid={`thumbs-down-button-${message.id}`}
            />
          </MessageActions>
        </MessageFooter>
      </BotMessageContent>
    </MessageContainer>
  );
};

export default ChatMessage; 