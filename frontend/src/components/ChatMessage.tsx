import React from 'react';
import styled from 'styled-components';
import ReactMarkdown from 'react-markdown';
import copyIconUrl from '../assets/images/copy_icon.svg';
import refreshIconUrl from '../assets/images/sync_icon.svg';
import thumbsUpIconUrl from '../assets/images/thumbs_up_icon.svg';
import thumbsDownIconUrl from '../assets/images/thumbs_down_icon.svg';
import sourceIconUrl from '../assets/images/source_icon.svg';
import { Message } from '../types';

const MessageContainer = styled.div<{ isUser: boolean }>`
  display: flex;
  flex-direction: column;
  width: 100%;
  align-items: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  text-align: ${props => props.isUser ? 'right' : 'left'};
  margin-top: ${props => props.isUser ? '8px' : '16px'};
  max-width: ${props => props.isUser ? '80%' : '100%'};
`;

const UserMessageContent = styled.div`
  background-color: #F5F5F5;
  color: #11171C;
  padding: 12px 16px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.5;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
`;

const BotMessageContent = styled.div`
  background-color: #FFFFFF;
  border-radius: 12px;
  width: 100%;
  padding: 12px 16px;
  word-wrap: break-word;
  overflow-wrap: break-word;
  white-space: normal;
  text-align: left;
`;

const ModelInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding: 4px 0;
`;

const ModelIcon = styled.div`
  width: 20px;
  height: 20px;
  background-color: #f0f0f0;
  border-radius: 50%;
`;

const ModelName = styled.span`
  font-size: 14px;
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
  gap: 16px;
  margin-top: 4px;
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
`;

const RefreshButton = styled(ActionButton)`
  background-image: url(${refreshIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
`;

const ThumbsUpButton = styled(ActionButton)`
  background-image: url(${thumbsUpIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
`;

const ThumbsDownButton = styled(ActionButton)`
  background-image: url(${thumbsDownIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
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
  background-image: url(${sourceIconUrl});
  background-size: 14px;
  background-repeat: no-repeat;
  background-position: left 4px center;
  padding-left: 24px;
  
  &:hover {
    color: #11171C;
    background-color: #F5F5F5;
  }
`;

const SourcesRow = styled.div`
  display: flex;
  gap: 8px;
`;

interface ChatMessageProps {
  message: Message;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';
  
  if (isUser) {
    return (
      <MessageContainer isUser={true}>
        <UserMessageContent>{message.content}</UserMessageContent>
      </MessageContainer>
    );
  }
  
  return (
    <MessageContainer isUser={false}>
      <ModelInfo>
        <ModelIcon />
        <ModelName>Databricks LLM</ModelName>
      </ModelInfo>
      
      <BotMessageContent>
        <ReactMarkdown>{message.content}</ReactMarkdown>
        <MessageFooter>
          <SourcesRow>
            <SourcesButton>Sources</SourcesButton>
          </SourcesRow>
          <MessageActions>
            <CopyButton title="Copy" />
            <RefreshButton title="Regenerate" />
            <ThumbsUpButton title="Thumbs Up" />
            <ThumbsDownButton title="Thumbs Down" />
          </MessageActions>
        </MessageFooter>
      </BotMessageContent>
    </MessageContainer>
  );
};

export default ChatMessage; 