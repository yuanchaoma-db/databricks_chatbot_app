import React from 'react';
import styled from 'styled-components';
import { useChat } from '../context/ChatContext';

interface SidebarContainerProps {
  isOpen: boolean;
  'data-testid'?: string;
}

const SidebarContainer = styled.div<SidebarContainerProps>`
  display: flex;
  flex-direction: column;
  height: calc(100vh - 48px);
  width: 100%;
  overflow: hidden;
  padding: ${props => props.isOpen ? '8px 16px 24px 16px' : '0'};
  white-space: nowrap;
`;

const SidebarHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  margin-bottom: 8px;
`;

const SidebarHeaderText = styled.div`
  font-size: 18px;
  font-weight: 600;
  color: #333333;
`;

const ChatList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0;
`;

const ChatItem = styled.div<{ active: boolean }>`
  padding: 6px 12px;
  cursor: pointer;
  font-size: 13px;
  color: ${props => props.active ? '#0E538B' : '#11171C'};
  height: 32px;
  display: flex;
  align-items: center;
  margin-bottom: 0;
  box-shadow: none;
  position: relative;
  overflow: hidden;
  white-space: nowrap;
  background-color: ${props => props.active ? 'rgba(34, 114, 180, 0.08)' : 'transparent'};
  border-radius: 4px;
  
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
  }
`;

const ChatItemText = styled.span`
  position: relative;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  flex: 1;
`;

const Sidebar: React.FC = () => {
  const { chats, currentChat, selectChat, isSidebarOpen, createChat } = useChat();
  //TODO: Add a new chat button
  const handleNewChat = async () => {
    try {
      await createChat();
    } catch (error)  {
      console.error('Failed to create new chat:', error);
    }
  };
  
  return (
    <SidebarContainer isOpen={isSidebarOpen} data-testid="sidebar">
      {isSidebarOpen && (
        <>
          <SidebarHeader data-testid="sidebar-header">
            <SidebarHeaderText data-testid="sidebar-header-text">Recent chats</SidebarHeaderText>
          </SidebarHeader>
          <ChatList data-testid="chat-list">
            {chats.map(chat => (
              <ChatItem 
                key={chat.id} 
                active={currentChat?.id === chat.id}
                onClick={() => selectChat(chat.id)}
                data-testid={`chat-item-${chat.id}`}
              >
                <ChatItemText>{chat.title}</ChatItemText>
              </ChatItem>
            ))}
          </ChatList>
        </>
      )}
    </SidebarContainer>
  );
};

export default Sidebar; 