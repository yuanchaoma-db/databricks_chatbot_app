import React from 'react';
import styled from 'styled-components';
import { useChat } from '../context/ChatContext';
import newChatIconUrl from '../assets/images/newchat_icon.svg';

interface SidebarProps {
  hideNewChatButton?: boolean;
}

interface SidebarContainerProps {
  isOpen: boolean;
  'data-testid'?: string;
}

const SidebarContainer = styled.div<SidebarContainerProps>`
  display: flex;
  flex-direction: column;
  height: calc(100vh - 48px);
  width: 100%;
  background-color: #FFFFFF;
  overflow: hidden;
  padding: ${props => props.isOpen ? '8px 16px 24px 16px' : '0'};
  white-space: nowrap;
  opacity: ${props => props.isOpen ? '1' : '0'};
  transition: opacity 0.2s ease;
`;

const SidebarHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0;
  margin-bottom: 16px;
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
  
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
  }
  
  &::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    height: 100%;
    width: 30px;
    background: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,1) 90%);
    pointer-events: none;
  }
`;

const NewChatButton = styled.button`
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  border: 1px solid #C0CDD8;
  border-radius: 4px;
  background-color: transparent;
  color: #11171C;
  font-size: 13px;
  cursor: pointer;
  box-shadow: 0px 1px 0px rgba(0, 0, 0, 0.05);
  height: 32px;
  margin-top: 6px;
  
  &:hover {
    background-color: #F5F5F5;
  }
`;

const NewChatIcon = styled.div`
  width: 16px;
  height: 16px;
  background-image: url(${newChatIconUrl});
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
`;

const Sidebar: React.FC<SidebarProps> = ({ hideNewChatButton = false }) => {
  const { chats, currentChat, selectChat, isSidebarOpen, createChat } = useChat();
  
  const handleNewChat = async () => {
    try {
      await createChat();
    } catch (error) {
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
                {chat.title}
              </ChatItem>
            ))}
          </ChatList>
          {!hideNewChatButton && (
            <NewChatButton 
              onClick={handleNewChat} 
              data-testid="sidebar-new-chat-button"
            >
              <NewChatIcon />
              <span>New chat</span>
            </NewChatButton>
          )}
        </>
      )}
    </SidebarContainer>
  );
};

export default Sidebar; 