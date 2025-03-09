import React from 'react';
import styled from 'styled-components';
import { useChat } from '../context/ChatContext';
import menuIconUrl from '../assets/images/menu_icon.svg';
import newChatIconUrl from '../assets/images/newchat_icon.svg';
import Sidebar from './Sidebar';

interface LeftComponentProps {
  'data-testid'?: string;
}

interface LeftContainerProps {
  isOpen: boolean;
  'data-testid'?: string;
}

const LeftContainer = styled.div<LeftContainerProps>`
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  z-index: 10;
  height: 100vh;
  width: ${props => props.isOpen ? '300px' : '100px'};
  background-color: #FFFFFF;
  border-right: ${props => props.isOpen ? '1px solid #DCDCDC' : 'none'};
  transition: width 0.3s ease;
`;

const NavLeft = styled.div`
  display: flex;
  align-items: center;
  height: 48px;
  padding: 0 8px;
  gap: 16px;
  justify-content: space-between;
`;

const MenuButton = styled.button`
  width: 32px;
  height: 32px;
  margin-left: 12px;
  border: none;
  background-color: transparent;
  background-image: url(${menuIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  cursor: pointer;
  border-radius: 4px;
  
  &:hover {
    background-color: #F5F5F5;
  }
`;

const NewChatButton = styled.button`
  width: 32px;
  height: 32px;
  border: none;
  background-color: transparent;
  background-image: url(${newChatIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  cursor: pointer;
  border-radius: 4px;
  
  &:hover {
    background-color: #F5F5F5;
  }
`;

const LeftComponent: React.FC<LeftComponentProps> = (props) => {
  const { isSidebarOpen, toggleSidebar, createChat } = useChat();
  
  const handleNewChat = async () => {
    try {
      await createChat();
    } catch (error) {
      console.error('Failed to create new chat:', error);
    }
  };
  
  return (
    <LeftContainer isOpen={isSidebarOpen} data-testid="left-component">
      <NavLeft data-testid="nav-left">
        <MenuButton onClick={toggleSidebar} data-testid="menu-button" />
        <NewChatButton onClick={handleNewChat} data-testid="new-chat-button" />
      </NavLeft>
      <Sidebar />
    </LeftContainer>
  );
};

export default LeftComponent; 