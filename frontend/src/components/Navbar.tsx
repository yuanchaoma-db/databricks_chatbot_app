import React from 'react';
import styled from 'styled-components';
import { useChat } from '../context/ChatContext';
import menuIconUrl from '../assets/images/menu_icon.svg';
import speechIconUrl from '../assets/images/speech_icon.svg';

interface NavContainerProps {
  'data-testid'?: string;
}

const NavContainer = styled.div<NavContainerProps>`
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 48px;
  padding: 0 16px;
  gap: 8px;
  border-bottom: 1px solid #DCDCDC;
`;

interface NavLeftProps {
  isOpen: boolean;
  'data-testid'?: string;
}

const NavLeft = styled.div<NavLeftProps>`
  display: flex;
  align-items: center;
  height: 32px;
  gap: 28px;
  justify-content: ${props => props.isOpen ? 'flex-start' : 'space-between'};
  min-width: ${props => props.isOpen ? '280px' : 'auto'};
  margin: 8px 16px;
  padding-right: ${props => props.isOpen ? '16px' : '0'};
`;

const NavCenter = styled.div`
  margin-left: 100px;
  justify-content: flex-start;
  display: flex;
  align-items: center;
`;

const NavRight = styled.div`
  justify-content: flex-end;
  margin-right: 16px;
  display: flex;
  align-items: center;
  height: 32px;
  gap: 28px;
`;

const NavButton = styled.button`
  width: 32px;
  height: 32px;
  border: none;
  background: transparent;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  margin-right: 0px;
  border-radius: 4px;
  
  &:hover {
    background-color: #F5F5F5;
  }
`;

const MenuButton = styled(NavButton)`
  background-image: url(${menuIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
`;

const SpeechButton = styled(NavButton)`
  background-image: url(${speechIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
`;

const LogoContainer = styled.div<{ isOpen: boolean }>`
  height: 32px;
  padding: 0 4px;
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: ${props => props.isOpen ? '200px' : '0'};
`;

const UserAvatar = styled.div`
  width: 24px;
  height: 24px;
  background-color: #434A93;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
  font-size: 12px;
  font-weight: 600;
`;

const Logo = styled.div`
  font-weight: bold;
  font-size: 18px;
  color: #333;
`;

const Navbar: React.FC = () => {
  const { toggleSidebar, isSidebarOpen } = useChat();
  
  return (
    <NavContainer data-testid="navbar">
      <NavLeft isOpen={isSidebarOpen} data-testid="nav-left">
        <MenuButton onClick={toggleSidebar} data-testid="menu-button" />
        {!isSidebarOpen && <SpeechButton data-testid="speech-button" />}
      </NavLeft>
      
      <NavCenter>
        <LogoContainer isOpen={isSidebarOpen}>
          <Logo>Databricks</Logo>
        </LogoContainer>
      </NavCenter>
      
      <NavRight>
        <UserAvatar>S</UserAvatar>
      </NavRight>
    </NavContainer>
  );
};

export default Navbar; 