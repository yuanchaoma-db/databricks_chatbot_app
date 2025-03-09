import React from 'react';
import styled from 'styled-components';
import databricksLogo from '../assets/images/databricks_icon.svg';
import databricksText from '../assets/images/databricks_text.svg';
import userIconUrl from '../assets/images/user_icon.svg';

interface ChatTopNavProps {
  'data-testid'?: string;
}

const TopNavContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 48px;
  padding: 0 16px;
  background-color: #FFFFFF;
  width: 100%;
`;

const LogoContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
  padding-left: 8px;
  height: 32px;
`;

const LogoIcon = styled.img`
  height: 16px;
`;

const LogoText = styled.img`
  height: 13px;
  margin-left: 4px;
`;

const UserContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const UserIcon = styled.div`
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
    cursor: pointer;
`;

const ChatTopNav: React.FC<ChatTopNavProps> = (props) => {
  return (
    <TopNavContainer data-testid="chat-top-nav">
      <LogoContainer>
        <LogoIcon src={databricksLogo} alt="Databricks Logo" />
        <LogoText src={databricksText} alt="Databricks" />
      </LogoContainer>
      <UserContainer>
        <UserIcon>S</UserIcon>
      </UserContainer>
    </TopNavContainer>
  );
};

export default ChatTopNav; 