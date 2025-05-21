import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useChat } from '../context/ChatContext';
import databricksLogo from '../assets/images/databricks_icon.svg';
import databricksText from '../assets/images/databricks_text.svg';
import { fetchUserInfo, getServingEndpoints, ServingEndpoint } from '../api/chatApi';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronDown } from '@fortawesome/free-solid-svg-icons';

const UserMenuContainer = styled.div`
  position: relative;
`;

const Avatar = styled.button`
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #434A93;
  color: white;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
`;

const MenuDropdown = styled.div<{ isOpen: boolean }>`
  display: ${props => props.isOpen ? 'flex' : 'none'};
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 8px;
  width: 240px;
  background: #F6F7F9;
  box-shadow: 0px 4px 8px rgba(27, 49, 57, 0.04);
  border-radius: 2px;
  border: 1px solid #D1D9E1;
  flex-direction: column;
  z-index: 100;
`;

const UserInfo = styled.div`
  padding: 10px;
  color: #5F7281;
  font-size: 13px;
  line-height: 16px;
  border-bottom: 1px solid #D1D9E1;
  margin: 4px 2px;
`;

const MenuItem = styled.button`
  margin: 2px;
  width: 100%;
  padding: 10px;
  text-align: left;
  background: none;
  border: none;
  font-size: 12px;
  color: #11171C;
  cursor: pointer;
  line-height: 20px;

  &:hover {
    background: rgba(0, 0, 0, 0.04);
  }
`;

const LogoDropdownContainer = styled.div`
  position: relative;
  display: flex;
  align-items: center;
`;

const LogoDropdownButton = styled.button`
  display: flex;
  align-items: center;
  gap: 4px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 15px 10px;
  border-radius: 4px;
  &:hover {
    background: rgba(67, 74, 147, 0.08);
  }
`;

const ModelDropdown = styled.div<{ isOpen: boolean }>`
  display: ${props => props.isOpen ? 'block' : 'none'};
  position: absolute;
  top: 110%;
  left: 0;
  min-width: 180px;
  background: #fff;
  box-shadow: 0px 4px 8px rgba(27, 49, 57, 0.08);
  border-radius: 2px;
  border: 1px solid #D1D9E1;
  z-index: 200;
  padding: 4px 0;
  max-height: 240px;
  overflow-y: auto;
`;

const ModelDropdownItem = styled.button<{ selected: boolean }>`
  width: 100%;
  padding: 8px 16px;
  background: ${props => props.selected ? 'rgba(67, 74, 147, 0.08)' : 'none'};
  border: none;
  text-align: left;
  font-size: 15px;
  color: #11171C;
  cursor: pointer;
  &:hover {
    background: rgba(67, 74, 147, 0.12);
  }
`;

const LogoIcon = styled.img`
  height: 22px;
`;

const LogoText = styled.img`
  height: 22px;
  margin-left: 4px;
`;

const SelectedEndpointText = styled.span`
  margin-left: 10px;
  font-size: 15px;
  color: #5F7281;
  align-self: center;
  margin-bottom: -10px;
`;

const ChevronIcon = styled.span<{ open: boolean }>`
  display: inline-block;
  align-self: center;
  margin-left: 8px;
  margin-top: 6px;
  font-size: 13px;
  color: #5F7281;
`;

const UserMenu: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const logoDropdownRef = useRef<HTMLDivElement>(null);
  const { logout, currentEndpoint, setCurrentEndpoint } = useChat();
  const [userInfo, setUserInfo] = useState<{username: string, email: string, displayName: string} | null>(null);
  const [availableEndpoints, setAvailableEndpoints] = useState<ServingEndpoint[]>([]);
  const [isLoadingEndpoints, setIsLoadingEndpoints] = useState(false);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
      if (logoDropdownRef.current && !logoDropdownRef.current.contains(event.target as Node)) {
        setIsModelDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    const getUserInfo = async () => {
      try {
        const userInfo = await fetchUserInfo();
        setUserInfo(userInfo);
      } catch (error) {
        console.error('Failed to fetch user info:', error);
      }
    };
    getUserInfo();
  }, []);

  useEffect(() => {
    const fetchModels = async () => {
      setIsLoadingEndpoints(true);
      try {
        const endpoints = await getServingEndpoints();
        setAvailableEndpoints(endpoints);
      } catch (error) {
        console.error('Failed to fetch models:', error);
      } finally {
        setIsLoadingEndpoints(false);
      }
    };
    fetchModels();
  }, []);

  const handleLogout = () => {
    try {
      logout();
    } catch (error) {
      console.error('Failed to logout:', error);
    }
    setIsOpen(false);
  };

  if(!userInfo) {
    return null;
  }

  return (
    <>
      <LogoDropdownContainer ref={logoDropdownRef}>
        <LogoDropdownButton
          onClick={() => setIsModelDropdownOpen(open => !open)}
          aria-haspopup="listbox"
          aria-expanded={isModelDropdownOpen}
          data-testid="logo-dropdown-trigger"
        >
          <LogoIcon src={databricksLogo} alt="Databricks Logo" data-testid="logo-icon"/>
          <LogoText src={databricksText} alt="Databricks" data-testid="logo-text"/>
          <SelectedEndpointText data-testid="selected-endpoint-text">
            {currentEndpoint || (isLoadingEndpoints ? 'Loading...' : 'Select model')}
          </SelectedEndpointText>
          <ChevronIcon open={isModelDropdownOpen} data-testid="dropdown-chevron">
            <FontAwesomeIcon icon={faChevronDown} />
          </ChevronIcon>
        </LogoDropdownButton>
        <ModelDropdown isOpen={isModelDropdownOpen} data-testid="model-dropdown">
          {isLoadingEndpoints ? (
            <ModelDropdownItem selected={false} disabled>
              {currentEndpoint || 'Loading endpoints...'}
            </ModelDropdownItem>
          ) : (
            availableEndpoints.map((endpoint) => (
              <ModelDropdownItem
                key={endpoint.name}
                selected={endpoint.name === currentEndpoint}
                onClick={() => {
                  setCurrentEndpoint(endpoint.name);
                  setIsModelDropdownOpen(false);
                }}
                data-testid={`model-dropdown-item-${endpoint.name}`}
              >
                {endpoint.name}
              </ModelDropdownItem>
            ))
          )}
        </ModelDropdown>
      </LogoDropdownContainer>
      <UserMenuContainer ref={menuRef}>
        <Avatar onClick={() => setIsOpen(!isOpen)}>{userInfo.username.charAt(0).toUpperCase()}</Avatar>
        <MenuDropdown isOpen={isOpen}>
          <UserInfo>
            {userInfo.displayName}<br />
            <span style={{fontSize: '12px', color: '#5F7281', display: 'block', marginTop: '2px'}}>{userInfo.email}</span>
          </UserInfo>
          <MenuItem onClick={handleLogout}>Log out</MenuItem>
        </MenuDropdown>
      </UserMenuContainer>
    </>
  );
};

export default UserMenu; 