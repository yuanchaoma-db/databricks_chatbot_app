import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useChat } from '../context/ChatContext';
import atIconUrl from '../assets/images/at_icon.svg';
import clipIconUrl from '../assets/images/clip_icon.svg';
import sendIconUrl from '../assets/images/send_icon.svg';
import { fetchServingEndpoints } from '../api/chatApi';
import type { ServingEndpoint } from '../types';
import Select from 'react-select';

interface InputContainerProps {
  'data-testid'?: string;
}

const InputContainer = styled.div<InputContainerProps>`
  width: 100%;
  max-width: 680px;
  min-height: 50px;
  height: 100px;
  position: relative;
  border: 1px solid #C0CDD8;
  border-radius: 12px;
  padding: 10px 12px;
  background-color: white;
  box-shadow: 0px 1px 3px -1px rgba(0, 0, 0, 0.05), 0px 2px 0px 0px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 8px;
`;

const TextArea = styled.textarea`
  width: 100%;
  border: none;
  outline: none;
  font-size: 13px;
  padding: 6px 0;
  color: #11171C;
  white-space: pre-wrap;
  word-break: break-word;
  overflow-wrap: break-word;
  min-height: 40px;
  max-height: 90px;
  overflow-y: auto;
  display: block;
  background-color: transparent;
  font-family: inherit;
  margin-bottom: 30px;
  resize: none;
  box-sizing: border-box;
`;

const ButtonsLeft = styled.div`
  display: flex;
  align-items: center;
  position: absolute;
  bottom: 12px;
  left: 12px;
  z-index: 2;
`;

const ButtonsRight = styled.div`
  display: flex;
  align-items: center;
  position: absolute;
  bottom: 12px;
  right: 12px;
  z-index: 2;
`;

const InputButton = styled.button`
  width: 24px;
  height: 24px;
  border: none;
  background: transparent;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  margin: 0 4px;
  
  &:hover {
    background-color: #F0F0F0;
    border-radius: 4px;
  }
`;

const SendButton = styled(InputButton)`
  background-image: url(${sendIconUrl});
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  &:hover {
    background-color: rgba(34, 114, 180, 0.08);
    color: #0E538B;
  }
`;


interface OptionType {
  value: string;
  label: string;
}

const StyledSelect = styled.div`
  width: 200px;
  
  .select-container {
    font-size: 11px;
    
    .select__control {
      border: 1px solid #C0CDD8;
      border-radius: 6px;
      background-color: white;
      min-height: 32px;
      height: 32px;
    }

    .select__value-container {
      padding: 0 8px;
      height: 30px;
    }

    .select__indicators {
      height: 30px;
    }

    .select__menu {
      font-size: 11px;
      border: 1px solid #C0CDD8;
    }

    .select__option {
      padding: 4px 8px;
      &:hover {
        background-color: #F0F0F0;
      }
      &--is-selected {
        background-color: #2272B4;
      }
    }
  }
`;

interface ChatInputProps {
  fixed?: boolean;
  setIsRegenerating: (value: boolean) => void;
}

const ChatInput: React.FC<ChatInputProps> = ({ fixed = false, setIsRegenerating }) => {
  const [inputValue, setInputValue] = useState('');
  const [endpoints, setEndpoints] = useState<OptionType[]>([]);
  const { sendMessage, loading, selectedModel, setSelectedModel } = useChat();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  useEffect(() => {
    const loadEndpoints = async () => {
      try {
        if (!selectedModel) {
          const endpointList = await fetchServingEndpoints();
          const options = endpointList.names.map(name => ({
            value: name,
            label: name
          }));
          setEndpoints(options);
          if (options.length > 0) {
            setSelectedModel(options[0].value);
          }
        }
        } catch (error) {
          console.error('Failed to load endpoints:', error);
        }
    };
    loadEndpoints();
  }, [setSelectedModel]);
  
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const newHeight = Math.max(50, Math.min(textareaRef.current.scrollHeight, 100));
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, [inputValue]);
  
  const handleSubmit = async () => {
    console.log(`Selected Model====: ${selectedModel}`);
    if (inputValue.trim() && !loading) {
      setIsRegenerating(false);
      await sendMessage(inputValue, selectedModel);
      setInputValue('');
      if (textareaRef.current) {
        textareaRef.current.style.height = '50px';
      }
    }
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  
  return (
    <InputContainer data-testid="chat-input-container">
      <ButtonsLeft data-testid="buttons-left">
        <StyledSelect>
          <Select
            className="select-container"
            classNamePrefix="select"
            value={endpoints.find(opt => opt.value === selectedModel)}
            onChange={(option) => setSelectedModel(option?.value || '')}
            options={endpoints}
            isDisabled={loading}
            isSearchable={true}
            menuPlacement="auto"
          />
        </StyledSelect>
      </ButtonsLeft>
      <TextArea
        ref={textareaRef}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        placeholder="Ask anything"
        onKeyDown={handleKeyDown}
        data-testid="chat-input-textarea"
      />
      <ButtonsRight data-testid="buttons-right">
        <SendButton onClick={handleSubmit} disabled={loading} data-testid="send-button" />
      </ButtonsRight>
    </InputContainer>
  );
};

export default ChatInput; 