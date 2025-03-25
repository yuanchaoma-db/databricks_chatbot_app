export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp?: Date;
  isThinking?: boolean;
}

export interface Chat {
  id: string;
  sessionId: string;
  firstQuery: string;
  messages: Message[];
  timestamp: Date;
  isActive?: boolean;
} 