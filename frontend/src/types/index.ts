export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp?: Date;
  isThinking?: boolean;
  model?: string;
}

export interface Chat {
  id: string;
  sessionId: string;
  firstQuery: string;
  messages: Message[];
  timestamp: Date;
  isActive?: boolean;
} 