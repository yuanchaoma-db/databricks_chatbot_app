export interface Message {
  content: string;
  role: 'user' | 'assistant';
  timestamp?: Date;
}

export interface Chat {
  id: string;
  sessionId: string;
  firstQuery: string;
  messages: Message[];
  timestamp: Date;
  isActive?: boolean;
} 