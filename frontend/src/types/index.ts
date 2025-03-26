export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp?: Date;
  isThinking?: boolean;
  model?: string;
  sources?: any[] | null;
  metrics?: {
    timeToFirstToken?: number;
    totalTime?: number;
  } | null;
}

export interface Chat {
  id: string;
  sessionId: string;
  firstQuery: string;
  messages: Message[];
  timestamp: Date;
  isActive?: boolean;
}

export interface ServingEndpoint {
  names: string[];
} 