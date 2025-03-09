export interface Message {
  content: string;
  role: 'user' | 'assistant';
}

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
} 