import { Message, Chat } from '../types';

//export const API_URL = '/chat-api';
export const API_URL = 'http://localhost:8000/chat-api';


export const sendMessage = async (
  content: string, 
  sessionId: string,
  onChunk: (chunk: { 
    message_id: string,
    content?: string, 
    sources?: any[],
    metrics?: {
      timeToFirstToken?: number;
      totalTime?: number;
    },
    model?: string
  }) => void
): Promise<void> => {
  try {
    const response = await fetch(
      `${API_URL}/chat?session_id=${sessionId}`, 
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({ content })
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No reader available');

    const decoder = new TextDecoder();
    let accumulatedContent = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6);
          if (jsonStr && jsonStr !== '{}') {
            try {
              const data = JSON.parse(jsonStr);
              if (data.content) {
                accumulatedContent += data.content;
              }
              console.log('data', data);
              console.log('accumulatedContent', accumulatedContent);
              onChunk({
                message_id: data.message_id,
                content: accumulatedContent,
                sources: data.sources,
                metrics: data.metrics
              });
            } catch (e) {
              console.error('Error parsing JSON:', e);
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error sending message:', error);
    throw error;
  }
};

export const getChatHistory = async (): Promise<{ sessions: Chat[] }> => {
  try {
    const response = await fetch(`${API_URL}/chats`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching chat history:', error);
    return { sessions: [] };
  }
};
export const getModel = async (): Promise<string> => {
  try {
    const response = await fetch(`${API_URL}/model`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.model;
  } catch (error) {
    console.error('Error fetching model:', error);
    return '';
  }
};

export const postError = async (
  sessionId: string,
  errorMessage: Message,
): Promise<void> => {
  try {
    const response = await fetch(
      `${API_URL}/error?session_id=${sessionId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: errorMessage.message_id,
          content: errorMessage.content,
          role: errorMessage.role,
          timestamp: errorMessage.timestamp,
          sources: errorMessage.sources,
          metrics: errorMessage.metrics
        })
      }
    );

    if (!response.ok) {
      console.error('Failed to post error to backend');
    }
  } catch (error) {
    console.error('Error posting error message:', error);
  }
};

export const regenerateMessage = async (
  content: string,
  sessionId: string,
  messageId: string,
  onChunk: (chunk: {
    content?: string,
    sources?: any[],
    metrics?: {
      timeToFirstToken?: number;
      totalTime?: number;
    }
  }) => void
): Promise<void> => {
  try {
    const response = await fetch(
      `${API_URL}/regenerate?session_id=${sessionId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({ 
          message_id: messageId,
          original_content: content 
        })
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No reader available');

    const decoder = new TextDecoder();
    let accumulatedContent = '';
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.slice(6);
          if (jsonStr && jsonStr !== '{}') {
            try {
              const data = JSON.parse(jsonStr);
              if (data.content) {
                accumulatedContent += data.content;
              }
              onChunk({
                ...data,
                content: accumulatedContent // Send accumulated content instead of just the chunk
              });
            } catch (e) {
              console.error('Error parsing JSON:', e);
            }
          }
        }
      }
    }
  } catch (error) {
    console.error('Error regenerating message:', error);
    throw error;
  }
};

export const postRegenerateError = async (
  sessionId: string,
  messageId: string,
  errorMessage: Message,
): Promise<void> => {
  try {
    const response = await fetch(
      `${API_URL}/regenerate/error?session_id=${sessionId}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message_id: messageId,
          content: errorMessage.content,
          role: errorMessage.role,
          timestamp: errorMessage.timestamp,
          sources: errorMessage.sources,
          metrics: errorMessage.metrics
        })
      }
    );

    if (!response.ok) {
      console.error('Failed to post regenerate error to backend');
    }
  } catch (error) {
    console.error('Error posting regenerate error message:', error);
  }
};

export const fetchUserInfo = async (): Promise<{ username: string; email: string }> => {
  try {
    const response = await fetch(`${API_URL}/login`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    console.log('data--->', data);
    return data.user_info;
  } catch (error) {
    console.error('Error fetching user info:', error);
    throw error;
  }
};

// export const getSessionMessages = async (sessionId: string): Promise<{ messages: Message[] }> => {
//   try {
//     const response = await fetch(`${API_URL}/sessions/${sessionId}/messages`);
//     if (!response.ok) {
//       throw new Error(`HTTP error! status: ${response.status}`);
//     }
//     const data = await response.json();
//     return data.messages;
//   } catch (error) {
//     console.error('Error fetching session messages:', error);
//     throw error;
//   }
// };