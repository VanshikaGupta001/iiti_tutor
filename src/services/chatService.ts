
import { getBackendUrl } from '@/config/backend';
import { ServiceResponse } from '@/types/chat';

export const sendMessage = async (prompt: string, file?: File): Promise<ServiceResponse> => {
  try {
    const formData = new FormData();
    formData.append('prompt', prompt);
    
    if (file) {
      formData.append('file', file);
    }

    const response = await fetch(getBackendUrl('/route'), {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const contentType = response.headers.get('content-type');

    if (contentType && contentType.includes('multipart/mixed')) {
      const boundary = contentType.split('boundary=')[1];
      const text = await response.text();
      const parts = text.split(`--${boundary}`).filter(part => part.trim() && !part.startsWith('--'));

      let textResult: string | null = null;
      let fileBlob: Blob | null = null;

      for (const part of parts) {
        const [header, body] = part.split('\r\n\r\n', 2);
        if (!header || !body) continue;

        if (header.includes('application/json')) {
          try {
            const jsonData = JSON.parse(body);
            textResult = jsonData.text || jsonData.message || null;
          } catch (e) {
            console.error('Error parsing JSON part:', e);
          }
        } else if (header.includes('application/pdf')) {
          try {
            const pdfArrayBuffer = new TextEncoder().encode(body).buffer;
            fileBlob = new Blob([pdfArrayBuffer], { type: 'application/pdf' });
          } catch (e) {
            console.error('Error processing PDF part:', e);
          }
        }
      }

      return { text: textResult, file: fileBlob };
    }

    if (contentType && contentType.includes('application/json')) {
      const data = await response.json();
      return { text: data.text || data.message || null, file: null };
    }

    return { text: 'Unexpected response format.', file: null };
  } catch (error) {
    console.error('Error sending message:', error);
    throw new Error('Failed to send message. Please check backend status.');
  }
};
