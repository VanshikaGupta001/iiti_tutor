import { getBackendUrl } from '../config/backend';
import { ServiceResponse } from '../types/chat';

export const sendMessage = async (prompt: string, file?: File): Promise<ServiceResponse> => {
  try {
    const formData = new FormData();
    formData.append('prompt', prompt);
    
    if (file) {
      formData.append('file', file);
    }

    // 1. Send Request
    const response = await fetch(getBackendUrl('/route'), {
      method: 'POST',
      body: formData,
      credentials: 'include',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 2. Parse JSON (Backend now ALWAYS returns JSON)
    const data = await response.json();
    
    let fileBlob: Blob | null = null;

    // 3. Check for File (Base64)
    // The new backend sends 'file_base64' and 'mime_type'
    if (data.file_base64) {
      try {
        // Convert Base64 string -> Binary Blob
        const byteCharacters = atob(data.file_base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        
        // Use the dynamic mime type from backend (or default to PDF)
        const mimeType = data.mime_type || 'application/pdf';
        fileBlob = new Blob([byteArray], { type: mimeType });

        // Hack: Attach filename to the blob so UI can read it for download
        (fileBlob as any).name = data.filename || "Nexus_Output.pdf";
        
      } catch (e) {
        console.error('Error processing file from backend:', e);
      }
    }

    // 4. Return Clean Result
    return { 
      text: data.text || "Here is the response.", 
      file: fileBlob 
    };

  } catch (error) {
    console.error('Error sending message:', error);
    throw new Error('Failed to send message. Please check backend status.');
  }
};