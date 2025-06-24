
import React, { useState, useRef } from 'react';
import { Send, Paperclip } from 'lucide-react';
import { sendMessage } from '../services/chatService';
import { ServiceResponse } from '../types/chat';

interface ChatInputProps {
  onSendMessage: (message: string, isBot?: boolean, file?: File) => void;
  onReceiveResponse: (response: ServiceResponse) => void;
  isLoading: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({ onSendMessage, onReceiveResponse, isLoading }) => {
  const [message, setMessage] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showAttachMenu, setShowAttachMenu] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!message.trim() || isLoading) return;

    const messageText = message.trim();
    const fileToSend = uploadedFile;

    // Clear input immediately for better UX
    setMessage('');
    setUploadedFile(null);

    // Show user message in chat with file if present
    onSendMessage(messageText, false, fileToSend || undefined);

    try {
      // Send to backend
      const response = await sendMessage(messageText, fileToSend || undefined);
      console.log('Backend response:', response);
      
      // Handle the response
      onReceiveResponse(response);
    } catch (error) {
      console.error('Message send failed:', error);
      // Handle error - could show error message to user
      onReceiveResponse({ 
        text: 'Sorry, there was an error sending your message. Please try again.', 
        file: null 
      });
    }
  };

  const handleAttachClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
  };

  return (
    <div className=" border-cyan-500/30 bg-black/80 backdrop-blur-sm ">
      {uploadedFile && (
        <div className="mb-2 text-sm text-white flex items-center space-x-2">
          {uploadedFile.type.startsWith('image') ? (
            <img
              src={URL.createObjectURL(uploadedFile)}
              alt="preview"
              className="h-12 w-12 object-cover rounded-md"
            />
          ) : (
            <span className="truncate max-w-xs">📎 {uploadedFile.name}</span>
          )}
          <button
            type="button"
            onClick={() => setUploadedFile(null)}
            className="text-red-400 hover:underline text-xs"
          >
            Remove
          </button>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex items-end space-x-3">
        <div className="flex-1 relative">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask NEXUS anything related to your studies..."
            className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/30 rounded-2xl 
                     text-white placeholder-gray-400 focus:outline-none focus:border-cyan-500/50 
                     focus:bg-gray-800/70 transition-all duration-200 resize-none min-h-[48px] max-h-32 overflow-y-auto"
            rows={1}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSubmit(e);
              }
            }}
          />

          <div className="absolute right-3 bottom-3 flex items-center space-x-2">
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowAttachMenu((prev) => !prev)}
                className="p-1.5 rounded-lg hover:bg-gray-700/50 transition-colors"
              >
                <Paperclip className="w-4 h-4 text-gray-400 hover:text-cyan-400" />
              </button>

              {showAttachMenu && (
                <div className="absolute bottom-10 right-0 z-50 w-48 bg-gray-800 border border-cyan-500/20 rounded-lg shadow-xl p-2 space-y-1">
                  <button
                    onClick={() => {
                      handleAttachClick();
                      setShowAttachMenu(false);
                    }}
                    className="w-full text-left text-sm text-white hover:text-cyan-400 px-3 py-2 rounded hover:bg-gray-700"
                  >
                    Upload from Computer
                  </button>
                </div>
              )}
            </div>

            <input
              type="file"
              accept="application/pdf,image/*"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={!message.trim() || isLoading}
          className="p-3 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 text-white 
                   disabled:opacity-50 disabled:cursor-not-allowed hover:shadow-lg 
                   hover:shadow-cyan-500/25 transition-all duration-200 hover:scale-105 
                   active:scale-95"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
    </div>
  );
};

export default ChatInput;
