
import React from 'react';
import { Bot, User } from 'lucide-react';
import FileButton from './FileButton';

interface MessageProps {
  message: string;
  isBot: boolean;
  timestamp: string;
  file?: Blob;
  fileName?: string;
}

const Message: React.FC<MessageProps> = ({ 
  message, 
  isBot, 
  timestamp, 
  file, 
  fileName 
}) => {
  const handleFileDownload = () => {
    if (file && fileName) {
      const url = URL.createObjectURL(file);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  return (
    <div className={`flex items-start space-x-3 mb-6 ${isBot ? '' : 'flex-row-reverse space-x-reverse'}`}>
      <div className={`
        flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center
        ${isBot 
          ? 'bg-gradient-to-r from-cyan-500 to-purple-500 shadow-lg shadow-cyan-500/25' 
          : 'bg-gradient-to-r from-green-500 to-blue-500 shadow-lg shadow-green-500/25'
        }
      `}>
        {isBot ? (
          <Bot className="w-5 h-5 text-white" />
        ) : (
          <User className="w-5 h-5 text-white" />
        )}
      </div>
      
      <div className={`flex-1 max-w-md ${isBot ? '' : 'flex flex-col items-end'}`}>
        {/* File attachment display */}
        {file && fileName && (
          <div className={`mb-2 ${isBot ? '' : 'flex justify-end'}`}>
            <FileButton
              fileName={fileName}
              fileBlob={file}
              onDownload={handleFileDownload}
            />
          </div>
        )}
        
        <div className={`
          px-4 py-3 rounded-2xl shadow-lg backdrop-blur-sm max-h-96 overflow-y-auto transition-colors duration-300
          ${isBot 
            ? 'bg-gray-100 dark:bg-gray-800/80 border border-gray-200 dark:border-cyan-500/20 text-gray-900 dark:text-white rounded-tl-none' 
            : 'bg-gradient-to-r from-cyan-600 to-purple-600 text-white rounded-tr-none'
          }
        `}>
          <pre className="text-sm leading-relaxed whitespace-pre-wrap break-words font-mono text-inherit m-0">
            {message}
          </pre>
        </div>
        <div className={`text-xs text-gray-500 dark:text-gray-400 mt-1 px-2 ${isBot ? 'text-left' : 'text-right'}`}>
          {timestamp}
        </div>
      </div>
    </div>
  );
};

export default Message;
