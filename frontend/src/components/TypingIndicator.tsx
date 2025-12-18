import React from 'react';

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex items-start space-x-3 mb-6">
      <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 
                      flex items-center justify-center shadow-lg shadow-cyan-500/25">
        <div className="w-5 h-5 rounded-full bg-white/20 animate-pulse"></div>
      </div>
      
      <div className="px-4 py-3 rounded-2xl rounded-tl-none bg-gray-800/80 border border-cyan-500/20 backdrop-blur-sm">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce"></div>
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
          <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;