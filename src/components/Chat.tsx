

import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import ChatInput from './ChatInput';
import TypingIndicator from './TypingIndicator';

interface ChatMessage {
  id: string;
  message: string;
  isBot: boolean;
  timestamp: string;
}

interface ChatProps {
  selectedFeature: string | null;
}

const Chat: React.FC<ChatProps> = ({ selectedFeature }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      message: "Welcome to NEXUS! I'm your AI tutor for the IITI community...",
      isBot: true,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (selectedFeature) {
      const prompts: Record<string, string> = {
        doubt: "I'm ready to help with your course doubts...",
        schedule: "Let's create a personalized study schedule...",
        solve: "I can help you solve question papers...",
        generate: "I'll generate custom question papers...",
        upload: "Upload your study materials or textbook pages...",
        tutor: "I'm your personal AI tutor!"
      };
      handleSendMessage(prompts[selectedFeature] || '', true);
    }
  }, [selectedFeature]);

  const handleSendMessage = (message: string, isBot = false) => {
    const newMessage: ChatMessage = {
      id: Date.now().toString(),
      message,
      isBot,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    setMessages(prev => [...prev, newMessage]);

    if (!isBot) {
      setIsLoading(true);
      setTimeout(() => {
        const responses = [
          "That's an excellent question...",
          "Here's how we can approach this...",
          "Based on IITI curriculum...",
        ];
        setMessages(prev => [...prev, {
          ...newMessage,
          id: (Date.now() + 1).toString(),
          message: responses[Math.floor(Math.random() * responses.length)],
          isBot: true
        }]);
        setIsLoading(false);
      }, 1500);
    }
  };

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 pt-[81px] space-y-4 scroll-smooth">

        {messages.map((msg) => (
          <div key={msg.id} className="max-w-4xl mx-auto">
            <Message message={msg.message} isBot={msg.isBot} timestamp={msg.timestamp} />
          </div>
        ))}
        {isLoading && <div className="max-w-4xl mx-auto"><TypingIndicator /></div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="w-full border-t border-gray-700 bg-black px-4 py-2">
        <div className="max-w-4xl mx-auto">
          <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </div>
      </div>
    </div>
  );
};

export default Chat;






  
