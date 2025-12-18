
import React from 'react';
import { Zap, Settings, User } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

const Header: React.FC = () => {
  return (
    <header className="top-0 left-0 right-0 z-50 bg-white/80 dark:bg-black/80 backdrop-blur-sm 
                     border-b border-gray-200 dark:border-cyan-500/30 px-6 py-1 transition-colors duration-300">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Zap className="w-8 h-8 text-cyan-500 dark:text-cyan-400 animate-pulse" />
            <div className="absolute inset-0 w-8 h-8 bg-cyan-500/20 dark:bg-cyan-400/20 rounded-full blur-md"></div>
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-purple-600 dark:from-cyan-400 dark:to-purple-400 bg-clip-text text-transparent">
              NEXUS
            </h1>
            <p className="text-xs text-gray-600 dark:text-gray-400 uppercase tracking-widest mt-0.5">AI Tutor • IITI Community</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <ThemeToggle />
          <button className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800/50 hover:bg-gray-200 dark:hover:bg-gray-700/50 
                           transition-colors duration-300 border border-gray-200 dark:border-gray-600/30 
                           hover:border-gray-300 dark:hover:border-cyan-500/50">
            <Settings className="w-5 h-5 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-cyan-400 transition-colors" />
          </button>
          <button className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800/50 hover:bg-gray-200 dark:hover:bg-gray-700/50 
                           transition-colors duration-300 border border-gray-200 dark:border-gray-600/30 
                           hover:border-gray-300 dark:hover:border-cyan-500/50">
            <User className="w-5 h-5 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-cyan-400 transition-colors" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
