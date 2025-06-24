
import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

const ThemeToggle: React.FC = () => {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="p-2 rounded-lg bg-gray-200 dark:bg-gray-800/50 hover:bg-gray-300 dark:hover:bg-gray-700/50 
                 transition-all duration-300 border border-gray-300 dark:border-gray-600/30 
                 hover:border-gray-400 dark:hover:border-cyan-500/50 group"
      aria-label="Toggle theme"
    >
      <div className="relative w-5 h-5">
        <Sun 
          className={`w-5 h-5 text-yellow-500 absolute transition-all duration-300 ${
            isDarkMode ? 'opacity-0 rotate-90 scale-0' : 'opacity-100 rotate-0 scale-100'
          }`} 
        />
        <Moon 
          className={`w-5 h-5 text-blue-400 absolute transition-all duration-300 ${
            isDarkMode ? 'opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-0'
          }`} 
        />
      </div>
    </button>
  );
};

export default ThemeToggle;
