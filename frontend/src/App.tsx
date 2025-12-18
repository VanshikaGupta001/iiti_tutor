import 'katex/dist/katex.min.css';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const handleFeatureSelect = (feature: string) => {
    setSelectedFeature(feature);
    setSidebarOpen(false);
  };

  return (
    <ThemeProvider>
      <div className="h-screen w-screen bg-gradient-to-br from-gray-100 via-white to-gray-100 dark:from-gray-900 dark:via-black dark:to-gray-900 
                      text-gray-900 dark:text-white flex flex-col overflow-hidden transition-colors duration-300">
        <Header />

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar toggle */}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="md:hidden fixed top-4 left-4 z-50 p-2 rounded-lg bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm 
                      border border-gray-300 dark:border-cyan-500/30 hover:border-gray-400 dark:hover:border-cyan-500/50 
                      transition-colors duration-300"
          >
            {sidebarOpen ? (
              <X className="w-5 h-5 text-gray-600 dark:text-cyan-400" />
            ) : (
              <Menu className="w-5 h-5 text-gray-600 dark:text-cyan-400" />
            )}
          </button>

          {sidebarOpen && (
            <div 
              className="md:hidden fixed inset-0 bg-black/30 dark:bg-black/50 backdrop-blur-sm z-30 transition-colors duration-300"
              onClick={() => setSidebarOpen(false)}
            />
          )}

          <Sidebar isOpen={sidebarOpen} onFeatureSelect={handleFeatureSelect} />

          <div className="flex-1 flex flex-col overflow-hidden">
            <Chat selectedFeature={selectedFeature} />
          </div>
        </div>
      </div>
    </ThemeProvider>
  );
}

export default App;
