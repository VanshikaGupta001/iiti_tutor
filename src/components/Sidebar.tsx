
import React from 'react';
import { 
  Calendar, 
  FileText, 
  Upload, 
  Brain,
  Target,
  Lightbulb,
  Zap
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onFeatureSelect: (feature: string) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onFeatureSelect }) => {
  const features = [
    { id: 'doubt', icon: Lightbulb, label: 'Ask Doubt', color: 'text-yellow-500 dark:text-yellow-400' },
    { id: 'schedule', icon: Calendar, label: 'Study Schedule', color: 'text-green-500 dark:text-green-400' },
    { id: 'solve', icon: Target, label: 'Solve Papers', color: 'text-blue-500 dark:text-blue-400' },
    { id: 'generate', icon: FileText, label: 'Generate Papers', color: 'text-purple-500 dark:text-purple-400' },
    { id: 'upload', icon: Upload, label: 'Upload Book', color: 'text-pink-500 dark:text-pink-400' },
    { id: 'tutor', icon: Brain, label: 'AI Tutor', color: 'text-cyan-500 dark:text-cyan-400' },
  ];

  return (
    <div className={`
      fixed left-0 top-0 h-full bg-white/90 dark:bg-black/90 backdrop-blur-sm 
      border-r border-gray-200 dark:border-cyan-500/30 
      transform transition-all duration-300 z-40
      ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      w-80 md:relative md:translate-x-0
    `}>
      <div className="flex flex-col h-full p-6 pt-20 md:pt-6 space-y-20">
        <div className="space-y-8">
          <div>
            <h2 className="text-lg font-semibold text-gray-800 dark:text-white mb-2">Features</h2>
            <div className="w-12 h-0.5 bg-gradient-to-r from-cyan-500 dark:from-cyan-400 to-transparent"></div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <button
                  key={feature.id}
                  onClick={() => onFeatureSelect(feature.id)}
                  className="flex flex-col items-center justify-center p-3 rounded-lg bg-gray-50 dark:bg-gray-800/30 
                             hover:bg-gray-100 dark:hover:bg-gray-700/50 
                             transition-all duration-200 border border-gray-200 dark:border-gray-600/20 
                             hover:border-gray-300 dark:hover:border-cyan-500/30
                             group hover:scale-[1.02] active:scale-[0.98] text-center"
                >
                  <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-800/50 ${feature.color} group-hover:scale-110 transition-transform mb-1`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="text-xs font-medium text-gray-800 dark:text-white group-hover:text-cyan-600 dark:group-hover:text-cyan-400 leading-tight">
                    {feature.label}
                  </div>
                  <div className="text-[10px] text-gray-500 dark:text-gray-400">Click</div>
                </button>
              );
            })}
          </div>
        </div>

        <div className="mt-10 p-5 rounded-lg bg-gradient-to-r from-cyan-500/10 to-purple-500/10 
                        border border-cyan-200 dark:border-cyan-500/20">
          <div className="flex items-center space-x-2 mb-2">
            <Zap className="w-4 h-4 text-cyan-500 dark:text-cyan-400" />
            <span className="text-sm font-semibold text-cyan-600 dark:text-cyan-400 tracking-wide">AI Status</span>
          </div>

          <div className="text-xs text-gray-600 dark:text-gray-300 mb-1">
            Neural networks online and ready to assist.
          </div>

          <div className="flex items-center mt-3">
            <div className="w-2 h-2 bg-green-500 dark:bg-green-400 rounded-full animate-pulse mr-2"></div>
            <span className="text-xs text-green-600 dark:text-green-400">Connected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
