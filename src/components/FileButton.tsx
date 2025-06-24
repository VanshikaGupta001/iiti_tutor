
import React from 'react';
import { FileText, Download } from 'lucide-react';

interface FileButtonProps {
  fileName: string;
  fileBlob?: Blob;
  onDownload: () => void;
  className?: string;
}

const FileButton: React.FC<FileButtonProps> = ({ 
  fileName, 
  onDownload, 
  className = "" 
}) => {
  const getFileIcon = () => {
    if (fileName.toLowerCase().endsWith('.pdf')) {
      return <FileText className="w-4 h-4 text-red-500" />;
    }
    return <FileText className="w-4 h-4 text-gray-500" />;
  };

  return (
    <div className={`
      inline-flex items-center space-x-2 px-3 py-2 
      bg-gray-700/50 border border-gray-600/30 rounded-lg 
      hover:bg-gray-600/50 hover:border-gray-500/50 
      transition-all duration-200 cursor-pointer
      ${className}
    `} onClick={onDownload}>
      {getFileIcon()}
      <span className="text-sm text-gray-200 truncate max-w-[200px]">
        {fileName}
      </span>
      <Download className="w-3 h-3 text-gray-400" />
    </div>
  );
};

export default FileButton;
