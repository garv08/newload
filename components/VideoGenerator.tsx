import React from 'react';
const VideoGenerator: React.FC<{ onClose: () => void }> = ({ onClose }) => (
  <div className="fixed inset-0 bg-black/80 z-[110] flex items-center justify-center">
    <div className="bg-white p-8 rounded-xl w-80 text-center">
      <h3 className="font-bold mb-4">Video Generator</h3>
      <p className="text-xs text-gray-500 mb-6">Select a project key to generate cinematic workflow visualizations.</p>
      <button onClick={onClose} className="bg-blue-600 text-white px-4 py-2 rounded">Close</button>
    </div>
  </div>
);
export default VideoGenerator;