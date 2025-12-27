import React from 'react';
const LinkedInPost: React.FC<{ onBack: () => void }> = ({ onBack }) => (
  <div className="min-h-screen bg-gray-100 p-20 flex justify-center">
    <div className="bg-white w-96 p-6 rounded shadow border">
      <button onClick={onBack} className="text-blue-600 text-xs mb-4">Back</button>
      <div className="font-bold text-sm mb-4">GenAI PHY Architect</div>
      <p className="text-xs mb-4">🚀 Transforming 5G Engineering with Multi-Agent RAG. High compliance, low latency.</p>
      <div className="h-40 bg-gray-200 rounded"></div>
    </div>
  </div>
);
export default LinkedInPost;