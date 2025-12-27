import React from 'react';
const PresentationViewer: React.FC<{ isOpen: boolean; onClose: () => void }> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 bg-black z-[100] text-white p-12 flex flex-col items-center justify-center">
      <button onClick={onClose} className="absolute top-8 right-8 font-bold">Close Presentation</button>
      <div className="text-center">
        <h1 className="text-6xl font-black mb-4">PHY Layer AI</h1>
        <p className="text-2xl text-gray-400">Autonomous 5G/6G Development</p>
      </div>
    </div>
  );
};
export default PresentationViewer;