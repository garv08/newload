import React from 'react';
import { NodeDetail } from '../constants';

const DetailsModal: React.FC<{ isOpen: boolean; onClose: () => void; data: NodeDetail | null }> = ({ isOpen, onClose, data }) => {
  if (!isOpen || !data) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose}></div>
      <div className="relative w-full max-w-lg bg-white rounded-2xl shadow-2xl overflow-hidden">
        <div className={`p-6 border-b ${data.color.split(' ')[0]}`}>
          <div className="flex items-center gap-3">
            <span className="text-4xl">{data.icon}</span>
            <h2 className={`text-xl font-bold ${data.color.split(' ')[1]}`}>{data.title}</h2>
          </div>
        </div>
        <div className="p-6 space-y-4">
          <p className="text-gray-600 text-sm">{data.description}</p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-50 p-3 rounded">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">Inputs</h4>
              <ul className="text-xs space-y-1">{data.inputs.map((d, i) => <li key={i}>• {d}</li>)}</ul>
            </div>
            <div className="bg-gray-50 p-3 rounded">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-2">Outputs</h4>
              <ul className="text-xs space-y-1">{data.outputs.map((d, i) => <li key={i}>• {d}</li>)}</ul>
            </div>
          </div>
          <button onClick={onClose} className="w-full py-2 bg-gray-900 text-white rounded font-bold text-sm">Close</button>
        </div>
      </div>
    </div>
  );
};
export default DetailsModal;