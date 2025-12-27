import React from 'react';
const SequenceExplanation: React.FC<{ onBack: () => void }> = ({ onBack }) => (
  <div className="p-12 max-w-2xl mx-auto">
    <button onClick={onBack} className="mb-8 font-bold">← Back</button>
    <h1 className="text-2xl font-black mb-6">Workflow Steps</h1>
    <div className="space-y-6">
      <div className="border-l-4 border-blue-500 pl-4">
        <h3 className="font-bold">1. Input Ingestion</h3>
        <p className="text-sm text-gray-600">Data from specs and logs is converted into embeddings.</p>
      </div>
      <div className="border-l-4 border-orange-500 pl-4">
        <h3 className="font-bold">2. Semantic Search</h3>
        <p className="text-sm text-gray-600">The Router selects the best store for the current intent.</p>
      </div>
    </div>
  </div>
);
export default SequenceExplanation;