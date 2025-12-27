import React from 'react';

interface TechnicalDeepDiveProps {
  onBack: () => void;
}

const TechnicalDeepDive: React.FC<TechnicalDeepDiveProps> = ({ onBack }) => {
  return (
    <div className="min-h-screen bg-white text-slate-800 font-sans pb-20">
      <nav className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <button onClick={onBack} className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg font-bold transition-colors text-sm shadow-md">
          <span>←</span> Back to App
        </button>
        <div className="text-xs font-black text-slate-400 uppercase tracking-widest">
          Technical Specification & Math Models
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-12">
        <header className="mb-12">
          <h1 className="text-4xl font-black text-slate-900 mb-4 tracking-tight">System Specification</h1>
          <p className="text-lg text-slate-600">Detailed mathematical modeling and algorithmic pseudo-code for the Tri-Hybrid PHY Agentic Architecture.</p>
        </header>

        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <span className="w-10 h-10 rounded-lg bg-orange-100 text-orange-600 flex items-center justify-center font-bold">01</span>
            <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Semantic Cache (CAG)</h2>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
              <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 tracking-widest">Mathematical Model</h3>
              <p className="mb-4 text-slate-700 leading-relaxed">
                {'Let $q$ be the user query vector and $C = \{c_1, c_2, ..., c_n\}$ be the set of cached query vectors.'}
              </p>
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-center font-serif text-xl mb-4 italic">
                {'S(q, c_i) = \\frac{q \\cdot c_i}{\\|q\\| \\|c_i\\|}'}
              </div>
              <p className="text-slate-700 leading-relaxed">
                {'The cache returns result $A_{best}$ if:'}
              </p>
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-center font-serif text-xl italic">
                {'\\max_{c_i \\in C} S(q, c_i) > \\tau'}
              </div>
              <p className="mt-4 text-xs text-slate-500 italic">where $\tau = 0.95$ is the defined similarity threshold.</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default TechnicalDeepDive;