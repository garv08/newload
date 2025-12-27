import React from 'react';
import KaTeX from './KaTeX';

const Paper2CodeView: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center pb-24">
      {/* Navigation Header */}
      <nav className="w-full bg-slate-900 border-b border-white/10 sticky top-0 z-50 px-8 py-4 flex items-center justify-between">
        <button 
          onClick={onBack} 
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg font-bold transition-all text-xs uppercase tracking-widest shadow-xl"
        >
          <span>←</span> Back to System
        </button>
        <div className="flex items-center gap-3">
          <div className="px-3 py-1 bg-blue-500/20 text-blue-400 text-[10px] font-black rounded-full uppercase tracking-widest border border-blue-500/30">
            Paper2Code Engine v0.9-Alpha
          </div>
        </div>
      </nav>

      <div className="max-w-6xl w-full px-6 py-12">
        <header className="mb-20 text-left border-l-4 border-blue-600 pl-8">
          <h1 className="text-4xl md:text-7xl font-black mb-4 tracking-tighter uppercase leading-none">
            Paper <span className="text-blue-500">2</span> Code
          </h1>
          <p className="text-slate-400 text-xl font-light max-w-2xl leading-relaxed italic">
            "Automated numerical equivalence: Transforming complex mathematical publications into executable PHY simulations."
          </p>
        </header>

        {/* Logic Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mb-24">
          
          {/* Step 1: Semantic Parsing */}
          <div className="bg-slate-900 p-10 rounded-[2.5rem] border border-white/5 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-48 h-48 bg-blue-600/5 rounded-full blur-[80px] group-hover:bg-blue-600/10 transition-colors"></div>
            <div className="w-12 h-12 bg-blue-600 text-white rounded-2xl flex items-center justify-center font-black mb-8 shadow-lg shadow-blue-600/20">01</div>
            <h3 className="text-2xl font-black mb-6 uppercase tracking-tight">Equation Extraction</h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-8">
              The utility scans LaTeX tokens or PDF bitmaps to reconstruct the mathematical computational graph. Each equation is mapped to a directed acyclic graph (DAG).
            </p>
            <div className="p-6 bg-black rounded-2xl border border-white/10 mb-6">
              <div className="text-[10px] font-black text-blue-500 uppercase tracking-[0.2em] mb-4">Input Probability Model</div>
              <KaTeX math={'P(\\text{node} | \\text{paper}) = \\prod_{i=1}^n \\text{Softmax}(\\frac{v_i \\cdot w^T}{\\sqrt{d}})'} block />
            </div>
            <ul className="text-xs text-slate-500 space-y-3 font-mono">
              <li>• Token Alignment with arXiv Metadata</li>
              <li>• Parameter Boundary Detection</li>
              <li>• Reference Cross-Checking</li>
            </ul>
          </div>

          {/* Step 2: AST Reconstruction */}
          <div className="bg-slate-900 p-10 rounded-[2.5rem] border border-white/5 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-600/5 rounded-full blur-[80px] group-hover:bg-emerald-600/10 transition-colors"></div>
            <div className="w-12 h-12 bg-emerald-600 text-white rounded-2xl flex items-center justify-center font-black mb-8 shadow-lg shadow-emerald-600/20">02</div>
            <h3 className="text-2xl font-black mb-6 uppercase tracking-tight">Code Synthesis (AST)</h3>
            <p className="text-slate-400 text-sm leading-relaxed mb-8">
              Logic is synthesized using an Abstract Syntax Tree (AST) that enforces numeric stability and 3GPP data-type constraints (e.g., fixed-point arithmetic for DSP).
            </p>
            <div className="p-6 bg-black rounded-2xl border border-white/10 mb-6 font-mono text-[10px] text-emerald-400">
{`class PHYSimulation(nn.Module):
    def forward(self, x, noise_floor):
        # Derivation from Eq. 4.1 (Spectral Density)
        SNR = calculate_snr(x, noise_floor)
        capacity = np.log2(1 + SNR) 
        return capacity`}
            </div>
            <ul className="text-xs text-slate-500 space-y-3 font-mono">
              <li>• C++/Python Template Selection</li>
              <li>• Vectorized Kernel Optimization</li>
              <li>• Type-Safety Verification</li>
            </ul>
          </div>
        </div>

        {/* Verification Section */}
        <section className="bg-white text-slate-950 rounded-[3rem] p-12 md:p-20 shadow-2xl relative">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl font-black mb-8 uppercase tracking-tighter">Deterministic Verification</h2>
              <p className="text-slate-600 leading-relaxed mb-8 italic">
                The Paper2Code agent doesn't just "guess" the code; it validates generated outputs by comparing the resulting simulation curves against the original paper data using an MSE-based similarity check.
              </p>
              <div className="p-8 bg-slate-50 rounded-3xl border border-slate-200">
                <div className="text-[10px] font-black text-slate-400 uppercase mb-4 tracking-widest">Equivalence Proof</div>
                <KaTeX math={'\\mathcal{L}_{sim} = \\frac{1}{N} \\sum_{i=1}^N \\| \\hat{y}_i(code) - y_i(paper) \\|^2'} block />
                <div className="mt-6 flex items-center gap-4">
                  <div className="px-3 py-1 bg-emerald-100 text-emerald-700 text-[10px] font-black rounded-full uppercase">Target: < 0.05</div>
                  <div className="text-[10px] text-slate-400 font-bold uppercase">Simulation Divergence</div>
                </div>
              </div>
            </div>
            <div className="relative">
              <div className="bg-slate-900 rounded-[2rem] p-6 aspect-video flex flex-col">
                <div className="flex justify-between items-center mb-6">
                  <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Verification Plot: SNR vs BLER</div>
                  <div className="flex gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                    <div className="w-2 h-2 rounded-full bg-white opacity-20"></div>
                  </div>
                </div>
                <div className="flex-grow flex items-end gap-2 px-4">
                  {/* Mock Chart */}
                  {[30, 45, 60, 40, 75, 90, 85, 95].map((h, i) => (
                    <div key={i} className="flex-grow bg-blue-600/40 border-t-2 border-blue-400 rounded-t-sm" style={{ height: `${h}%` }}></div>
                  ))}
                </div>
                <div className="mt-4 flex justify-between text-[8px] font-black text-slate-600 uppercase">
                  <span>-10dB</span>
                  <span>0dB</span>
                  <span>10dB</span>
                  <span>20dB</span>
                </div>
                <div className="mt-4 text-center text-emerald-400 text-[10px] font-bold animate-pulse">
                   SUCCESS: 98.4% ALIGNMENT WITH SOURCE PAPER
                </div>
              </div>
            </div>
          </div>
        </section>

        <footer className="mt-24 text-center">
          <div className="w-16 h-1 bg-slate-800 mx-auto mb-10"></div>
          <p className="text-[10px] font-black uppercase tracking-[0.5em] text-slate-600">
             Integrated GitHub Utility • Paper2Code Protocol v1.5
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Paper2CodeView;
