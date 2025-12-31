import React from 'react';
import KaTeX from './KaTeX';
import MermaidDiagram from './MermaidDiagram';

const LSTM_CELL_DIAGRAM = `
graph TD
    subgraph CellLogic [" LSTM Internal Micro-Architecture "]
        xt[Input x_t] --> F[Forget Gate σ]
        ht_prev[Hidden h_t-1] --> F
        
        xt --> I[Input Gate σ]
        ht_prev --> I
        
        xt --> C_tilde[Candidate Cell tanh]
        ht_prev --> C_tilde
        
        xt --> O[Output Gate σ]
        ht_prev --> O
        
        Ct_prev[Previous Cell C_t-1] --> Mult1["⊗"]
        F --> Mult1
        
        I --> Mult2["⊗"]
        C_tilde --> Mult2
        
        Mult1 --> Add["⊕"]
        Mult2 --> Add
        
        Add --> Ct[New Cell C_t]
        
        Ct --> TanhFinal[tanh]
        TanhFinal --> Mult3["⊗"]
        O --> Mult3
        Mult3 --> ht[New Hidden h_t]
    end

    style CellLogic fill:#f0fdf4,stroke:#16a34a,stroke-width:2px
    style Mult1 fill:#fff,stroke:#16a34a
    style Mult2 fill:#fff,stroke:#16a34a
    style Mult3 fill:#fff,stroke:#16a34a
    style Add fill:#fff,stroke:#16a34a
`;

/**
 * TechnicalDeepDiveAudit Component
 * Provides a mathematical audit and proofs for the underlying architecture.
 */
const TechnicalDeepDiveAudit: React.FC<{ onBack: () => void }> = ({ onBack }) => {
  return (
    <div className="min-h-screen bg-slate-50 flex flex-col items-center pb-24">
      {/* Navigation Header */}
      <nav className="w-full bg-white/90 backdrop-blur-md border-b border-slate-200 sticky top-0 z-50 px-8 py-4 flex items-center justify-between shadow-sm">
        <button 
          onClick={onBack} 
          className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white hover:bg-black rounded-lg font-bold transition-all text-xs uppercase tracking-widest shadow-xl hover:shadow-slate-200"
        >
          <span>←</span> Back to System
        </button>
        <div className="flex items-center gap-3">
          <div className="px-3 py-1 bg-blue-100 text-blue-700 text-[10px] font-black rounded-full uppercase tracking-tighter">
            Mathematical Verification v1.9
          </div>
        </div>
      </nav>

      <div className="max-w-5xl w-full px-6 py-12">
        <header className="mb-16 text-center">
          <h1 className="text-4xl md:text-6xl font-black text-slate-900 mb-4 tracking-tighter uppercase">
            Technical <span className="text-blue-600">Audit</span> & Proofs
          </h1>
          <p className="text-slate-500 font-medium max-w-2xl mx-auto leading-relaxed italic text-sm">
            "Formalizing the transition from sequential recurrence to global attention kernels through the lens of gradient stability."
          </p>
        </header>

        {/* --- EVOLUTIONARY TRANSITION SECTION --- */}
        <section className="mb-24">
          <div className="flex items-center gap-4 mb-12">
            <div className="h-px bg-slate-200 flex-grow"></div>
            <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.4em]">The Core Transition: RNN → LSTM → GPT</h2>
            <div className="h-px bg-slate-200 flex-grow"></div>
          </div>

          <div className="space-y-12">
            
            {/* Logic Step 1: RNN & The Vanishing Gradient Problem */}
            <div className="bg-white rounded-[2.5rem] p-10 border border-slate-100 shadow-xl overflow-hidden relative group">
              <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
                <div className="text-[80px] font-black text-slate-900 select-none uppercase">RNN</div>
              </div>
              <h3 className="text-xl font-black text-slate-900 mb-6 uppercase flex items-center gap-3">
                <span className="w-8 h-8 rounded-lg bg-amber-100 text-amber-600 flex items-center justify-center text-xs">01</span>
                The Sequential Recurrence Bottleneck
              </h3>
              <p className="text-sm text-slate-600 mb-8 leading-relaxed max-w-3xl">
                Standard RNNs suffer from the vanishing gradient problem. During backpropagation through time (BPTT), gradients are repeatedly multiplied by weights, causing them to decay exponentially.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div className="p-8 bg-slate-50 rounded-3xl border border-slate-100">
                  <div className="text-[10px] font-black text-slate-400 uppercase mb-4 tracking-widest">Hidden State Update</div>
                  <KaTeX math={'h_t = \\sigma(W_{hh} h_{t-1} + W_{xh} x_t + b)'} block />
                </div>
                <div className="p-8 bg-slate-900 rounded-3xl border border-slate-800 text-white text-center">
                  <div className="text-[10px] font-black text-slate-500 uppercase mb-4 tracking-widest">Gradient Decay Law</div>
                  <KaTeX math={'\\frac{\\partial h_T}{\\partial h_t} = \\prod_{k=t+1}^T W_{hh}^T \\text{diag}(\\sigma\'(...))'} block />
                </div>
              </div>
            </div>

            {/* Logic Step 2: LSTM - WHERE GATING SOLVED THE GRADIENT */}
            <div className="bg-white rounded-[2.5rem] p-10 border border-emerald-100 shadow-xl overflow-hidden relative group border-2">
              <div className="absolute top-0 right-0 p-8 opacity-5 group-hover:opacity-10 transition-opacity">
                <div className="text-[80px] font-black text-emerald-900 select-none uppercase">LSTM</div>
              </div>
              <h3 className="text-xl font-black text-slate-900 mb-6 uppercase flex items-center gap-3">
                <span className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-600 flex items-center justify-center text-xs">02</span>
                The Constant Error Carousel
              </h3>
              <p className="text-sm text-slate-600 mb-8 leading-relaxed max-w-3xl">
                LSTM introduces the "Cell State" and "Gating" mechanisms to selectively pass information. This creates a "Constant Error Carousel" that allows gradients to flow over long sequences without vanishing.
              </p>
              
              <div className="mb-10">
                <MermaidDiagram chart={LSTM_CELL_DIAGRAM} height="400px" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                <div className="p-8 bg-slate-50 rounded-3xl border border-slate-100">
                  <div className="text-[10px] font-black text-slate-400 uppercase mb-4 tracking-widest">Forget Gate</div>
                  <KaTeX math={'f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)'} block />
                </div>
                <div className="p-8 bg-emerald-900 rounded-3xl border border-emerald-800 text-white text-center">
                  <div className="text-[10px] font-black text-emerald-500 uppercase mb-4 tracking-widest">Cell State Update</div>
                  <KaTeX math={'C_t = f_t \\otimes C_{t-1} + i_t \\otimes \\tilde{C}_t'} block />
                </div>
              </div>
            </div>
          </div>
        </section>

        <footer className="mt-24 text-center">
          <div className="w-16 h-1 bg-slate-300 mx-auto mb-10"></div>
          <p className="text-[10px] font-black uppercase tracking-[0.5em] text-slate-400">
             End of Mathematical Proofs • Technical Audit v1.9
          </p>
        </footer>
      </div>
    </div>
  );
};

// Fixed: Added missing default export to resolve "Module has no default export" error in App.tsx
export default TechnicalDeepDiveAudit;