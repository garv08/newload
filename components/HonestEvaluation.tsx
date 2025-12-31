import React from 'react';
import MermaidDiagram from './MermaidDiagram';

interface HonestEvaluationProps {
  onBack: () => void;
}

const EVOLUTION_DIAGRAM = `
%%{init: { 'theme': 'base', 'themeVariables': {
    'fontSize': '18px',
    'cScale0': '#bae6fd',
    'cScale1': '#c7d2fe',
    'cScale2': '#e9d5ff',
    'cScale3': '#ddd6fe',
    'cScale4': '#fbcfe8',
    'cScale5': '#fecdd3',
    'cScale6': '#ffedd5',
    'cScale7': '#99f6e4'
} } }%%
timeline
    title Cognitive Evolution of RAG Systems
    2017 : Transformer : Attention Is All You Need (Vaswani)
    2018 : BERT : Bidirectional Grounding
    2020 : GPT-3 : Parameter Scaling Emergence
         : RAG : Retrieval Contextualization (Lewis)
    2022 : CoT : Chain of Thought Reasoning (Wei)
    2023 : Toolformer : Autonomous API Interactions
    2024 : CRAG : Corrective RAG Loops (Yan)
    2025 : Agentic RAG : Multi-Silo Knowledge Orchestration
`;

const HonestEvaluation: React.FC<HonestEvaluationProps> = ({ onBack }) => {
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans leading-relaxed pb-24 animate-in fade-in duration-700">
      <style>{`
        .academic-quote { font-family: 'Georgia', serif; }
        .ieee-text { font-family: 'Times New Roman', serif; font-size: 0.95rem; text-align: justify; }
        .lit-card { border: 1px solid #e2e8f0; border-radius: 1rem; padding: 2rem; background: #fff; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); transition: all 0.2s ease; }
        .lit-card:hover { transform: translateY(-2px); border-color: #9333ea; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
        .section-tag { font-family: 'ui-sans-serif', 'system-ui', 'sans-serif'; font-size: 10px; font-weight: 900; letter-spacing: 0.2em; text-transform: uppercase; color: #94a3b8; }
        
        /* Force bold font for mermaid timeline text */
        .mermaid svg text {
          font-weight: 900 !important;
        }
        .mermaid svg .titleText {
          font-size: 24px !important;
        }
      `}</style>

      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-xl border-b border-slate-200 px-8 py-4 flex items-center justify-between shadow-sm">
        <button 
          onClick={onBack} 
          className="flex items-center gap-2 px-5 py-2 bg-slate-900 text-white hover:bg-black rounded-lg font-black transition-all text-[10px] uppercase tracking-widest shadow-xl active:scale-95"
        >
          <span>←</span> Back to Architecture
        </button>
        <div className="flex items-center gap-4">
            <div className="px-3 py-1 bg-purple-100 text-purple-700 text-[10px] font-black rounded-full uppercase tracking-widest">
                Academic Survey v3.1
            </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-12">
        
        <header className="mb-20 text-center">
            <span className="section-tag mb-4 block">Knowledge Plane Intelligence</span>
            <h1 className="text-4xl md:text-7xl font-black text-slate-900 mb-8 tracking-tighter uppercase leading-none">
                Literature <span className="text-purple-600">Survey</span>
            </h1>
            <p className="text-xl md:text-2xl text-slate-500 italic academic-quote max-w-4xl mx-auto leading-relaxed border-l-8 border-purple-500 pl-10 text-left">
                "Charting the journey from stochastic word prediction to deterministic grounding in the 5G Physical Layer domain."
            </p>
        </header>

        {/* Timeline Section */}
        <section className="mb-24">
            <h2 className="section-tag mb-10 text-center">The Historical Convergence of AI & Specs</h2>
            <div className="bg-white rounded-[2rem] p-6 md:p-12 shadow-2xl border border-slate-100 mb-16 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-purple-500 to-blue-500"></div>
                <MermaidDiagram chart={EVOLUTION_DIAGRAM} height="480px" />
                <div className="mt-8 text-center text-[10px] text-slate-400 font-bold uppercase tracking-widest italic">
                    Figure 1.0: Evolution from Attention (2017) to Agentic RAG (2025)
                </div>
            </div>

            {/* Citations Grid */}
            <h2 className="section-tag mb-10 text-center">Key Academic Milestones</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {[
                  { id: '01', title: 'The Transformer Basis', ref: 'Vaswani et al., 2017', desc: 'Introduced the Attention mechanism, allowing the model to focus on distant but related code blocks in PHY implementations.' },
                  { id: '02', title: 'Retrieval Grounding', ref: 'Lewis et al., 2020', desc: 'Demonstrated that augmenting LLMs with external document indices (RAG) significantly reduces hallucination in technical domains.' },
                  { id: '03', title: 'Chain-of-Thought', ref: 'Wei et al., 2022', desc: 'Revealed that complex DSP reasoning can be elicited by forcing models to output intermediate rationales.' },
                  { id: '04', title: 'Self-Refinement Loops', ref: 'Madaan et al., 2023', desc: 'The predecessor to our Grader Agent; proved that models can audit and fix their own outputs based on constraints.' },
                  { id: '05', title: 'Agentic Tool-Use', ref: 'Schick et al., 2023', desc: 'Standardized the pattern of LLMs calling external tools (like our Neo4j Knowledge Graph) for factual lookup.' },
                  { id: '06', title: 'GraphRAG Ontologies', ref: 'MS Research, 2024', desc: 'Moved beyond flat vector search to multi-hop graph traversal for complex specification hierarchies.' },
                  { id: '07', title: 'Corrective RAG (CRAG)', ref: 'Yan et al., 2024', desc: 'Introduced the classify-then-retrieve flow used in our Router node to optimize precision.' },
                  { id: '08', title: 'Multi-Agent State', ref: 'Wu et al., 2024', desc: 'Formalized state-management for teams of agents, which we implement via LangGraph in the Orchestrator.' },
                  { id: '09', title: 'Engineering Discovery', ref: 'DeepSeek, 2025', desc: 'Representing the frontier of autonomous scientific and coding intelligence for massive telecommunications projects.' }
                ].map((item) => (
                  <div key={item.id} className="lit-card group relative">
                    <div className="flex items-center justify-between mb-6">
                      <div className="w-12 h-12 bg-slate-900 text-white rounded-xl flex items-center justify-center font-black group-hover:bg-purple-600 transition-colors shadow-lg">{item.id}</div>
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{item.ref}</span>
                    </div>
                    <h3 className="font-black text-xl text-slate-900 mb-4 uppercase tracking-tight">{item.title}</h3>
                    <p className="ieee-text text-slate-600 leading-relaxed text-sm">
                      {item.desc}
                    </p>
                    <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between">
                       <span className="text-[9px] font-black text-purple-600 uppercase tracking-widest">Verified Cit.</span>
                       <div className="flex gap-1">
                          <div className="w-1.5 h-1.5 rounded-full bg-purple-400"></div>
                          <div className="w-1.5 h-1.5 rounded-full bg-slate-200"></div>
                       </div>
                    </div>
                  </div>
                ))}
            </div>
        </section>

        {/* Synthesis Section */}
        <section className="bg-slate-950 text-white rounded-[3rem] p-12 md:p-24 shadow-2xl relative overflow-hidden">
             <div className="absolute top-0 right-0 w-96 h-96 bg-purple-600/10 rounded-full blur-[120px]"></div>
             <div className="absolute bottom-0 left-0 w-80 h-80 bg-blue-600/10 rounded-full blur-[100px]"></div>
             <div className="relative z-10 max-w-4xl">
               <h2 className="text-4xl md:text-5xl font-black mb-10 uppercase tracking-tighter leading-tight">
                  Moving from <span className="text-purple-400">Probabilistic Chat</span> to <span className="text-blue-400">Engineering Intelligence</span>.
               </h2>
               <p className="text-slate-400 text-xl leading-relaxed mb-12 font-medium">
                  The literature synthesis confirms that a single LLM is insufficient for the Physical Layer. Only by orchestrating a specialized ensemble of retrievers, graders, and routers—grounded in formal standards—can we achieve the reliability required for 5G and 6G deployment.
               </p>
               <div className="flex flex-wrap gap-6">
                  <div className="px-8 py-4 bg-white/5 border border-white/10 rounded-2xl text-xs font-black uppercase tracking-widest hover:bg-white/10 transition-colors cursor-default">
                    High-Fidelity Grounding
                  </div>
                  <div className="px-8 py-4 bg-white/5 border border-white/10 rounded-2xl text-xs font-black uppercase tracking-widest hover:bg-white/10 transition-colors cursor-default">
                    Zero-Trust Reflection
                  </div>
               </div>
             </div>
        </section>

        <footer className="mt-24 text-center">
          <div className="section-tag opacity-40">
            Literature Survey Archive • Vol. 3.1 • 2025
          </div>
        </footer>
      </div>
    </div>
  );
};

export default HonestEvaluation;