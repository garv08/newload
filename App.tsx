
import React, { useState, useEffect } from 'react';
import MermaidDiagram, { FlowDefinition } from './components/MermaidDiagram';
import DetailsModal from './components/DetailsModal';
import PresentationViewer from './components/PresentationViewer';
import WhitePaper from './components/WhitePaper';
import LinkedInPost from './components/LinkedInPost';
import ChatSimulation from './components/ChatSimulation';
import SequenceExplanation from './components/SequenceExplanation';
import TechnicalDeepDiveAudit from './components/TechnicalDeepDiveAudit';
import HonestEvaluation from './components/HonestEvaluation';
import VideoGenerator from './components/VideoGenerator';
import Paper2CodeView from './components/Paper2CodeView';
import { getDiagramDefinition, SEQUENCE_DIAGRAM_DEFINITION, NODE_DETAILS, NodeDetail } from './constants';
import { LOGO_3GPP, LOGO_ORAN, LOGO_FAPI, LOGO_NEO4J, LOGO_CACHE, LOGO_BOOK } from './logos';

const FLOWS: Record<string, FlowDefinition> = {
  ingest: {
    nodes: ['U', 'A', 'B', 'C', 'D', 'E1', 'E2', 'E3', 'SC', 'SK'],
    edges: [
      ['A', 'SC'], ['B', 'SC'], ['D', 'SC'], ['C', 'SC'],
      ['E1', 'SK'], ['E2', 'SK'], ['E3', 'SK'], ['C', 'SK'],
      ['E1', 'SC'], ['E2', 'SC'], ['E3', 'SC']
    ]
  },
  vector_rag: {
    nodes: ['U', 'CACHE', 'SB', 'QT', 'SC', 'RR', 'AI'],
    edges: [['U', 'CACHE'], ['CACHE', 'SB'], ['SB', 'QT'], ['QT', 'SC'], ['SC', 'RR'], ['RR', 'SB'], ['SB', 'AI'], ['AI', 'SB']]
  },
  graph_rag: {
    nodes: ['U', 'CACHE', 'SB', 'QT', 'SK', 'RR', 'AI'],
    edges: [['U', 'CACHE'], ['CACHE', 'SB'], ['SB', 'QT'], ['QT', 'SK'], ['SK', 'RR'], ['RR', 'SB'], ['SB', 'AI'], ['AI', 'SB']]
  },
  paper2code: {
    nodes: ['C', 'E1', 'P2C', 'SB', 'I'],
    edges: [['C', 'P2C'], ['E1', 'P2C'], ['P2C', 'SB'], ['SB', 'I']]
  },
  correction: {
    nodes: ['SB', 'SE', 'QT'],
    edges: [['SB', 'SE'], ['SE', 'SB'], ['SE', 'QT']]
  },
  output: {
    nodes: ['U', 'SB', 'F', 'G', 'H', 'I'],
    edges: [['SB', 'F'], ['SB', 'G'], ['SB', 'H'], ['SB', 'I'], ['SB', 'U']]
  }
};

const FLOW_COLORS: Record<string, string> = {
  ingest: '#16a34a',
  vector_rag: '#2563eb',
  graph_rag: '#9333ea',
  paper2code: '#3b82f6',
  correction: '#e11d48',
  output: '#4f46e5'
};

const App: React.FC = () => {
  const [activeFlowKey, setActiveFlowKey] = useState<string | null>(null);
  const [diagramDirection, setDiagramDirection] = useState<'LR' | 'TD'>('LR');
  const [viewMode, setViewMode] = useState<'architecture' | 'sequence'>('architecture');
  const [customFlow, setCustomFlow] = useState<FlowDefinition | null>(null);
  const [selectedNodeData, setSelectedNodeData] = useState<NodeDetail | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isPresentationMode, setIsPresentationMode] = useState(false);
  const [showWhitePaper, setShowWhitePaper] = useState(false);
  const [showLinkedInPost, setShowLinkedInPost] = useState(false);
  const [showSequenceExplanation, setShowSequenceExplanation] = useState(false);
  const [showDeepDiveAudit, setShowDeepDiveAudit] = useState(false);
  const [showHonestEvaluation, setShowHonestEvaluation] = useState(false);
  const [showVideoGenerator, setShowVideoGenerator] = useState(false);
  const [showPaper2Code, setShowPaper2Code] = useState(false);
  const [telemetry, setTelemetry] = useState({ latency: 240, cacheHit: 92, agents: 3 });

  useEffect(() => {
    const interval = setInterval(() => {
      setTelemetry({
        latency: 200 + Math.floor(Math.random() * 150),
        cacheHit: 88 + Math.floor(Math.random() * 10),
        agents: activeFlowKey ? 5 : 3
      });
    }, 3000);
    return () => clearInterval(interval);
  }, [activeFlowKey]);

  const handleFlowClick = (key: string) => {
    setCustomFlow(null);
    setActiveFlowKey(activeFlowKey === key ? null : key);
  };

  const handleNodeClick = (nodeId: string) => {
    const details = NODE_DETAILS[nodeId];
    if (details) {
      setSelectedNodeData(details);
      setIsModalOpen(true);
    }
  };

  if (showPaper2Code) return <Paper2CodeView onBack={() => setShowPaper2Code(false)} />;
  if (showDeepDiveAudit) return <TechnicalDeepDiveAudit onBack={() => setShowDeepDiveAudit(false)} />;
  if (showHonestEvaluation) return <HonestEvaluation onBack={() => setShowHonestEvaluation(false)} />;
  if (showWhitePaper) return <WhitePaper onBack={() => setShowWhitePaper(false)} />;
  if (showLinkedInPost) return <LinkedInPost onBack={() => setShowLinkedInPost(false)} />;
  if (showSequenceExplanation) return <SequenceExplanation onBack={() => setShowSequenceExplanation(false)} />;

  return (
    <div className="min-h-screen bg-neutral-100 flex flex-col items-center justify-start p-2 md:p-6 pb-24 font-sans">
      <ChatSimulation 
        onSimulateStep={(nodes, edges) => {
          setActiveFlowKey(null);
          setCustomFlow({ nodes, edges });
        }}
        onReset={() => {
          setCustomFlow(null);
          setActiveFlowKey(null);
        }}
      />

      <DetailsModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} data={selectedNodeData} />
      <PresentationViewer isOpen={isPresentationMode} onClose={() => setIsPresentationMode(false)} />
      {showVideoGenerator && <VideoGenerator onClose={() => setShowVideoGenerator(false)} />}

      <div className="fixed bottom-0 left-0 right-0 bg-gray-950 text-white border-t border-white/10 z-30 px-6 py-2 flex items-center justify-between text-[10px] md:text-xs font-mono">
         <div className="flex items-center gap-4 md:gap-6">
             <div className="flex items-center gap-2">
                 <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                 <span className="opacity-70">SYSTEM:</span>
                 <span className="font-bold text-green-400 uppercase">Production Ready</span>
             </div>
             <div className="hidden sm:flex items-center gap-2">
                 <span className="opacity-70">LATENCY:</span>
                 <span className="font-bold">{telemetry.latency}ms</span>
             </div>
             <div className="hidden sm:flex items-center gap-2">
                 <span className="opacity-70">CACHE:</span>
                 <span className="font-bold text-orange-400">{telemetry.cacheHit}%</span>
             </div>
         </div>
         <div className="flex items-center gap-4 opacity-50">
             <span className="hidden lg:inline text-blue-400">GEMINI_AI_ENABLED</span>
             <span>PHY-RAG v1.5.0</span>
         </div>
      </div>

      <main className="bg-white rounded-2xl shadow-2xl border border-slate-200 p-4 md:p-10 max-w-[1600px] w-full mx-auto relative overflow-hidden">
        
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-12">
          <div className="flex-1">
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-slate-900 text-white text-[10px] font-black rounded-full uppercase tracking-widest mb-4">
              <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></span>
              PHY Architecture Intelligence
            </div>
            <h1 className="text-3xl md:text-5xl font-black text-slate-900 tracking-tighter leading-none mb-2">
              PHY Layer <span className="text-blue-600">Multi-Agent</span> Platform
            </h1>
            <p className="text-slate-500 text-sm max-w-2xl font-medium">
              Advanced Hybrid-RAG platform utilizing Large Language Models for autonomous 5G/6G development.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button onClick={() => setShowVideoGenerator(true)} className="px-4 py-2 bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-700 rounded-xl transition-all shadow-sm text-xs font-bold flex items-center gap-2 group">
              <span className="group-hover:rotate-12 transition-transform">🎬</span> Simulation Studio
            </button>
            <button onClick={() => setIsPresentationMode(true)} className="px-4 py-2 bg-slate-900 text-white rounded-xl hover:bg-black transition-all shadow-md text-xs font-bold">📺 Start Presentation</button>
          </div>
        </div>

        <div className="bg-slate-50 rounded-3xl p-6 mb-10 border border-slate-200">
           <div className="flex items-center gap-3 mb-6">
              <div className="h-px bg-slate-300 flex-grow"></div>
              <h2 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.4em]">Integrated Intelligence Nodes</h2>
              <div className="h-px bg-slate-300 flex-grow"></div>
           </div>
           <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              {[
                { label: "Literature Survey", icon: "📜", color: "purple", onClick: () => setShowHonestEvaluation(true), desc: "Evolutionary path of Attention and RAG." },
                { label: "Technical Paper", icon: "📄", color: "blue", onClick: () => setShowWhitePaper(true), desc: "Formal architecture specification." },
                { label: "Math Audit", icon: "⚖️", color: "emerald", onClick: () => setShowDeepDiveAudit(true), desc: "Gradient stability and proofs." },
                { label: "Paper2Code", icon: "⚡", color: "sky", onClick: () => setShowPaper2Code(true), desc: "Autonomous simulation synthesis.", dark: true }
              ].map((item, i) => (
                <button 
                  key={i}
                  onClick={item.onClick}
                  className={`group flex flex-col items-start p-6 ${item.dark ? 'bg-slate-900 text-white' : 'bg-white text-slate-900'} border border-slate-200 rounded-2xl shadow-sm hover:shadow-xl transition-all text-left relative overflow-hidden`}
                >
                  <div className={`w-12 h-12 ${item.dark ? 'bg-white/10' : `bg-${item.color}-100 text-${item.color}-600`} rounded-xl flex items-center justify-center text-2xl mb-4 group-hover:scale-110 transition-transform`}>{item.icon}</div>
                  <h3 className="font-black uppercase tracking-tight mb-2 text-sm">{item.label}</h3>
                  <p className={`text-[10px] ${item.dark ? 'text-slate-400' : 'text-slate-500'} leading-relaxed mb-4`}>{item.desc}</p>
                  <span className={`text-[10px] font-black ${item.dark ? 'text-blue-400' : `text-${item.color}-600`} uppercase tracking-widest group-hover:translate-x-1 transition-transform`}>Access →</span>
                </button>
              ))}
           </div>
        </div>

        <div className="flex justify-center mb-8">
           <div className="bg-slate-100 p-1.5 rounded-2xl inline-flex shadow-inner border border-slate-200">
             <button onClick={() => { setViewMode('architecture'); setActiveFlowKey(null); }} className={`px-8 py-2.5 rounded-xl text-[10px] font-black transition-all uppercase tracking-widest ${viewMode === 'architecture' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-500 hover:text-slate-700'}`}>System Architecture</button>
             <button onClick={() => { setViewMode('sequence'); setActiveFlowKey(null); }} className={`px-8 py-2.5 rounded-xl text-[10px] font-black transition-all uppercase tracking-widest ${viewMode === 'sequence' ? 'bg-white text-slate-900 shadow-md' : 'text-slate-500 hover:text-slate-700'}`}>Protocol Workflow</button>
           </div>
        </div>

        {viewMode === 'architecture' && (
          <div className="flex flex-wrap justify-center gap-2 mb-8 animate-fade-in">
             <button onClick={() => { setActiveFlowKey(null); setCustomFlow(null); }} className={`px-4 py-2 rounded-full font-black text-[10px] uppercase tracking-widest transition-all shadow-sm ${activeFlowKey === null && customFlow === null ? 'bg-slate-900 text-white shadow-xl scale-105' : 'bg-white text-slate-600 border border-slate-200 hover:bg-slate-50'}`}>Complete Network</button>
             {[
               {id: 'ingest', label: 'Ingestion', emoji: '📥', color: 'bg-green-600'},
               {id: 'vector_rag', label: 'Vector RAG', emoji: '⚡', color: 'bg-blue-600'},
               {id: 'graph_rag', label: 'Graph KAG', emoji: '🕸️', color: 'bg-purple-600'},
               {id: 'paper2code', label: 'Paper2Code', emoji: '📄', color: 'bg-sky-600'},
               {id: 'correction', label: 'Grader', emoji: '🛡️', color: 'bg-rose-600'},
               {id: 'output', label: 'Delivery', emoji: '📤', color: 'bg-indigo-600'}
             ].map(flow => (
               <button 
                 key={flow.id} 
                 onClick={() => handleFlowClick(flow.id)} 
                 className={`px-5 py-2.5 rounded-full font-black text-[10px] uppercase tracking-widest transition-all shadow-sm flex items-center gap-2 ${activeFlowKey === flow.id ? `${flow.color} text-white shadow-xl scale-105` : `bg-white text-slate-600 border border-slate-200 hover:bg-slate-50`}`}
               >
                 <span>{flow.emoji}</span> {flow.label}
               </button>
             ))}
          </div>
        )}
        
        <div className="w-full mb-10 relative group">
          {viewMode === 'architecture' && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10 bg-white/90 backdrop-blur-md p-1.5 rounded-xl border border-slate-200 shadow-xl flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <button onClick={() => setDiagramDirection('LR')} className={`px-4 py-1.5 text-[9px] font-black rounded-lg uppercase tracking-widest transition-all ${diagramDirection === 'LR' ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>Horizontal View</button>
              <button onClick={() => setDiagramDirection('TD')} className={`px-4 py-1.5 text-[9px] font-black rounded-lg uppercase tracking-widest transition-all ${diagramDirection === 'TD' ? 'bg-slate-900 text-white' : 'text-slate-500 hover:bg-slate-100'}`}>Vertical View</button>
            </div>
          )}

          <MermaidDiagram 
            chart={viewMode === 'architecture' ? getDiagramDefinition(diagramDirection) : SEQUENCE_DIAGRAM_DEFINITION} 
            activeFlow={customFlow || (activeFlowKey && viewMode === 'architecture' ? FLOWS[activeFlowKey] : null)}
            flowColor={customFlow ? '#3b82f6' : (activeFlowKey && viewMode === 'architecture' ? FLOW_COLORS[activeFlowKey] : undefined)}
            onNodeClick={viewMode === 'architecture' ? handleNodeClick : undefined}
            nodeDetails={viewMode === 'architecture' ? NODE_DETAILS : undefined}
            height="650px"
          />
        </div>

        <div className="border-t border-slate-100 pt-8">
          <div className="flex flex-wrap justify-center gap-8 md:gap-16 opacity-40 hover:opacity-100 transition-all pb-6">
            {[
              {src: LOGO_3GPP, label: '3GPP Compliance'},
              {src: LOGO_ORAN, label: 'O-RAN Alliance'},
              {src: LOGO_FAPI, label: 'SCF Interface'},
              {src: LOGO_BOOK, label: 'Scientific Literature'},
              {src: LOGO_CACHE, label: 'Semantic Cache'},
              {src: LOGO_NEO4J, label: 'Knowledge Graph'}
            ].map((logo, i) => (
              <div key={i} className="flex flex-col items-center gap-2 group cursor-help grayscale hover:grayscale-0 transition-all">
                <img src={logo.src} alt={logo.label} className="h-8 w-auto brightness-90 group-hover:brightness-100 group-hover:scale-110 transition-transform" />
                <span className="text-[7px] font-black text-slate-400 uppercase tracking-widest">{logo.label}</span>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
