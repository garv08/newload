import React, { useState, useEffect } from 'react';
import MermaidDiagram from './components/MermaidDiagram.tsx';
import DetailsModal from './components/DetailsModal.tsx';
import PresentationViewer from './components/PresentationViewer.tsx';
import WhitePaper from './components/WhitePaper.tsx';
import LinkedInPost from './components/LinkedInPost.tsx';
import ChatSimulation from './components/ChatSimulation.tsx';
import SequenceExplanation from './components/SequenceExplanation.tsx';
import TechnicalDeepDiveAudit from './components/TechnicalDeepDiveAudit.tsx';
import HonestEvaluation from './components/HonestEvaluation.tsx';
import VideoGenerator from './components/VideoGenerator.tsx';
import Paper2CodeView from './components/Paper2CodeView.tsx';
import { 
  getDiagramDefinition, 
  SEQUENCE_DIAGRAM_DEFINITION, 
  NODE_DETAILS
} from './constants.ts';
import type { FlowDefinition, NodeDetail } from './constants.ts';

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

  const activeFlow = customFlow || (activeFlowKey ? FLOWS[activeFlowKey] : null);
  const activeColor = activeFlowKey ? FLOW_COLORS[activeFlowKey] : '#3b82f6';

  return (
    <div className="min-h-screen bg-neutral-100 flex flex-col items-center justify-start p-2 md:p-6 pb-24 font-sans text-neutral-900">
      <ChatSimulation 
        onSimulateStep={(nodes, edges) => {
          setActiveFlowKey(null);
          setCustomFlow({ nodes, edges: edges as [string, string][] });
        }}
        onReset={() => {
          setCustomFlow(null);
          setActiveFlowKey(null);
        }}
      />

      <DetailsModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} data={selectedNodeData} />
      <PresentationViewer isOpen={isPresentationMode} onClose={() => setIsPresentationMode(false)} />
      {showVideoGenerator && <VideoGenerator onClose={() => setShowVideoGenerator(false)} />}

      <header className="w-full max-w-7xl mb-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-black tracking-tighter uppercase text-slate-900">
            PHY Layer <span className="text-blue-600">Agentic RAG</span>
          </h1>
          <p className="text-sm font-medium text-slate-500 uppercase tracking-widest">
            Autonomous 5G/6G Development & Log Triage
          </p>
        </div>

        <div className="flex gap-4">
          <button onClick={() => setShowWhitePaper(true)} className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-slate-50 transition-all shadow-sm">Whitepaper</button>
          <button onClick={() => setShowHonestEvaluation(true)} className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-[10px] font-black uppercase tracking-widest hover:bg-slate-50 transition-all shadow-sm">Research Survey</button>
        </div>
      </header>

      <main className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <section className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Operational Flows</h3>
            <div className="space-y-2">
              {Object.keys(FLOWS).map(key => (
                <button
                  key={key}
                  onClick={() => handleFlowClick(key)}
                  className={`w-full text-left px-4 py-3 rounded-xl text-xs font-bold transition-all border ${
                    activeFlowKey === key 
                      ? 'bg-slate-900 text-white border-slate-900 shadow-lg' 
                      : 'bg-white text-slate-600 border-slate-100 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="capitalize">{key.replace('_', ' ')}</span>
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: FLOW_COLORS[key] }}></div>
                  </div>
                </button>
              ))}
            </div>
          </section>

          <section className="bg-white p-6 rounded-3xl border border-slate-200 shadow-sm">
            <h3 className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em] mb-4">Real-time Telemetry</h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase mb-1">
                  <span>Inference Latency</span>
                  <span>{telemetry.latency}ms</span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-blue-500 transition-all duration-1000" style={{ width: `${(telemetry.latency / 400) * 100}%` }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[10px] font-bold text-slate-400 uppercase mb-1">
                  <span>Semantic Cache Hit Rate</span>
                  <span>{telemetry.cacheHit}%</span>
                </div>
                <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 transition-all duration-1000" style={{ width: `${telemetry.cacheHit}%` }}></div>
                </div>
              </div>
            </div>
          </section>

          <div className="grid grid-cols-2 gap-2">
            <button onClick={() => setShowDeepDiveAudit(true)} className="p-4 bg-slate-900 text-white rounded-2xl text-[9px] font-black uppercase tracking-widest hover:bg-black transition-all">Math Audit</button>
            <button onClick={() => setShowPaper2Code(true)} className="p-4 bg-blue-600 text-white rounded-2xl text-[9px] font-black uppercase tracking-widest hover:bg-blue-700 transition-all">Paper2Code</button>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-6">
          <div className="bg-white p-4 rounded-[2.5rem] border border-slate-200 shadow-sm relative">
            <div className="absolute top-8 left-8 z-10 flex gap-2">
              <button 
                onClick={() => setViewMode('architecture')}
                className={`px-4 py-2 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${
                  viewMode === 'architecture' ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-400'
                }`}
              >Topology</button>
              <button 
                onClick={() => setViewMode('sequence')}
                className={`px-4 py-2 rounded-full text-[10px] font-black uppercase tracking-widest transition-all ${
                  viewMode === 'sequence' ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-400'
                }`}
              >Sequence</button>
            </div>

            <MermaidDiagram 
              chart={viewMode === 'architecture' ? getDiagramDefinition(diagramDirection) : SEQUENCE_DIAGRAM_DEFINITION} 
              activeFlow={activeFlow}
              flowColor={activeColor}
              onNodeClick={handleNodeClick}
              height="650px"
            />
          </div>
        </div>
      </main>

      <footer className="fixed bottom-0 w-full bg-white/80 backdrop-blur-md border-t border-slate-200 py-3 px-6 flex justify-between items-center text-[9px] font-black text-slate-400 uppercase tracking-[0.3em] z-30">
        <div>System Status: <span className="text-emerald-500">Nominal</span></div>
        <div className="hidden md:block">3GPP TS 38.211 Grounded Intelligence • REL-18 Compliance</div>
        <div>V2.4.0-STABLE</div>
      </footer>
    </div>
  );
};

export default App;