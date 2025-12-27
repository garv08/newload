
import React, { useState } from 'react';
import { GoogleGenAI, Type } from "@google/genai";

const PRESET_QUERIES = [
  { label: "Analyze FAPI Log", text: "Why did throughput drop in Slot 4?", nodes: ['U', 'CACHE', 'SB', 'QT', 'SC', 'RR', 'G', 'U'] },
  { label: "3GPP Spec", text: "What are the DMRS ports for Rel 16?", nodes: ['U', 'CACHE', 'SB', 'QT', 'SK', 'RR', 'SE', 'U'] },
  { label: "Code Generation", text: "Generate a C++ kernel for polar encoding based on TS 38.212", nodes: ['U', 'SB', 'QT', 'SK', 'P2C', 'I', 'SE', 'U'] }
];

const AVAILABLE_NODES = ['U', 'CACHE', 'A', 'B', 'C', 'D', 'E1', 'E2', 'E3', 'SC', 'SK', 'RR', 'SB', 'SE', 'QT', 'P2C', 'AI', 'F', 'G', 'H', 'I'];

const ChatSimulation: React.FC<{ onSimulateStep: (n: string[], e: string[][]) => void; onReset: () => void }> = ({ onSimulateStep, onReset }) => {
  const [minimized, setMinimized] = useState(true);
  const [messages, setMessages] = useState<{role: 'user'|'system'|'ai', text: string}[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);

  const runSim = async (q: { text: string, nodes: string[] }) => {
    setMessages([{ role: 'user', text: q.text }]);
    for(let i=0; i<q.nodes.length; i++) {
      onSimulateStep([q.nodes[i]], []);
      await new Promise(r => setTimeout(r, 800));
    }
    setMessages(prev => [...prev, { role: 'system', text: "Sequence complete. Diagnostic nodes identified and verified." }]);
  };

  const handleAiInference = async () => {
    if (!userInput.trim()) return;
    setIsGenerating(true);
    setMessages([{ role: 'user', text: userInput }]);
    
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
      const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: `Analyze this user query for a PHY Layer RAG system: "${userInput}". 
        Identify the sequence of nodes that would be involved in processing this request. 
        Available nodes: ${AVAILABLE_NODES.join(', ')}.
        Nodes represent: U (User), CACHE (Semantic Cache), SB (Orchestrator), QT (Router), SC (Vector DB), SK (Knowledge Graph), RR (Reranker), SE (Self-Correction), P2C (Paper2Code), AI (Core), F (Reviews), G (Logs), H (TVs), I (CodeGen), A,B,C,D,E (Inputs).
        Return a JSON array of node IDs in sequence.`,
        config: {
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              nodes: {
                type: Type.ARRAY,
                items: { type: Type.STRING }
              },
              explanation: { type: Type.STRING }
            }
          }
        }
      });

      const data = JSON.parse(response.text || '{"nodes":[], "explanation":""}');
      const nodesToHighlight = data.nodes.filter((n: string) => AVAILABLE_NODES.includes(n));
      
      setMessages(prev => [...prev, { role: 'ai', text: data.explanation || "AI-orchestrated workflow activated." }]);
      
      for(let i=0; i<nodesToHighlight.length; i++) {
        onSimulateStep([nodesToHighlight[i]], []);
        await new Promise(r => setTimeout(r, 800));
      }
      
    } catch (err) {
      console.error("AI Simulation Error:", err);
      setMessages(prev => [...prev, { role: 'system', text: "Error in AI Orchestration. Reverting to manual mode." }]);
    } finally {
      setIsGenerating(false);
      setUserInput('');
    }
  };

  return (
    <div className="fixed bottom-12 right-6 z-40 w-80 md:w-96 bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden pointer-events-auto transition-all duration-500">
      <div 
        className="bg-slate-900 text-white p-4 flex justify-between items-center cursor-pointer hover:bg-black transition-colors" 
        onClick={() => setMinimized(!minimized)}
      >
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span className="font-black text-[10px] uppercase tracking-widest">Agentic Simulator</span>
        </div>
        <span className="text-xl leading-none">{minimized ? '⊕' : '⊖'}</span>
      </div>
      
      {!minimized && (
        <div className="flex flex-col h-[500px]">
          <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50">
            {messages.length === 0 && (
              <div className="text-center py-10">
                <div className="text-4xl mb-4">🤖</div>
                <p className="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Awaiting query input...</p>
              </div>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}>
                <div className={`max-w-[85%] text-[11px] p-3 rounded-2xl shadow-sm ${
                  m.role === 'user' ? 'bg-blue-600 text-white rounded-tr-none' : 
                  m.role === 'ai' ? 'bg-purple-100 text-purple-900 border border-purple-200 rounded-tl-none' :
                  'bg-white text-slate-600 border border-slate-200 rounded-tl-none'
                }`}>
                  <div className="font-black uppercase tracking-tighter mb-1 opacity-50 text-[9px]">
                    {m.role === 'user' ? 'Developer' : m.role === 'ai' ? 'AI Orchestrator' : 'System'}
                  </div>
                  {m.text}
                </div>
              </div>
            ))}
            {isGenerating && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-slate-200 h-10 w-24 rounded-2xl"></div>
              </div>
            )}
          </div>

          <div className="p-4 border-t border-slate-100 space-y-4 bg-white">
            <div className="flex flex-wrap gap-1.5">
              {PRESET_QUERIES.map(q => (
                <button 
                  key={q.label} 
                  onClick={() => runSim(q)} 
                  disabled={isGenerating}
                  className="text-[9px] font-black uppercase tracking-tight bg-slate-100 hover:bg-slate-200 text-slate-600 px-3 py-1.5 rounded-lg transition-colors border border-slate-200/50 disabled:opacity-50"
                >
                  {q.label}
                </button>
              ))}
            </div>
            
            <div className="relative">
              <input 
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleAiInference()}
                placeholder="Ask the system architecture..."
                className="w-full text-xs bg-slate-100 border border-slate-200 rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all"
              />
              <button 
                onClick={handleAiInference}
                disabled={isGenerating || !userInput.trim()}
                className="absolute right-2 top-2 w-8 h-8 bg-slate-900 text-white rounded-lg flex items-center justify-center hover:bg-black transition-colors disabled:opacity-30"
              >
                {isGenerating ? '...' : '→'}
              </button>
            </div>
            
            <button 
              onClick={() => { setMessages([]); onReset(); }} 
              className="w-full text-[9px] font-black text-slate-400 uppercase tracking-widest hover:text-slate-600 transition-colors py-1"
            >
              Reset Topology Diagram
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatSimulation;
