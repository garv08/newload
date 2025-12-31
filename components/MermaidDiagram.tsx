import React, { useEffect, useRef, useState } from 'react';
import mermaid from 'mermaid';
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch';
import { NodeDetail } from '../constants';

// Initialize mermaid with vibrant base theme instead of neutral gray
mermaid.initialize({
  startOnLoad: false,
  theme: 'base',
  securityLevel: 'loose',
  fontFamily: 'ui-sans-serif, system-ui, sans-serif',
  htmlLabels: true,
  themeVariables: {
    primaryColor: '#f0f9ff',
    primaryTextColor: '#0c4a6e',
    primaryBorderColor: '#7dd3fc',
    lineColor: '#94a3b8',
    secondaryColor: '#f8fafc',
    tertiaryColor: '#fff',
    cScale0: '#bae6fd',
    cScale1: '#c7d2fe',
    cScale2: '#e9d5ff',
    cScale3: '#ddd6fe',
    cScale4: '#fbcfe8',
    cScale5: '#fecdd3',
    cScale6: '#ffedd5',
    cScale7: '#99f6e4'
  },
  flowchart: { 
    useMaxWidth: false, 
    curve: 'basis',
    htmlLabels: true
  }
});

export interface FlowDefinition {
  nodes: string[];
  edges: string[][];
}

interface MermaidDiagramProps {
  chart: string;
  activeFlow?: FlowDefinition | null;
  flowColor?: string;
  onNodeClick?: (nodeId: string) => void;
  nodeDetails?: Record<string, NodeDetail>;
  height?: string;
}

const MermaidDiagram: React.FC<MermaidDiagramProps> = ({ 
  chart, 
  activeFlow, 
  flowColor = '#3b82f6', 
  onNodeClick, 
  nodeDetails,
  height = "520px"
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isRendered, setIsRendered] = useState(false);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    
    const render = async () => {
      if (!chart || !containerRef.current) return;
      
      try {
        setRenderError(null);
        setIsRendered(false);
        
        const uniqueId = `mermaid-${Math.random().toString(36).slice(2, 11)}`;
        const tempDiv = document.createElement('div');
        tempDiv.id = uniqueId;
        tempDiv.style.position = 'absolute';
        tempDiv.style.visibility = 'hidden';
        tempDiv.style.top = '-10000px';
        tempDiv.textContent = chart.trim();
        document.body.appendChild(tempDiv);

        if (containerRef.current) {
          containerRef.current.innerHTML = '<div class="flex items-center justify-center h-full text-slate-400 text-[10px] font-mono tracking-widest animate-pulse">OPTIMIZING TOPOLOGY ENGINE...</div>';
        }

        await new Promise(resolve => setTimeout(resolve, 100));
        
        if (!isMounted) {
          if (document.body.contains(tempDiv)) document.body.removeChild(tempDiv);
          return;
        }

        await mermaid.run({
          nodes: [tempDiv],
          suppressErrors: false
        });

        if (isMounted && containerRef.current) {
          const svg = tempDiv.querySelector('svg');
          if (svg) {
            containerRef.current.innerHTML = '';
            containerRef.current.appendChild(svg.cloneNode(true));
            setIsRendered(true);
          } else {
            throw new Error("SVG generation produced no output");
          }
        }

        if (document.body.contains(tempDiv)) document.body.removeChild(tempDiv);

      } catch (error) {
        console.error("Mermaid Render Critical Error:", error);
        if (isMounted) {
          setRenderError(error instanceof Error ? error.message : "Internal Logic Fault");
        }
      }
    };

    render();
    return () => { isMounted = false; };
  }, [chart]);

  useEffect(() => {
    if (!isRendered || !containerRef.current) return;

    const svg = containerRef.current.querySelector('svg');
    if (!svg) return;

    try {
      svg.setAttribute('width', '100%');
      svg.setAttribute('height', '100%');
      if (svg.style) {
        svg.style.maxWidth = 'none';
        svg.style.maxHeight = 'none';
      }

      const allNodes = svg.querySelectorAll('.node');
      allNodes.forEach(node => {
        const el = node as unknown as SVGElement;
        if (!el || !el.style) return;

        el.style.opacity = activeFlow ? '0.2' : '1';
        el.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
        el.style.cursor = onNodeClick ? 'pointer' : 'default';
        
        const logicalId = el.id?.split('-')[1] || el.getAttribute('data-id');
        
        if (logicalId && onNodeClick) {
          el.onclick = (e) => {
            e.stopPropagation();
            onNodeClick(logicalId);
          };
        }
      });

      const allEdges = svg.querySelectorAll('.edgePath');
      allEdges.forEach(edge => {
        const el = edge as unknown as SVGElement;
        if (el && el.style) {
          el.style.opacity = activeFlow ? '0.1' : '1';
          el.style.transition = 'opacity 0.4s ease';
        }
      });

      if (activeFlow) {
        activeFlow.nodes.forEach(nodeId => {
          const nodeEl = svg.querySelector(`[id*="-${nodeId}-"], .node-${nodeId}`);
          if (nodeEl) {
            const el = nodeEl as unknown as SVGElement;
            if (el && el.style) {
              el.style.opacity = '1';
              el.style.filter = `drop-shadow(0 0 12px ${flowColor}66)`;
              
              const shapes = el.querySelectorAll('rect, polygon, circle, path, ellipse');
              shapes.forEach(shape => {
                const s = shape as unknown as SVGElement;
                if (s && s.style) {
                  s.style.stroke = flowColor;
                  s.style.strokeWidth = '4px';
                }
              });
            }
          }
        });

        activeFlow.edges.forEach(([from, to]) => {
          const edgeSelector = `[id*="${from}"][id*="${to}"].edgePath, .edge-${from}-${to}`;
          const edgeEl = svg.querySelector(edgeSelector);
          if (edgeEl) {
            const el = edgeEl as unknown as SVGElement;
            if (el && el.style) {
              el.style.opacity = '1';
              const paths = el.querySelectorAll('path');
              paths.forEach(p => {
                const ps = p as unknown as SVGElement;
                if (ps && ps.style) {
                  ps.style.stroke = flowColor;
                  ps.style.strokeWidth = '4px';
                }
              });
            }
          }
        });
      }
    } catch (err) {
      console.warn("Mermaid Post-Processing Warning:", err);
    }
  }, [isRendered, onNodeClick, activeFlow, flowColor]);

  if (renderError) {
    return (
      <div className="w-full bg-rose-50 border border-rose-100 rounded-3xl flex flex-col items-center justify-center p-12 text-center" style={{ height }}>
        <div className="w-20 h-20 bg-rose-100 text-rose-600 rounded-full flex items-center justify-center text-4xl mb-6 animate-bounce">🛠️</div>
        <h3 className="text-rose-950 font-black text-lg uppercase tracking-widest mb-3">Kernel Panic: Render Fault</h3>
        <p className="text-rose-700/70 text-[11px] font-mono max-w-md mb-8 leading-relaxed px-4">
          The visualization engine failed to initialize the topology. <br/>
          <span className="font-bold text-rose-900 bg-white/50 px-2 py-1 rounded inline-block mt-2">{renderError}</span>
        </p>
        <button 
          onClick={() => window.location.reload()} 
          className="px-8 py-3 bg-rose-600 text-white text-xs font-black rounded-xl hover:bg-rose-700 transition-all shadow-xl hover:shadow-rose-200 uppercase tracking-widest"
        >
          Cold Reboot System
        </button>
      </div>
    );
  }

  return (
    <div className="w-full border border-gray-200 rounded-3xl bg-white overflow-hidden relative shadow-sm group" style={{ height }}>
      <TransformWrapper 
        centerOnInit 
        minScale={0.05} 
        maxScale={10} 
        limitToBounds={false}
        initialScale={0.8}
      >
        <TransformComponent wrapperClass="!w-full !h-full" contentClass="!w-full !h-full flex items-center justify-center">
          <div 
            ref={containerRef} 
            className="mermaid w-full h-full p-4 flex items-center justify-center select-none" 
          />
        </TransformComponent>
      </TransformWrapper>
      
      {activeFlow && (
        <div className="absolute top-8 left-8 z-10 bg-white/80 backdrop-blur-xl px-5 py-2.5 rounded-2xl border border-gray-100 shadow-2xl flex items-center gap-4 animate-in fade-in slide-in-from-left-6 duration-1000">
           <div className="relative flex h-3 w-3">
             <div className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></div>
             <div className="relative inline-flex rounded-full h-3 w-3 bg-blue-600"></div>
           </div>
           <span className="text-[11px] font-black text-slate-900 uppercase tracking-[0.2em]">Signal Trace Active</span>
        </div>
      )}

      <div className="absolute bottom-8 right-8 z-10 opacity-0 group-hover:opacity-100 transition-all duration-500 transform translate-y-2 group-hover:translate-y-0 pointer-events-none">
        <div className="bg-slate-900/10 text-slate-600 text-[9px] font-black px-4 py-2 rounded-xl backdrop-blur-md border border-white/20 uppercase tracking-wider">
          Pinch Zoom • Pan Discovery Mode
        </div>
      </div>
    </div>
  );
};

export default MermaidDiagram;