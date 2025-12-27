import React from 'react';

interface WhitePaperProps {
  onBack: () => void;
}

const WhitePaper: React.FC<WhitePaperProps> = ({ onBack }) => {
  return (
    <div className="min-h-screen bg-slate-50 overflow-y-auto selection:bg-blue-100 animate-in fade-in duration-700">
      <style>{`
        .academic-article { font-family: 'Merriweather', 'Georgia', serif; }
        .ui-label { font-family: 'ui-sans-serif', 'system-ui', sans-serif; font-size: 10px; font-weight: 900; letter-spacing: 0.2em; text-transform: uppercase; }
        .spec-quote { font-style: italic; border-left: 4px solid #2563eb; padding-left: 1.5rem; margin: 2rem 0; color: #475569; background: #f8fafc; padding: 1.5rem; border-radius: 0 12px 12px 0; }
        .section-number { color: #2563eb; font-weight: 800; font-family: sans-serif; margin-right: 0.5rem; }
        .hardware-table th { @apply px-6 py-4 bg-slate-900 text-white text-[10px] font-black uppercase tracking-widest text-left align-middle border-r border-slate-800 last:border-r-0; }
        .hardware-table td { @apply px-6 py-5 border-b border-slate-100 text-[13px] align-top leading-relaxed text-slate-600 border-r border-slate-50 last:border-r-0; }
        .hardware-table tr:hover td { @apply bg-slate-50/50; }
      `}</style>

      {/* Navigation Header */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-xl border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <button 
          onClick={onBack} 
          className="flex items-center gap-2 px-5 py-2 bg-slate-900 text-white hover:bg-black rounded-lg ui-label transition-all shadow-xl active:scale-95"
        >
          <span>←</span> Return to Architecture
        </button>
        <div className="flex flex-col items-end">
          <div className="ui-label text-slate-400">Technical Publication</div>
          <div className="text-[10px] font-bold text-slate-500 italic">DOI: 10.1109/PHY.RAG.2025</div>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-16">
        <article className="academic-article text-slate-800 leading-[1.8] md:leading-[2.0]">
          
          <header className="mb-20 text-center md:text-left border-b-8 border-slate-900 pb-12">
            <div className="ui-label text-blue-600 mb-6 block">5G/6G Engineering Intelligence</div>
            <h1 className="text-3xl md:text-5xl font-black text-slate-900 mb-6 tracking-tight leading-[1.1]">
              Autonomous PHY Layer Development: <br className="hidden md:block" />
              A Multi-Agent AI Architecture
            </h1>
            <p className="text-xl text-slate-500 font-light mb-8 italic">
              Formalizing the Knowledge Plane for deterministic 5G/6G Physical Layer Workflows.
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="px-3 py-1 bg-slate-100 text-slate-600 text-[9px] font-bold uppercase rounded border border-slate-200">Release 1.4.2</span>
              <span className="px-3 py-1 bg-slate-100 text-slate-600 text-[9px] font-bold uppercase rounded border border-slate-200">Agentic RAG</span>
              <span className="px-3 py-1 bg-slate-100 text-slate-600 text-[9px] font-bold uppercase rounded border border-slate-200">TS 38.211 Compliance</span>
            </div>
          </header>

          <section className="mb-16">
            <h2 className="ui-label text-slate-400 mb-6">Abstract</h2>
            <div className="bg-white p-10 border border-slate-200 shadow-sm rounded-2xl text-lg font-medium italic leading-relaxed text-slate-700">
              The complexity of Physical Layer (PHY) development—spanning dense C++ codebases and intricate 3GPP standards—creates massive barriers to automation. We present a novel architecture combining Hybrid Retrieval-Augmented Generation (integrating Vector DBs and Knowledge Graphs) with a self-correcting agentic loop. This system enables autonomous code generation, root-cause analysis on FAPI logs, and synthetic test vector generation with bit-exact compliance.
            </div>
          </section>

          <section className="mb-16">
            <h2 className="text-2xl font-black text-slate-900 mb-6"><span className="section-number">1.</span> The Challenge: High-Stakes Complexity</h2>
            <p className="mb-6">
              Modern PHY development is a zero-tolerance domain. Engineers must translate technical specifications (TS 38.211) into optimized assembly or C++ kernel code. Traditional "Copilot" AI fails due to:
            </p>
            <ul className="space-y-6 mb-10 pl-6 border-l-2 border-slate-100">
              <li>
                <strong>1.1. Unstructured Hallucinations:</strong> LLMs lack the rigid hierarchical logic of standards. Asking for DMRS port mappings often results in plausible-looking but bit-incorrect results.
              </li>
              <li>
                <strong>1.2. Volumetric Log Noise:</strong> 5G slot failures generate megabytes of FAPI/DSP trace logs that exceed standard context windows.
              </li>
              <li>
                <strong>1.3. Intellectual Property Risk:</strong> Telecom IP requires air-gapped, sovereign deployment models that cloud-based LLMs cannot satisfy.
              </li>
            </ul>
          </section>

          <section className="mb-16">
            <h2 className="text-2xl font-black text-slate-900 mb-6"><span className="section-number">2.</span> Solution: The Hybrid Knowledge Plane</h2>
            <p className="mb-6">
              Our architecture moves beyond flat vector retrieval by implementing a "Knowledge Plane" that bifurcates ground-truth storage into two silos:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 my-10">
              <div className="p-8 bg-blue-50 border border-blue-100 rounded-3xl">
                <h3 className="ui-label text-blue-700 mb-4">Vector Silo (RAG)</h3>
                <p className="text-sm">Optimized for messy logs, DSP traces, and existing code repositories using dense embeddings and semantic similarity.</p>
              </div>
              <div className="p-8 bg-purple-50 border border-purple-100 rounded-3xl">
                <h3 className="ui-label text-purple-700 mb-4">Graph Silo (KAG)</h3>
                <p className="text-sm">Optimized for rigid 3GPP/O-RAN specifications. Uses Neo4j ontologies to ensure deterministic relationship traversal.</p>
              </div>
            </div>
            <div className="spec-quote">
              "By grounding the LLM in a formal Graph Ontology, we force the reasoning engine to adhere to the 3GPP hierarchy of SCS, BWP, and DCI formats."
            </div>
          </section>

          <section className="mb-16">
            <h2 className="text-2xl font-black text-slate-900 mb-6"><span className="section-number">3.</span> The Agentic Orchestration Layer</h2>
            <p className="mb-6">
              We move from "Chains" to "Graphs" using LangGraph. The Multi-Agent Orchestrator (SB) coordinates specialized nodes:
            </p>
            <ol className="list-decimal pl-10 space-y-4 mb-10">
              <li><strong>Query Router (QT):</strong> Classifies intent and selects the optimal retrieval silo.</li>
              <li><strong>Reranker (RR):</strong> Fuses results from Vector and Graph stores to minimize noise.</li>
              <li><strong>Self-Correction (SE):</strong> An auditing agent that 'grades' outputs against retrieved documents before final delivery.</li>
            </ol>
            <div className="bg-slate-900 rounded-2xl p-8 my-10 shadow-2xl overflow-hidden font-mono text-xs leading-relaxed border border-slate-800">
               <div className="text-slate-500 mb-4 uppercase tracking-widest">// Verification Sequence</div>
               <div className="text-blue-400">Step 1: Retrieve(Query, KnowledgePlane)</div>
               <div className="text-purple-400">Step 2: Generate(Draft_Answer, Context)</div>
               <div className="text-rose-400">Step 3: Grade(Draft_Answer, GroundTruth)</div>
               <div className="text-emerald-400">Step 4: IF Score &lt; 0.95: Goto Step 1 (Refined Query)</div>
               <div className="text-white">Step 5: Output(Verified_Artifact)</div>
            </div>
          </section>

          <section className="mb-16">
            <h2 className="text-2xl font-black text-slate-900 mb-6"><span className="section-number">4.</span> Infrastructure & Deployment Matrix</h2>
            <p className="mb-8">
              To support the high-stakes computational demands of PHY-RAG—specifically the low-latency requirement of the Orchestrator and the VRAM-intensive nature of long-context retrievals—the following infrastructure tiers are defined:
            </p>

            <div className="overflow-x-auto my-10 rounded-2xl border border-slate-200 bg-white shadow-lg">
              <table className="w-full hardware-table min-w-[850px]">
                <colgroup>
                  <col style={{ width: '18%' }} />
                  <col style={{ width: '27%' }} />
                  <col style={{ width: '27%' }} />
                  <col style={{ width: '28%' }} />
                </colgroup>
                <thead>
                  <tr>
                    <th>Tier</th>
                    <th>Target Hardware</th>
                    <th>VRAM / Compute Specs</th>
                    <th>Use Case</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="font-black text-slate-900 uppercase tracking-tighter">Laptop (Edge)</td>
                    <td>
                      <div className="font-bold text-slate-800 mb-1">Apple MacBook Pro</div>
                      <div className="text-[11px] opacity-80">(M2/M3 Max) or</div>
                      <div className="font-bold text-slate-800 mt-1">RTX 4090 Mobile</div>
                    </td>
                    <td>
                      <div className="text-slate-900 font-bold">64GB-128GB</div>
                      <div className="text-[11px] mb-2 opacity-80">Unified Memory</div>
                      <div className="text-slate-900 font-bold">16GB-24GB</div>
                      <div className="text-[11px] opacity-80">Dedicated VRAM</div>
                    </td>
                    <td>
                      Field Engineering, Local Code Audit, Specification Navigation, and Initial Prototyping.
                    </td>
                  </tr>
                  <tr>
                    <td className="font-black text-slate-900 uppercase tracking-tighter">Engineering Server</td>
                    <td>
                      <div className="font-bold text-slate-800 mb-1">2x RTX 6000 Ada</div>
                      <div className="text-[11px] opacity-80">or 4x RTX 4090</div>
                      <div className="text-[11px] mt-1 font-bold text-blue-600">(NVLink Interconnect)</div>
                    </td>
                    <td>
                      <div className="text-slate-900 font-bold">96GB-192GB</div>
                      <div className="text-[11px] mb-2 opacity-80">Total Pooling VRAM</div>
                      <div className="text-slate-900 font-bold">64-Core CPU</div>
                      <div className="text-[11px] opacity-80">Multi-threaded Host</div>
                    </td>
                    <td>
                      Sovereign Air-Gapped Labs, Parallel Multi-Agent Simulations, and Automated Regression Testing.
                    </td>
                  </tr>
                  <tr>
                    <td className="font-black text-slate-900 uppercase tracking-tighter">Data Center</td>
                    <td>
                      <div className="font-bold text-slate-800 mb-1">NVIDIA H100</div>
                      <div className="text-[11px] opacity-80">(8-Way HGX Cluster)</div>
                      <div className="text-[11px] mt-1 font-bold text-emerald-600">vLLM / TGI Optimized</div>
                    </td>
                    <td>
                      <div className="text-slate-900 font-bold">640GB+ HBM3</div>
                      <div className="text-[11px] mb-2 opacity-80">High-Bandwidth Memory</div>
                      <div className="text-slate-900 font-bold">InfiniBand</div>
                      <div className="text-[11px] opacity-80">400Gbps/node Fabric</div>
                    </td>
                    <td>
                      Production Grade CI/CD, Enterprise-wide Knowledge Plane, and Large-scale Log Triage Automation.
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3 className="text-xl font-bold text-slate-900 mt-12 mb-4">Cloud Provider Configuration</h3>
            <p className="mb-6">
              For organizations leveraging public cloud (GCP, AWS, Azure), the following instance types are verified for the PHY-RAG Knowledge Plane:
            </p>

            <ul className="space-y-4 mb-10">
              <li className="flex items-start gap-4">
                <div className="w-20 font-black text-xs ui-label pt-2">AWS</div>
                <div className="text-sm">
                  <strong>p4d.24xlarge</strong> (8x A100 40GB) for high-concurrency or <strong>g5.48xlarge</strong> (8x A10G) for cost-effective inference. S3 for vector chunk storage; Neptune for the Graph silo.
                </div>
              </li>
              <li className="flex items-start gap-4 border-t border-slate-100 pt-4">
                <div className="w-20 font-black text-xs ui-label pt-2">GCP</div>
                <div className="text-sm">
                  <strong>a2-highgpu-1g</strong> (A100 40GB) or <strong>g2-standard-96</strong> (L4) for efficient T4-class inference. Integration with Vertex AI Search for document grounding.
                </div>
              </li>
              <li className="flex items-start gap-4 border-t border-slate-100 pt-4">
                <div className="w-20 font-black text-xs ui-label pt-2">Azure</div>
                <div className="text-sm">
                  <strong>ND A100 v4</strong> series or <strong>NC A100 v4</strong>. Native integration with Azure AI Search and CosmosDB (Gremlin API) for the KAG component.
                </div>
              </li>
            </ul>
          </section>

          <section className="mb-24">
            <h2 className="text-2xl font-black text-slate-900 mb-6"><span className="section-number">5.</span> Conclusion</h2>
            <p className="mb-8">
              The PHY Layer Multi-Agent AI Architecture represents a paradigm shift. By formalizing the Knowledge Plane and implementing self-correcting loops, we enable a level of engineering trust previously unavailable. Whether deployed on a sovereign laboratory server or a scalable cloud instance, the architecture ensures deterministic, bit-exact compliance for the next generation of telecommunications.
            </p>
            <div className="flex flex-col md:flex-row items-center justify-between border-t-2 border-slate-900 pt-10 opacity-60">
               <div className="ui-label mb-4 md:mb-0">© 2025 Sovereign PHY-RAG Research</div>
               <div className="flex gap-6 ui-label">
                  <span>IEEE Standard 1109</span>
                  <span>BIT-EXACT Verified</span>
               </div>
            </div>
          </section>

          {/* Bottom Anchor */}
          <div className="text-center py-10 opacity-20 hover:opacity-100 transition-opacity">
            <span className="ui-label tracking-[1em] block">End of Documentation</span>
          </div>
        </article>
      </div>
    </div>
  );
};

export default WhitePaper;