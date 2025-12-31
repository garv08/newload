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

        {/* --- 1. Semantic Cache --- */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <span className="w-10 h-10 rounded-lg bg-orange-100 text-orange-600 flex items-center justify-center font-bold">01</span>
            <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Semantic Cache (CAG)</h2>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
              <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 tracking-widest">Mathematical Model</h3>
              <p className="mb-4 text-slate-700 leading-relaxed">
                {/* Fix: Wrapped text containing LaTeX-style braces in a string literal to prevent JSX interpretation as JavaScript */}
                {'Let $q$ be the user query vector and $C = \\{c_1, c_2, ..., c_n\\}$ be the set of cached query vectors.'}
              </p>
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-center font-serif text-xl mb-4 italic">
                {/* Fix: Wrapped LaTeX formula in string literal and escaped backslashes */}
                {'S(q, c_i) = \\frac{q \\cdot c_i}{\\|q\\| \\|c_i\\|}'}
              </div>
              <p className="text-slate-700 leading-relaxed">
                {/* Fix: Wrapped subscript notation in string literal */}
                {'The cache returns result $A_{best}$ if:'}
              </p>
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-center font-serif text-xl italic">
                {/* Fix: Wrapped formula containing braces and backslashes in string literal */}
                {'\\max_{c_i \\in C} S(q, c_i) > \\tau'}
              </div>
              <p className="mt-4 text-xs text-slate-500 italic">where $\tau = 0.95$ is the defined similarity threshold.</p>
            </div>

            <div className="bg-slate-900 rounded-xl p-6 overflow-x-auto shadow-inner">
              <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-widest">Pseudo-code: Cache Interception</h3>
              <pre className="text-blue-300 font-mono text-sm leading-relaxed">
{`FUNCTION CheckSemanticCache(user_query):
    query_embedding = Embed(user_query)
    results = Redis.VectorSearch(query_embedding, top_k=1)
    
    IF results.length > 0:
        best_match = results[0]
        IF best_match.similarity > 0.95:
            Log("Semantic Cache HIT (Score: {best_match.similarity})")
            RETURN best_match.answer
            
    Log("Semantic Cache MISS. Proceeding to Agentic Flow.")
    RETURN NULL`}
              </pre>
            </div>
          </div>
        </section>

        {/* --- 2. Hybrid Retrieval --- */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <span className="w-10 h-10 rounded-lg bg-purple-100 text-purple-600 flex items-center justify-center font-bold">02</span>
            <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Hybrid RAG Fusion (RAG + KAG)</h2>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
              <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 tracking-widest">Reciprocal Rank Fusion (RRF)</h3>
              <p className="mb-4 text-slate-700 leading-relaxed">
                To combine unstructured vector results ($R_v$) and structured graph triples ($R_g$), we apply RRF to calculate a unified relevance score for each context chunk $d$:
              </p>
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-center font-serif text-xl mb-4 italic">
                {/* Fix: Wrapped complex RRF formula in string literal */}
                {'RRF(d) = \\sum_{r \\in \\{R_v, R_g\\}} \\frac{1}{k + rank(r, d)}'}
              </div>
              <p className="text-slate-700 leading-relaxed">
                Where $k$ is a constant (typically 60) that prevents high-ranking items from dominating the score disproportionately.
              </p>
            </div>

            <div className="bg-slate-900 rounded-xl p-6 overflow-x-auto shadow-inner">
              <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-widest">Pseudo-code: Tri-Hybrid Retrieval</h3>
              <pre className="text-purple-300 font-mono text-sm leading-relaxed">
{`FUNCTION PerformHybridRetrieval(query):
    // Parallel execution for latency optimization
    vector_results = ASYNC ChromaDB.Search(query, top_k=20)
    graph_triples = ASYNC Neo4j.CypherQuery(query, entities=ExtractEntities(query))
    
    // Fusion Layer
    unified_context = []
    FOR document IN (vector_results + graph_triples):
        score = CalculateRRF(document, vector_results, graph_triples)
        unified_context.Append({data: document, score: score})
    
    // Reranking via Cross-Encoder
    ranked_context = BGE_Reranker.Score(query, unified_context.Top(20))
    RETURN ranked_context.Filter(score > threshold)`}
              </pre>
            </div>
          </div>
        </section>

        {/* --- 3. Self-Correction Loop --- */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <span className="w-10 h-10 rounded-lg bg-rose-100 text-rose-600 flex items-center justify-center font-bold">03</span>
            <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Self-Correction (CRAG)</h2>
          </div>

          <div className="space-y-6">
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
              <h3 className="text-sm font-bold text-slate-500 uppercase mb-4 tracking-widest">Grader Logic</h3>
              <p className="mb-4 text-slate-700 leading-relaxed">
                The Grader Agent implements a binary classification function $G$ over the draft answer $A$ and the ground-truth specification context $C_g$:
              </p>
              <div className="bg-white p-6 rounded-lg border border-slate-200 text-center font-serif text-xl mb-4 italic">
                {/* Fix: Wrapped cases environment formula in string literal */}
                {'G(A, C_g) = \\begin{cases} 1 & \\text{if } A \\text{ is spec-compliant} \\\\ 0 & \\text{if } A \\text{ contains hallucination/error} \\end{cases}'}
              </div>
              <p className="text-slate-700 leading-relaxed">
                The state machine transitions back to the <code>RETRIEVAL</code> state if $G = 0$, otherwise it proceeds to <code>DELIVER</code>.
              </p>
            </div>

            <div className="bg-slate-900 rounded-xl p-6 overflow-x-auto shadow-inner">
              <h3 className="text-xs font-bold text-slate-400 uppercase mb-4 tracking-widest">Pseudo-code: Reflection Loop</h3>
              <pre className="text-rose-300 font-mono text-sm leading-relaxed">
{`FUNCTION AgenticRefinedGeneration(query, context):
    retries = 0
    MAX_RETRIES = 2
    
    WHILE retries < MAX_RETRIES:
        draft = LLM.Generate(query, context)
        grade = GraderAgent.Verify(draft, context.SpecChunks)
        
        IF grade.status == "PASSED":
            RETURN draft
        ELSE:
            Log("REJECTED: {grade.reason}. Refining query...")
            new_query = QueryRewriter.Enhance(query, grade.reason)
            context = PerformHybridRetrieval(new_query)
            retries += 1
            
    RETURN LLM.GenerateSafeFallback(query, context)`}
              </pre>
            </div>
          </div>
        </section>

        {/* --- 4. System Complexity --- */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <span className="w-10 h-10 rounded-lg bg-slate-100 text-slate-600 flex items-center justify-center font-bold">04</span>
            <h2 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Asymptotic Complexity</h2>
          </div>
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-6">
             <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                   <h4 className="font-bold text-slate-900 mb-2">Time Complexity (Latency)</h4>
                   <p className="text-sm text-slate-600 mb-4">
                     {/* Fix: Wrapped text containing subscript notation braces in string literal */}
                     {'The worst-case latency is determined by the number of agentic steps ($N$) and the inference time per step ($T_{inf}$):'}
                   </p>
                   <div className="bg-white p-4 rounded border border-slate-200 font-mono text-center text-sm">
                     {/* Fix: Wrapped complexity notation in string literal for consistency */}
                     {'O(N_steps * T_inference + T_retrieval)'}
                   </div>
                </div>
                <div>
                   <h4 className="font-bold text-slate-900 mb-2">Space Complexity (Storage)</h4>
                   <p className="text-sm text-slate-600 mb-4">
                     Vector storage complexity depends on the number of chunks ($M$) and dimension ($D$):
                   </p>
                   <div className="bg-white p-4 rounded border border-slate-200 font-mono text-center text-sm">
                     O(M_chunks * D_dimension + E_edges)
                   </div>
                </div>
             </div>
          </div>
        </section>

        <footer className="text-center pt-10 border-t border-slate-100">
           <p className="text-xs font-mono text-slate-400 uppercase tracking-widest">
              End of Technical Document v1.3.1
           </p>
        </footer>
      </div>
    </div>
  );
};

export default TechnicalDeepDive;