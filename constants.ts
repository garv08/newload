import { 
  LOGO_3GPP, 
  LOGO_ORAN, 
  LOGO_FAPI, 
  LOGO_LANGCHAIN, 
  LOGO_CHROMA, 
  LOGO_OPENAI, 
  LOGO_LLAMA, 
  LOGO_NEO4J, 
  LOGO_RERANKER, 
  LOGO_USER, 
  LOGO_WEIGHTS, 
  LOGO_GEAR, 
  LOGO_SHIELD, 
  LOGO_COMPASS, 
  LOGO_CACHE,
  LOGO_BOOK
} from './logos';

export interface NodeDetail {
  title: string;
  icon: string;
  color: string;
  description: string;
  inputs: string[];
  outputs: string[];
  specs: string[];
}

export const NODE_DETAILS: Record<string, NodeDetail> = {
  U: {
    title: "End User",
    icon: "💻",
    color: "bg-sky-50 text-sky-700",
    description: "The primary actor initiating queries related to PHY layer development, log analysis, or specification clarification.",
    inputs: ["Natural Language Queries", "Log Files (FAPI/DSP)", "Code Snippets"],
    outputs: ["Optimized Code", "Root Cause Reports", "Synthetic Test Vectors"],
    specs: ["User Interface", "REST API", "Local Environment"]
  },
  CACHE: {
    title: "Semantic Cache",
    icon: "⚡",
    color: "bg-orange-50 text-orange-700",
    description: "A Redis-based vector storage that intercepts common queries to provide sub-200ms responses.",
    inputs: ["Query Embeddings"],
    outputs: ["Cached Responses", "Cache Miss Signal"],
    specs: ["Redis Stack", "Cosine Similarity", "Threshold: 0.95"]
  },
  A: {
    title: "PHY Codebase",
    icon: "💻",
    color: "bg-amber-50 text-amber-700",
    description: "The core L1 C/C++ and Assembly implementation including signal processing kernels.",
    inputs: ["Source Code", "Header Files"],
    outputs: ["Vector Embeddings", "Code Snippets"],
    specs: ["C++20", "SIMD (AVX-512)", "Assembly"]
  },
  B: {
    title: "Logs & Traces",
    icon: "📊",
    color: "bg-amber-50 text-amber-700",
    description: "High-volume operational data from FAPI interfaces and DSP traces.",
    inputs: ["Binary Logs", "ASCII Traces"],
    outputs: ["Parsed Events", "Error Patterns"],
    specs: ["SCF FAPI", "Circular Buffers", "PCAP"]
  },
  C: {
    title: "Books & Web",
    icon: "📖",
    color: "bg-amber-50 text-amber-700",
    description: "Conceptual knowledge harvested from technical literature.",
    inputs: ["HTML", "PDFs", "Markdown"],
    outputs: ["Concepts", "Explanatory Context"],
    specs: ["Semantic Parsing", "OCR", "Web Scraping"]
  },
  D: {
    title: "Test Vectors",
    icon: "🧪",
    color: "bg-amber-50 text-amber-700",
    description: "Standardized test data used to verify implementation against expected bit-exact results.",
    inputs: ["JSON", "MATLAB Files"],
    outputs: ["Vector Embeddings", "Bit-exact Samples"],
    specs: ["3GPP 38.141", "IQ Samples", "Parity Checks"]
  },
  E1: {
    title: "3GPP Specs",
    icon: "📜",
    color: "bg-amber-50 text-amber-700",
    description: "Definitive technical specifications for the 5G NR Physical Layer.",
    inputs: ["TS Documents"],
    outputs: ["Ontology Triples", "Formal Constraints"],
    specs: ["Release 15-18", "PDF/Word", "HTML"]
  },
  E2: {
    title: "O-RAN Specs",
    icon: "📜",
    color: "bg-amber-50 text-amber-700",
    description: "Open RAN Alliance specifications defining the 7.2x split.",
    inputs: ["O-RAN WGs"],
    outputs: ["Interface Mappings", "AAL Specs"],
    specs: ["WG4/WG8", "C-Plane/U-Plane", "LLS-CU"]
  },
  E3: {
    title: "SCF FAPI",
    icon: "📜",
    color: "bg-amber-50 text-amber-700",
    description: "Small Cell Forum Functional Application Platform Interface.",
    inputs: ["SCF 222 Docs"],
    outputs: ["Message Schemas", "P7/P5 Specs"],
    specs: ["Rel 10.04", "Tag-Length-Value", "Timing"]
  },
  SC: {
    title: "Vector DB",
    icon: "🟩",
    color: "bg-emerald-50 text-emerald-700",
    description: "ChromaDB instance storing dense vector representations of unstructured data.",
    inputs: ["Text Chunks", "Embeddings"],
    outputs: ["Relevant Chunks", "Metadata"],
    specs: ["HNSW Index", "Cosine Similarity", "Top-K Search"]
  },
  SK: {
    title: "Knowledge Graph",
    icon: "🕸️",
    color: "bg-purple-50 text-purple-700",
    description: "Neo4j graph database storing standards as a formal ontology.",
    inputs: ["Ontology Models", "Entities"],
    outputs: ["Structured Triples", "Relations"],
    specs: ["Cypher Queries", "LPG Model", "Entity Resolution"]
  },
  RR: {
    title: "Reranker",
    icon: "⚖️",
    color: "bg-orange-50 text-orange-700",
    description: "A cross-encoder model that scores retrieved context against the query.",
    inputs: ["Raw Chunks", "Graph Triples"],
    outputs: ["Ranked Context", "Scores"],
    specs: ["BGE-Reranker", "Cross-Attention", "Thresholding"]
  },
  SB: {
    title: "Multi-Agent Orchestrator",
    icon: "🧠",
    color: "bg-rose-50 text-rose-700",
    description: "The central intelligence using LangGraph to coordinate specialized agents.",
    inputs: ["Query", "Context"],
    outputs: ["Draft Response", "Tool Calls"],
    specs: ["LangGraph", "ReAct Pattern", "State Management"]
  },
  SE: {
    title: "Self-Correction & Grader",
    icon: "🛡️",
    color: "bg-rose-50 text-rose-700",
    description: "A reflection agent that audits generated code and answers against retrieved ground truth.",
    inputs: ["Draft Artifact", "Ground Truth"],
    outputs: ["Approved/Reject", "Feedback"],
    specs: ["Reflection Loop", "CRAG Pattern", "Deterministic Checks"]
  },
  QT: {
    title: "Query Router",
    icon: "🧭",
    color: "bg-rose-50 text-rose-700",
    description: "A classification agent that directs queries to the appropriate data silo.",
    inputs: ["User Query"],
    outputs: ["Transformed Query", "Target Store"],
    specs: ["Semantic Routing", "Prompt Engineering", "NLI"]
  },
  P2C: {
    title: "Paper2Code Agent",
    icon: "📄💻",
    color: "bg-blue-50 text-blue-700",
    description: "Utility for parsing academic papers and 3GPP proposals into executable simulations.",
    inputs: ["PDF Papers", "LaTeX Sources", "Equation Chunks"],
    outputs: ["Python/C++ Simulations", "Verification Graphs"],
    specs: ["Paper2Code CLI", "AST Mapping", "Numeric Equivalence"]
  },
  AI: {
    title: "AI Systems Core",
    icon: "⚙️",
    color: "bg-slate-50 text-slate-700",
    description: "The underlying inference engines (GPT-4o, Llama 3).",
    inputs: ["Prompts", "Context"],
    outputs: ["Token Streams", "JSON"],
    specs: ["GPT-4o", "Llama-3-70B", "quantization"]
  },
  F: {
    title: "Code Reviews",
    icon: "✅",
    color: "bg-sky-50 text-sky-700",
    description: "Automated review reports focusing on logic bugs and spec compliance.",
    inputs: ["Draft Code"],
    outputs: ["PR Comments", "Scores"],
    specs: ["Static Analysis", "Spec Mapping", "Performance"]
  },
  G: {
    title: "Log Triage",
    icon: "🔍",
    color: "bg-sky-50 text-sky-700",
    description: "Root cause analysis reports detailing the exact moment of failure.",
    inputs: ["Binary/Text Logs"],
    outputs: ["RCA Report", "Timeline"],
    specs: ["Event Correlation", "Timing Analysis", "FSM Trace"]
  },
  H: {
    title: "TV Generation",
    icon: "🧬",
    color: "bg-sky-50 text-sky-700",
    description: "Generation of synthetic test vectors covering edge cases.",
    inputs: ["Scenario Parameters"],
    outputs: ["JSON TVs", "IQ Files"],
    specs: ["3GPP 38.104", "Channel Models", "Noise Floor"]
  },
  I: {
    title: "Code Gen",
    icon: "💻",
    color: "bg-sky-50 text-sky-700",
    description: "Creation of highly optimized C++ kernels or Assembly snippets.",
    inputs: ["Functional Specs"],
    outputs: ["C++/ASM Code", "Unit Tests"],
    specs: ["Intrinsics", "Memory Alignment", "Cache Locality"]
  }
};

export const SLIDE_DECK = [
  {
    title: "PHY Layer RAG Architecture",
    subtitle: "Accelerating 5G/6G Workflows with AI",
    icon: "📡",
    color: "bg-gray-900 text-white",
    content: [
      "A Multi-Agent system designed for autonomous PHY layer development.",
      "Combines Knowledge Graphs for strict specs and Vector DBs for messy logs.",
      "Implements a 'Self-Correction' loop to ensure 100% spec compliance.",
      "Features Semantic Caching to reduce latency for common queries."
    ]
  }
];

export const getDiagramDefinition = (direction: 'LR' | 'TD' = 'LR') => `
flowchart ${direction}
    
    U["<div style='text-align:center;background:#f0f9ff;padding:10px;border-radius:15px;border:2px solid #7dd3fc'><img src='${LOGO_USER}' width='60' height='60' class='logo-img'/><div><b style='font-size:24px;color:#0369a1'>End User</b></div></div>"]

    CACHE["<div style='text-align:center;background:#fff7ed;padding:10px;border-radius:15px;border:2px solid #fdba74'><img src='${LOGO_CACHE}' width='50' height='50' class='logo-img'/><div><b style='font-size:24px;color:#c2410c'>Semantic Cache</b><br/><span style='color:#ea580c'>Redis/Vector</span></div></div>"]

    subgraph InputLayer [" 📥 Input Ingestion Plane "]
        style InputLayer fill:#fffbeb,stroke:#fcd34d,stroke-width:2px,stroke-dasharray: 5 5
        A["<div style='text-align:center;background:#fffdfa;padding:10px;border-radius:12px;border:1px solid #fde68a'><div style='font-size:30px'>💻</div><b style='font-size:24px;color:#92400e'>PHY Codebase</b><br/>C++, Assembly</div>"]
        B["<div style='text-align:center;background:#fffdfa;padding:10px;border-radius:12px;border:1px solid #fde68a'><div style='font-size:30px'>📊</div><b style='font-size:24px;color:#92400e'>Logs & Traces</b><br/>FAPI, DSP</div>"]
        C["<div style='text-align:center;background:#fffdfa;padding:10px;border-radius:12px;border:1px solid #fde68a'><img src='${LOGO_BOOK}' width='50' height='50'/><br/><b style='font-size:24px;color:#92400e'>Books & Web</b></div>"]
        D["<div style='text-align:center;background:#fffdfa;padding:10px;border-radius:12px;border:1px solid #fde68a'><div style='font-size:30px'>🧪</div><b style='font-size:24px;color:#92400e'>Test Vectors</b></div>"]
        
        subgraph Standards [" 📜 Specifications "]
            style Standards fill:#fffbeb,stroke:#fef3c7,stroke-width:1px,stroke-dasharray: 3 3
            E1["<div style='text-align:center;background:#fff;padding:8px;border-radius:10px;border:1px solid #fcd34d'><img src='${LOGO_3GPP}' width='60' height='50'/><br/><b style='font-size:22px;color:#713f12'>3GPP</b></div>"]
            E2["<div style='text-align:center;background:#fff;padding:8px;border-radius:10px;border:1px solid #fcd34d'><img src='${LOGO_ORAN}' width='60' height='40'/><br/><b style='font-size:22px;color:#713f12'>O-RAN</b></div>"]
            E3["<div style='text-align:center;background:#fff;padding:8px;border-radius:10px;border:1px solid #fcd34d'><img src='${LOGO_FAPI}' width='50' height='50'/><br/><b style='font-size:22px;color:#713f12'>SCF FAPI</b></div>"]
        end
    end

    subgraph DataLayer [" 🗄️ Knowledge & Retrieval Plane "]
        style DataLayer fill:#f5f3ff,stroke:#c4b5fd,stroke-width:2px,stroke-dasharray: 5 5
        SC["<div style='text-align:center;background:#f0fdf4;padding:10px;border-radius:15px;border:2px solid #86efac'><img src='${LOGO_CHROMA}' width='70' height='70' class='logo-img'/><div><b style='font-size:24px;color:#065f46'>Vector DB</b></div></div>"]
        SK["<div style='text-align:center;background:#f5f3ff;padding:10px;border-radius:15px;border:2px solid #c4b5fd'><img src='${LOGO_NEO4J}' width='70' height='70' class='logo-img'/><div><b style='font-size:24px;color:#5b21b6'>Knowledge Graph</b></div></div>"]
        RR["<div style='text-align:center;background:#fff7ed;padding:10px;border-radius:15px;border:2px solid #fdba74'><img src='${LOGO_RERANKER}' width='60' height='60' class='logo-img'/><div><b style='font-size:24px;color:#9a3412'>Reranker</b></div></div>"]
    end

    subgraph AgentLayer [" 🧠 Agentic Control Plane "]
        style AgentLayer fill:#fff1f2,stroke:#fda4af,stroke-width:2px,stroke-dasharray: 5 5
        SB["<div style='text-align:center;background:#fff1f2;padding:10px;border-radius:15px;border:2px solid #fda4af'><img src='${LOGO_GEAR}' width='60' height='60' class='logo-img'/><div><b style='font-size:24px;color:#9f1239'>Multi-Agent Orchestrator</b></div></div>"]
        SE["<div style='text-align:center;background:#fff1f2;padding:10px;border-radius:15px;border:2px solid #fda4af'><img src='${LOGO_SHIELD}' width='60' height='60' class='logo-img'/><div><b style='font-size:24px;color:#9f1239'>Self-Correction</b></div></div>"]
        QT["<div style='text-align:center;background:#fff1f2;padding:10px;border-radius:15px;border:2px solid #fda4af'><img src='${LOGO_COMPASS}' width='60' height='60' class='logo-img'/><div><b style='font-size:24px;color:#9f1239'>Query Router</b></div></div>"]
        P2C["<div style='text-align:center;background:#eff6ff;padding:10px;border-radius:15px;border:2px solid #3b82f6'><div style='font-size:24px'>📄💻</div><div><b style='font-size:24px;color:#1d4ed8'>Paper2Code</b><br/><span style='font-size:12px;color:#2563eb'>Simulation Gen</span></div></div>"]
        AI["<div style='text-align:center;background:#f8fafc;padding:15px;min-width:240px;border-radius:20px;border:2px solid #94a3b8'><div style='display:flex;align-items:center;justify-content:center;gap:15px;margin-bottom:15px'><img src='${LOGO_LANGCHAIN}' width='45' height='45'/><img src='${LOGO_OPENAI}' width='45' height='45'/><img src='${LOGO_LLAMA}' width='45' height='45'/></div><b style='font-size:26px;color:#334155'>AI Systems Core</b><br/><span style='font-size:16px;color:#64748b'>GPT-4o / Llama 3</span></div>"]
    end

    subgraph OutputLayer [" 📤 Output Delivery Plane "]
        style OutputLayer fill:#f0f9ff,stroke:#7dd3fc,stroke-width:2px,stroke-dasharray: 5 5
        F["<div style='text-align:center;background:#f8fafc;padding:10px;border-radius:12px;border:1px solid #7dd3fc'><div style='font-size:28px'>✅</div><b style='font-size:24px;color:#0369a1'>Code Reviews</b></div>"]
        G["<div style='text-align:center;background:#f8fafc;padding:10px;border-radius:12px;border:1px solid #7dd3fc'><div style='font-size:28px'>🔍</div><b style='font-size:24px;color:#0369a1'>Log Triage</b></div>"]
        H["<div style='text-align:center;background:#f8fafc;padding:10px;border-radius:12px;border:1px solid #7dd3fc'><div style='font-size:28px'>🧬</div><b style='font-size:24px;color:#0369a1'>TV Generation</b></div>"]
        I["<div style='text-align:center;background:#f8fafc;padding:10px;border-radius:12px;border:1px solid #7dd3fc'><div style='font-size:28px'>💻</div><b style='font-size:24px;color:#0369a1'>Code Gen</b></div>"]
    end

    A --> SC
    B --> SC
    C --> SC
    D --> SC
    E1 --> SK
    E2 --> SK
    E3 --> SK
    E1 --> SC
    E2 --> SC
    E3 --> SC
    U --> CACHE
    CACHE --> U
    CACHE --> SB
    SB --> U
    SB --> AI
    AI --> SB
    SB --> SE
    SE --> SB
    SE --> QT
    SB --> QT
    QT --> SC
    QT --> SK
    SC --> RR
    SK --> RR
    RR --> SB
    C --> P2C
    E1 --> P2C
    P2C --> SB
    SB --> F
    SB --> G
    SB --> H
    SB --> I
`;

export const SEQUENCE_DIAGRAM_DEFINITION = `
sequenceDiagram
    autonumber
    actor U as End User
    participant C as Semantic Cache
    participant O as Orchestrator
    participant P as Paper2Code
    participant R as Router
    participant V as Vector DB
    participant K as Knowledge Graph
    participant RR as Reranker
    participant G as Grader
    
    U->>C: Query
    alt Cache Hit
        C-->>U: Cached Response
    else Cache Miss
        C->>O: Forward
        O->>R: Route
        par Parallel
            R->>V: Vector Search
            R->>K: Graph Search
            R->>P: Extract Logic from PDF
        end
        V-->>RR: Chunks
        K-->>RR: Triples
        P-->>RR: AST Map
        RR-->>O: Fused Context
        O->>O: Draft Simulation
        O->>G: Verify vs Paper Curves
        G-->>O: OK
        O-->>U: Output (Code + Graphs)
        O->>C: Update Cache
    end
`;
