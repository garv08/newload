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
} from './logos.ts';

export type FlowDefinition = {
  nodes: string[];
  edges: [string, string][];
};

export type NodeDetail = {
  title: string;
  description: string;
  logo?: string;
  // Necessary fields for current UI implementation
  icon: string;
  color: string;
  inputs: string[];
  outputs: string[];
  specs: string[];
};

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
      "Implements self-correction loops to ensure bit-exact engineering compliance."
    ]
  }
];

export const getDiagramDefinition = (direction: 'LR' | 'TD') => `
flowchart ${direction}
    U["${NODE_DETAILS.U.icon} ${NODE_DETAILS.U.title}"]
    
    subgraph Inputs ["Technical Knowledge Sources"]
        direction ${direction}
        A["${NODE_DETAILS.A.icon} ${NODE_DETAILS.A.title}"]
        B["${NODE_DETAILS.B.icon} ${NODE_DETAILS.B.title}"]
        C["${NODE_DETAILS.C.icon} ${NODE_DETAILS.C.title}"]
        D["${NODE_DETAILS.D.icon} ${NODE_DETAILS.D.title}"]
    end

    subgraph Specs ["Standardization Frameworks"]
        direction ${direction}
        E1["${NODE_DETAILS.E1.icon} ${NODE_DETAILS.E1.title}"]
        E2["${NODE_DETAILS.E2.icon} ${NODE_DETAILS.E2.title}"]
        E3["${NODE_DETAILS.E3.icon} ${NODE_DETAILS.E3.title}"]
    end

    CACHE["${NODE_DETAILS.CACHE.icon} ${NODE_DETAILS.CACHE.title}"]
    
    subgraph Storage ["Ground Truth Engines"]
        SC[("${NODE_DETAILS.SC.icon} ${NODE_DETAILS.SC.title}")]
        SK["${NODE_DETAILS.SK.icon} ${NODE_DETAILS.SK.title}"]
    end

    QT{"${NODE_DETAILS.QT.icon} ${NODE_DETAILS.QT.title}"}
    RR["${NODE_DETAILS.RR.icon} ${NODE_DETAILS.RR.title}"]
    SB["${NODE_DETAILS.SB.icon} ${NODE_DETAILS.SB.title}"]
    AI["${NODE_DETAILS.AI.icon} ${NODE_DETAILS.AI.title}"]
    SE["${NODE_DETAILS.SE.icon} ${NODE_DETAILS.SE.title}"]
    P2C["${NODE_DETAILS.P2C.icon} ${NODE_DETAILS.P2C.title}"]

    subgraph OutputNodes ["Engineering Artifacts"]
        F["${NODE_DETAILS.F.icon} ${NODE_DETAILS.F.title}"]
        G["${NODE_DETAILS.G.icon} ${NODE_DETAILS.G.title}"]
        H["${NODE_DETAILS.H.icon} ${NODE_DETAILS.H.title}"]
        I["${NODE_DETAILS.I.icon} ${NODE_DETAILS.I.title}"]
    end

    U <--> CACHE
    CACHE <--> SB
    SB <--> QT
    QT --> SC
    QT --> SK
    SC --> RR
    SK --> RR
    RR --> SB
    SB <--> AI
    SB <--> SE
    SB --> F
    SB --> G
    SB --> H
    SB --> I
    SB --> U
    C --> P2C
    E1 --> P2C
    P2C --> SB
    
    A --> SC
    B --> SC
    D --> SC
    E1 --> SK
    E2 --> SK
    E3 --> SK

    classDef default font-family:ui-sans-serif, system-ui, sans-serif, font-weight:700;
`;

export const SEQUENCE_DIAGRAM_DEFINITION = `
sequenceDiagram
    autonumber
    participant U as Developer
    participant C as Cache
    participant SB as Orchestrator
    participant QT as Router
    participant SC as VectorDB
    participant SK as GraphDB
    participant AI as LLM Core
    participant SE as Auditor

    U->>C: Submit Query
    alt Cache Hit
        C-->>U: Cached Response (200ms)
    else Cache Miss
        C->>SB: Forward Intent
        SB->>QT: Identify Silos
        par
            QT->>SC: Vector Retrieval
            QT->>SK: Graph Traversal
        end
        SC-->>SB: Unstructured Context
        SK-->>SB: Formal Triples
        SB->>AI: Generate Draft
        AI-->>SB: Raw Response
        SB->>SE: Audit against Ground Truth
        alt Hallucination Detected
            SE->>SB: Reject & Feedback
            SB->>AI: Re-generate with constraints
        end
        SE-->>SB: Verified Approval
        SB-->>U: Delivery Verified Artifact
    end
`;