# PHY Layer Agentic RAG - Local Development Guide

## Overview
This is a React/TypeScript application for visualizing and simulating autonomous 5G/6G development and log triage using a multi-agent RAG (Retrieval-Augmented Generation) architecture. The application features interactive diagrams, real-time telemetry, and specialized components for various telecommunications workflows.

## Prerequisites
- Node.js (version 18 or higher)
- npm (comes with Node.js)
- A modern browser that supports ES modules and import maps

## Local Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Environment Configuration
The application uses a Gemini API key for AI functionality. Create a `.env.local` file in the root directory:

```bash
# .env.local
GEMINI_API_KEY=your-gemini-api-key-here
```

### 4. Run the Application
```bash
npm run dev
```

The application will start on `http://localhost:3000`.

## Application Architecture

### Core Components
- **Main App (`App.tsx`)**: The main application component with multiple operational flows (ingest, vector_rag, graph_rag, paper2code, correction, output)
- **Mermaid Diagram Component**: Interactive visualization of system architecture and sequence diagrams
- **Specialized Views**: White paper, LinkedIn post, technical deep dive, chat simulation, and more
- **Real-time Telemetry**: Shows inference latency and semantic cache hit rates

### Key Features
1. **Multi-Agent Architecture**: Visualizes different operational flows in a telecommunications RAG system
2. **Interactive Diagrams**: Toggle between architecture and sequence diagrams
3. **Node Details**: Click on any node to see detailed information about its function
4. **Real-time Metrics**: Shows live telemetry data for system performance
5. **Specialized Tools**: Paper2Code, Technical Deep Dive, Honesty Evaluation, and other domain-specific tools

### Data Flow Nodes
The application visualizes a complex telecommunications RAG system with these key nodes:
- **End User (U)**: Primary actor initiating queries
- **Knowledge Sources (A-I)**: Codebase, logs, books, test vectors, specifications
- **Storage Systems (SC, SK)**: Vector database and Knowledge graph
- **Processing Agents (SB, QT, RR, SE, AI)**: Multi-agent orchestrator, query router, reranker, self-correction, AI core
- **Output Artifacts (F-I)**: Code reviews, log triage, test vector generation, code generation

## Memory Constraints Workaround

If you encounter memory constraints during installation (exit code 137), try these approaches:

### Option 1: Increase Node.js Memory Limit
```bash
export NODE_OPTIONS="--max-old-space-size=4096"
npm install
```

### Option 2: Install with Production Dependencies Only
```bash
npm install --production
```

### Option 3: Install Dependencies Individually
```bash
npm install react react-dom mermaid react-zoom-pan-pinch @google/genai katex
npm install -D @vitejs/plugin-react typescript vite vitest @vitest/ui @testing-library/react @testing-library/jest-dom jsdom
```

### Option 4: Use Alternative Package Manager
```bash
# Using yarn instead of npm
yarn install
yarn dev

# Or using pnpm
pnpm install
pnpm dev
```

## Alternative: Standalone HTML Version

The application can potentially run as a standalone HTML file using ES modules from CDNs. The `index.html` file already includes:
- Tailwind CSS via CDN
- KaTeX for mathematical expressions
- Import maps for React and related libraries

However, this approach may have limitations with complex components and may not work in all browsers.

## Development Scripts

- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run preview`: Preview production build
- `npm test`: Run tests with Vitest
- `npm test:ui`: Run tests with UI
- `npm test:run`: Run tests in CLI mode

## Testing

The application includes comprehensive tests using Vitest and React Testing Library. Tests cover:
- Main App component
- Mermaid diagram component
- Constants and data structures

Run tests with:
```bash
npm test
```

## Troubleshooting

### Common Issues:
1. **Memory Issues**: Increase Node.js memory limit or use production dependencies only
2. **Import Map Issues**: Ensure you're using a modern browser that supports import maps
3. **API Key Issues**: Ensure GEMINI_API_KEY is properly set in environment variables
4. **Dependency Conflicts**: Clear node_modules and reinstall: `rm -rf node_modules package-lock.json && npm install`

### Browser Support:
- The application uses import maps which are supported in modern browsers (Chrome 89+, Edge 89+, Firefox 102+)
- For older browsers, consider using a bundler build instead of the CDN approach

## Key Technologies Used

- **Frontend**: React 19, TypeScript
- **Styling**: Tailwind CSS
- **Visualization**: Mermaid.js for diagrams
- **Math Rendering**: KaTeX
- **Build Tool**: Vite
- **Testing**: Vitest, React Testing Library
- **Zoom/Pan**: react-zoom-pan-pinch

## Project Structure

```
/workspace/
├── App.tsx                 # Main application component
├── index.tsx               # Application entry point
├── index.html              # HTML template with CDN imports
├── constants.ts            # Node definitions and diagram constants
├── components/             # Reusable UI components
├── package.json            # Dependencies and scripts
├── vite.config.ts          # Vite configuration
├── tsconfig.json           # TypeScript configuration
├── README.md               # Project overview
├── TESTING.md              # Testing documentation
└── start.sh                # Startup script
```

## Next Steps for Local Development

1. Ensure Node.js and npm are installed
2. Clone the repository
3. Install dependencies with `npm install`
4. Set up environment variables
5. Start the development server with `npm run dev`
6. Explore the different operational flows and components
7. Modify components as needed for your specific use case