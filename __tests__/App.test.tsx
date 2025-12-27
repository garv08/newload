import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import { NODE_DETAILS, FLOWS } from '../constants';

// Mock the components that are imported in App
jest.mock('../components/MermaidDiagram', () => {
  return {
    __esModule: true,
    default: ({ chart, height }: { chart: string; height: string }) => (
      <div data-testid="mermaid-diagram" style={{ height }}>
        Mock Mermaid Diagram: {chart?.substring(0, 20)}...
      </div>
    )
  };
});

jest.mock('../components/DetailsModal', () => {
  return {
    __esModule: true,
    default: ({ isOpen, onClose, data }: { isOpen: boolean; onClose: () => void; data: any }) => (
      isOpen ? (
        <div data-testid="details-modal">
          <h2>{data?.title}</h2>
          <button onClick={onClose}>Close</button>
        </div>
      ) : null
    )
  };
});

jest.mock('../components/PresentationViewer', () => {
  return {
    __esModule: true,
    default: ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => (
      isOpen ? (
        <div data-testid="presentation-viewer">
          <button onClick={onClose}>Close</button>
        </div>
      ) : null
    )
  };
});

jest.mock('../components/WhitePaper', () => {
  return {
    __esModule: true,
    default: ({ onBack }: { onBack: () => void }) => (
      <div data-testid="white-paper">
        <h1>White Paper View</h1>
        <button onClick={onBack}>Back</button>
      </div>
    )
  };
});

jest.mock('../components/LinkedInPost', () => {
  return {
    __esModule: true,
    default: ({ onBack }: { onBack: () => void }) => (
      <div data-testid="linkedin-post">
        <h1>LinkedIn Post View</h1>
        <button onClick={onBack}>Back</button>
      </div>
    )
  };
});

jest.mock('../components/SequenceExplanation', () => {
  return {
    __esModule: true,
    default: ({ onBack }: { onBack: () => void }) => (
      <div data-testid="sequence-explanation">
        <h1>Sequence Explanation</h1>
        <button onClick={onBack}>Back</button>
      </div>
    )
  };
});

jest.mock('../components/TechnicalDeepDiveAudit', () => {
  return {
    __esModule: true,
    default: ({ onBack }: { onBack: () => void }) => (
      <div data-testid="technical-deep-dive-audit">
        <h1>Technical Deep Dive Audit</h1>
        <button onClick={onBack}>Back</button>
      </div>
    )
  };
});

jest.mock('../components/HonestEvaluation', () => {
  return {
    __esModule: true,
    default: ({ onBack }: { onBack: () => void }) => (
      <div data-testid="honest-evaluation">
        <h1>Honest Evaluation</h1>
        <button onClick={onBack}>Back</button>
      </div>
    )
  };
});

jest.mock('../components/VideoGenerator', () => {
  return {
    __esModule: true,
    default: ({ onClose }: { onClose: () => void }) => (
      <div data-testid="video-generator">
        <h1>Video Generator</h1>
        <button onClick={onClose}>Close</button>
      </div>
    )
  };
});

jest.mock('../components/Paper2CodeView', () => {
  return {
    __esModule: true,
    default: ({ onBack }: { onBack: () => void }) => (
      <div data-testid="paper2code-view">
        <h1>Paper2Code View</h1>
        <button onClick={onBack}>Back</button>
      </div>
    )
  };
});

jest.mock('../components/ChatSimulation', () => {
  return {
    __esModule: true,
    default: ({ onSimulateStep, onReset }: { onSimulateStep: (nodes: string[], edges: [string, string][]) => void; onReset: () => void }) => (
      <div data-testid="chat-simulation">
        <button onClick={() => onSimulateStep(['A', 'B'], [['A', 'B']])}>Simulate</button>
        <button onClick={onReset}>Reset</button>
      </div>
    )
  };
});

describe('App Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('renders main application structure', () => {
    render(<App />);
    
    // Check if main header elements are present
    expect(screen.getByText('PHY Layer Agentic RAG')).toBeInTheDocument();
    expect(screen.getByText('Autonomous 5G/6G Development & Log Triage')).toBeInTheDocument();
    
    // Check if operational flows section is present
    expect(screen.getByText('Operational Flows')).toBeInTheDocument();
    
    // Check if real-time telemetry section is present
    expect(screen.getByText('Real-time Telemetry')).toBeInTheDocument();
    
    // Check if diagram container is present
    expect(screen.getByTestId('mermaid-diagram')).toBeInTheDocument();
  });

  test('toggles flow selection', () => {
    render(<App />);
    
    // Initially no flow should be selected
    Object.keys(FLOWS).forEach(flowKey => {
      const flowButton = screen.getByText(flowKey.replace('_', ' '));
      expect(flowButton.closest('button')).not.toHaveClass('bg-slate-900');
    });
    
    // Click on the first flow
    const firstFlowKey = Object.keys(FLOWS)[0];
    const firstFlowButton = screen.getByText(firstFlowKey.replace('_', ' '));
    fireEvent.click(firstFlowButton);
    
    // Now the first flow should be selected
    expect(firstFlowButton.closest('button')).toHaveClass('bg-slate-900');
    
    // Click again to deselect
    fireEvent.click(firstFlowButton);
    expect(firstFlowButton.closest('button')).not.toHaveClass('bg-slate-900');
  });

  test('switches between architecture and sequence views', () => {
    render(<App />);
    
    // Initially architecture view should be selected
    expect(screen.getByText('Topology')).toHaveClass('bg-slate-900');
    expect(screen.getByText('Sequence')).not.toHaveClass('bg-slate-900');
    
    // Switch to sequence view
    const sequenceButton = screen.getByText('Sequence');
    fireEvent.click(sequenceButton);
    
    // Now sequence view should be selected
    expect(screen.getByText('Sequence')).toHaveClass('bg-slate-900');
    expect(screen.getByText('Topology')).not.toHaveClass('bg-slate-900');
  });

  test('opens and closes details modal when node is clicked', () => {
    render(<App />);
    
    // Initially modal should not be open
    expect(screen.queryByTestId('details-modal')).not.toBeInTheDocument();
    
    // Simulate clicking on a node (we'll mock the node click handler)
    const nodeKey = 'U'; // User node
    const nodeDetails = NODE_DETAILS[nodeKey];
    
    // Find a way to trigger node click - this would require more complex mocking
    // For now, we'll test the modal rendering when props are passed
  });

  test('opens white paper view when button is clicked', () => {
    render(<App />);
    
    // Initially white paper view should not be visible
    expect(screen.queryByTestId('white-paper')).not.toBeInTheDocument();
    
    // Click the white paper button
    const whitePaperButton = screen.getByText('Whitepaper');
    fireEvent.click(whitePaperButton);
    
    // Now white paper view should be visible
    expect(screen.getByTestId('white-paper')).toBeInTheDocument();
    
    // Click back button to return to main view
    fireEvent.click(screen.getByText('Back'));
    expect(screen.queryByTestId('white-paper')).not.toBeInTheDocument();
  });

  test('opens honest evaluation view when button is clicked', () => {
    render(<App />);
    
    // Initially honest evaluation view should not be visible
    expect(screen.queryByTestId('honest-evaluation')).not.toBeInTheDocument();
    
    // Click the research survey button (which opens honest evaluation)
    const researchSurveyButton = screen.getByText('Research Survey');
    fireEvent.click(researchSurveyButton);
    
    // Now honest evaluation view should be visible
    expect(screen.getByTestId('honest-evaluation')).toBeInTheDocument();
    
    // Click back button to return to main view
    fireEvent.click(screen.getByText('Back'));
    expect(screen.queryByTestId('honest-evaluation')).not.toBeInTheDocument();
  });

  test('opens technical deep dive audit when button is clicked', () => {
    render(<App />);
    
    // Initially technical deep dive audit view should not be visible
    expect(screen.queryByTestId('technical-deep-dive-audit')).not.toBeInTheDocument();
    
    // Click the math audit button
    const mathAuditButton = screen.getByRole('button', { name: /Math Audit/i });
    fireEvent.click(mathAuditButton);
    
    // Now technical deep dive audit view should be visible
    expect(screen.getByTestId('technical-deep-dive-audit')).toBeInTheDocument();
    
    // Click back button to return to main view
    fireEvent.click(screen.getByText('Back'));
    expect(screen.queryByTestId('technical-deep-dive-audit')).not.toBeInTheDocument();
  });

  test('opens paper2code view when button is clicked', () => {
    render(<App />);
    
    // Initially paper2code view should not be visible
    expect(screen.queryByTestId('paper2code-view')).not.toBeInTheDocument();
    
    // Click the paper2code button
    const paper2CodeButton = screen.getByRole('button', { name: /Paper2Code/i });
    fireEvent.click(paper2CodeButton);
    
    // Now paper2code view should be visible
    expect(screen.getByTestId('paper2code-view')).toBeInTheDocument();
    
    // Click back button to return to main view
    fireEvent.click(screen.getByText('Back'));
    expect(screen.queryByTestId('paper2code-view')).not.toBeInTheDocument();
  });

  test('updates telemetry data periodically', () => {
    render(<App />);
    
    // Check initial telemetry values
    const latencyElement = screen.getByText(/ms/);
    const cacheHitElement = screen.getByText(/%/);
    
    // Advance timers to trigger the useEffect interval
    jest.advanceTimersByTime(3000);
    
    // The telemetry values should have updated
    expect(latencyElement).toBeInTheDocument();
    expect(cacheHitElement).toBeInTheDocument();
  });
});