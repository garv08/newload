import React from 'react';
import { render, screen } from '@testing-library/react';
import MermaidDiagram from '../components/MermaidDiagram';

// Mock the mermaid library
jest.mock('mermaid', () => ({
  initialize: jest.fn(),
  render: jest.fn(() => '<svg></svg>'),
}));

// Mock DOMPurify
jest.mock('dompurify', () => ({
  sanitize: (html: string) => html,
}));

describe('MermaidDiagram Component', () => {
  test('renders with default props', () => {
    render(<MermaidDiagram chart="graph TD; A-->B;" height="400px" onNodeClick={jest.fn()} activeFlow={null} flowColor="#3b82f6" />);
    
    // Check if the diagram container is present
    const diagramContainer = screen.getByRole('group');
    expect(diagramContainer).toBeInTheDocument();
    
    // Check if the loading state is rendered initially
    expect(screen.getByText(/OPTIMIZING TOPOLOGY ENGINE/i)).toBeInTheDocument();
  });

  test('applies custom height when provided', () => {
    render(<MermaidDiagram chart="graph TD; A-->B;" height="500px" onNodeClick={jest.fn()} activeFlow={null} flowColor="#3b82f6" />);
    
    const diagramContainer = screen.getByRole('group');
    expect(diagramContainer).toHaveStyle('height: 500px');
  });

  test('applies default height when not provided', () => {
    render(<MermaidDiagram chart="graph TD; A-->B;" onNodeClick={jest.fn()} activeFlow={null} flowColor="#3b82f6" />);
    
    const diagramContainer = screen.getByRole('group');
    expect(diagramContainer).toHaveStyle('height: 400px');
  });

  test('renders with active flow styling', () => {
    const activeFlow = {
      nodes: ['A', 'B'],
      edges: [['A', 'B']]
    };
    
    render(<MermaidDiagram chart="graph TD; A-->B;" activeFlow={activeFlow} flowColor="#ff0000" onNodeClick={jest.fn()} height="400px" />);
    
    const diagramContainer = screen.getByRole('group');
    expect(diagramContainer).toBeInTheDocument();
  });

  test('handles empty chart gracefully', () => {
    render(<MermaidDiagram chart="" onNodeClick={jest.fn()} activeFlow={null} flowColor="#3b82f6" height="400px" />);
    
    const diagramContainer = screen.getByRole('group');
    expect(diagramContainer).toBeInTheDocument();
  });
});