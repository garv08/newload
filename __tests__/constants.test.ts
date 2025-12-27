import { NODE_DETAILS, FLOWS, getDiagramDefinition, SEQUENCE_DIAGRAM_DEFINITION } from '../constants';

describe('Constants Module', () => {
  describe('NODE_DETAILS', () => {
    test('has expected node keys', () => {
      const expectedNodeKeys = ['U', 'CACHE', 'A', 'B', 'C', 'D', 'E1', 'E2', 'E3', 'SC', 'SK', 'RR', 'SB', 'AI', 'SE', 'QT', 'P2C', 'F', 'G', 'H', 'I'];
      
      expectedNodeKeys.forEach(key => {
        expect(NODE_DETAILS).toHaveProperty(key);
      });
    });

    test('each node has required properties', () => {
      Object.entries(NODE_DETAILS).forEach(([key, node]) => {
        expect(node).toHaveProperty('title');
        expect(node).toHaveProperty('description');
        expect(node).toHaveProperty('icon');
        expect(node).toHaveProperty('color');
        expect(node).toHaveProperty('inputs');
        expect(node).toHaveProperty('outputs');
        expect(node).toHaveProperty('specs');
        
        // Verify that inputs, outputs, and specs are arrays
        expect(Array.isArray(node.inputs)).toBe(true);
        expect(Array.isArray(node.outputs)).toBe(true);
        expect(Array.isArray(node.specs)).toBe(true);
      });
    });

    test('node properties have correct types', () => {
      const firstNode = NODE_DETAILS[Object.keys(NODE_DETAILS)[0]];
      
      expect(typeof firstNode.title).toBe('string');
      expect(typeof firstNode.description).toBe('string');
      expect(typeof firstNode.icon).toBe('string');
      expect(typeof firstNode.color).toBe('string');
      expect(Array.isArray(firstNode.inputs)).toBe(true);
      expect(Array.isArray(firstNode.outputs)).toBe(true);
      expect(Array.isArray(firstNode.specs)).toBe(true);
    });
  });

  describe('FLOWS', () => {
    test('has expected flow keys', () => {
      const expectedFlowKeys = ['ingest', 'vector_rag', 'graph_rag', 'paper2code', 'correction', 'output'];
      
      expectedFlowKeys.forEach(key => {
        expect(FLOWS).toHaveProperty(key);
      });
    });

    test('each flow has required properties', () => {
      Object.entries(FLOWS).forEach(([key, flow]) => {
        expect(flow).toHaveProperty('nodes');
        expect(flow).toHaveProperty('edges');
        
        // Verify that nodes and edges are arrays
        expect(Array.isArray(flow.nodes)).toBe(true);
        expect(Array.isArray(flow.edges)).toBe(true);
        
        // Verify that edges are tuples of strings
        flow.edges.forEach(edge => {
          expect(Array.isArray(edge)).toBe(true);
          expect(edge.length).toBe(2);
          expect(typeof edge[0]).toBe('string');
          expect(typeof edge[1]).toBe('string');
        });
      });
    });
  });

  describe('getDiagramDefinition function', () => {
    test('returns diagram definition with LR direction', () => {
      const diagram = getDiagramDefinition('LR');
      expect(diagram).toContain('flowchart LR');
      expect(diagram).toContain('U[');
      expect(diagram).toContain('SC[');
      expect(diagram).toContain('SK[');
    });

    test('returns diagram definition with TD direction', () => {
      const diagram = getDiagramDefinition('TD');
      expect(diagram).toContain('flowchart TD');
      expect(diagram).toContain('U[');
      expect(diagram).toContain('SC[');
      expect(diagram).toContain('SK[');
    });

    test('includes all expected nodes in diagram', () => {
      const diagram = getDiagramDefinition('LR');
      
      Object.keys(NODE_DETAILS).forEach(nodeKey => {
        // Check if the node is included in the diagram
        if (nodeKey !== 'F' && nodeKey !== 'G' && nodeKey !== 'H' && nodeKey !== 'I') {
          // Most nodes should be in the main diagram
          expect(diagram).toContain(`${nodeKey}[`);
        }
      });
      
      // Output nodes are in a subgraph
      expect(diagram).toContain('F[');
      expect(diagram).toContain('G[');
      expect(diagram).toContain('H[');
      expect(diagram).toContain('I[');
    });
  });

  describe('SEQUENCE_DIAGRAM_DEFINITION', () => {
    test('contains expected sequence diagram structure', () => {
      expect(SEQUENCE_DIAGRAM_DEFINITION).toContain('sequenceDiagram');
      expect(SEQUENCE_DIAGRAM_DEFINITION).toContain('participant U as Developer');
      expect(SEQUENCE_DIAGRAM_DEFINITION).toContain('participant C as Cache');
      expect(SEQUENCE_DIAGRAM_DEFINITION).toContain('participant SB as Orchestrator');
      expect(SEQUENCE_DIAGRAM_DEFINITION).toContain('U->>C: Submit Query');
      expect(SEQUENCE_DIAGRAM_DEFINITION).toContain('C-->>U: Cached Response');
    });
  });
});