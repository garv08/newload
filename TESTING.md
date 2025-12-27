# Testing Setup for PHY Layer Agentic RAG

This document explains how to set up and run unit tests for the PHY Layer Agentic RAG application.

## Test Structure

The test suite includes:

1. **Component Tests**: Test individual React components
2. **Integration Tests**: Test how components work together
3. **Constants Tests**: Test data structures and utility functions
4. **Application Tests**: Test the main App component and its functionality

## Test Files

- `__tests__/App.test.tsx`: Tests for the main application component
- `__tests__/constants.test.ts`: Tests for constants and data structures
- `__tests__/MermaidDiagram.test.tsx`: Tests for the Mermaid diagram component
- `__tests__/setup.ts`: Test setup and configuration
- `vitest.config.ts`: Vitest configuration

## Required Dependencies

To run the tests, you need to install the following development dependencies:

```bash
npm install --save-dev vitest @vitest/ui @testing-library/react @testing-library/jest-dom jsdom
```

## Running Tests

Once dependencies are installed, you can run tests using:

```bash
# Run all tests in watch mode
npm test

# Run all tests once
npm run test:run

# Open Vitest UI
npm run test:ui
```

## Test Configuration

The testing setup uses:

- **Vitest**: Fast test runner
- **Testing Library**: React testing utilities
- **JSDOM**: Browser environment simulation
- **Mocking**: Components are mocked to isolate units under test

## Current Test Coverage

### App Component Tests
- Renders main application structure
- Toggles flow selection
- Switches between architecture and sequence views
- Opens and closes various views (white paper, honest evaluation, etc.)
- Updates telemetry data periodically

### Constants Tests
- Verifies NODE_DETAILS structure and properties
- Verifies FLOWS structure and properties
- Tests getDiagramDefinition function
- Tests SEQUENCE_DIAGRAM_DEFINITION

### Component Tests
- Tests MermaidDiagram rendering
- Tests component props handling
- Tests error handling

## Mocking Strategy

Components that are difficult to test (like those that interact with external libraries) are mocked to focus on the component's behavior rather than its dependencies.