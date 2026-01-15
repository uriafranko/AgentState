# AgentState TypeScript SDK

TypeScript SDK for AgentState - A semantic state engine for AI agents with a unified intent-based API.

> **Note:** This is a placeholder implementation. Full functionality with native Rust bindings is coming soon.

## Installation

```bash
npm install @agent-state/sdk
# or
yarn add @agent-state/sdk
# or
pnpm add @agent-state/sdk
```

## Quick Start

```typescript
import { AgentEngine } from "@agent-state/sdk";

// Create an engine instance
const engine = new AgentEngine();

// Store information - intent is auto-detected
engine.process("My favorite color is blue"); // Stores as MEMORY
engine.process("Remind me to call John at 5pm"); // Stores as TASK

// Query information - intent is auto-detected
const result = engine.process("What is my favorite color?");
console.log(result); // Returns query result
```

## Features

- **Unified API**: One method (`process()`) handles both storage and retrieval
- **Intent Detection**: Automatically classifies whether input is a store or query operation
- **Data Type Classification**: Distinguishes between tasks and memories
- **TypeScript First**: Full type definitions included
- **Zero Dependencies**: Lightweight placeholder implementation

## API Reference

### AgentEngine

The main class for interacting with AgentState.

```typescript
const engine = new AgentEngine(options?: AgentEngineOptions);
```

#### Methods

- `process(text: string): AgentResponse` - Process natural language input
- `store(text: string, dataType?: DataType): StoredResponse` - Explicitly store data
- `search(query: string, dataType?: DataType, limit?: number): AgentResponse` - Explicitly search

### Types

```typescript
enum DataType {
  TASK = "task",
  MEMORY = "memory",
}

type AgentResponse = StoredResponse | QueryResultResponse | NotFoundResponse;

interface StoredResponse {
  type: "stored";
  id: number;
  dataType: DataType;
  content: string;
}

interface QueryResultResponse {
  type: "query_result";
  results: string[];
  count: number;
  dataType: DataType | null;
}

interface NotFoundResponse {
  type: "not_found";
  query: string;
}
```

### Helper Functions

- `toAgentString(response: AgentResponse): string` - Convert response to human-readable string

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run tests
npm test

# Type check
npm run typecheck
```

## License

MIT
