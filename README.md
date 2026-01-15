# AgentState

A **Semantic State Engine** for AI agents with a unified intent-based API.

> **One API to handle everything** - agents just express intent naturally, the engine figures out the rest.

## Packages

This monorepo contains the following packages:

| Package | Language | Description | Status |
|---------|----------|-------------|--------|
| [agent-state-rs](./packages/agent-state-rs/) | Rust | Core semantic state engine | Stable |
| [agent-state-py](./packages/agent-state-py/) | Python | Python SDK with native bindings | Alpha |
| [agent-state-ts](./packages/agent-state-ts/) | TypeScript | TypeScript/JavaScript SDK | Placeholder |

## Quick Start

### Rust

```rust
use agent_brain::AgentEngine;

let engine = AgentEngine::new()?;

// Store information - intent is auto-detected
engine.process("My favorite color is blue");      // Auto: STORE + MEMORY
engine.process("Remind me to call John");         // Auto: STORE + TASK

// Query information - intent is auto-detected
engine.process("What is my favorite color?");     // Auto: QUERY
engine.process("Who should I call?");             // Auto: QUERY
```

### Python

```python
from agent_state import AgentEngine

engine = AgentEngine()

# Store and query with auto-detection
engine.process("My favorite color is blue")
result = engine.process("What is my favorite color?")
```

### TypeScript

```typescript
import { AgentEngine } from "@agent-state/sdk";

const engine = new AgentEngine();

engine.process("My favorite color is blue");
const result = engine.process("What is my favorite color?");
```

## Core Concepts

### Intent Classification

The engine automatically classifies:

1. **Action**: Is this a store request or query request?
2. **DataType**: Is this a task (action item) or memory (fact)?

### Response Format

```rust
enum AgentResponse {
    Stored { id, data_type, content },      // Data was saved
    QueryResult { results, count, data_type }, // Query returned matches
    NotFound { query },                      // No results found
}
```

## Development

### Prerequisites

- Rust 1.70+
- Python 3.9+ (for Python SDK)
- Node.js 18+ (for TypeScript SDK)

### Building

```bash
# Rust core
cd packages/agent-state-rs
cargo build --release

# Python SDK
cd packages/agent-state-py
pip install maturin
maturin develop

# TypeScript SDK
cd packages/agent-state-ts
npm install
npm run build
```

### Testing

```bash
# Rust
cd packages/agent-state-rs
cargo test

# Python
cd packages/agent-state-py
pytest

# TypeScript
cd packages/agent-state-ts
npm test
```

## Architecture

```
packages/
├── agent-state-rs/     # Rust core engine
│   ├── src/
│   │   ├── lib.rs      # Public API
│   │   ├── brain.rs    # AI model & classification
│   │   ├── storage.rs  # SQLite vector storage
│   │   └── ...
│   └── tests/
├── agent-state-py/     # Python SDK
│   ├── python/         # Pure Python code
│   └── src/            # Rust bindings (PyO3)
└── agent-state-ts/     # TypeScript SDK
    └── src/            # TypeScript implementation
```

## License

MIT
