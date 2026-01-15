# AgentState Python SDK

Python SDK for AgentState - A semantic state engine for AI agents with a unified intent-based API.

## Installation

```bash
pip install agent-state
```

## Quick Start

```python
from agent_state import AgentEngine

# Create an engine instance
engine = AgentEngine()

# Store information - intent is auto-detected
engine.process("My favorite color is blue")      # Stores as MEMORY
engine.process("Remind me to call John at 5pm")  # Stores as TASK

# Query information - intent is auto-detected
result = engine.process("What is my favorite color?")
print(result)  # Returns: "blue"

result = engine.process("What tasks do I have?")
print(result)  # Returns task list
```

## Features

- **Unified API**: One method (`process()`) handles both storage and retrieval
- **Intent Detection**: Automatically classifies whether input is a store or query operation
- **Data Type Classification**: Distinguishes between tasks and memories
- **Semantic Search**: Find information based on meaning, not just keywords
- **Local Processing**: All AI runs locally - no external API calls needed

## Development

### Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
ruff check .
```

### Building from Source

This package uses [maturin](https://github.com/PyO3/maturin) to build Python bindings for the Rust core.

```bash
# Install maturin
pip install maturin

# Build and install locally
maturin develop

# Build wheel
maturin build --release
```

## API Reference

### AgentEngine

The main class for interacting with AgentState.

#### Methods

- `process(text: str) -> AgentResponse` - Process natural language input
- `store(text: str, data_type: DataType | None = None) -> AgentResponse` - Explicitly store data
- `search(query: str, data_type: DataType | None = None, limit: int = 5) -> AgentResponse` - Explicitly search

### AgentResponse

Response object returned from engine operations.

#### Variants

- `Stored` - Data was successfully stored
- `QueryResult` - Query returned results
- `NotFound` - No matching results found

## License

MIT
