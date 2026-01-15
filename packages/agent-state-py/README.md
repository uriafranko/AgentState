# AgentState Python SDK

Python SDK for AgentState - A semantic state engine for AI agents with a unified intent-based API.

**All AI processing (embeddings, intent classification) is powered by the Rust core.**

## Installation

```bash
pip install agent-state
```

## Quick Start

```python
from agent_state import AgentEngine, StoredResponse, QueryResultResponse

# Create an engine instance (downloads model on first run, ~90MB)
engine = AgentEngine()

# Store information - intent is auto-detected
response = engine.process("My favorite color is blue")      # Auto: STORE + MEMORY
response = engine.process("Remind me to call John at 5pm")  # Auto: STORE + TASK

# Query information - intent is auto-detected
response = engine.process("What is my favorite color?")
if isinstance(response, QueryResultResponse):
    for result in response:
        print(result)

# Use to_agent_string() for simple text output
print(response.to_agent_string())
```

## Features

- **Unified API**: One method (`process()`) handles both storage and retrieval
- **Intent Detection**: Automatically classifies whether input is a store or query operation
- **Data Type Classification**: Distinguishes between tasks, memories, preferences, relationships, and events
- **Semantic Search**: Find information based on meaning, not just keywords
- **Local Processing**: All AI runs locally via the Rust core - no external API calls
- **High Performance**: Rust-powered embeddings with LRU caching

## Development

### Building from Source

This package uses [maturin](https://github.com/PyO3/maturin) to build Python bindings for the Rust core.

```bash
# Install maturin
pip install maturin

# Build and install locally (from packages/agent-state-py/)
maturin develop

# Build release wheel
maturin build --release
```

### Testing

Tests use mock mode which doesn't require the ML model:

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=agent_state
```

## API Reference

### AgentEngine

The main class for interacting with AgentState.

```python
# Create with real model (downloads on first run)
engine = AgentEngine()
engine = AgentEngine(db_path="agent.db")  # Persistent storage

# Create with mock embeddings (for testing, no download needed)
engine = AgentEngine.mock()
engine = AgentEngine(mock=True)
```

#### Methods

| Method | Description |
|--------|-------------|
| `process(text)` | Process natural language, auto-detect intent |
| `store(text)` | Explicitly store data (auto-classifies type) |
| `store_as(text, data_type)` | Store with explicit data type |
| `store_batch(texts)` | Batch store multiple items efficiently |
| `search(query, limit=5)` | Search all stored data |
| `search_by_type(query, data_type, limit=5)` | Search filtered by type |
| `search_tasks(query, limit=5)` | Search only tasks |
| `search_memories(query, limit=5)` | Search only memories |
| `smart_search(query, limit=5)` | Auto-detect time filters in query |
| `classify(text)` | Classify intent without storing |
| `count()` | Get total item count |
| `delete(id)` | Delete item by ID |
| `clear()` | Clear all stored data |

### Response Types

```python
from agent_state import (
    StoredResponse,
    QueryResultResponse,
    NotFoundResponse,
    NeedsClarificationResponse,
)

# Check response type
if isinstance(response, StoredResponse):
    print(f"Stored with ID: {response.id}")
    print(f"Type: {response.data_type}")
    print(f"Content: {response.content}")
    print(f"Latency: {response.latency_ms}ms")

elif isinstance(response, QueryResultResponse):
    print(f"Found {response.count} results")
    for result in response:  # Iterable
        print(result)

elif isinstance(response, NotFoundResponse):
    print(f"No results for: {response.query}")

elif isinstance(response, NeedsClarificationResponse):
    print(f"Ambiguous: {response.message}")
```

### Data Types

```python
from agent_state import DataType

DataType.Task        # Action items, reminders, todos
DataType.Memory      # General facts and information
DataType.Preference  # User preferences and likes/dislikes
DataType.Relationship # Relationships between entities
DataType.Event       # Time-based events and appointments
```

### Time Filters

```python
from agent_state import TimeFilter

engine.search_with_time("meetings", TimeFilter.Today)
engine.search_with_time("tasks", TimeFilter.LastWeek)
engine.search_with_time("notes", TimeFilter.LastMonth)
```

### Intent Classification

```python
intent = engine.classify("What is my name?")
print(intent.action)           # Action.Query
print(intent.data_type)        # DataType.Memory
print(intent.action_confidence)
print(intent.data_type_confidence)
print(intent.is_ambiguous())   # True if low confidence
```

## License

MIT
