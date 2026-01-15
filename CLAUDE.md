# Claude Development Guidelines

This document contains guidelines for AI assistants (Claude) working on this codebase.

## Project Vision

AgentState is a **Semantic State Engine** for AI agents. The key principle is:

> **One API to handle everything** - agents just express intent naturally, the engine figures out the rest.

### Core Concept

```rust
// Agents don't need to specify "store" vs "query" - just talk naturally
engine.process("My favorite color is blue");      // Auto-detected as STORE + MEMORY
engine.process("Remind me to call John");         // Auto-detected as STORE + TASK
engine.process("What is my favorite color?");     // Auto-detected as QUERY
engine.process("Who should I call?");             // Auto-detected as QUERY
```

### Response Format

The `AgentResponse` enum provides structured feedback:
- `Stored { id, data_type, content }` - Data was saved
- `QueryResult { results, count, data_type }` - Query returned matches
- `NotFound { query }` - No results found

Use `response.to_agent_string()` for simple text output.

## Testing Requirements

### Always Run Tests After Significant Changes

**CRITICAL**: After making any significant changes to the codebase, you MUST run the test suite to verify nothing is broken.

```bash
# Run all tests (unit + integration)
cargo test

# Run tests with output visible
cargo test -- --nocapture

# Run only unit tests (faster, no model download)
cargo test --lib

# Run only integration tests
cargo test --test storage_tests
```

### What Counts as "Significant Changes"

Run tests after:
- Adding new functions or methods
- Modifying existing logic
- Changing data structures
- Updating database schemas
- Refactoring code
- Fixing bugs
- Adding new dependencies

### Test Categories

1. **Unit Tests** (in `src/*.rs` files)
   - Fast, isolated tests
   - Located in `#[cfg(test)]` modules
   - Tests marked `#[ignore]` require model download

2. **Integration Tests** (in `tests/` directory)
   - `tests/storage_tests.rs` - Database operation tests
   - Test actual data insertion, retrieval, and integrity

### Before Committing

Always ensure:
1. `cargo check` passes (no compilation errors)
2. `cargo test --lib` passes (unit tests)
3. `cargo test --test storage_tests` passes (integration tests)
4. `cargo clippy` has no warnings (if available)

```bash
# Quick pre-commit check
cargo check && cargo test
```

## Code Quality

### Compilation Checks

```bash
# Check for compilation errors
cargo check

# Check with all warnings
cargo check 2>&1 | head -50
```

### Running the Application

```bash
# Build and run
cargo run

# Build release version
cargo build --release
```

## Architecture Overview

```
src/
├── brain.rs     # AI model loading, embeddings, intent classification
├── storage.rs   # SQLite database with vector storage
├── lib.rs       # Public API (AgentEngine, AgentResponse)
└── main.rs      # CLI demo/interactive mode

tests/
└── storage_tests.rs  # Comprehensive DB tests
```

### Key Components

- **Brain**:
  - Loads MiniLM model (384-dim embeddings)
  - Classifies **Action** (Store vs Query)
  - Classifies **DataType** (Task vs Memory)
  - Uses zero-shot classification via anchor vectors

- **Storage**:
  - SQLite with blob vector storage
  - Cosine similarity search
  - Category-filtered queries

- **AgentEngine**:
  - `process()` - The unified API (auto-routes by intent)
  - `store()` / `search()` - Explicit methods when needed
  - Returns `AgentResponse` for structured handling

### Intent Classification

The Brain uses anchor vectors to classify:
1. **Action**: Is this a store request or query request?
2. **DataType**: Is this a task (action item) or memory (fact)?

```
Input: "What is my name?"
  -> Action: QUERY (matches "what is, who is, find, search...")
  -> DataType: MEMORY (matches "fact, preference, information...")
```

## Common Tasks

### Adding a New Feature

1. Write the code
2. Add unit tests in the same file
3. Add integration tests if it affects storage
4. Run `cargo test`
5. Verify all tests pass before committing

### Fixing a Bug

1. Write a failing test that reproduces the bug
2. Fix the bug
3. Verify the test passes
4. Run full test suite
5. Commit with test and fix together

### Modifying Database Schema

1. Update schema in `Storage::new()`
2. Update affected queries
3. Run `cargo test --test storage_tests`
4. Verify all storage tests pass
5. Consider migration strategy for existing databases

## Notes on Tests

### Tests Requiring Model Download

Some tests in `src/brain.rs` and `src/lib.rs` are marked `#[ignore]` because they require downloading the MiniLM model from HuggingFace. To run these:

```bash
# Run ignored tests (will download ~90MB model on first run)
cargo test -- --ignored
```

### In-Memory Database Tests

Storage tests use `:memory:` SQLite databases for speed and isolation. Each test gets a fresh database.

## Reminders

- **ALWAYS** run `cargo test` after significant changes
- Check for warnings with `cargo check`
- Keep tests fast by using in-memory databases
- Write tests that verify actual data, not just "no errors"
