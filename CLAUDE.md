# Claude Development Guidelines

This document contains guidelines for AI assistants (Claude) working on this codebase.

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
├── brain.rs     # AI model loading and inference (MiniLM)
├── storage.rs   # SQLite database with vector storage
├── lib.rs       # Public API (AgentEngine)
└── main.rs      # CLI demo/test runner

tests/
└── storage_tests.rs  # Comprehensive DB tests
```

### Key Components

- **Brain**: Loads MiniLM model, generates 384-dim embeddings, classifies intent
- **Storage**: SQLite with blob vector storage, cosine similarity search
- **AgentEngine**: Combines Brain + Storage, provides clean API

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
