//! AgentState - A Semantic State Engine for AI Agents
//!
//! AgentState is a high-performance semantic state engine built specifically for
//! autonomous AI agents. Unlike traditional databases that are passive storage,
//! AgentState is an active "nervous system" that automatically interprets natural
//! language intent and routes it to the correct storage format.
//!
//! # The Problem This Solves
//!
//! - **Amnesia**: Agents forget context between sessions
//! - **Coordination**: Multiple agents can't share state efficiently
//! - **Complexity**: Traditional DBs require schema setup and manual routing
//!
//! # The Solution
//!
//! One unified API where agents just express intent in natural language:
//!
//! ```no_run
//! use agent_brain::{AgentEngine, AgentResponse};
//!
//! let mut engine = AgentEngine::new("agent.db").unwrap();
//!
//! // Just talk to it - the engine figures out what to do
//! engine.process("Remember that John's favorite color is blue").unwrap();
//! engine.process("Remind me to call John tomorrow").unwrap();
//!
//! // Query naturally
//! let response = engine.process("What do I know about John?").unwrap();
//! if let AgentResponse::QueryResult { results, .. } = response {
//!     for item in results {
//!         println!("{}", item);
//!     }
//! }
//! ```

mod brain;
mod storage;

pub use brain::{Action, Brain, DataType, Intent};
pub use storage::{KnowledgeItem, Storage};

use anyhow::{Context, Result};

/// Response from the AgentEngine after processing a request
#[derive(Debug, Clone)]
pub enum AgentResponse {
    /// Data was stored successfully
    Stored {
        /// The ID of the stored item
        id: i64,
        /// What type of data was stored
        data_type: DataType,
        /// The original content that was stored
        content: String,
    },
    /// Query returned results
    QueryResult {
        /// The matching items (most relevant first)
        results: Vec<String>,
        /// The data type that was searched (if filtered)
        data_type: Option<DataType>,
        /// Number of results returned
        count: usize,
    },
    /// Query returned no results
    NotFound {
        /// The original query
        query: String,
    },
}

impl AgentResponse {
    /// Returns true if this is a successful store operation
    pub fn is_stored(&self) -> bool {
        matches!(self, AgentResponse::Stored { .. })
    }

    /// Returns true if this is a query with results
    pub fn has_results(&self) -> bool {
        matches!(self, AgentResponse::QueryResult { count, .. } if *count > 0)
    }

    /// Converts the response to a simple string for agent consumption
    pub fn to_agent_string(&self) -> String {
        match self {
            AgentResponse::Stored { data_type, content, .. } => {
                format!("Stored as {:?}: {}", data_type, content)
            }
            AgentResponse::QueryResult { results, count, .. } => {
                if *count == 0 {
                    "No results found.".to_string()
                } else {
                    results.join("\n")
                }
            }
            AgentResponse::NotFound { query } => {
                format!("No information found for: {}", query)
            }
        }
    }
}

/// The main Agent Engine - a semantic state manager for AI agents
///
/// This is the primary interface. Agents interact through a single `process()`
/// method that automatically understands intent and routes accordingly.
pub struct AgentEngine {
    brain: Brain,
    storage: Storage,
}

impl AgentEngine {
    /// Creates a new AgentEngine instance
    ///
    /// Downloads and caches the MiniLM model on first run (~90MB).
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file
    pub fn new(db_path: &str) -> Result<Self> {
        let brain = Brain::new().context("Failed to initialize Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;
        Ok(Self { brain, storage })
    }

    /// Creates an AgentEngine with an in-memory database (useful for testing)
    pub fn new_in_memory() -> Result<Self> {
        Self::new(":memory:")
    }

    // =========================================================================
    // THE UNIFIED API - This is the main interface for agents
    // =========================================================================

    /// Process natural language input and automatically route to store or query
    ///
    /// This is the **primary API** for agents. Just pass natural language and
    /// the engine will figure out what to do:
    ///
    /// - "Remember X" / "Note that X" / "My name is X" → Store
    /// - "What is X?" / "Who is X?" / "Find X" → Query
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::{AgentEngine, AgentResponse};
    /// # let mut engine = AgentEngine::new(":memory:").unwrap();
    /// // Store operations
    /// engine.process("My favorite food is pizza").unwrap();
    /// engine.process("Remind me to buy groceries").unwrap();
    ///
    /// // Query operations
    /// let response = engine.process("What is my favorite food?").unwrap();
    /// ```
    pub fn process(&mut self, input: &str) -> Result<AgentResponse> {
        // 1. Generate embedding for the input
        let vector_tensor = self
            .brain
            .embed(input)
            .context("Failed to generate embedding")?;

        // 2. Classify the full intent (action + data type)
        let intent = self
            .brain
            .classify(&vector_tensor)
            .context("Failed to classify intent")?;

        // 3. Convert tensor to vector for storage/search
        let vector_flat: Vec<f32> = vector_tensor
            .flatten_all()?
            .to_vec1()
            .context("Failed to flatten embedding")?;

        // 4. Route based on detected action
        match intent.action {
            Action::Store => {
                let category = match intent.data_type {
                    DataType::Task => "task",
                    DataType::Memory => "memory",
                };

                let id = self
                    .storage
                    .save(input, category, &vector_flat)
                    .context("Failed to save")?;

                Ok(AgentResponse::Stored {
                    id,
                    data_type: intent.data_type,
                    content: input.to_string(),
                })
            }
            Action::Query => {
                let results = self.storage.search(&vector_flat, 5)?;

                if results.is_empty() {
                    Ok(AgentResponse::NotFound {
                        query: input.to_string(),
                    })
                } else {
                    let count = results.len();
                    Ok(AgentResponse::QueryResult {
                        results,
                        data_type: None,
                        count,
                    })
                }
            }
        }
    }

    // =========================================================================
    // EXPLICIT STORE/QUERY - For when you know what you want
    // =========================================================================

    /// Explicitly store information (bypasses action detection)
    ///
    /// Use when you know you want to store, not query.
    pub fn store(&mut self, content: &str) -> Result<AgentResponse> {
        let vector_tensor = self.brain.embed(content)?;
        let data_type = self.brain.classify_data_type(&vector_tensor)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

        let category = match data_type {
            DataType::Task => "task",
            DataType::Memory => "memory",
        };

        let id = self.storage.save(content, category, &vector_flat)?;

        Ok(AgentResponse::Stored {
            id,
            data_type,
            content: content.to_string(),
        })
    }

    /// Explicitly store as a specific type
    pub fn store_as(&mut self, content: &str, data_type: DataType) -> Result<AgentResponse> {
        let vector_tensor = self.brain.embed(content)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

        let category = match data_type {
            DataType::Task => "task",
            DataType::Memory => "memory",
        };

        let id = self.storage.save(content, category, &vector_flat)?;

        Ok(AgentResponse::Stored {
            id,
            data_type,
            content: content.to_string(),
        })
    }

    /// Explicitly query (bypasses action detection)
    ///
    /// Use when you know you want to search, not store.
    pub fn search(&self, query: &str) -> Result<AgentResponse> {
        self.search_with_limit(query, 5)
    }

    /// Query with custom result limit
    pub fn search_with_limit(&self, query: &str, limit: usize) -> Result<AgentResponse> {
        let vector_tensor = self.brain.embed(query)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        let results = self.storage.search(&vector_flat, limit)?;

        if results.is_empty() {
            Ok(AgentResponse::NotFound {
                query: query.to_string(),
            })
        } else {
            let count = results.len();
            Ok(AgentResponse::QueryResult {
                results,
                data_type: None,
                count,
            })
        }
    }

    /// Query only tasks
    pub fn search_tasks(&self, query: &str, limit: usize) -> Result<AgentResponse> {
        let vector_tensor = self.brain.embed(query)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        let results = self.storage.search_by_category(&vector_flat, "task", limit)?;

        if results.is_empty() {
            Ok(AgentResponse::NotFound {
                query: query.to_string(),
            })
        } else {
            let count = results.len();
            Ok(AgentResponse::QueryResult {
                results,
                data_type: Some(DataType::Task),
                count,
            })
        }
    }

    /// Query only memories
    pub fn search_memories(&self, query: &str, limit: usize) -> Result<AgentResponse> {
        let vector_tensor = self.brain.embed(query)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        let results = self.storage.search_by_category(&vector_flat, "memory", limit)?;

        if results.is_empty() {
            Ok(AgentResponse::NotFound {
                query: query.to_string(),
            })
        } else {
            let count = results.len();
            Ok(AgentResponse::QueryResult {
                results,
                data_type: Some(DataType::Memory),
                count,
            })
        }
    }

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    /// Get all stored tasks
    pub fn get_tasks(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("task")
    }

    /// Get all stored memories
    pub fn get_memories(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("memory")
    }

    /// Get total count of stored items
    pub fn count(&self) -> Result<usize> {
        self.storage.count()
    }

    /// Get count by data type
    pub fn count_by_type(&self, data_type: DataType) -> Result<usize> {
        let category = match data_type {
            DataType::Task => "task",
            DataType::Memory => "memory",
        };
        self.storage.count_by_category(category)
    }

    /// Delete an item by ID
    pub fn delete(&mut self, id: i64) -> Result<bool> {
        self.storage.delete(id)
    }

    /// Clear all stored data
    pub fn clear(&mut self) -> Result<()> {
        self.storage.clear()
    }

    /// Classify intent without storing (useful for debugging/preview)
    pub fn classify(&self, text: &str) -> Result<Intent> {
        let vector = self.brain.embed(text)?;
        self.brain.classify(&vector)
    }

    // =========================================================================
    // BACKWARD COMPATIBILITY - Old API still works
    // =========================================================================

    /// Add information (old API - use `process()` or `store()` instead)
    #[deprecated(since = "0.2.0", note = "Use process() or store() instead")]
    pub fn add(&mut self, text: &str) -> Result<String> {
        let response = self.store(text)?;
        if let AgentResponse::Stored { data_type, .. } = response {
            Ok(format!(
                "processed_as_{}",
                match data_type {
                    DataType::Task => "task",
                    DataType::Memory => "memory",
                }
            ))
        } else {
            Ok("processed".to_string())
        }
    }

    /// Query information (old API - use `process()` or `search()` instead)
    #[deprecated(since = "0.2.0", note = "Use process() or search() instead")]
    pub fn query(&self, question: &str) -> Result<Vec<String>> {
        let response = self.search(question)?;
        match response {
            AgentResponse::QueryResult { results, .. } => Ok(results),
            AgentResponse::NotFound { .. } => Ok(vec![]),
            _ => Ok(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require the model to be downloaded
    // Run with: cargo test -- --ignored

    #[test]
    #[ignore]
    fn test_engine_creation() {
        let _engine = AgentEngine::new_in_memory().expect("Engine should initialize");
    }

    #[test]
    #[ignore]
    fn test_process_store() {
        let mut engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        let response = engine
            .process("My favorite color is blue")
            .expect("Process should succeed");

        assert!(response.is_stored());
        if let AgentResponse::Stored { data_type, .. } = response {
            assert_eq!(data_type, DataType::Memory);
        }
    }

    #[test]
    #[ignore]
    fn test_process_query() {
        let mut engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        // First store something
        engine.store("My name is Alice").unwrap();

        // Then query for it
        let response = engine.process("What is my name?").expect("Process should succeed");

        assert!(response.has_results());
    }

    #[test]
    #[ignore]
    fn test_unified_flow() {
        let mut engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        // Store via process
        engine.process("Remember that John likes pizza").unwrap();
        engine.process("Remind me to call John tomorrow").unwrap();

        // Query via process
        let response = engine.process("What do I know about John?").unwrap();

        match response {
            AgentResponse::QueryResult { results, count, .. } => {
                assert!(count > 0);
                assert!(!results.is_empty());
            }
            _ => panic!("Expected QueryResult"),
        }
    }
}
