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
pub mod federation;
pub mod metrics;
mod storage;

pub use brain::{Action, Backend, Brain, BrainConfig, DataType, EmbeddingModel, Intent};
pub use federation::{FederatedEngine, FederationConfig};
pub use metrics::{Metrics, Operation, OperationStats};
pub use storage::{KnowledgeItem, Storage, TimeFilter};

use anyhow::{Context, Result};
use std::time::Instant;

/// Response from the AgentEngine after processing a request
///
/// This enum is designed for cloud-ready stateless operation. Instead of maintaining
/// agent context on the server, we return clear messages telling the agent what we need.
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
        /// Latency of this operation in milliseconds
        latency_ms: f64,
    },
    /// Query returned results
    QueryResult {
        /// The matching items (most relevant first)
        results: Vec<String>,
        /// The data type that was searched (if filtered)
        data_type: Option<DataType>,
        /// Number of results returned
        count: usize,
        /// Latency of this operation in milliseconds
        latency_ms: f64,
    },
    /// Query returned no results
    NotFound {
        /// The original query
        query: String,
        /// Latency of this operation in milliseconds
        latency_ms: f64,
    },
    /// The engine needs more context to understand the request
    ///
    /// This is the key response for cloud-ready stateless operation.
    /// Instead of maintaining context, we return a message telling the agent
    /// exactly what information we need to proceed.
    NeedsClarification {
        /// The original input that was ambiguous
        original_input: String,
        /// Human-readable message explaining what clarification is needed
        message: String,
        /// The detected intent (even if ambiguous, for debugging)
        detected_intent: Intent,
        /// Latency of this operation in milliseconds
        latency_ms: f64,
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

    /// Returns true if this response requires clarification from the agent
    pub fn needs_clarification(&self) -> bool {
        matches!(self, AgentResponse::NeedsClarification { .. })
    }

    /// Get the latency of this operation in milliseconds
    pub fn latency_ms(&self) -> f64 {
        match self {
            AgentResponse::Stored { latency_ms, .. } => *latency_ms,
            AgentResponse::QueryResult { latency_ms, .. } => *latency_ms,
            AgentResponse::NotFound { latency_ms, .. } => *latency_ms,
            AgentResponse::NeedsClarification { latency_ms, .. } => *latency_ms,
        }
    }

    /// Converts the response to a simple string for agent consumption
    ///
    /// This is the primary interface for cloud/stateless operation.
    /// The returned string tells the agent exactly what happened or what is needed.
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
            AgentResponse::NotFound { query, .. } => {
                format!("No information found for: {}", query)
            }
            AgentResponse::NeedsClarification { message, .. } => {
                format!("I need more context: {}", message)
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
    metrics: Metrics,
}

impl AgentEngine {
    /// Creates a new AgentEngine instance with default configuration
    ///
    /// Uses BGE-Small model (best accuracy for 384 dimensions) with the best available backend.
    /// Downloads and caches the model on first run (~90MB).
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file
    pub fn new(db_path: &str) -> Result<Self> {
        Self::with_config(db_path, BrainConfig::default())
    }

    /// Creates a new AgentEngine with custom Brain configuration
    ///
    /// # Example
    /// ```no_run
    /// use agent_brain::{AgentEngine, BrainConfig, EmbeddingModel};
    ///
    /// // Use the most accurate model (768 dimensions)
    /// let config = BrainConfig::accurate();
    /// let engine = AgentEngine::with_config("agent.db", config)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_config(db_path: &str, config: BrainConfig) -> Result<Self> {
        let brain = Brain::with_config(config).context("Failed to initialize Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;
        let metrics = Metrics::new();
        Ok(Self { brain, storage, metrics })
    }

    /// Creates an AgentEngine with an in-memory database (useful for testing)
    pub fn new_in_memory() -> Result<Self> {
        Self::new(":memory:")
    }

    /// Creates an AgentEngine optimized for speed (MiniLM-L6)
    pub fn new_fast(db_path: &str) -> Result<Self> {
        Self::with_config(db_path, BrainConfig::fast())
    }

    /// Creates an AgentEngine optimized for accuracy (BGE-Base, 768 dims)
    pub fn new_accurate(db_path: &str) -> Result<Self> {
        Self::with_config(db_path, BrainConfig::accurate())
    }

    /// Creates an AgentEngine with metrics disabled
    pub fn new_without_metrics(db_path: &str) -> Result<Self> {
        let brain = Brain::new().context("Failed to initialize Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;
        let metrics = Metrics::disabled();
        Ok(Self { brain, storage, metrics })
    }

    /// Creates an AgentEngine in mock mode for testing
    ///
    /// Uses hash-based deterministic embeddings instead of the ML model.
    /// This allows testing all functionality without requiring the actual model files.
    pub fn new_mock(db_path: &str) -> Result<Self> {
        let brain = Brain::new_mock().context("Failed to initialize mock Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;
        let metrics = Metrics::new();
        Ok(Self { brain, storage, metrics })
    }

    /// Creates an in-memory AgentEngine in mock mode for testing
    ///
    /// Combines in-memory database with mock embeddings for fast, isolated testing.
    pub fn new_mock_in_memory() -> Result<Self> {
        Self::new_mock(":memory:")
    }

    /// Returns whether this engine is running in mock mode
    pub fn is_mock(&self) -> bool {
        self.brain.is_mock()
    }

    /// Returns the configured embedding model
    pub fn model(&self) -> EmbeddingModel {
        self.brain.model()
    }

    /// Returns the configured backend
    pub fn backend(&self) -> Backend {
        self.brain.backend()
    }

    /// Returns the embedding dimension for the configured model
    pub fn embedding_dim(&self) -> usize {
        self.brain.embedding_dim()
    }

    // =========================================================================
    // METRICS API
    // =========================================================================

    /// Get the metrics collector
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Get a summary of all collected metrics
    pub fn metrics_summary(&self) -> String {
        self.metrics.summary()
    }

    /// Reset all collected metrics
    pub fn reset_metrics(&self) {
        self.metrics.reset();
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
    /// # Cloud-Ready Stateless Design
    ///
    /// If the intent is ambiguous (low confidence), returns `NeedsClarification`
    /// with a message telling the agent what information we need. This enables
    /// stateless operation - no agent context is stored on the server.
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
    ///
    /// // Handle clarification requests
    /// if let AgentResponse::NeedsClarification { message, .. } = response {
    ///     println!("Agent should clarify: {}", message);
    /// }
    /// ```
    pub fn process(&mut self, input: &str) -> Result<AgentResponse> {
        let total_start = Instant::now();

        // 1. Generate embedding for the input
        let embed_start = Instant::now();
        let vector_flat = self
            .brain
            .embed_to_vec(input)
            .context("Failed to generate embedding")?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        // 2. Classify the full intent (action + data type)
        let classify_start = Instant::now();
        let intent = self
            .brain
            .classify_vec(&vector_flat)
            .context("Failed to classify intent")?;
        self.metrics.record(Operation::Classify, classify_start.elapsed());

        // 3. Check if intent is ambiguous - return clarification request
        // This is the key for cloud-ready stateless operation!
        if intent.is_ambiguous() {
            let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
            let message = intent.clarification_message().unwrap_or_else(|| {
                "Your request is ambiguous. Please be more specific about what you want to do.".to_string()
            });
            self.metrics.record(Operation::Process, total_start.elapsed());
            return Ok(AgentResponse::NeedsClarification {
                original_input: input.to_string(),
                message,
                detected_intent: intent,
                latency_ms,
            });
        }

        // 4. Route based on detected action
        let response = match intent.action {
            Action::Store => {
                let category = intent.data_type.as_category();

                let db_start = Instant::now();
                let id = self
                    .storage
                    .save(input, category, &vector_flat)
                    .context("Failed to save")?;
                self.metrics.record(Operation::DbSave, db_start.elapsed());

                let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
                AgentResponse::Stored {
                    id,
                    data_type: intent.data_type,
                    content: input.to_string(),
                    latency_ms,
                }
            }
            Action::Query => {
                let db_start = Instant::now();
                let results = self.storage.search(&vector_flat, 5)?;
                self.metrics.record(Operation::DbSearch, db_start.elapsed());

                let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
                if results.is_empty() {
                    AgentResponse::NotFound {
                        query: input.to_string(),
                        latency_ms,
                    }
                } else {
                    let count = results.len();
                    AgentResponse::QueryResult {
                        results,
                        data_type: None,
                        count,
                        latency_ms,
                    }
                }
            }
        };

        self.metrics.record(Operation::Process, total_start.elapsed());
        Ok(response)
    }

    /// Process with forced action (bypasses ambiguity check)
    ///
    /// Use when the agent explicitly specifies what it wants to do.
    /// This is useful after receiving a NeedsClarification response.
    pub fn process_as(&mut self, input: &str, action: Action) -> Result<AgentResponse> {
        match action {
            Action::Store => self.store(input),
            Action::Query => self.search(input),
        }
    }

    // =========================================================================
    // EXPLICIT STORE/QUERY - For when you know what you want
    // =========================================================================

    /// Explicitly store information (bypasses action detection)
    ///
    /// Use when you know you want to store, not query.
    pub fn store(&mut self, content: &str) -> Result<AgentResponse> {
        let total_start = Instant::now();

        let embed_start = Instant::now();
        let vector_flat = self.brain.embed_to_vec(content)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let classify_start = Instant::now();
        let data_type = self.brain.classify_data_type_vec(&vector_flat)?;
        self.metrics.record(Operation::Classify, classify_start.elapsed());

        let category = data_type.as_category();

        let db_start = Instant::now();
        let id = self.storage.save(content, category, &vector_flat)?;
        self.metrics.record(Operation::DbSave, db_start.elapsed());

        let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record(Operation::Store, total_start.elapsed());

        Ok(AgentResponse::Stored {
            id,
            data_type,
            content: content.to_string(),
            latency_ms,
        })
    }

    /// Explicitly store as a specific type
    pub fn store_as(&mut self, content: &str, data_type: DataType) -> Result<AgentResponse> {
        let total_start = Instant::now();

        let embed_start = Instant::now();
        let vector_flat = self.brain.embed_to_vec(content)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let category = data_type.as_category();

        let db_start = Instant::now();
        let id = self.storage.save(content, category, &vector_flat)?;
        self.metrics.record(Operation::DbSave, db_start.elapsed());

        let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record(Operation::Store, total_start.elapsed());

        Ok(AgentResponse::Stored {
            id,
            data_type,
            content: content.to_string(),
            latency_ms,
        })
    }

    /// Batch store multiple items efficiently (10-100x faster than individual stores)
    ///
    /// Items are auto-classified by data type. All items are stored in a single
    /// database transaction for maximum performance.
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::AgentEngine;
    /// # let mut engine = AgentEngine::new(":memory:").unwrap();
    /// let items = vec![
    ///     "My favorite color is blue",
    ///     "Remind me to call John",
    ///     "Meeting tomorrow at 3pm",
    /// ];
    /// let ids = engine.store_batch(&items).unwrap();
    /// println!("Stored {} items", ids.len());
    /// ```
    pub fn store_batch(&mut self, contents: &[&str]) -> Result<Vec<i64>> {
        let total_start = Instant::now();

        // Generate all embeddings
        let embed_start = Instant::now();
        let embeddings = self.brain.embed_batch(contents)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        // Classify and prepare batch items
        let mut items = Vec::with_capacity(contents.len());
        for (content, embedding) in contents.iter().zip(embeddings.iter()) {
            let data_type = self.brain.classify_data_type_vec(embedding)?;
            items.push((*content, data_type.as_category(), embedding.as_slice()));
        }

        // Batch save to database
        let db_start = Instant::now();
        let ids = self.storage.save_batch(items)?;
        self.metrics.record(Operation::DbSave, db_start.elapsed());

        self.metrics.record(Operation::Store, total_start.elapsed());
        Ok(ids)
    }

    /// Batch store with explicit data types (skips classification)
    pub fn store_batch_as(&mut self, items: &[(&str, DataType)]) -> Result<Vec<i64>> {
        let total_start = Instant::now();

        // Generate all embeddings
        let contents: Vec<&str> = items.iter().map(|(c, _)| *c).collect();
        let embed_start = Instant::now();
        let embeddings = self.brain.embed_batch(&contents)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        // Prepare batch items with specified types
        let batch_items: Vec<(&str, &str, &[f32])> = items
            .iter()
            .zip(embeddings.iter())
            .map(|((content, dt), emb)| (*content, dt.as_category(), emb.as_slice()))
            .collect();

        // Batch save to database
        let db_start = Instant::now();
        let ids = self.storage.save_batch(batch_items)?;
        self.metrics.record(Operation::DbSave, db_start.elapsed());

        self.metrics.record(Operation::Store, total_start.elapsed());
        Ok(ids)
    }

    /// Explicitly query (bypasses action detection)
    ///
    /// Use when you know you want to search, not store.
    pub fn search(&mut self, query: &str) -> Result<AgentResponse> {
        self.search_with_limit(query, 5)
    }

    /// Query with custom result limit
    pub fn search_with_limit(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        let total_start = Instant::now();

        let embed_start = Instant::now();
        let vector_flat = self.brain.embed_to_vec(query)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let db_start = Instant::now();
        let results = self.storage.search(&vector_flat, limit)?;
        self.metrics.record(Operation::DbSearch, db_start.elapsed());

        let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record(Operation::Search, total_start.elapsed());

        if results.is_empty() {
            Ok(AgentResponse::NotFound {
                query: query.to_string(),
                latency_ms,
            })
        } else {
            let count = results.len();
            Ok(AgentResponse::QueryResult {
                results,
                data_type: None,
                count,
                latency_ms,
            })
        }
    }

    /// Query only tasks
    pub fn search_tasks(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        self.search_by_type(query, DataType::Task, limit)
    }

    /// Query only memories
    pub fn search_memories(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        self.search_by_type(query, DataType::Memory, limit)
    }

    /// Query only preferences
    pub fn search_preferences(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        self.search_by_type(query, DataType::Preference, limit)
    }

    /// Query only relationships
    pub fn search_relationships(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        self.search_by_type(query, DataType::Relationship, limit)
    }

    /// Query only events
    pub fn search_events(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        self.search_by_type(query, DataType::Event, limit)
    }

    /// Query by specific data type (generic version)
    pub fn search_by_type(&mut self, query: &str, data_type: DataType, limit: usize) -> Result<AgentResponse> {
        self.search_filtered(query, Some(data_type), TimeFilter::All, limit)
    }

    /// Search with time filter (e.g., "last week", "today")
    ///
    /// Automatically detects time references in the query text.
    /// Example: "What did I save yesterday?" will filter to yesterday's items.
    pub fn search_with_time(&mut self, query: &str, time_filter: TimeFilter, limit: usize) -> Result<AgentResponse> {
        self.search_filtered(query, None, time_filter, limit)
    }

    /// Search with both type and time filters
    pub fn search_filtered(
        &mut self,
        query: &str,
        data_type: Option<DataType>,
        time_filter: TimeFilter,
        limit: usize,
    ) -> Result<AgentResponse> {
        let total_start = Instant::now();

        let embed_start = Instant::now();
        let vector_flat = self.brain.embed_to_vec(query)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let db_start = Instant::now();
        let category = data_type.map(|dt| dt.as_category());
        let results = self.storage.search_filtered(&vector_flat, category, time_filter, limit)?;
        self.metrics.record(Operation::DbSearch, db_start.elapsed());

        let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record(Operation::Search, total_start.elapsed());

        if results.is_empty() {
            Ok(AgentResponse::NotFound {
                query: query.to_string(),
                latency_ms,
            })
        } else {
            let count = results.len();
            Ok(AgentResponse::QueryResult {
                results,
                data_type,
                count,
                latency_ms,
            })
        }
    }

    /// Smart search that auto-detects time references in natural language
    ///
    /// Example queries:
    /// - "What tasks did I add today?" -> filters to today
    /// - "Show me recent memories" -> filters to last week
    /// - "What did I save last month?" -> filters to last 30 days
    pub fn smart_search(&mut self, query: &str, limit: usize) -> Result<AgentResponse> {
        let time_filter = TimeFilter::from_text(query).unwrap_or(TimeFilter::All);
        self.search_with_time(query, time_filter, limit)
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

    /// Get all stored preferences
    pub fn get_preferences(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("preference")
    }

    /// Get all stored relationships
    pub fn get_relationships(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("relationship")
    }

    /// Get all stored events
    pub fn get_events(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("event")
    }

    /// Get items by data type (generic version)
    pub fn get_by_type(&self, data_type: DataType) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category(data_type.as_category())
    }

    /// Get total count of stored items
    pub fn count(&self) -> Result<usize> {
        self.storage.count()
    }

    /// Get count by data type
    pub fn count_by_type(&self, data_type: DataType) -> Result<usize> {
        self.storage.count_by_category(data_type.as_category())
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
    pub fn classify(&mut self, text: &str) -> Result<Intent> {
        self.brain.classify_text(text)
    }

    // =========================================================================
    // BACKWARD COMPATIBILITY - Old API still works
    // =========================================================================

    /// Add information (old API - use `process()` or `store()` instead)
    #[deprecated(since = "0.2.0", note = "Use process() or store() instead")]
    pub fn add(&mut self, text: &str) -> Result<String> {
        let response = self.store(text)?;
        if let AgentResponse::Stored { data_type, .. } = response {
            Ok(format!("processed_as_{}", data_type.as_category()))
        } else {
            Ok("processed".to_string())
        }
    }

    /// Query information (old API - use `process()` or `search()` instead)
    #[deprecated(since = "0.2.0", note = "Use process() or search() instead")]
    pub fn query(&mut self, question: &str) -> Result<Vec<String>> {
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
        assert!(response.latency_ms() > 0.0);
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
        assert!(response.latency_ms() > 0.0);
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

    #[test]
    #[ignore]
    fn test_metrics_collection() {
        let mut engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        engine.process("My favorite color is blue").unwrap();
        engine.process("What is my favorite color?").unwrap();

        let stats = engine.metrics().get_all_stats();

        // Should have recorded embed, classify, and process operations
        assert!(stats.contains_key(&Operation::Embed));
        assert!(stats.contains_key(&Operation::Classify));
        assert!(stats.contains_key(&Operation::Process));

        // Embed should have been called twice
        let embed_stats = stats.get(&Operation::Embed).unwrap();
        assert_eq!(embed_stats.count, 2);
    }
}
