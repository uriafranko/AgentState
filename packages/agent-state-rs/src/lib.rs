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
pub mod schema;
mod storage;

pub use brain::{Action, Brain, DataType, Intent};
pub use federation::{FederatedEngine, FederationConfig};
pub use metrics::{Metrics, Operation, OperationStats};
pub use schema::{
    ExtractedData, Extractor, FieldDefinition, FieldType, Schema, SchemaRegistry,
    ValidationResult, contact_schema, event_schema, note_schema, task_schema,
};
pub use storage::{KnowledgeItem, Storage, StructuredDataItem, TimeFilter};

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

/// Response from storing structured data
#[derive(Debug, Clone)]
pub struct StructuredResponse {
    /// The ID of the stored item
    pub id: i64,
    /// The schema that was used
    pub schema_name: String,
    /// The extracted structured data
    pub extracted_data: ExtractedData,
    /// Validation result
    pub validation: ValidationResult,
    /// Latency of this operation in milliseconds
    pub latency_ms: f64,
}

impl StructuredResponse {
    /// Returns true if extraction and validation were successful
    pub fn is_valid(&self) -> bool {
        self.validation.is_valid
    }

    /// Returns true if all required fields were extracted
    pub fn is_complete(&self) -> bool {
        self.extracted_data.is_complete()
    }

    /// Gets an extracted field value
    pub fn get_field(&self, name: &str) -> Option<&String> {
        self.extracted_data.get(name)
    }

    /// Gets all extracted fields
    pub fn fields(&self) -> &std::collections::HashMap<String, String> {
        &self.extracted_data.fields
    }

    /// Returns the extraction confidence (0.0 to 1.0)
    pub fn confidence(&self) -> f32 {
        self.extracted_data.confidence
    }

    /// Returns missing required fields
    pub fn missing_fields(&self) -> &[String] {
        &self.extracted_data.missing_fields
    }

    /// Converts to agent-friendly string
    pub fn to_agent_string(&self) -> String {
        if self.is_valid() {
            let fields: Vec<String> = self.extracted_data.fields
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect();
            format!(
                "Stored {} (confidence: {:.0}%): {}",
                self.schema_name,
                self.confidence() * 100.0,
                fields.join(", ")
            )
        } else {
            let mut issues = Vec::new();
            if !self.extracted_data.missing_fields.is_empty() {
                issues.push(format!("Missing: {}", self.extracted_data.missing_fields.join(", ")));
            }
            for (field, error) in &self.validation.errors {
                issues.push(format!("{}: {}", field, error));
            }
            format!("Partial {} stored with issues: {}", self.schema_name, issues.join("; "))
        }
    }
}

/// Result of processing with schema detection
#[derive(Debug)]
pub enum ProcessedWithSchema {
    /// Structured data was extracted and stored
    Structured(StructuredResponse),
    /// Regular processing (no schema matched)
    Regular(AgentResponse),
}

impl ProcessedWithSchema {
    /// Returns the knowledge item ID if stored
    pub fn id(&self) -> Option<i64> {
        match self {
            ProcessedWithSchema::Structured(r) => Some(r.id),
            ProcessedWithSchema::Regular(AgentResponse::Stored { id, .. }) => Some(*id),
            _ => None,
        }
    }

    /// Returns whether this was structured data
    pub fn is_structured(&self) -> bool {
        matches!(self, ProcessedWithSchema::Structured(_))
    }

    /// Converts to agent-friendly string
    pub fn to_agent_string(&self) -> String {
        match self {
            ProcessedWithSchema::Structured(r) => r.to_agent_string(),
            ProcessedWithSchema::Regular(r) => r.to_agent_string(),
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
    schema_registry: SchemaRegistry,
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
        let metrics = Metrics::new();
        let schema_registry = SchemaRegistry::new();
        Ok(Self { brain, storage, metrics, schema_registry })
    }

    /// Creates an AgentEngine with an in-memory database (useful for testing)
    pub fn new_in_memory() -> Result<Self> {
        Self::new(":memory:")
    }

    /// Creates an AgentEngine with metrics disabled
    pub fn new_without_metrics(db_path: &str) -> Result<Self> {
        let brain = Brain::new().context("Failed to initialize Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;
        let metrics = Metrics::disabled();
        let schema_registry = SchemaRegistry::new();
        Ok(Self { brain, storage, metrics, schema_registry })
    }

    /// Creates an AgentEngine in mock mode for testing
    ///
    /// Uses hash-based deterministic embeddings instead of the ML model.
    /// This allows testing all functionality without requiring the actual model files.
    pub fn new_mock(db_path: &str) -> Result<Self> {
        let brain = Brain::new_mock().context("Failed to initialize mock Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;
        let metrics = Metrics::new();
        let schema_registry = SchemaRegistry::new();
        Ok(Self { brain, storage, metrics, schema_registry })
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
        let vector_tensor = self
            .brain
            .embed(input)
            .context("Failed to generate embedding")?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        // 2. Classify the full intent (action + data type)
        let classify_start = Instant::now();
        let intent = self
            .brain
            .classify(&vector_tensor)
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

        // 4. Convert tensor to vector for storage/search
        let vector_flat: Vec<f32> = vector_tensor
            .flatten_all()?
            .to_vec1()
            .context("Failed to flatten embedding")?;

        // 5. Route based on detected action
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
        let vector_tensor = self.brain.embed(content)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let classify_start = Instant::now();
        let data_type = self.brain.classify_data_type(&vector_tensor)?;
        self.metrics.record(Operation::Classify, classify_start.elapsed());

        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

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
        let vector_tensor = self.brain.embed(content)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

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
            let tensor = candle_core::Tensor::new(embedding.as_slice(), &candle_core::Device::Cpu)?
                .unsqueeze(0)?;
            let data_type = self.brain.classify_data_type(&tensor)?;
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
        let vector_tensor = self.brain.embed(query)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

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
        let vector_tensor = self.brain.embed(query)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

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
        let vector = self.brain.embed(text)?;
        self.brain.classify(&vector)
    }

    // =========================================================================
    // STRUCTURED DATA API - Schema-based extraction and validation
    // =========================================================================

    /// Registers a schema for structured data extraction
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::{AgentEngine, Schema, FieldType};
    /// # let mut engine = AgentEngine::new(":memory:").unwrap();
    /// let contact = Schema::new("contact")
    ///     .field("name", FieldType::String)
    ///     .field("email", FieldType::Email)
    ///     .optional_field("company", FieldType::String);
    ///
    /// engine.register_schema(contact).unwrap();
    /// ```
    pub fn register_schema(&mut self, schema: Schema) -> Result<()> {
        // Save to database for persistence
        let json = schema.to_json()?;
        self.storage.save_schema(&schema.name, &json)?;
        // Also keep in memory for fast access
        self.schema_registry.register(schema);
        Ok(())
    }

    /// Gets a registered schema by name
    pub fn get_schema(&self, name: &str) -> Option<&Schema> {
        self.schema_registry.get(name)
    }

    /// Lists all registered schema names
    pub fn list_schemas(&self) -> Vec<&str> {
        self.schema_registry.list()
    }

    /// Unregisters a schema
    pub fn unregister_schema(&mut self, name: &str) -> Result<bool> {
        self.schema_registry.unregister(name);
        self.storage.delete_schema(name)
    }

    /// Stores content with structured data extraction
    ///
    /// Extracts fields from natural language based on the specified schema,
    /// stores both the raw content and the structured data.
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::{AgentEngine, Schema, FieldType, contact_schema};
    /// # let mut engine = AgentEngine::new_mock(":memory:").unwrap();
    /// engine.register_schema(contact_schema()).unwrap();
    ///
    /// let response = engine.store_structured(
    ///     "Add John Smith, email john@example.com, works at Acme",
    ///     "contact"
    /// ).unwrap();
    /// ```
    pub fn store_structured(&mut self, content: &str, schema_name: &str) -> Result<StructuredResponse> {
        let total_start = Instant::now();

        // Get the schema
        let schema = self.schema_registry.get(schema_name)
            .ok_or_else(|| anyhow::anyhow!("Schema '{}' not found", schema_name))?
            .clone();

        // Extract structured data
        let extracted = Extractor::extract(content, &schema);

        // Validate the extraction
        let validation = Extractor::validate(&extracted, &schema);

        // Generate embedding and classify
        let embed_start = Instant::now();
        let vector_tensor = self.brain.embed(content)?;
        self.metrics.record(Operation::Embed, embed_start.elapsed());

        let classify_start = Instant::now();
        let data_type = self.brain.classify_data_type(&vector_tensor)?;
        self.metrics.record(Operation::Classify, classify_start.elapsed());

        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;

        // Store the raw content
        let db_start = Instant::now();
        let knowledge_id = self.storage.save(content, data_type.as_category(), &vector_flat)?;
        self.metrics.record(Operation::DbSave, db_start.elapsed());

        // Store the structured data
        let fields_json = extracted.fields_to_json()?;
        self.storage.save_structured_data(
            knowledge_id,
            schema_name,
            &fields_json,
            extracted.confidence,
        )?;

        let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        self.metrics.record(Operation::Store, total_start.elapsed());

        Ok(StructuredResponse {
            id: knowledge_id,
            schema_name: schema_name.to_string(),
            extracted_data: extracted,
            validation,
            latency_ms,
        })
    }

    /// Queries structured data by schema and optional field filters
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::{AgentEngine, contact_schema};
    /// # let mut engine = AgentEngine::new_mock(":memory:").unwrap();
    /// engine.register_schema(contact_schema()).unwrap();
    ///
    /// // Get all contacts
    /// let contacts = engine.query_structured("contact", None).unwrap();
    ///
    /// // Filter by field value
    /// let acme_contacts = engine.query_structured("contact", Some(("company", "Acme"))).unwrap();
    /// ```
    pub fn query_structured(
        &self,
        schema_name: &str,
        field_filter: Option<(&str, &str)>,
    ) -> Result<Vec<StructuredDataItem>> {
        if let Some((field, value)) = field_filter {
            self.storage.search_structured_by_field(schema_name, field, value)
        } else {
            self.storage.get_structured_by_schema(schema_name)
        }
    }

    /// Queries structured data with the original content
    pub fn query_structured_with_content(
        &self,
        schema_name: &str,
    ) -> Result<Vec<(StructuredDataItem, KnowledgeItem)>> {
        self.storage.get_structured_with_content(schema_name)
    }

    /// Gets structured data for a specific knowledge item
    pub fn get_structured(&self, knowledge_id: i64) -> Result<Option<StructuredDataItem>> {
        self.storage.get_structured_data(knowledge_id)
    }

    /// Counts structured data items by schema
    pub fn count_structured(&self, schema_name: &str) -> Result<usize> {
        self.storage.count_structured_by_schema(schema_name)
    }

    /// Processes natural language with automatic schema detection
    ///
    /// If the input matches a registered schema pattern, extracts structured data.
    /// Otherwise, falls back to regular processing.
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::{AgentEngine, contact_schema};
    /// # let mut engine = AgentEngine::new_mock(":memory:").unwrap();
    /// engine.register_schema(contact_schema()).unwrap();
    ///
    /// // This will auto-detect as a contact and extract fields
    /// let response = engine.process_with_schema_detection(
    ///     "Add contact: John at john@example.com"
    /// ).unwrap();
    /// ```
    pub fn process_with_schema_detection(&mut self, input: &str) -> Result<ProcessedWithSchema> {
        let input_lower = input.to_lowercase();

        // Try to detect which schema best matches the input
        let mut best_match: Option<(String, f32)> = None;

        for schema_name in self.schema_registry.list() {
            if let Some(schema) = self.schema_registry.get(schema_name) {
                // Try extraction and use confidence as match score
                let extracted = Extractor::extract(input, schema);
                if extracted.confidence > 0.5 {
                    if best_match.as_ref().map_or(true, |(_, c)| extracted.confidence > *c) {
                        best_match = Some((schema_name.to_string(), extracted.confidence));
                    }
                }
            }
        }

        // Also check for explicit schema hints in the input
        for schema_name in self.schema_registry.list() {
            if input_lower.contains(schema_name) {
                best_match = Some((schema_name.to_string(), 1.0));
                break;
            }
        }

        if let Some((schema_name, _)) = best_match {
            let structured_response = self.store_structured(input, &schema_name)?;
            Ok(ProcessedWithSchema::Structured(structured_response))
        } else {
            let response = self.process(input)?;
            Ok(ProcessedWithSchema::Regular(response))
        }
    }

    /// Loads schemas from the database (call on startup to restore persisted schemas)
    pub fn load_schemas_from_db(&mut self) -> Result<usize> {
        let schema_names = self.storage.list_schemas()?;
        let mut loaded = 0;

        for name in schema_names {
            if let Ok(Some(json)) = self.storage.get_schema(&name) {
                if let Ok(schema) = Schema::from_json(&json) {
                    self.schema_registry.register(schema);
                    loaded += 1;
                }
            }
        }

        Ok(loaded)
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
