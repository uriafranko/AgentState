//! Agent Brain - A Rust-based AI agent core engine
//!
//! This library provides a "Smart" knowledge store that automatically classifies
//! and stores information using local AI embeddings and vector search.
//!
//! # Architecture
//!
//! - **Brain**: Handles AI model loading (MiniLM), embedding generation, and intent classification
//! - **Storage**: Manages SQLite database with vector search for semantic retrieval
//! - **AgentEngine**: The main API that combines Brain and Storage
//!
//! # Example
//!
//! ```no_run
//! use agent_brain::AgentEngine;
//!
//! let engine = AgentEngine::new("my_agent.db").unwrap();
//!
//! // Add information - automatically classified
//! engine.add("Remind me to buy milk").unwrap(); // -> Task
//! engine.add("My favorite color is blue").unwrap(); // -> Memory
//!
//! // Query semantically
//! let results = engine.query("What should I buy?").unwrap();
//! ```

mod brain;
mod storage;

pub use brain::{Brain, Intent};
pub use storage::{KnowledgeItem, Storage};

use anyhow::{Context, Result};

/// The main Agent Engine that combines AI and storage capabilities
///
/// This is the primary interface for interacting with the agent.
/// It handles automatic intent classification and semantic storage/retrieval.
pub struct AgentEngine {
    brain: Brain,
    storage: Storage,
}

/// Result of adding new information to the engine
#[derive(Debug, Clone)]
pub struct AddResult {
    /// The ID of the stored item
    pub id: i64,
    /// The classified intent
    pub intent: Intent,
    /// The category string ("task" or "memory")
    pub category: String,
}

impl AgentEngine {
    /// Creates a new AgentEngine instance
    ///
    /// This will:
    /// 1. Download and load the MiniLM model (cached after first run)
    /// 2. Initialize the SQLite database with vector search
    ///
    /// # Arguments
    /// * `db_path` - Path to the SQLite database file
    ///
    /// # Example
    /// ```no_run
    /// use agent_brain::AgentEngine;
    /// let engine = AgentEngine::new("my_agent.db").unwrap();
    /// ```
    pub fn new(db_path: &str) -> Result<Self> {
        let brain = Brain::new().context("Failed to initialize Brain")?;
        let storage = Storage::new(db_path).context("Failed to initialize Storage")?;

        Ok(Self { brain, storage })
    }

    /// Creates an AgentEngine with an in-memory database (useful for testing)
    pub fn new_in_memory() -> Result<Self> {
        Self::new(":memory:")
    }

    /// Adds new information to the agent
    ///
    /// The text is automatically:
    /// 1. Embedded into a vector representation
    /// 2. Classified as either a Task or Memory
    /// 3. Stored with its embedding for later retrieval
    ///
    /// # Arguments
    /// * `text` - The text to process and store
    ///
    /// # Returns
    /// A string indicating how the input was processed (e.g., "processed_as_task")
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::AgentEngine;
    /// # let engine = AgentEngine::new(":memory:").unwrap();
    /// let result = engine.add("Remind me to call mom").unwrap();
    /// assert_eq!(result, "processed_as_task");
    /// ```
    pub fn add(&mut self, text: &str) -> Result<String> {
        let result = self.add_with_details(text)?;
        Ok(format!("processed_as_{}", result.category))
    }

    /// Adds new information and returns detailed results
    ///
    /// Similar to `add()` but returns more information about the stored item.
    pub fn add_with_details(&mut self, text: &str) -> Result<AddResult> {
        // 1. Generate embedding
        let vector_tensor = self
            .brain
            .embed(text)
            .context("Failed to generate embedding")?;

        // 2. Classify intent
        let intent = self
            .brain
            .classify(&vector_tensor)
            .context("Failed to classify intent")?;

        let category = match intent {
            Intent::Task => "task",
            Intent::Memory => "memory",
        };

        // 3. Convert tensor to Vec<f32>
        let vector_flat: Vec<f32> = vector_tensor
            .flatten_all()?
            .to_vec1()
            .context("Failed to flatten embedding tensor")?;

        // 4. Store in database
        let id = self
            .storage
            .save(text, category, &vector_flat)
            .context("Failed to save to storage")?;

        Ok(AddResult {
            id,
            intent,
            category: category.to_string(),
        })
    }

    /// Forces adding as a specific category (bypasses classification)
    ///
    /// # Arguments
    /// * `text` - The text to store
    /// * `category` - Either "task" or "memory"
    pub fn add_as(&mut self, text: &str, category: &str) -> Result<i64> {
        let vector_tensor = self.brain.embed(text)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        self.storage.save(text, category, &vector_flat)
    }

    /// Queries the knowledge store for relevant information
    ///
    /// # Arguments
    /// * `question` - The query text
    ///
    /// # Returns
    /// Up to 5 most relevant items, ordered by semantic similarity
    ///
    /// # Example
    /// ```no_run
    /// # use agent_brain::AgentEngine;
    /// # let mut engine = AgentEngine::new(":memory:").unwrap();
    /// # engine.add("Buy groceries tomorrow").unwrap();
    /// let results = engine.query("What should I buy?").unwrap();
    /// ```
    pub fn query(&self, question: &str) -> Result<Vec<String>> {
        self.query_with_limit(question, 5)
    }

    /// Queries with a custom limit
    pub fn query_with_limit(&self, question: &str, limit: usize) -> Result<Vec<String>> {
        let vector_tensor = self.brain.embed(question)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        self.storage.search(&vector_flat, limit)
    }

    /// Queries and returns results with similarity scores
    pub fn query_with_scores(&self, question: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        let vector_tensor = self.brain.embed(question)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        self.storage.search_with_scores(&vector_flat, limit)
    }

    /// Queries only within a specific category
    ///
    /// # Arguments
    /// * `question` - The query text
    /// * `category` - Either "task" or "memory"
    /// * `limit` - Maximum number of results
    pub fn query_category(
        &self,
        question: &str,
        category: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        let vector_tensor = self.brain.embed(question)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        self.storage.search_by_category(&vector_flat, category, limit)
    }

    /// Gets all tasks
    pub fn get_tasks(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("task")
    }

    /// Gets all memories
    pub fn get_memories(&self) -> Result<Vec<KnowledgeItem>> {
        self.storage.get_by_category("memory")
    }

    /// Gets the total count of stored items
    pub fn count(&self) -> Result<usize> {
        self.storage.count()
    }

    /// Gets count by category
    pub fn count_by_category(&self, category: &str) -> Result<usize> {
        self.storage.count_by_category(category)
    }

    /// Deletes an item by ID
    pub fn delete(&mut self, id: i64) -> Result<bool> {
        self.storage.delete(id)
    }

    /// Clears all stored data
    pub fn clear(&mut self) -> Result<()> {
        self.storage.clear()
    }

    /// Classifies text without storing it
    ///
    /// Useful for previewing how text would be categorized.
    pub fn classify(&self, text: &str) -> Result<Intent> {
        let vector = self.brain.embed(text)?;
        self.brain.classify(&vector)
    }

    /// Generates an embedding for text without storing it
    ///
    /// Returns the raw 384-dimensional vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let vector_tensor = self.brain.embed(text)?;
        let vector_flat: Vec<f32> = vector_tensor.flatten_all()?.to_vec1()?;
        Ok(vector_flat)
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
    fn test_add_and_query() {
        let mut engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        engine.add("Buy milk from the store").unwrap();
        engine.add("My favorite color is blue").unwrap();

        let results = engine.query("What should I purchase?").unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    #[ignore]
    fn test_task_classification() {
        let engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        let intent = engine.classify("Remind me to call mom tomorrow").unwrap();
        assert_eq!(intent, Intent::Task);
    }

    #[test]
    #[ignore]
    fn test_memory_classification() {
        let engine = AgentEngine::new_in_memory().expect("Engine should initialize");

        let intent = engine.classify("I was born in New York").unwrap();
        assert_eq!(intent, Intent::Memory);
    }
}
