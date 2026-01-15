//! Federation Layer - Multi-database support for cloud scalability
//!
//! This module enables running multiple isolated databases (shards) that can be
//! addressed individually or queried collectively. Perfect for:
//! - Multi-tenant cloud deployments (each agent gets its own database)
//! - Horizontal scaling (distribute load across multiple databases)
//! - Data isolation (keep different projects/contexts separate)
//!
//! # Cloud-Ready Design
//!
//! The federation is stateless - each request includes the shard identifier.
//! No session state is maintained on the server, enabling easy horizontal scaling.

use crate::{AgentEngine, AgentResponse, DataType, TimeFilter};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Configuration for a federated engine
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Base directory for shard databases
    pub base_path: PathBuf,
    /// Maximum number of shards to keep open (LRU eviction)
    pub max_open_shards: usize,
    /// Whether to create shards on demand
    pub auto_create_shards: bool,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./shards"),
            max_open_shards: 100,
            auto_create_shards: true,
        }
    }
}

/// A federated engine that manages multiple database shards
///
/// Each shard is an independent AgentEngine with its own SQLite database.
/// Shards are identified by string IDs (typically agent IDs or tenant IDs).
///
/// # Example
/// ```no_run
/// use agent_brain::federation::{FederatedEngine, FederationConfig};
///
/// let config = FederationConfig::default();
/// let mut federation = FederatedEngine::new(config).unwrap();
///
/// // Each agent gets its own isolated database
/// federation.process("agent_001", "My favorite color is blue").unwrap();
/// federation.process("agent_002", "I prefer dark mode").unwrap();
///
/// // Queries are isolated to each agent's shard
/// let response = federation.process("agent_001", "What is my favorite color?").unwrap();
/// ```
pub struct FederatedEngine {
    config: FederationConfig,
    /// Map of shard_id -> AgentEngine (wrapped in RwLock for thread safety)
    shards: Arc<RwLock<HashMap<String, AgentEngine>>>,
    /// Order of shard access for LRU eviction
    access_order: Arc<RwLock<Vec<String>>>,
    /// Whether to use mock mode for shards (testing without ML model)
    mock_mode: bool,
}

impl FederatedEngine {
    /// Creates a new federated engine with the given configuration
    pub fn new(config: FederationConfig) -> Result<Self> {
        // Ensure base path exists
        if !config.base_path.exists() {
            std::fs::create_dir_all(&config.base_path)
                .with_context(|| format!("Failed to create shard directory: {:?}", config.base_path))?;
        }

        Ok(Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            mock_mode: false,
        })
    }

    /// Creates a federated engine with in-memory shards (for testing)
    pub fn new_in_memory() -> Result<Self> {
        let config = FederationConfig {
            base_path: PathBuf::from(":memory:"),
            max_open_shards: 100,
            auto_create_shards: true,
        };
        Ok(Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            mock_mode: false,
        })
    }

    /// Creates a federated engine in mock mode for testing
    ///
    /// Uses hash-based deterministic embeddings instead of the ML model.
    /// This allows testing federation without requiring model files.
    pub fn new_mock_in_memory() -> Result<Self> {
        let config = FederationConfig {
            base_path: PathBuf::from(":memory:"),
            max_open_shards: 100,
            auto_create_shards: true,
        };
        Ok(Self {
            config,
            shards: Arc::new(RwLock::new(HashMap::new())),
            access_order: Arc::new(RwLock::new(Vec::new())),
            mock_mode: true,
        })
    }

    /// Returns whether this federation is running in mock mode
    pub fn is_mock(&self) -> bool {
        self.mock_mode
    }

    /// Gets the database path for a shard
    fn shard_path(&self, shard_id: &str) -> PathBuf {
        if self.config.base_path.to_string_lossy() == ":memory:" {
            PathBuf::from(":memory:")
        } else {
            self.config.base_path.join(format!("{}.db", shard_id))
        }
    }

    /// Gets or creates a shard engine
    fn get_or_create_shard(&self, shard_id: &str) -> Result<()> {
        // Check if shard already exists
        {
            let shards = self.shards.read().unwrap();
            if shards.contains_key(shard_id) {
                // Update access order
                let mut order = self.access_order.write().unwrap();
                order.retain(|id| id != shard_id);
                order.push(shard_id.to_string());
                return Ok(());
            }
        }

        // Need to create or load the shard
        if !self.config.auto_create_shards {
            let path = self.shard_path(shard_id);
            if !path.exists() && path.to_string_lossy() != ":memory:" {
                anyhow::bail!("Shard '{}' does not exist and auto_create_shards is disabled", shard_id);
            }
        }

        // Evict oldest shard if at capacity
        self.maybe_evict_shard()?;

        // Create new shard (use mock mode if federation is in mock mode)
        let path = self.shard_path(shard_id);
        let engine = if self.mock_mode {
            AgentEngine::new_mock(path.to_string_lossy().as_ref())
                .with_context(|| format!("Failed to create mock shard: {}", shard_id))?
        } else {
            AgentEngine::new(path.to_string_lossy().as_ref())
                .with_context(|| format!("Failed to create shard: {}", shard_id))?
        };

        // Insert shard
        {
            let mut shards = self.shards.write().unwrap();
            shards.insert(shard_id.to_string(), engine);
        }

        // Update access order
        {
            let mut order = self.access_order.write().unwrap();
            order.push(shard_id.to_string());
        }

        Ok(())
    }

    /// Evicts the least recently used shard if at capacity
    fn maybe_evict_shard(&self) -> Result<()> {
        let shards_count = self.shards.read().unwrap().len();

        if shards_count >= self.config.max_open_shards {
            let oldest = {
                let order = self.access_order.read().unwrap();
                order.first().cloned()
            };

            if let Some(shard_id) = oldest {
                let mut shards = self.shards.write().unwrap();
                shards.remove(&shard_id);

                let mut order = self.access_order.write().unwrap();
                order.retain(|id| id != &shard_id);
            }
        }

        Ok(())
    }

    /// Process a request for a specific shard
    ///
    /// This is the main API for federated operation. Each request includes
    /// the shard ID, making the operation completely stateless.
    pub fn process(&self, shard_id: &str, input: &str) -> Result<AgentResponse> {
        self.get_or_create_shard(shard_id)?;

        let mut shards = self.shards.write().unwrap();
        let engine = shards.get_mut(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.process(input)
    }

    /// Store data in a specific shard
    pub fn store(&self, shard_id: &str, content: &str) -> Result<AgentResponse> {
        self.get_or_create_shard(shard_id)?;

        let mut shards = self.shards.write().unwrap();
        let engine = shards.get_mut(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.store(content)
    }

    /// Store with explicit type in a specific shard
    pub fn store_as(&self, shard_id: &str, content: &str, data_type: DataType) -> Result<AgentResponse> {
        self.get_or_create_shard(shard_id)?;

        let mut shards = self.shards.write().unwrap();
        let engine = shards.get_mut(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.store_as(content, data_type)
    }

    /// Search within a specific shard
    pub fn search(&self, shard_id: &str, query: &str) -> Result<AgentResponse> {
        self.get_or_create_shard(shard_id)?;

        let mut shards = self.shards.write().unwrap();
        let engine = shards.get_mut(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.search(query)
    }

    /// Search with time filter in a specific shard
    pub fn search_with_time(
        &self,
        shard_id: &str,
        query: &str,
        time_filter: TimeFilter,
        limit: usize,
    ) -> Result<AgentResponse> {
        self.get_or_create_shard(shard_id)?;

        let mut shards = self.shards.write().unwrap();
        let engine = shards.get_mut(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.search_with_time(query, time_filter, limit)
    }

    /// Batch store in a specific shard
    pub fn store_batch(&self, shard_id: &str, contents: &[&str]) -> Result<Vec<i64>> {
        self.get_or_create_shard(shard_id)?;

        let mut shards = self.shards.write().unwrap();
        let engine = shards.get_mut(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.store_batch(contents)
    }

    /// Search across all open shards (global search)
    ///
    /// Returns results from all shards, tagged with their shard IDs.
    /// Useful for cross-agent searches or global analytics.
    pub fn global_search(&self, query: &str, limit_per_shard: usize) -> Result<Vec<(String, AgentResponse)>> {
        let shard_ids: Vec<String> = {
            let shards = self.shards.read().unwrap();
            shards.keys().cloned().collect()
        };

        let mut results = Vec::new();

        for shard_id in shard_ids {
            self.get_or_create_shard(&shard_id)?;

            let search_result = {
                let mut shards = self.shards.write().unwrap();
                let engine = shards.get_mut(&shard_id)
                    .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;
                engine.search_with_limit(query, limit_per_shard)
            };

            match search_result {
                Ok(response) => results.push((shard_id, response)),
                Err(e) => {
                    // Log error but continue with other shards
                    eprintln!("Error searching shard {}: {}", shard_id, e);
                }
            }
        }

        Ok(results)
    }

    /// Get the number of open shards
    pub fn shard_count(&self) -> usize {
        self.shards.read().unwrap().len()
    }

    /// List all open shard IDs
    pub fn list_shards(&self) -> Vec<String> {
        self.shards.read().unwrap().keys().cloned().collect()
    }

    /// Check if a shard exists (either open or on disk)
    pub fn shard_exists(&self, shard_id: &str) -> bool {
        // Check if open
        if self.shards.read().unwrap().contains_key(shard_id) {
            return true;
        }

        // Check if on disk
        let path = self.shard_path(shard_id);
        path.exists()
    }

    /// Delete a shard and its data
    pub fn delete_shard(&self, shard_id: &str) -> Result<bool> {
        // Remove from memory
        {
            let mut shards = self.shards.write().unwrap();
            shards.remove(shard_id);
        }

        {
            let mut order = self.access_order.write().unwrap();
            order.retain(|id| id != shard_id);
        }

        // Remove from disk
        let path = self.shard_path(shard_id);
        if path.exists() && path.to_string_lossy() != ":memory:" {
            std::fs::remove_file(&path)?;
            return Ok(true);
        }

        Ok(false)
    }

    /// Get item count for a specific shard
    pub fn count(&self, shard_id: &str) -> Result<usize> {
        self.get_or_create_shard(shard_id)?;

        let shards = self.shards.read().unwrap();
        let engine = shards.get(shard_id)
            .ok_or_else(|| anyhow::anyhow!("Shard not found: {}", shard_id))?;

        engine.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_config_default() {
        let config = FederationConfig::default();
        assert_eq!(config.max_open_shards, 100);
        assert!(config.auto_create_shards);
    }

    // Note: Full tests require model download
    // Run with: cargo test -- --ignored
}
