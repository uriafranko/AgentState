//! Storage Layer - Handles persistent storage with vector search capabilities
//!
//! This module manages the SQLite database for storing knowledge items (tasks and memories)
//! along with their embedding vectors for semantic search.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, Transaction};

/// Time-based filter for queries
///
/// Enables natural language time expressions like "from last week" or "yesterday"
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeFilter {
    /// All time (no filter)
    All,
    /// Last N hours
    LastHours(u32),
    /// Today only
    Today,
    /// Yesterday only
    Yesterday,
    /// Last 7 days
    LastWeek,
    /// Last 30 days
    LastMonth,
    /// Last 365 days
    LastYear,
    /// Custom: last N days
    LastDays(u32),
}

impl TimeFilter {
    /// Converts the time filter to a SQL WHERE clause fragment
    /// Returns (clause, params_needed)
    pub fn to_sql_clause(&self) -> (&'static str, Option<i64>) {
        match self {
            TimeFilter::All => ("1=1", None),
            TimeFilter::LastHours(h) => {
                // We'll use a placeholder and pass the value
                ("created_at >= datetime('now', '-' || ?1 || ' hours')", Some(*h as i64))
            }
            TimeFilter::Today => (
                "date(created_at) = date('now')",
                None,
            ),
            TimeFilter::Yesterday => (
                "date(created_at) = date('now', '-1 day')",
                None,
            ),
            TimeFilter::LastWeek => (
                "created_at >= datetime('now', '-7 days')",
                None,
            ),
            TimeFilter::LastMonth => (
                "created_at >= datetime('now', '-30 days')",
                None,
            ),
            TimeFilter::LastYear => (
                "created_at >= datetime('now', '-365 days')",
                None,
            ),
            TimeFilter::LastDays(d) => {
                ("created_at >= datetime('now', '-' || ?1 || ' days')", Some(*d as i64))
            }
        }
    }

    /// Parse from natural language hints
    /// Returns None if no time reference detected
    pub fn from_text(text: &str) -> Option<Self> {
        let lower = text.to_lowercase();

        if lower.contains("today") {
            Some(TimeFilter::Today)
        } else if lower.contains("yesterday") {
            Some(TimeFilter::Yesterday)
        } else if lower.contains("last week") || lower.contains("this week") {
            Some(TimeFilter::LastWeek)
        } else if lower.contains("last month") || lower.contains("this month") {
            Some(TimeFilter::LastMonth)
        } else if lower.contains("last year") || lower.contains("this year") {
            Some(TimeFilter::LastYear)
        } else if lower.contains("recent") || lower.contains("lately") {
            Some(TimeFilter::LastWeek) // Default "recent" to last week
        } else {
            None
        }
    }
}

/// Storage handles all database operations for the agent
pub struct Storage {
    conn: Connection,
}

/// A stored knowledge item with its metadata
#[derive(Debug, Clone)]
pub struct KnowledgeItem {
    pub id: i64,
    pub content: String,
    pub category: String,
    pub created_at: String,
}

impl Storage {
    /// Creates a new Storage instance, initializing the database schema
    ///
    /// # Arguments
    /// * `path` - Path to the SQLite database file (will be created if it doesn't exist)
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open database at: {}", path))?;

        // Enable WAL mode for better concurrent performance
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
            .context("Failed to set PRAGMA options")?;

        // Create the main knowledge table with extended categories
        conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('task', 'memory', 'preference', 'relationship', 'event')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        )
        .context("Failed to create knowledge table")?;

        // Create the vectors table for storing embeddings
        // We store vectors as BLOBs (raw float arrays) for efficiency
        conn.execute(
            "CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY,
                knowledge_id INTEGER NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
            )",
            [],
        )
        .context("Failed to create vectors table")?;

        // Create index for faster lookups
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category)",
            [],
        )
        .context("Failed to create category index")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_knowledge_id ON vectors(knowledge_id)",
            [],
        )
        .context("Failed to create vectors index")?;

        // Create entities table for knowledge graph support
        // Extracted entities like people, places, projects, etc.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (knowledge_id) REFERENCES knowledge(id) ON DELETE CASCADE
            )",
            [],
        )
        .context("Failed to create entities table")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_knowledge_id ON entities(knowledge_id)",
            [],
        )
        .context("Failed to create entities knowledge_id index")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)",
            [],
        )
        .context("Failed to create entities type index")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entities_value ON entities(entity_value)",
            [],
        )
        .context("Failed to create entities value index")?;

        // Create relationships table for linking entities
        // Enables queries like "What do I know about John?"
        conn.execute(
            "CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES knowledge(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES knowledge(id) ON DELETE CASCADE
            )",
            [],
        )
        .context("Failed to create relationships table")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)",
            [],
        )
        .context("Failed to create relationships source index")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)",
            [],
        )
        .context("Failed to create relationships target index")?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_type ON relationships(relation_type)",
            [],
        )
        .context("Failed to create relationships type index")?;

        Ok(Self { conn })
    }

    /// Saves a piece of knowledge along with its embedding vector
    ///
    /// # Arguments
    /// * `text` - The content to store
    /// * `category` - One of: "task", "memory", "preference", "relationship", "event"
    /// * `vector` - The embedding vector (dimension depends on model: 384 for small models, 768 for base models)
    pub fn save(&mut self, text: &str, category: &str, vector: &[f32]) -> Result<i64> {
        let tx = self.conn.transaction()?;
        let row_id = Self::save_in_transaction(&tx, text, category, vector)?;
        tx.commit()?;
        Ok(row_id)
    }

    /// Batch save multiple items in a single transaction (much faster for bulk inserts)
    ///
    /// # Arguments
    /// * `items` - Vec of (text, category, vector) tuples
    ///
    /// # Returns
    /// Vec of inserted IDs in the same order as input
    pub fn save_batch(&mut self, items: Vec<(&str, &str, &[f32])>) -> Result<Vec<i64>> {
        let tx = self.conn.transaction()?;
        let mut ids = Vec::with_capacity(items.len());

        for (text, category, vector) in items {
            let id = Self::save_in_transaction(&tx, text, category, vector)?;
            ids.push(id);
        }

        tx.commit()?;
        Ok(ids)
    }

    /// Internal method to save within a transaction
    fn save_in_transaction(
        tx: &Transaction,
        text: &str,
        category: &str,
        vector: &[f32],
    ) -> Result<i64> {
        // Insert the knowledge content
        tx.execute(
            "INSERT INTO knowledge (content, category) VALUES (?1, ?2)",
            params![text, category],
        )
        .context("Failed to insert knowledge")?;

        let row_id = tx.last_insert_rowid();

        // Convert vector to raw bytes for storage
        let vec_bytes: &[u8] = bytemuck::cast_slice(vector);

        // Insert the embedding vector
        tx.execute(
            "INSERT INTO vectors (knowledge_id, embedding) VALUES (?1, ?2)",
            params![row_id, vec_bytes],
        )
        .context("Failed to insert vector")?;

        Ok(row_id)
    }

    /// Searches for similar content using cosine similarity
    ///
    /// # Arguments
    /// * `query_vec` - The query embedding vector
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// A vector of content strings ordered by similarity (most similar first)
    pub fn search(&self, query_vec: &[f32], limit: usize) -> Result<Vec<String>> {
        let results = self.search_with_scores(query_vec, limit)?;
        Ok(results.into_iter().map(|(content, _)| content).collect())
    }

    /// Searches for similar content and returns similarity scores
    ///
    /// # Arguments
    /// * `query_vec` - The query embedding vector
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// A vector of (content, score) tuples ordered by similarity
    pub fn search_with_scores(&self, query_vec: &[f32], limit: usize) -> Result<Vec<(String, f32)>> {
        // Fetch all vectors and compute similarity in Rust
        // This is efficient for small to medium datasets
        // For larger datasets, consider using sqlite-vec extension
        let mut stmt = self.conn.prepare(
            "SELECT k.content, v.embedding
             FROM vectors v
             JOIN knowledge k ON k.id = v.knowledge_id"
        )?;

        let mut results: Vec<(String, f32)> = stmt
            .query_map([], |row| {
                let content: String = row.get(0)?;
                let embedding_blob: Vec<u8> = row.get(1)?;
                Ok((content, embedding_blob))
            })?
            .filter_map(|r| r.ok())
            .map(|(content, embedding_blob)| {
                let stored_vec: &[f32] = bytemuck::cast_slice(&embedding_blob);
                let similarity = cosine_similarity(query_vec, stored_vec);
                (content, similarity)
            })
            .collect();

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top results
        results.truncate(limit);
        Ok(results)
    }

    /// Searches within a specific category
    pub fn search_by_category(
        &self,
        query_vec: &[f32],
        category: &str,
        limit: usize,
    ) -> Result<Vec<String>> {
        self.search_filtered(query_vec, Some(category), TimeFilter::All, limit)
    }

    /// Searches with time filter
    pub fn search_with_time(
        &self,
        query_vec: &[f32],
        time_filter: TimeFilter,
        limit: usize,
    ) -> Result<Vec<String>> {
        self.search_filtered(query_vec, None, time_filter, limit)
    }

    /// Searches with both category and time filter
    pub fn search_filtered(
        &self,
        query_vec: &[f32],
        category: Option<&str>,
        time_filter: TimeFilter,
        limit: usize,
    ) -> Result<Vec<String>> {
        let results = self.search_filtered_with_scores(query_vec, category, time_filter, limit)?;
        Ok(results.into_iter().map(|(content, _)| content).collect())
    }

    /// Searches with filters and returns similarity scores
    pub fn search_filtered_with_scores(
        &self,
        query_vec: &[f32],
        category: Option<&str>,
        time_filter: TimeFilter,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let (time_clause, time_param) = time_filter.to_sql_clause();

        // Build query dynamically based on filters
        let query = if category.is_some() {
            format!(
                "SELECT k.content, v.embedding
                 FROM vectors v
                 JOIN knowledge k ON k.id = v.knowledge_id
                 WHERE k.category = ?1 AND {}",
                time_clause
            )
        } else {
            format!(
                "SELECT k.content, v.embedding
                 FROM vectors v
                 JOIN knowledge k ON k.id = v.knowledge_id
                 WHERE {}",
                time_clause
            )
        };

        let mut stmt = self.conn.prepare(&query)?;

        // Collect rows into a vec first to avoid closure type issues
        let rows: Vec<(String, Vec<u8>)> = if let Some(cat) = category {
            if let Some(tp) = time_param {
                stmt.query_map(params![cat, tp], |row| {
                    let content: String = row.get(0)?;
                    let embedding_blob: Vec<u8> = row.get(1)?;
                    Ok((content, embedding_blob))
                })?
                .filter_map(|r| r.ok())
                .collect()
            } else {
                stmt.query_map(params![cat], |row| {
                    let content: String = row.get(0)?;
                    let embedding_blob: Vec<u8> = row.get(1)?;
                    Ok((content, embedding_blob))
                })?
                .filter_map(|r| r.ok())
                .collect()
            }
        } else if let Some(tp) = time_param {
            stmt.query_map(params![tp], |row| {
                let content: String = row.get(0)?;
                let embedding_blob: Vec<u8> = row.get(1)?;
                Ok((content, embedding_blob))
            })?
            .filter_map(|r| r.ok())
            .collect()
        } else {
            stmt.query_map([], |row| {
                let content: String = row.get(0)?;
                let embedding_blob: Vec<u8> = row.get(1)?;
                Ok((content, embedding_blob))
            })?
            .filter_map(|r| r.ok())
            .collect()
        };

        let mut results: Vec<(String, f32)> = rows
            .into_iter()
            .map(|(content, embedding_blob)| {
                let stored_vec: &[f32] = bytemuck::cast_slice(&embedding_blob);
                let similarity = cosine_similarity(query_vec, stored_vec);
                (content, similarity)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results)
    }

    /// Retrieves all items of a specific category
    pub fn get_by_category(&self, category: &str) -> Result<Vec<KnowledgeItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, category, created_at FROM knowledge WHERE category = ?1 ORDER BY created_at DESC"
        )?;

        let items = stmt
            .query_map(params![category], |row| {
                Ok(KnowledgeItem {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    category: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(items)
    }

    /// Gets the total count of stored items
    pub fn count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM knowledge", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Gets the count of items by category
    pub fn count_by_category(&self, category: &str) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM knowledge WHERE category = ?1",
            params![category],
            |row| row.get(0),
        )?;
        Ok(count as usize)
    }

    /// Deletes an item by ID
    pub fn delete(&mut self, id: i64) -> Result<bool> {
        // Delete from vectors first (foreign key)
        self.conn
            .execute("DELETE FROM vectors WHERE knowledge_id = ?1", params![id])?;

        let affected = self
            .conn
            .execute("DELETE FROM knowledge WHERE id = ?1", params![id])?;

        Ok(affected > 0)
    }

    /// Clears all data from the database
    pub fn clear(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute("DELETE FROM relationships", [])?;
        tx.execute("DELETE FROM entities", [])?;
        tx.execute("DELETE FROM vectors", [])?;
        tx.execute("DELETE FROM knowledge", [])?;
        tx.commit()?;
        Ok(())
    }

    // =========================================================================
    // ENTITY METHODS - For knowledge graph support
    // =========================================================================

    /// Adds an entity associated with a knowledge item
    ///
    /// # Arguments
    /// * `knowledge_id` - The ID of the knowledge item
    /// * `entity_type` - Type of entity (e.g., "person", "project", "location")
    /// * `entity_value` - The entity value (e.g., "John", "ProjectX", "New York")
    pub fn add_entity(&mut self, knowledge_id: i64, entity_type: &str, entity_value: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO entities (knowledge_id, entity_type, entity_value) VALUES (?1, ?2, ?3)",
            params![knowledge_id, entity_type, entity_value],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Gets all entities for a knowledge item
    pub fn get_entities(&self, knowledge_id: i64) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT entity_type, entity_value FROM entities WHERE knowledge_id = ?1"
        )?;

        let entities = stmt
            .query_map(params![knowledge_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(entities)
    }

    /// Finds knowledge items by entity
    ///
    /// Example: Find all items mentioning "John"
    pub fn find_by_entity(&self, entity_value: &str) -> Result<Vec<KnowledgeItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT k.id, k.content, k.category, k.created_at
             FROM knowledge k
             JOIN entities e ON e.knowledge_id = k.id
             WHERE e.entity_value = ?1
             ORDER BY k.created_at DESC"
        )?;

        let items = stmt
            .query_map(params![entity_value], |row| {
                Ok(KnowledgeItem {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    category: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(items)
    }

    /// Finds knowledge items by entity type
    ///
    /// Example: Find all items mentioning any person
    pub fn find_by_entity_type(&self, entity_type: &str) -> Result<Vec<KnowledgeItem>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT k.id, k.content, k.category, k.created_at
             FROM knowledge k
             JOIN entities e ON e.knowledge_id = k.id
             WHERE e.entity_type = ?1
             ORDER BY k.created_at DESC"
        )?;

        let items = stmt
            .query_map(params![entity_type], |row| {
                Ok(KnowledgeItem {
                    id: row.get(0)?,
                    content: row.get(1)?,
                    category: row.get(2)?,
                    created_at: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(items)
    }

    // =========================================================================
    // RELATIONSHIP METHODS - For linking knowledge items
    // =========================================================================

    /// Creates a relationship between two knowledge items
    ///
    /// # Arguments
    /// * `source_id` - The source knowledge item ID
    /// * `target_id` - The target knowledge item ID
    /// * `relation_type` - Type of relationship (e.g., "related_to", "depends_on", "part_of")
    /// * `strength` - Relationship strength (0.0 to 1.0, default 1.0)
    pub fn add_relationship(
        &mut self,
        source_id: i64,
        target_id: i64,
        relation_type: &str,
        strength: f32,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO relationships (source_id, target_id, relation_type, strength) VALUES (?1, ?2, ?3, ?4)",
            params![source_id, target_id, relation_type, strength],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Gets all related knowledge items for a given item
    pub fn get_related(&self, knowledge_id: i64) -> Result<Vec<(KnowledgeItem, String, f32)>> {
        let mut stmt = self.conn.prepare(
            "SELECT k.id, k.content, k.category, k.created_at, r.relation_type, r.strength
             FROM knowledge k
             JOIN relationships r ON (r.target_id = k.id OR r.source_id = k.id)
             WHERE (r.source_id = ?1 OR r.target_id = ?1) AND k.id != ?1
             ORDER BY r.strength DESC"
        )?;

        let items = stmt
            .query_map(params![knowledge_id], |row| {
                Ok((
                    KnowledgeItem {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        category: row.get(2)?,
                        created_at: row.get(3)?,
                    },
                    row.get::<_, String>(4)?,
                    row.get::<_, f32>(5)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        Ok(items)
    }

    /// Auto-links related items based on semantic similarity
    ///
    /// Finds items with high semantic similarity and creates relationships
    pub fn auto_link(&mut self, knowledge_id: i64, query_vec: &[f32], threshold: f32) -> Result<usize> {
        let similar = self.search_with_scores(query_vec, 10)?;
        let mut linked_count = 0;

        for (content, score) in similar {
            if score >= threshold {
                // Get the ID of the similar item
                let similar_id: Option<i64> = self.conn.query_row(
                    "SELECT id FROM knowledge WHERE content = ?1 AND id != ?2",
                    params![content, knowledge_id],
                    |row| row.get(0),
                ).ok();

                if let Some(sid) = similar_id {
                    // Create bidirectional relationship
                    self.add_relationship(knowledge_id, sid, "related_to", score)?;
                    linked_count += 1;
                }
            }
        }

        Ok(linked_count)
    }
}

/// Computes cosine similarity between two vectors
/// Both vectors should already be normalized for this to equal dot product
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

    // If vectors are already normalized, dot product equals cosine similarity
    // Otherwise, we need to divide by the product of magnitudes
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a < 1e-10 || mag_b < 1e-10 {
        return 0.0;
    }

    dot_product / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_storage() -> Storage {
        Storage::new(":memory:").expect("Should create in-memory storage")
    }

    fn random_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect()
    }

    #[test]
    fn test_storage_creation() {
        let storage = create_test_storage();
        assert_eq!(storage.count().unwrap(), 0);
    }

    #[test]
    fn test_save_and_count() {
        let mut storage = create_test_storage();
        let vec = random_vector(384);

        storage.save("Test task", "task", &vec).unwrap();
        assert_eq!(storage.count().unwrap(), 1);
        assert_eq!(storage.count_by_category("task").unwrap(), 1);
        assert_eq!(storage.count_by_category("memory").unwrap(), 0);

        storage.save("Test memory", "memory", &vec).unwrap();
        assert_eq!(storage.count().unwrap(), 2);
        assert_eq!(storage.count_by_category("memory").unwrap(), 1);
    }

    #[test]
    fn test_search() {
        let mut storage = create_test_storage();
        let vec1: Vec<f32> = (0..384).map(|i| if i < 192 { 1.0 } else { 0.0 }).collect();
        let vec2: Vec<f32> = (0..384).map(|i| if i >= 192 { 1.0 } else { 0.0 }).collect();

        storage.save("First item", "task", &vec1).unwrap();
        storage.save("Second item", "memory", &vec2).unwrap();

        // Search with vec1 should return "First item" first
        let results = storage.search(&vec1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "First item");
    }

    #[test]
    fn test_delete() {
        let mut storage = create_test_storage();
        let vec = random_vector(384);

        let id = storage.save("To delete", "task", &vec).unwrap();
        assert_eq!(storage.count().unwrap(), 1);

        let deleted = storage.delete(id).unwrap();
        assert!(deleted);
        assert_eq!(storage.count().unwrap(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.001);
    }
}
