//! Storage Layer - Handles persistent storage with vector search capabilities
//!
//! This module manages the SQLite database for storing knowledge items (tasks and memories)
//! along with their embedding vectors for semantic search.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, Transaction};

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

        // Create the main knowledge table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL CHECK(category IN ('task', 'memory')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        Ok(Self { conn })
    }

    /// Saves a piece of knowledge along with its embedding vector
    ///
    /// # Arguments
    /// * `text` - The content to store
    /// * `category` - Either "task" or "memory"
    /// * `vector` - The 384-dimensional embedding vector
    pub fn save(&mut self, text: &str, category: &str, vector: &[f32]) -> Result<i64> {
        let tx = self.conn.transaction()?;
        let row_id = Self::save_in_transaction(&tx, text, category, vector)?;
        tx.commit()?;
        Ok(row_id)
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
        let mut stmt = self.conn.prepare(
            "SELECT k.content, v.embedding
             FROM vectors v
             JOIN knowledge k ON k.id = v.knowledge_id
             WHERE k.category = ?1"
        )?;

        let mut results: Vec<(String, f32)> = stmt
            .query_map(params![category], |row| {
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

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        Ok(results.into_iter().map(|(content, _)| content).collect())
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
        tx.execute("DELETE FROM vectors", [])?;
        tx.execute("DELETE FROM knowledge", [])?;
        tx.commit()?;
        Ok(())
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
