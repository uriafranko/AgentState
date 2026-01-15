//! Comprehensive integration tests for the Storage layer
//!
//! These tests verify that data is correctly inserted, retrieved, and managed
//! in the SQLite database.

use agent_brain::Storage;

/// Helper to create a test vector with a specific pattern
fn create_test_vector(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i * seed + seed) % 100) as f32 / 100.0)
        .collect()
}

/// Helper to create an orthogonal vector (for testing similarity)
fn create_orthogonal_vectors(dim: usize) -> (Vec<f32>, Vec<f32>) {
    let mut vec1 = vec![0.0f32; dim];
    let mut vec2 = vec![0.0f32; dim];

    // First half is 1.0 for vec1, second half is 1.0 for vec2
    for i in 0..dim / 2 {
        vec1[i] = 1.0;
    }
    for i in dim / 2..dim {
        vec2[i] = 1.0;
    }

    (vec1, vec2)
}

// ============================================================================
// Database Creation Tests
// ============================================================================

#[test]
fn test_storage_creates_tables() {
    let storage = Storage::new(":memory:").expect("Should create storage");

    // Verify tables exist by checking count (would fail if tables don't exist)
    assert_eq!(storage.count().unwrap(), 0, "New database should be empty");
}

#[test]
fn test_storage_creates_file_database() {
    let temp_path = "/tmp/test_agent_storage.db";

    // Clean up any existing file
    let _ = std::fs::remove_file(temp_path);

    {
        let mut storage = Storage::new(temp_path).expect("Should create file storage");
        storage.save("Test content", "task", &create_test_vector(1, 384)).unwrap();
        assert_eq!(storage.count().unwrap(), 1);
    }

    // Verify file was created
    assert!(std::path::Path::new(temp_path).exists(), "Database file should exist");

    // Verify data persists
    {
        let storage = Storage::new(temp_path).expect("Should reopen storage");
        assert_eq!(storage.count().unwrap(), 1, "Data should persist across reopens");
    }

    // Clean up
    let _ = std::fs::remove_file(temp_path);
    let _ = std::fs::remove_file(format!("{}-wal", temp_path));
    let _ = std::fs::remove_file(format!("{}-shm", temp_path));
}

// ============================================================================
// Data Insertion Tests
// ============================================================================

#[test]
fn test_save_returns_correct_id() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let id1 = storage.save("First", "task", &vec).unwrap();
    let id2 = storage.save("Second", "task", &vec).unwrap();
    let id3 = storage.save("Third", "memory", &vec).unwrap();

    assert_eq!(id1, 1, "First ID should be 1");
    assert_eq!(id2, 2, "Second ID should be 2");
    assert_eq!(id3, 3, "Third ID should be 3");
}

#[test]
fn test_save_stores_correct_content() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("My task content", "task", &vec).unwrap();
    storage.save("My memory content", "memory", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    let memories = storage.get_by_category("memory").unwrap();

    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].content, "My task content");
    assert_eq!(tasks[0].category, "task");

    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].content, "My memory content");
    assert_eq!(memories[0].category, "memory");
}

#[test]
fn test_save_stores_correct_category() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("Task 1", "task", &vec).unwrap();
    storage.save("Task 2", "task", &vec).unwrap();
    storage.save("Memory 1", "memory", &vec).unwrap();

    assert_eq!(storage.count_by_category("task").unwrap(), 2);
    assert_eq!(storage.count_by_category("memory").unwrap(), 1);
    assert_eq!(storage.count().unwrap(), 3);
}

#[test]
fn test_save_preserves_special_characters() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let special_content = "Test with 'quotes', \"double quotes\", and émojis 🚀";
    storage.save(special_content, "task", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks[0].content, special_content);
}

#[test]
fn test_save_preserves_unicode() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let unicode_content = "中文测试 • Тест • مرحبا • 日本語";
    storage.save(unicode_content, "memory", &vec).unwrap();

    let memories = storage.get_by_category("memory").unwrap();
    assert_eq!(memories[0].content, unicode_content);
}

#[test]
fn test_save_handles_long_content() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let long_content = "x".repeat(10000);
    storage.save(&long_content, "task", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks[0].content.len(), 10000);
}

// ============================================================================
// Vector Storage Tests
// ============================================================================

#[test]
fn test_vector_stored_correctly() {
    let mut storage = Storage::new(":memory:").unwrap();

    // Create a distinctive vector
    let original_vec: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();

    storage.save("Test item", "task", &original_vec).unwrap();

    // Search with the same vector should return the item with high similarity
    let results = storage.search_with_scores(&original_vec, 1).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "Test item");
    // Similarity should be very high (close to 1.0) for identical vectors
    assert!(results[0].1 > 0.99, "Self-similarity should be ~1.0, got {}", results[0].1);
}

#[test]
fn test_different_vectors_have_different_similarity() {
    let mut storage = Storage::new(":memory:").unwrap();

    let (vec1, vec2) = create_orthogonal_vectors(384);

    storage.save("Item A", "task", &vec1).unwrap();
    storage.save("Item B", "memory", &vec2).unwrap();

    // Search with vec1 should rank Item A higher
    let results = storage.search_with_scores(&vec1, 2).unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, "Item A", "Item A should be most similar to vec1");
    assert!(results[0].1 > results[1].1, "Item A should have higher score than Item B");
}

// ============================================================================
// Search Tests
// ============================================================================

#[test]
fn test_search_returns_correct_order() {
    let mut storage = Storage::new(":memory:").unwrap();

    // Create vectors with varying similarity to query
    let query: Vec<f32> = (0..384).map(|i| if i < 192 { 1.0 } else { 0.0 }).collect();
    let similar: Vec<f32> = (0..384).map(|i| if i < 180 { 1.0 } else { 0.0 }).collect();
    let different: Vec<f32> = (0..384).map(|i| if i >= 192 { 1.0 } else { 0.0 }).collect();

    storage.save("Different item", "task", &different).unwrap();
    storage.save("Similar item", "task", &similar).unwrap();

    let results = storage.search(&query, 2).unwrap();

    assert_eq!(results[0], "Similar item", "Similar item should be first");
    assert_eq!(results[1], "Different item", "Different item should be second");
}

#[test]
fn test_search_respects_limit() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    for i in 0..10 {
        storage.save(&format!("Item {}", i), "task", &vec).unwrap();
    }

    let results_3 = storage.search(&vec, 3).unwrap();
    let results_5 = storage.search(&vec, 5).unwrap();
    let results_20 = storage.search(&vec, 20).unwrap();

    assert_eq!(results_3.len(), 3);
    assert_eq!(results_5.len(), 5);
    assert_eq!(results_20.len(), 10, "Should not exceed total items");
}

#[test]
fn test_search_empty_database() {
    let storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let results = storage.search(&vec, 5).unwrap();
    assert!(results.is_empty(), "Search on empty DB should return empty");
}

#[test]
fn test_search_by_category() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("Task 1", "task", &vec).unwrap();
    storage.save("Task 2", "task", &vec).unwrap();
    storage.save("Memory 1", "memory", &vec).unwrap();

    let task_results = storage.search_by_category(&vec, "task", 10).unwrap();
    let memory_results = storage.search_by_category(&vec, "memory", 10).unwrap();

    assert_eq!(task_results.len(), 2);
    assert_eq!(memory_results.len(), 1);
    assert!(task_results.iter().all(|r| r.starts_with("Task")));
    assert!(memory_results.iter().all(|r| r.starts_with("Memory")));
}

// ============================================================================
// Delete Tests
// ============================================================================

#[test]
fn test_delete_removes_item() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let id = storage.save("To be deleted", "task", &vec).unwrap();
    assert_eq!(storage.count().unwrap(), 1);

    let deleted = storage.delete(id).unwrap();
    assert!(deleted, "Delete should return true for existing item");
    assert_eq!(storage.count().unwrap(), 0, "Item should be removed");
}

#[test]
fn test_delete_removes_vector_too() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let id = storage.save("Item to delete", "task", &vec).unwrap();
    storage.delete(id).unwrap();

    // Search should return nothing
    let results = storage.search(&vec, 10).unwrap();
    assert!(results.is_empty(), "Vector should also be deleted");
}

#[test]
fn test_delete_nonexistent_returns_false() {
    let mut storage = Storage::new(":memory:").unwrap();

    let deleted = storage.delete(999).unwrap();
    assert!(!deleted, "Delete should return false for non-existent item");
}

#[test]
fn test_delete_only_affects_target() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let id1 = storage.save("Keep this", "task", &vec).unwrap();
    let id2 = storage.save("Delete this", "task", &vec).unwrap();
    let _id3 = storage.save("Keep this too", "memory", &vec).unwrap();

    storage.delete(id2).unwrap();

    assert_eq!(storage.count().unwrap(), 2);

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, id1);
}

// ============================================================================
// Clear Tests
// ============================================================================

#[test]
fn test_clear_removes_all_data() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("Task 1", "task", &vec).unwrap();
    storage.save("Task 2", "task", &vec).unwrap();
    storage.save("Memory 1", "memory", &vec).unwrap();

    assert_eq!(storage.count().unwrap(), 3);

    storage.clear().unwrap();

    assert_eq!(storage.count().unwrap(), 0);
    assert_eq!(storage.count_by_category("task").unwrap(), 0);
    assert_eq!(storage.count_by_category("memory").unwrap(), 0);
}

#[test]
fn test_clear_allows_new_inserts() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("Old item", "task", &vec).unwrap();
    storage.clear().unwrap();

    let new_id = storage.save("New item", "task", &vec).unwrap();
    assert!(new_id > 0, "Should be able to insert after clear");

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].content, "New item");
}

// ============================================================================
// Count Tests
// ============================================================================

#[test]
fn test_count_accuracy() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    assert_eq!(storage.count().unwrap(), 0);

    for i in 1..=5 {
        storage.save(&format!("Item {}", i), "task", &vec).unwrap();
        assert_eq!(storage.count().unwrap(), i);
    }
}

#[test]
fn test_count_by_category_accuracy() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("Task 1", "task", &vec).unwrap();
    storage.save("Task 2", "task", &vec).unwrap();
    storage.save("Task 3", "task", &vec).unwrap();
    storage.save("Memory 1", "memory", &vec).unwrap();
    storage.save("Memory 2", "memory", &vec).unwrap();

    assert_eq!(storage.count_by_category("task").unwrap(), 3);
    assert_eq!(storage.count_by_category("memory").unwrap(), 2);
    assert_eq!(storage.count_by_category("unknown").unwrap(), 0);
}

// ============================================================================
// Get By Category Tests
// ============================================================================

#[test]
fn test_get_by_category_returns_correct_items() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("Task A", "task", &vec).unwrap();
    storage.save("Memory B", "memory", &vec).unwrap();
    storage.save("Task C", "task", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks.len(), 2);
    assert!(tasks.iter().all(|t| t.category == "task"));

    let memories = storage.get_by_category("memory").unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0].category, "memory");
}

#[test]
fn test_get_by_category_includes_metadata() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    let id = storage.save("Test task", "task", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].id, id);
    assert_eq!(tasks[0].content, "Test task");
    assert_eq!(tasks[0].category, "task");
    assert!(!tasks[0].created_at.is_empty(), "Should have timestamp");
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_empty_content() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    // Empty string is technically valid
    storage.save("", "task", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].content, "");
}

#[test]
fn test_whitespace_content() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    storage.save("   ", "task", &vec).unwrap();
    storage.save("\n\t", "memory", &vec).unwrap();

    let tasks = storage.get_by_category("task").unwrap();
    let memories = storage.get_by_category("memory").unwrap();

    assert_eq!(tasks[0].content, "   ");
    assert_eq!(memories[0].content, "\n\t");
}

#[test]
fn test_concurrent_operations() {
    let mut storage = Storage::new(":memory:").unwrap();
    let vec = create_test_vector(1, 384);

    // Simulate multiple rapid operations
    for i in 0..100 {
        storage.save(&format!("Item {}", i), if i % 2 == 0 { "task" } else { "memory" }, &vec).unwrap();
    }

    assert_eq!(storage.count().unwrap(), 100);
    assert_eq!(storage.count_by_category("task").unwrap(), 50);
    assert_eq!(storage.count_by_category("memory").unwrap(), 50);
}
