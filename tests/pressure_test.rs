//! Pressure Test - Validates the system works for real AI agent usage
//!
//! Tests:
//! 1. Bulk insertions with varied free text
//! 2. Query accuracy and retrieval
//! 3. Edge cases (ambiguous, short, long inputs)
//! 4. Performance under load
//! 5. Federation with multiple shards

use agent_brain::{AgentEngine, AgentResponse, DataType, TimeFilter, FederatedEngine};
use std::time::Instant;

/// Realistic agent inputs - mix of stores and queries
const AGENT_INPUTS: &[&str] = &[
    // Tasks
    "Remind me to call mom tomorrow",
    "I need to buy groceries this weekend",
    "Schedule a dentist appointment for next month",
    "Don't forget to submit the report by Friday",
    "TODO: Review the pull request from John",
    "Need to renew my passport before the trip",
    "Pick up dry cleaning on Tuesday",
    "Book flight tickets for the conference",

    // Memories / Facts
    "My favorite programming language is Rust",
    "The API key for the production server is stored in vault",
    "John's birthday is on March 15th",
    "The company was founded in 2015",
    "Our main database runs on PostgreSQL",
    "The wifi password at the office is 'SecureNet2024'",
    "Alice's email is alice@example.com",
    "The project deadline was extended to December",

    // Preferences
    "I prefer dark mode in all applications",
    "My favorite coffee is a double espresso",
    "I like to start work early in the morning",
    "I prefer video calls over phone calls",
    "My favorite color is blue",
    "I enjoy working from home on Fridays",

    // Relationships
    "John is my team lead",
    "Alice works in the marketing department",
    "Bob is the CTO of the company",
    "Sarah is my mentor",
    "Mike is a colleague from the London office",
    "Lisa is the project manager for ProjectX",

    // Events
    "Team meeting every Monday at 10am",
    "Company all-hands on the first Friday of each month",
    "Performance review scheduled for next week",
    "Product launch event on November 15th",
    "Holiday party on December 20th",

    // Queries (should be detected as Query action)
    "What is my favorite color?",
    "Who is John?",
    "When is the next meeting?",
    "What tasks do I have pending?",
    "Tell me about Alice",
    "What's the wifi password?",
    "Find all my appointments",
    "What preferences do I have set?",
];

/// Edge cases that might be tricky
const EDGE_CASES: &[&str] = &[
    // Very short inputs
    "hi",
    "ok",
    "yes",
    "?",

    // Ambiguous inputs
    "maybe",
    "thing",
    "stuff",
    "whatever",

    // Long inputs
    "I need to remember that the quarterly financial report needs to be submitted to the board of directors by the end of next month, and it should include all the revenue projections from the sales team as well as the cost analysis from the operations department",

    // Unicode and special characters
    "Remember: café ☕ at 3pm",
    "Meeting with 田中さん tomorrow",
    "Budget is €5,000 for Q4",
    "Email: user+tag@example.com",

    // Numbers and dates
    "Phone: +1-555-123-4567",
    "Meeting ID: 123-456-789",
    "Temperature was 72°F today",

    // Code-like content
    "The error was: NullPointerException at line 42",
    "Run: npm install && npm test",
    "SELECT * FROM users WHERE id = 1",
];

#[test]
#[ignore] // Requires model download - run with: cargo test --test pressure_test -- --ignored
fn test_bulk_insertions() {
    println!("\n=== BULK INSERTION TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    let start = Instant::now();
    let mut stored_count = 0;
    let mut query_count = 0;
    let mut clarification_count = 0;
    let mut errors = Vec::new();

    // Process all agent inputs
    for (i, input) in AGENT_INPUTS.iter().enumerate() {
        match engine.process(input) {
            Ok(response) => {
                match &response {
                    AgentResponse::Stored { data_type, .. } => {
                        stored_count += 1;
                        println!("[{}] STORED as {:?}: {}", i, data_type, truncate(input, 50));
                    }
                    AgentResponse::QueryResult { count, .. } => {
                        query_count += 1;
                        println!("[{}] QUERY ({} results): {}", i, count, truncate(input, 50));
                    }
                    AgentResponse::NotFound { .. } => {
                        query_count += 1;
                        println!("[{}] QUERY (no results): {}", i, truncate(input, 50));
                    }
                    AgentResponse::NeedsClarification { message: _, detected_intent, .. } => {
                        clarification_count += 1;
                        println!("[{}] CLARIFICATION NEEDED: {} (conf: {:.1}%/{:.1}%)",
                            i, truncate(input, 30),
                            detected_intent.action_confidence * 100.0,
                            detected_intent.data_type_confidence * 100.0);
                    }
                }
            }
            Err(e) => {
                errors.push((i, input.to_string(), e.to_string()));
            }
        }
    }

    let elapsed = start.elapsed();

    println!("\n--- Results ---");
    println!("Total inputs: {}", AGENT_INPUTS.len());
    println!("Stored: {}", stored_count);
    println!("Queries: {}", query_count);
    println!("Clarifications: {}", clarification_count);
    println!("Errors: {}", errors.len());
    println!("Time: {:.2}s ({:.1}ms/input)", elapsed.as_secs_f64(), elapsed.as_millis() as f64 / AGENT_INPUTS.len() as f64);

    if !errors.is_empty() {
        println!("\nErrors:");
        for (i, input, err) in &errors {
            println!("  [{}] '{}': {}", i, truncate(input, 30), err);
        }
    }

    // Verify counts
    let total = engine.count().unwrap();
    println!("\nDatabase contains {} items", total);

    assert!(errors.is_empty(), "Should have no errors");
    assert!(stored_count > 0, "Should have stored some items");
}

#[test]
#[ignore]
fn test_query_accuracy() {
    println!("\n=== QUERY ACCURACY TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    // Store specific items
    let items = vec![
        ("My favorite color is blue", DataType::Preference),
        ("John's phone number is 555-1234", DataType::Memory),
        ("Meeting with Alice tomorrow at 3pm", DataType::Event),
        ("Remind me to buy milk", DataType::Task),
        ("Bob is my manager", DataType::Relationship),
    ];

    for (content, data_type) in &items {
        engine.store_as(content, *data_type).unwrap();
    }

    println!("Stored {} items\n", items.len());

    // Test queries
    let queries = vec![
        ("What is my favorite color?", "blue"),
        ("What is John's phone number?", "555-1234"),
        ("When is the meeting with Alice?", "tomorrow"),
        ("What do I need to buy?", "milk"),
        ("Who is my manager?", "Bob"),
    ];

    let mut success = 0;
    let mut failed = 0;

    for (query, expected_keyword) in &queries {
        match engine.search(query) {
            Ok(AgentResponse::QueryResult { results, .. }) => {
                let found = results.iter().any(|r| r.to_lowercase().contains(&expected_keyword.to_lowercase()));
                if found {
                    success += 1;
                    println!("✓ '{}' -> found '{}'", query, expected_keyword);
                } else {
                    failed += 1;
                    println!("✗ '{}' -> expected '{}', got: {:?}", query, expected_keyword, results);
                }
            }
            Ok(AgentResponse::NotFound { .. }) => {
                failed += 1;
                println!("✗ '{}' -> no results (expected '{}')", query, expected_keyword);
            }
            Ok(other) => {
                failed += 1;
                println!("✗ '{}' -> unexpected response: {:?}", query, other);
            }
            Err(e) => {
                failed += 1;
                println!("✗ '{}' -> error: {}", query, e);
            }
        }
    }

    println!("\n--- Results ---");
    println!("Success: {}/{}", success, queries.len());
    println!("Failed: {}", failed);

    assert!(success >= queries.len() / 2, "At least 50% of queries should succeed");
}

#[test]
#[ignore]
fn test_edge_cases() {
    println!("\n=== EDGE CASES TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    let mut results = Vec::new();

    for input in EDGE_CASES {
        let result = engine.process(input);
        let status = match &result {
            Ok(AgentResponse::Stored { data_type, .. }) => format!("STORED as {:?}", data_type),
            Ok(AgentResponse::QueryResult { count, .. }) => format!("QUERY ({} results)", count),
            Ok(AgentResponse::NotFound { .. }) => "NOT_FOUND".to_string(),
            Ok(AgentResponse::NeedsClarification { detected_intent, .. }) => {
                format!("CLARIFICATION (conf: {:.0}%/{:.0}%)",
                    detected_intent.action_confidence * 100.0,
                    detected_intent.data_type_confidence * 100.0)
            }
            Err(e) => format!("ERROR: {}", e),
        };

        println!("{:50} -> {}", truncate(input, 48), status);
        results.push((input, result));
    }

    // Count outcomes
    let errors: Vec<_> = results.iter().filter(|(_, r)| r.is_err()).collect();
    let clarifications: Vec<_> = results.iter()
        .filter(|(_, r)| matches!(r, Ok(AgentResponse::NeedsClarification { .. })))
        .collect();

    println!("\n--- Results ---");
    println!("Total edge cases: {}", EDGE_CASES.len());
    println!("Errors: {}", errors.len());
    println!("Clarifications needed: {}", clarifications.len());

    // Edge cases should not cause errors
    assert!(errors.is_empty(), "Edge cases should not cause errors");
}

#[test]
#[ignore]
fn test_batch_performance() {
    println!("\n=== BATCH PERFORMANCE TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    // Generate 100 items
    let items: Vec<&str> = (0..100)
        .map(|i| AGENT_INPUTS[i % AGENT_INPUTS.len()])
        .collect();

    // Test individual inserts
    let start = Instant::now();
    for item in &items[..50] {
        engine.store(item).unwrap();
    }
    let individual_time = start.elapsed();

    // Test batch insert
    let batch_items: Vec<&str> = items[50..].iter().copied().collect();
    let start = Instant::now();
    engine.store_batch(&batch_items).unwrap();
    let batch_time = start.elapsed();

    println!("Individual inserts (50 items): {:.2}s ({:.1}ms/item)",
        individual_time.as_secs_f64(),
        individual_time.as_millis() as f64 / 50.0);
    println!("Batch insert (50 items): {:.2}s ({:.1}ms/item)",
        batch_time.as_secs_f64(),
        batch_time.as_millis() as f64 / 50.0);
    println!("Speedup: {:.1}x", individual_time.as_secs_f64() / batch_time.as_secs_f64());

    // Verify all items stored
    assert_eq!(engine.count().unwrap(), 100);
}

#[test]
#[ignore]
fn test_time_filtering() {
    println!("\n=== TIME FILTERING TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    // Store some items
    engine.store("Item stored now").unwrap();
    engine.store("Another recent item").unwrap();

    // Test time-based queries
    let queries = vec![
        ("What did I save today?", TimeFilter::Today),
        ("Recent items", TimeFilter::LastWeek),
        ("Items from this month", TimeFilter::LastMonth),
    ];

    for (query, time_filter) in queries {
        match engine.search_with_time(query, time_filter, 10) {
            Ok(response) => {
                let count = match &response {
                    AgentResponse::QueryResult { count, .. } => *count,
                    _ => 0,
                };
                println!("{:?}: '{}' -> {} results", time_filter, query, count);
            }
            Err(e) => println!("{:?}: '{}' -> error: {}", time_filter, query, e),
        }
    }

    // Test smart_search auto-detection
    println!("\nSmart search (auto-detects time references):");
    let smart_queries = vec![
        "What did I save today?",
        "Show me recent items",
        "Items from last week",
    ];

    for query in smart_queries {
        match engine.smart_search(query, 10) {
            Ok(AgentResponse::QueryResult { count, .. }) => {
                println!("  '{}' -> {} results", query, count);
            }
            Ok(AgentResponse::NotFound { .. }) => {
                println!("  '{}' -> no results", query);
            }
            _ => {}
        }
    }
}

#[test]
#[ignore]
fn test_category_filtering() {
    println!("\n=== CATEGORY FILTERING TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    // Store items with explicit types
    engine.store_as("Call the dentist", DataType::Task).unwrap();
    engine.store_as("Buy groceries", DataType::Task).unwrap();
    engine.store_as("My favorite food is pizza", DataType::Preference).unwrap();
    engine.store_as("John is my colleague", DataType::Relationship).unwrap();
    engine.store_as("Meeting tomorrow at 3pm", DataType::Event).unwrap();
    engine.store_as("The password is secret123", DataType::Memory).unwrap();

    // Test category counts
    for dt in DataType::all() {
        let count = engine.count_by_type(*dt).unwrap();
        println!("{:?}: {} items", dt, count);
    }

    // Test category-filtered searches
    println!("\nFiltered searches:");
    let response = engine.search_tasks("what tasks", 10).unwrap();
    println!("Tasks search: {} results", match response { AgentResponse::QueryResult { count, .. } => count, _ => 0 });

    let response = engine.search_preferences("preferences", 10).unwrap();
    println!("Preferences search: {} results", match response { AgentResponse::QueryResult { count, .. } => count, _ => 0 });

    let response = engine.search_events("events", 10).unwrap();
    println!("Events search: {} results", match response { AgentResponse::QueryResult { count, .. } => count, _ => 0 });
}

#[test]
#[ignore]
fn test_federation_multi_agent() {
    println!("\n=== FEDERATION MULTI-AGENT TEST ===\n");

    let federation = FederatedEngine::new_mock_in_memory().expect("Failed to create federation");

    // Simulate multiple agents
    let agents = vec!["agent_alice", "agent_bob", "agent_charlie"];

    // Each agent stores different data
    for agent in &agents {
        federation.store(agent, &format!("My name is {}", agent.replace("agent_", ""))).unwrap();
        federation.store(agent, &format!("{}'s favorite color", agent)).unwrap();
    }

    println!("Created {} shards", federation.shard_count());

    // Each agent queries their own data
    for agent in &agents {
        match federation.search(agent, "What is my name?") {
            Ok(AgentResponse::QueryResult { results, .. }) => {
                println!("{}: found {} result(s)", agent, results.len());
                if let Some(first) = results.first() {
                    println!("  -> {}", truncate(first, 50));
                }
            }
            Ok(AgentResponse::NotFound { .. }) => {
                println!("{}: no results", agent);
            }
            _ => {}
        }
    }

    // Global search across all shards
    println!("\nGlobal search for 'favorite color':");
    match federation.global_search("favorite color", 5) {
        Ok(results) => {
            println!("Found results in {} shards", results.len());
            for (shard_id, response) in results {
                if let AgentResponse::QueryResult { count, .. } = response {
                    println!("  {}: {} result(s)", shard_id, count);
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

#[test]
#[ignore]
fn test_clarification_responses() {
    println!("\n=== CLARIFICATION RESPONSE TEST ===\n");

    let mut engine = AgentEngine::new_mock_in_memory().expect("Failed to create engine");

    // Store some data first
    engine.store("My name is Alice").unwrap();

    // Test various inputs to see clarification behavior
    let test_inputs = vec![
        "Hello",
        "Yes",
        "Maybe later",
        "Thing about stuff",
        "?",
        "ok",
    ];

    println!("Testing clarification triggers:\n");

    for input in test_inputs {
        match engine.process(input) {
            Ok(AgentResponse::NeedsClarification { message, detected_intent, .. }) => {
                println!("'{}' -> NEEDS CLARIFICATION", input);
                println!("  Action confidence: {:.1}%", detected_intent.action_confidence * 100.0);
                println!("  Type confidence: {:.1}%", detected_intent.data_type_confidence * 100.0);
                println!("  Message: {}", truncate(&message, 60));
            }
            Ok(AgentResponse::Stored { data_type, .. }) => {
                println!("'{}' -> Stored as {:?}", input, data_type);
            }
            Ok(AgentResponse::QueryResult { count, .. }) => {
                println!("'{}' -> Query ({} results)", input, count);
            }
            Ok(AgentResponse::NotFound { .. }) => {
                println!("'{}' -> Query (no results)", input);
            }
            Err(e) => {
                println!("'{}' -> Error: {}", input, e);
            }
        }
        println!();
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
