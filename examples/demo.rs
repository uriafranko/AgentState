//! AgentState Demo Script
//!
//! This script demonstrates how the Semantic State Engine works with
//! different types of data: memories, tasks, and reminders.
//!
//! Run with: cargo run --example demo

use agent_brain::{AgentEngine, AgentResponse, DataType};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║        AgentState - Semantic State Engine Demo            ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Initialize engine with in-memory database for demo
    print!("Loading AI model... ");
    let mut engine = match AgentEngine::new(":memory:") {
        Ok(e) => {
            println!("OK");
            e
        }
        Err(e) => {
            eprintln!("Failed: {}", e);
            std::process::exit(1);
        }
    };
    println!();

    // =========================================================================
    // PART 1: Storing Memories (facts, preferences, personal info)
    // =========================================================================
    section_header("PART 1: STORING MEMORIES");
    println!("Memories are facts, preferences, and personal information.\n");

    let memories = [
        "My name is Alex",
        "I was born in Seattle",
        "My favorite color is midnight blue",
        "I prefer dark mode in all applications",
        "My email is alex@example.com",
        "I speak English and Spanish fluently",
        "Coffee is my favorite drink in the morning",
        "I'm allergic to peanuts",
    ];

    println!("Storing {} memories:\n", memories.len());
    for memory in &memories {
        process_and_print(&mut engine, memory);
    }

    // =========================================================================
    // PART 2: Storing Tasks (action items, todos)
    // =========================================================================
    section_header("PART 2: STORING TASKS");
    println!("Tasks are action items and things to do.\n");

    let tasks = [
        "Buy groceries from the store",
        "Call mom this weekend",
        "Submit the quarterly report by Friday",
        "Schedule dentist appointment",
        "Fix the bug in the login module",
        "Review the pull request from Sarah",
    ];

    println!("Storing {} tasks:\n", tasks.len());
    for task in &tasks {
        process_and_print(&mut engine, task);
    }

    // =========================================================================
    // PART 3: Storing Reminders (time-sensitive tasks)
    // =========================================================================
    section_header("PART 3: STORING REMINDERS");
    println!("Reminders are time-sensitive tasks that need attention.\n");

    let reminders = [
        "Remind me to water the plants every Monday",
        "Don't forget to renew my passport next month",
        "Remember to pick up dry cleaning tomorrow",
        "Set reminder to pay rent on the 1st",
        "Alert me about the team meeting at 3pm",
    ];

    println!("Storing {} reminders:\n", reminders.len());
    for reminder in &reminders {
        process_and_print(&mut engine, reminder);
    }

    // =========================================================================
    // PART 4: Database Statistics
    // =========================================================================
    section_header("PART 4: DATABASE STATISTICS");
    print_stats(&engine);

    // =========================================================================
    // PART 5: Querying Memories
    // =========================================================================
    section_header("PART 5: QUERYING MEMORIES");
    println!("Let's search for stored memories using natural language.\n");

    let memory_queries = [
        "What is my name?",
        "Where was I born?",
        "What is my favorite color?",
        "What are my preferences?",
        "Am I allergic to anything?",
        "What languages do I speak?",
    ];

    for query in &memory_queries {
        query_and_print(&mut engine, query);
    }

    // =========================================================================
    // PART 6: Querying Tasks
    // =========================================================================
    section_header("PART 6: QUERYING TASKS");
    println!("Let's search for stored tasks using natural language.\n");

    let task_queries = [
        "What do I need to buy?",
        "Who should I call?",
        "What work tasks do I have?",
        "What meetings are scheduled?",
        "What bugs need fixing?",
    ];

    for query in &task_queries {
        query_and_print(&mut engine, query);
    }

    // =========================================================================
    // PART 7: Querying Reminders
    // =========================================================================
    section_header("PART 7: QUERYING REMINDERS");
    println!("Let's search for reminders.\n");

    let reminder_queries = [
        "What do I need to remember?",
        "What reminders do I have?",
        "What should I not forget?",
        "What needs to be renewed?",
    ];

    for query in &reminder_queries {
        query_and_print(&mut engine, query);
    }

    // =========================================================================
    // PART 8: Mixed Queries (semantic search)
    // =========================================================================
    section_header("PART 8: MIXED QUERIES (SEMANTIC SEARCH)");
    println!("The engine uses semantic similarity - not just keywords.\n");

    let semantic_queries = [
        "Tell me about myself",
        "What personal information do you know?",
        "What's on my todo list?",
        "Health related information",
        "Contact information",
    ];

    for query in &semantic_queries {
        query_and_print(&mut engine, query);
    }

    // =========================================================================
    // PART 9: Using Explicit API Methods
    // =========================================================================
    section_header("PART 9: EXPLICIT API METHODS");
    println!("You can also use explicit methods instead of auto-detection.\n");

    // Store as specific type
    println!("Using store_as() to explicitly store as TASK:");
    match engine.store_as("Learn Rust programming", DataType::Task) {
        Ok(response) => println!("  Result: {}\n", response.to_agent_string()),
        Err(e) => println!("  Error: {}\n", e),
    }

    println!("Using store_as() to explicitly store as MEMORY:");
    match engine.store_as("I started learning programming in 2020", DataType::Memory) {
        Ok(response) => println!("  Result: {}\n", response.to_agent_string()),
        Err(e) => println!("  Error: {}\n", e),
    }

    // Search specific types
    println!("Using search_tasks() to only search tasks:");
    match engine.search_tasks("programming", 3) {
        Ok(response) => {
            println!("  Query: 'programming'");
            print_query_results(&response);
        }
        Err(e) => println!("  Error: {}\n", e),
    }

    println!("Using search_memories() to only search memories:");
    match engine.search_memories("programming", 3) {
        Ok(response) => {
            println!("  Query: 'programming'");
            print_query_results(&response);
        }
        Err(e) => println!("  Error: {}\n", e),
    }

    // =========================================================================
    // PART 10: List All Items
    // =========================================================================
    section_header("PART 10: ALL STORED DATA");

    println!("All Tasks:");
    println!("-----------");
    match engine.get_tasks() {
        Ok(tasks) => {
            for task in &tasks {
                println!("  [{}] {}", task.id, task.content);
            }
            println!("  Total: {} tasks\n", tasks.len());
        }
        Err(e) => println!("  Error: {}\n", e),
    }

    println!("All Memories:");
    println!("--------------");
    match engine.get_memories() {
        Ok(memories) => {
            for memory in &memories {
                println!("  [{}] {}", memory.id, memory.content);
            }
            println!("  Total: {} memories\n", memories.len());
        }
        Err(e) => println!("  Error: {}\n", e),
    }

    // =========================================================================
    // PART 11: Performance Metrics
    // =========================================================================
    section_header("PART 11: PERFORMANCE METRICS");
    println!("{}", engine.metrics_summary());

    // =========================================================================
    // Done
    // =========================================================================
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                    Demo Complete!                         ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
}

// Helper functions

fn section_header(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!("{}", title);
    println!("═══════════════════════════════════════════════════════════");
    println!();
}

fn process_and_print(engine: &mut AgentEngine, input: &str) {
    match engine.process(input) {
        Ok(response) => {
            if let AgentResponse::Stored { data_type, latency_ms, .. } = &response {
                let type_str = match data_type {
                    DataType::Task => "TASK  ",
                    DataType::Memory => "MEMORY",
                };
                println!("  [{}] \"{}\" ({:.1}ms)", type_str, input, latency_ms);
            }
        }
        Err(e) => println!("  [ERROR] \"{}\": {}", input, e),
    }
}

fn query_and_print(engine: &mut AgentEngine, query: &str) {
    println!("  Q: \"{}\"", query);
    match engine.process(query) {
        Ok(response) => {
            print_query_results(&response);
        }
        Err(e) => println!("  Error: {}\n", e),
    }
}

fn print_query_results(response: &AgentResponse) {
    match response {
        AgentResponse::QueryResult { results, count, latency_ms, .. } => {
            println!("  Found {} result(s) ({:.1}ms):", count, latency_ms);
            for (i, result) in results.iter().take(3).enumerate() {
                println!("    {}. {}", i + 1, result);
            }
            if *count > 3 {
                println!("    ... and {} more", count - 3);
            }
            println!();
        }
        AgentResponse::NotFound { latency_ms, .. } => {
            println!("  No results found ({:.1}ms)\n", latency_ms);
        }
        _ => {}
    }
}

fn print_stats(engine: &AgentEngine) {
    match (
        engine.count(),
        engine.count_by_type(DataType::Task),
        engine.count_by_type(DataType::Memory),
    ) {
        (Ok(total), Ok(tasks), Ok(memories)) => {
            println!("  ┌─────────────────────────┐");
            println!("  │ Total items: {:>9} │", total);
            println!("  │ Tasks:       {:>9} │", tasks);
            println!("  │ Memories:    {:>9} │", memories);
            println!("  └─────────────────────────┘");
        }
        _ => println!("  Error getting stats"),
    }
}
