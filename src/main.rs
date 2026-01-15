//! Agent Brain CLI - Test runner and interactive demo
//!
//! This binary demonstrates the AgentEngine functionality with example inputs.

use agent_brain::AgentEngine;
use std::io::{self, BufRead, Write};

fn main() {
    println!("Agent Brain - Rust Core Engine");
    println!("================================");
    println!();

    // Initialize the engine
    print!("Loading AI model (this may take a moment on first run)... ");
    io::stdout().flush().unwrap();

    let mut engine = match AgentEngine::new("my_agent.db") {
        Ok(e) => {
            println!("Done!");
            e
        }
        Err(e) => {
            eprintln!("Failed to initialize engine: {}", e);
            std::process::exit(1);
        }
    };

    println!();
    println!("Engine loaded successfully!");
    println!();

    // Run demo tests
    run_demo(&mut engine);

    // Interactive mode
    println!();
    println!("-----------------------------------");
    println!("Interactive Mode");
    println!("Commands: add <text>, query <text>, tasks, memories, clear, quit");
    println!("-----------------------------------");
    println!();

    interactive_mode(&mut engine);
}

fn run_demo(engine: &mut AgentEngine) {
    println!("Running Demo Tests");
    println!("------------------");
    println!();

    // Test 1: Add a Task
    println!("Test 1: Adding a task...");
    match engine.add("Remind me to buy oat milk tomorrow") {
        Ok(result) => {
            println!("  Input:  'Remind me to buy oat milk tomorrow'");
            println!("  Result: {}", result);
            println!("  [Expected: processed_as_task]");
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Test 2: Add a Memory
    println!("Test 2: Adding a memory...");
    match engine.add("My favorite color is blue") {
        Ok(result) => {
            println!("  Input:  'My favorite color is blue'");
            println!("  Result: {}", result);
            println!("  [Expected: processed_as_memory]");
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Test 3: Add more examples
    println!("Test 3: Adding more examples...");
    let examples = [
        "Schedule a meeting with John for Friday",
        "The project deadline is March 15th",
        "I prefer dark mode in applications",
        "Call the dentist to reschedule appointment",
        "My phone number is 555-1234",
        "Don't forget to submit the report",
    ];

    for example in &examples {
        match engine.add(example) {
            Ok(result) => println!("  '{}' -> {}", example, result),
            Err(e) => println!("  '{}' -> Error: {}", example, e),
        }
    }
    println!();

    // Test 4: Semantic Search
    println!("Test 4: Semantic search...");
    match engine.query("What should I buy?") {
        Ok(results) => {
            println!("  Query: 'What should I buy?'");
            println!("  Results:");
            for (i, result) in results.iter().enumerate() {
                println!("    {}. {}", i + 1, result);
            }
            println!("  [Expected: 'Remind me to buy oat milk tomorrow' should be in results]");
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Test 5: Search by category
    println!("Test 5: Search only tasks...");
    match engine.query_category("appointments and meetings", "task", 3) {
        Ok(results) => {
            println!("  Query: 'appointments and meetings' (tasks only)");
            println!("  Results:");
            for (i, result) in results.iter().enumerate() {
                println!("    {}. {}", i + 1, result);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // Test 6: Stats
    println!("Test 6: Statistics...");
    match (engine.count(), engine.count_by_category("task"), engine.count_by_category("memory")) {
        (Ok(total), Ok(tasks), Ok(memories)) => {
            println!("  Total items: {}", total);
            println!("  Tasks: {}", tasks);
            println!("  Memories: {}", memories);
        }
        _ => println!("  Error getting stats"),
    }
    println!();

    println!("Demo complete!");
}

fn interactive_mode(engine: &mut AgentEngine) {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush().unwrap();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ' ').collect();
        let command = parts[0].to_lowercase();

        match command.as_str() {
            "quit" | "exit" | "q" => {
                println!("Goodbye!");
                break;
            }
            "add" => {
                if parts.len() < 2 {
                    println!("Usage: add <text>");
                    continue;
                }
                match engine.add(parts[1]) {
                    Ok(result) => println!("{}", result),
                    Err(e) => println!("Error: {}", e),
                }
            }
            "query" | "search" => {
                if parts.len() < 2 {
                    println!("Usage: query <text>");
                    continue;
                }
                match engine.query_with_scores(parts[1], 5) {
                    Ok(results) => {
                        if results.is_empty() {
                            println!("No results found.");
                        } else {
                            for (i, (content, score)) in results.iter().enumerate() {
                                println!("{}. [score: {:.3}] {}", i + 1, score, content);
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }
            "classify" => {
                if parts.len() < 2 {
                    println!("Usage: classify <text>");
                    continue;
                }
                match engine.classify(parts[1]) {
                    Ok(intent) => println!("{:?}", intent),
                    Err(e) => println!("Error: {}", e),
                }
            }
            "tasks" => match engine.get_tasks() {
                Ok(tasks) => {
                    if tasks.is_empty() {
                        println!("No tasks stored.");
                    } else {
                        println!("Tasks ({}):", tasks.len());
                        for task in &tasks {
                            println!("  [{}] {} ({})", task.id, task.content, task.created_at);
                        }
                    }
                }
                Err(e) => println!("Error: {}", e),
            },
            "memories" => match engine.get_memories() {
                Ok(memories) => {
                    if memories.is_empty() {
                        println!("No memories stored.");
                    } else {
                        println!("Memories ({}):", memories.len());
                        for memory in &memories {
                            println!("  [{}] {} ({})", memory.id, memory.content, memory.created_at);
                        }
                    }
                }
                Err(e) => println!("Error: {}", e),
            },
            "stats" => {
                match (
                    engine.count(),
                    engine.count_by_category("task"),
                    engine.count_by_category("memory"),
                ) {
                    (Ok(total), Ok(tasks), Ok(memories)) => {
                        println!("Total: {}, Tasks: {}, Memories: {}", total, tasks, memories);
                    }
                    _ => println!("Error getting stats"),
                }
            }
            "clear" => {
                print!("Are you sure? (y/n): ");
                stdout.flush().unwrap();
                let mut confirm = String::new();
                if stdin.lock().read_line(&mut confirm).is_ok() && confirm.trim().to_lowercase() == "y" {
                    match engine.clear() {
                        Ok(_) => println!("All data cleared."),
                        Err(e) => println!("Error: {}", e),
                    }
                } else {
                    println!("Cancelled.");
                }
            }
            "delete" => {
                if parts.len() < 2 {
                    println!("Usage: delete <id>");
                    continue;
                }
                match parts[1].parse::<i64>() {
                    Ok(id) => match engine.delete(id) {
                        Ok(true) => println!("Deleted item {}", id),
                        Ok(false) => println!("Item {} not found", id),
                        Err(e) => println!("Error: {}", e),
                    },
                    Err(_) => println!("Invalid ID"),
                }
            }
            "help" => {
                println!("Commands:");
                println!("  add <text>      - Add text (auto-classified as task/memory)");
                println!("  query <text>    - Search for similar content");
                println!("  classify <text> - Preview classification without storing");
                println!("  tasks           - List all tasks");
                println!("  memories        - List all memories");
                println!("  stats           - Show statistics");
                println!("  delete <id>     - Delete item by ID");
                println!("  clear           - Clear all data");
                println!("  quit            - Exit");
            }
            _ => {
                println!("Unknown command: '{}'. Type 'help' for available commands.", command);
            }
        }
    }
}
