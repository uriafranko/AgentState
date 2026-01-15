//! AgentState CLI - Demo and interactive testing
//!
//! This binary demonstrates the unified AgentEngine API where you just
//! express intent naturally and the engine figures out what to do.

use agent_brain::{AgentEngine, AgentResponse, DataType};
use std::io::{self, BufRead, Write};

fn main() {
    println!("AgentState - Semantic State Engine for AI Agents");
    println!("=================================================");
    println!();

    // Initialize the engine
    print!("Loading AI model (this may take a moment on first run)... ");
    io::stdout().flush().unwrap();

    let mut engine = match AgentEngine::new("agent_state.db") {
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

    // Run demo showing the unified API
    run_demo(&mut engine);

    // Interactive mode
    println!();
    println!("-----------------------------------------------------------");
    println!("Interactive Mode - Just talk naturally!");
    println!("-----------------------------------------------------------");
    println!("Examples:");
    println!("  'My favorite color is blue'  -> Stores as memory");
    println!("  'Remind me to call John'     -> Stores as task");
    println!("  'What is my favorite color?' -> Queries and returns results");
    println!("  'Who should I call?'         -> Queries and returns results");
    println!();
    println!("Commands: stats, tasks, memories, clear, classify <text>, quit");
    println!("-----------------------------------------------------------");
    println!();

    interactive_mode(&mut engine);
}

fn run_demo(engine: &mut AgentEngine) {
    println!("Demo: The Unified API");
    println!("---------------------");
    println!();
    println!("AgentState uses ONE method (process) for everything.");
    println!("Just express intent naturally - no need to specify store vs query.");
    println!();

    // Demo 1: Store operations (detected automatically)
    println!("1. Storing information (detected as STORE action):");
    println!();

    let store_examples = [
        "My name is Alice",
        "I prefer dark mode in applications",
        "John's phone number is 555-1234",
        "Remind me to buy groceries tomorrow",
        "Schedule a meeting with Bob on Friday",
        "Call the dentist to reschedule",
    ];

    for example in &store_examples {
        match engine.process(example) {
            Ok(response) => {
                if let AgentResponse::Stored { data_type, .. } = &response {
                    let type_str = match data_type {
                        DataType::Task => "TASK",
                        DataType::Memory => "MEMORY",
                    };
                    println!("   '{}'\n   -> Stored as {}", example, type_str);
                }
            }
            Err(e) => println!("   Error: {}", e),
        }
    }
    println!();

    // Demo 2: Query operations (detected automatically)
    println!("2. Querying information (detected as QUERY action):");
    println!();

    let query_examples = [
        "What is my name?",
        "Who should I call?",
        "What do I need to buy?",
        "Tell me about John",
    ];

    for example in &query_examples {
        match engine.process(example) {
            Ok(response) => {
                println!("   Query: '{}'", example);
                match &response {
                    AgentResponse::QueryResult { results, count, .. } => {
                        println!("   Found {} result(s):", count);
                        for (i, r) in results.iter().take(2).enumerate() {
                            println!("     {}. {}", i + 1, r);
                        }
                    }
                    AgentResponse::NotFound { .. } => {
                        println!("   No results found");
                    }
                    _ => {}
                }
            }
            Err(e) => println!("   Error: {}", e),
        }
        println!();
    }

    // Demo 3: Show the response can be used directly by an agent
    println!("3. Response format for agents:");
    println!();

    let response = engine.process("What preferences do I have?").unwrap();
    println!("   response.to_agent_string():");
    println!("   {}", response.to_agent_string());
    println!();

    // Stats
    println!("4. Current state:");
    if let (Ok(total), Ok(tasks), Ok(memories)) = (
        engine.count(),
        engine.count_by_type(DataType::Task),
        engine.count_by_type(DataType::Memory),
    ) {
        println!("   Total items: {}", total);
        println!("   Tasks: {}", tasks);
        println!("   Memories: {}", memories);
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

        // Check for special commands
        let lower = line.to_lowercase();

        if lower == "quit" || lower == "exit" || lower == "q" {
            println!("Goodbye!");
            break;
        }

        if lower == "stats" {
            print_stats(engine);
            continue;
        }

        if lower == "tasks" {
            print_tasks(engine);
            continue;
        }

        if lower == "memories" {
            print_memories(engine);
            continue;
        }

        if lower == "clear" {
            handle_clear(engine);
            continue;
        }

        if lower == "help" {
            print_help();
            continue;
        }

        if lower.starts_with("classify ") {
            let text = &line[9..];
            handle_classify(engine, text);
            continue;
        }

        if lower.starts_with("delete ") {
            let id_str = &line[7..];
            handle_delete(engine, id_str);
            continue;
        }

        // Default: process as natural language
        match engine.process(line) {
            Ok(response) => {
                print_response(&response);
            }
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
}

fn print_response(response: &AgentResponse) {
    match response {
        AgentResponse::Stored {
            id,
            data_type,
            content,
        } => {
            let type_str = match data_type {
                DataType::Task => "TASK",
                DataType::Memory => "MEMORY",
            };
            println!("Stored [{}] as {} (id: {})", content, type_str, id);
        }
        AgentResponse::QueryResult {
            results,
            count,
            data_type,
        } => {
            let filter = match data_type {
                Some(DataType::Task) => " (tasks only)",
                Some(DataType::Memory) => " (memories only)",
                None => "",
            };
            println!("Found {} result(s){}:", count, filter);
            for (i, result) in results.iter().enumerate() {
                println!("  {}. {}", i + 1, result);
            }
        }
        AgentResponse::NotFound { query } => {
            println!("No results found for: '{}'", query);
        }
    }
}

fn print_stats(engine: &AgentEngine) {
    match (
        engine.count(),
        engine.count_by_type(DataType::Task),
        engine.count_by_type(DataType::Memory),
    ) {
        (Ok(total), Ok(tasks), Ok(memories)) => {
            println!("Statistics:");
            println!("  Total items: {}", total);
            println!("  Tasks: {}", tasks);
            println!("  Memories: {}", memories);
        }
        _ => println!("Error getting stats"),
    }
}

fn print_tasks(engine: &AgentEngine) {
    match engine.get_tasks() {
        Ok(tasks) => {
            if tasks.is_empty() {
                println!("No tasks stored.");
            } else {
                println!("Tasks ({}):", tasks.len());
                for task in &tasks {
                    println!("  [{}] {}", task.id, task.content);
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

fn print_memories(engine: &AgentEngine) {
    match engine.get_memories() {
        Ok(memories) => {
            if memories.is_empty() {
                println!("No memories stored.");
            } else {
                println!("Memories ({}):", memories.len());
                for memory in &memories {
                    println!("  [{}] {}", memory.id, memory.content);
                }
            }
        }
        Err(e) => println!("Error: {}", e),
    }
}

fn handle_clear(engine: &mut AgentEngine) {
    print!("Are you sure you want to clear all data? (y/n): ");
    io::stdout().flush().unwrap();

    let stdin = io::stdin();
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

fn handle_classify(engine: &AgentEngine, text: &str) {
    match engine.classify(text) {
        Ok(intent) => {
            let action = match intent.action {
                agent_brain::Action::Store => "STORE",
                agent_brain::Action::Query => "QUERY",
            };
            let data_type = match intent.data_type {
                DataType::Task => "TASK",
                DataType::Memory => "MEMORY",
            };
            println!("Intent: {} as {}", action, data_type);
        }
        Err(e) => println!("Error: {}", e),
    }
}

fn handle_delete(engine: &mut AgentEngine, id_str: &str) {
    match id_str.parse::<i64>() {
        Ok(id) => match engine.delete(id) {
            Ok(true) => println!("Deleted item {}", id),
            Ok(false) => println!("Item {} not found", id),
            Err(e) => println!("Error: {}", e),
        },
        Err(_) => println!("Invalid ID: {}", id_str),
    }
}

fn print_help() {
    println!("AgentState - Natural Language State Engine");
    println!();
    println!("Just type naturally! The engine automatically detects your intent.");
    println!();
    println!("Store examples:");
    println!("  'My favorite color is blue'");
    println!("  'Remember that John likes pizza'");
    println!("  'Remind me to call mom tomorrow'");
    println!();
    println!("Query examples:");
    println!("  'What is my favorite color?'");
    println!("  'Who should I call?'");
    println!("  'What do I know about John?'");
    println!();
    println!("Commands:");
    println!("  stats           - Show statistics");
    println!("  tasks           - List all tasks");
    println!("  memories        - List all memories");
    println!("  classify <text> - Preview classification without storing");
    println!("  delete <id>     - Delete item by ID");
    println!("  clear           - Clear all data");
    println!("  help            - Show this help");
    println!("  quit            - Exit");
}
