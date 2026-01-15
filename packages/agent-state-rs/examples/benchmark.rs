//! Model Benchmark - Compare performance and accuracy of different embedding models
//!
//! This benchmark helps you choose the right model for your use case by measuring:
//! 1. Embedding generation speed
//! 2. Classification accuracy
//! 3. Search relevance
//!
//! Run with: cargo run --release --example benchmark

use agent_brain::{AgentEngine, AgentResponse, Backend, BrainConfig, EmbeddingModel, Action, DataType};
use std::time::{Duration, Instant};

/// Test cases for classification accuracy
const CLASSIFICATION_TESTS: &[(&str, Action, DataType)] = &[
    // Store - Tasks
    ("Remind me to call John tomorrow", Action::Store, DataType::Task),
    ("Buy groceries from the store", Action::Store, DataType::Task),
    ("Schedule dentist appointment", Action::Store, DataType::Task),
    ("Don't forget to pay the rent", Action::Store, DataType::Task),
    ("Pick up dry cleaning", Action::Store, DataType::Task),

    // Store - Memories
    ("My favorite color is blue", Action::Store, DataType::Memory),
    ("The capital of France is Paris", Action::Store, DataType::Memory),
    ("John's phone number is 555-1234", Action::Store, DataType::Memory),
    ("My birthday is March 15th", Action::Store, DataType::Memory),
    ("The project deadline is next Friday", Action::Store, DataType::Memory),

    // Store - Preferences
    ("I prefer dark mode in all apps", Action::Store, DataType::Preference),
    ("I like my coffee black", Action::Store, DataType::Preference),
    ("I enjoy hiking on weekends", Action::Store, DataType::Preference),
    ("I hate early morning meetings", Action::Store, DataType::Preference),

    // Store - Relationships
    ("John is my colleague", Action::Store, DataType::Relationship),
    ("Sarah works at Google", Action::Store, DataType::Relationship),
    ("Alice is my sister", Action::Store, DataType::Relationship),

    // Store - Events
    ("Team meeting tomorrow at 3pm", Action::Store, DataType::Event),
    ("Birthday party next Saturday", Action::Store, DataType::Event),
    ("Conference call on Monday morning", Action::Store, DataType::Event),

    // Query - various
    ("What is my name?", Action::Query, DataType::Memory),
    ("What's my favorite color?", Action::Query, DataType::Preference),
    ("Who should I call?", Action::Query, DataType::Task),
    ("What do I need to do today?", Action::Query, DataType::Task),
    ("Find my scheduled meetings", Action::Query, DataType::Event),
    ("Who works at Google?", Action::Query, DataType::Relationship),
    ("What are my preferences?", Action::Query, DataType::Preference),
    ("Tell me about John", Action::Query, DataType::Memory),
    ("Search for tasks", Action::Query, DataType::Task),
    ("List my reminders", Action::Query, DataType::Task),
];

/// Test cases for semantic search relevance
const SEARCH_TESTS: &[(&str, &[&str], &str)] = &[
    (
        "What color do I like?",
        &["My favorite color is blue", "I like green apples"],
        "My favorite color is blue",
    ),
    (
        "Who should I contact?",
        &["Call John tomorrow", "Send email to Sarah", "The phone is ringing"],
        "Call John tomorrow",
    ),
    (
        "What meetings do I have?",
        &["Team meeting at 3pm", "I met John yesterday", "The conference room is booked"],
        "Team meeting at 3pm",
    ),
];

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         AgentState Model Benchmark - Performance & Accuracy       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Run mock mode benchmark (always available, fast)
    println!("═══════════════════════════════════════════════════════════════════");
    println!("MOCK MODE BENCHMARK (baseline, no ML model)");
    println!("═══════════════════════════════════════════════════════════════════");
    run_benchmark(BrainConfig::mock(), "Mock (hash-based)");

    // Print model comparison table
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("AVAILABLE MODELS COMPARISON");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("| {:20} | {:6} | {:10} | {:30} |", "Model", "Dims", "MTEB Score", "Use Case");
    println!("|{:-<22}|{:-<8}|{:-<12}|{:-<32}|", "", "", "", "");

    for model in &[
        EmbeddingModel::MiniLmL6,
        EmbeddingModel::MiniLmL12,
        EmbeddingModel::BgeSmall,
        EmbeddingModel::BgeBase,
        EmbeddingModel::E5Small,
        EmbeddingModel::GteSmall,
    ] {
        let use_case = match model {
            EmbeddingModel::MiniLmL6 => "Fast, resource-constrained",
            EmbeddingModel::MiniLmL12 => "Better accuracy, same size",
            EmbeddingModel::BgeSmall => "Best small model (default)",
            EmbeddingModel::BgeBase => "Best accuracy overall",
            EmbeddingModel::E5Small => "Good alternative to BGE",
            EmbeddingModel::GteSmall => "Competitive small model",
        };
        println!(
            "| {:20} | {:6} | {:10.1} | {:30} |",
            format!("{:?}", model),
            model.embedding_dim(),
            model.mteb_score(),
            use_case
        );
    }
    println!();

    // Try to run real model benchmarks if models are available
    println!("═══════════════════════════════════════════════════════════════════");
    println!("REAL MODEL BENCHMARKS");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("Attempting to load models from HuggingFace Hub cache...");
    println!("(First run may download ~90MB per model)");
    println!();

    // Try each model config
    let configs = [
        (BrainConfig {
            model: EmbeddingModel::MiniLmL6,
            backend: Backend::Candle,
            local_model_dir: None,
        }, "MiniLM-L6 (Candle)"),
        (BrainConfig {
            model: EmbeddingModel::BgeSmall,
            backend: Backend::Candle,
            local_model_dir: None,
        }, "BGE-Small (Candle)"),
    ];

    for (config, name) in configs {
        println!("-------------------------------------------------------------------");
        if let Err(e) = run_benchmark(config, name) {
            println!("  Skipped: {} ({})", name, e);
        }
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("BENCHMARK COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();
    println!("Tips for better performance:");
    println!("  1. Use --release mode: cargo run --release --example benchmark");
    println!("  2. Download models locally to avoid network latency");
    println!("  3. Use BGE-Small for best accuracy with 384 dimensions");
    println!("  4. Use BGE-Base for maximum accuracy (768 dimensions)");
    println!();
}

fn run_benchmark(config: BrainConfig, name: &str) -> Result<(), String> {
    println!();
    println!("Benchmarking: {}", name);
    println!("  Model: {:?}, Backend: {:?}, Dims: {}",
             config.model, config.backend, config.model.embedding_dim());
    println!();

    // Initialize engine
    let init_start = Instant::now();
    let mut engine = AgentEngine::with_config(":memory:", config)
        .map_err(|e| e.to_string())?;
    let init_time = init_start.elapsed();
    println!("  Initialization: {:?}", init_time);

    // Benchmark embedding generation
    let embed_times = benchmark_embeddings(&mut engine);
    println!("  Embedding (first): {:?}", embed_times.0);
    println!("  Embedding (cached): {:?}", embed_times.1);
    println!("  Embedding (avg 10): {:?}", embed_times.2);

    // Benchmark classification accuracy
    let (action_acc, type_acc) = benchmark_classification(&mut engine);
    println!("  Action accuracy: {:.1}%", action_acc * 100.0);
    println!("  DataType accuracy: {:.1}%", type_acc * 100.0);

    // Benchmark search relevance
    let search_acc = benchmark_search(&mut engine);
    println!("  Search relevance: {:.1}%", search_acc * 100.0);

    Ok(())
}

fn benchmark_embeddings(engine: &mut AgentEngine) -> (Duration, Duration, Duration) {
    let test_text = "This is a test sentence for embedding generation.";

    // First embedding (cold)
    engine.classify(test_text).ok(); // Warm up (loads anchors)
    let start = Instant::now();
    engine.store(test_text).ok();
    let first = start.elapsed();

    // Cached embedding
    let start = Instant::now();
    engine.search(test_text).ok();
    let cached = start.elapsed();

    // Average over 10 different texts
    let texts = [
        "Hello world",
        "How are you today?",
        "The quick brown fox",
        "Machine learning is great",
        "Rust programming language",
        "Database optimization techniques",
        "Natural language processing",
        "Semantic search engine",
        "Vector embeddings work well",
        "Classification algorithms",
    ];

    let start = Instant::now();
    for text in texts {
        engine.classify(text).ok();
    }
    let avg = start.elapsed() / 10;

    (first, cached, avg)
}

fn benchmark_classification(engine: &mut AgentEngine) -> (f32, f32) {
    let mut action_correct = 0;
    let mut type_correct = 0;
    let total = CLASSIFICATION_TESTS.len();

    for (text, expected_action, expected_type) in CLASSIFICATION_TESTS {
        if let Ok(intent) = engine.classify(text) {
            if intent.action == *expected_action {
                action_correct += 1;
            }
            if intent.data_type == *expected_type {
                type_correct += 1;
            }
        }
    }

    (
        action_correct as f32 / total as f32,
        type_correct as f32 / total as f32,
    )
}

fn benchmark_search(engine: &mut AgentEngine) -> f32 {
    let mut correct = 0;
    let total = SEARCH_TESTS.len();

    for (query, corpus, expected_top) in SEARCH_TESTS {
        // Store all corpus items
        for item in *corpus {
            engine.store(item).ok();
        }

        // Search and check if expected is top result
        if let Ok(AgentResponse::QueryResult { results, .. }) = engine.search(query) {
            if !results.is_empty() && results[0] == *expected_top {
                correct += 1;
            }
        }

        // Clear for next test
        engine.clear().ok();
    }

    correct as f32 / total as f32
}
