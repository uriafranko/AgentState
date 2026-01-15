//! Intelligence Layer - Handles AI model loading, embedding generation, and intent classification
//!
//! This module implements zero-shot intent routing using a MiniLM sentence transformer model.
//! It classifies input text by action (Store vs Query) and by data type (Task vs Memory)
//! by comparing embeddings against pre-computed anchor vectors.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::{HashMap, VecDeque};
use tokenizers::Tokenizer;

/// Maximum number of embeddings to cache (LRU eviction when exceeded)
const EMBEDDING_CACHE_SIZE: usize = 1000;

/// The Brain handles all AI-related operations including embedding generation and intent classification
pub struct Brain {
    /// Model for real mode (None in mock mode)
    model: Option<BertModel>,
    /// Tokenizer for real mode (None in mock mode)
    tokenizer: Option<Tokenizer>,
    device: Device,
    /// Whether running in mock mode (hash-based embeddings for testing)
    mock_mode: bool,
    /// Anchor vector representing "Task" data type
    anchor_task: Tensor,
    /// Anchor vector representing "Memory" data type
    anchor_memory: Tensor,
    /// Anchor vector representing "Preference" data type
    anchor_preference: Tensor,
    /// Anchor vector representing "Relationship" data type
    anchor_relationship: Tensor,
    /// Anchor vector representing "Event" data type
    anchor_event: Tensor,
    /// Anchor vector representing "Store" action
    anchor_store: Tensor,
    /// Anchor vector representing "Query" action
    anchor_query: Tensor,
    /// Cache for computed embeddings (text -> embedding vector as f32 array)
    embedding_cache: HashMap<String, Vec<f32>>,
    /// Order of cache insertions for LRU eviction (VecDeque for O(1) removal from front)
    cache_order: VecDeque<String>,
}

/// The action the agent wants to perform
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Store information (save, remember, add, note)
    Store,
    /// Query/retrieve information (find, search, what is, who is)
    Query,
}

/// The type of data being stored or queried
///
/// Extended data types enable richer semantic classification for AI agents.
/// The engine auto-detects the most appropriate type based on content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Action items, reminders, todos - things that need to be done
    /// Examples: "Remind me to call John", "I need to buy groceries"
    Task,
    /// General facts and information to remember
    /// Examples: "The capital of France is Paris", "John's phone number is 555-1234"
    Memory,
    /// User preferences and likes/dislikes
    /// Examples: "I prefer dark mode", "My favorite color is blue"
    Preference,
    /// Relationships between entities (people, things, concepts)
    /// Examples: "John is my colleague", "Alice works at Acme Corp"
    Relationship,
    /// Time-based events and appointments
    /// Examples: "Meeting tomorrow at 3pm", "Birthday party on Saturday"
    Event,
}

impl DataType {
    /// Returns the category string for database storage
    pub fn as_category(&self) -> &'static str {
        match self {
            DataType::Task => "task",
            DataType::Memory => "memory",
            DataType::Preference => "preference",
            DataType::Relationship => "relationship",
            DataType::Event => "event",
        }
    }

    /// Parse from category string
    pub fn from_category(s: &str) -> Option<Self> {
        match s {
            "task" => Some(DataType::Task),
            "memory" => Some(DataType::Memory),
            "preference" => Some(DataType::Preference),
            "relationship" => Some(DataType::Relationship),
            "event" => Some(DataType::Event),
            _ => None,
        }
    }

    /// Returns all available data types
    pub fn all() -> &'static [DataType] {
        &[
            DataType::Task,
            DataType::Memory,
            DataType::Preference,
            DataType::Relationship,
            DataType::Event,
        ]
    }
}

/// Full intent classification result
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intent {
    /// What action to perform (store or query)
    pub action: Action,
    /// What type of data this relates to
    pub data_type: DataType,
    /// Confidence score for action classification (0.0 to 1.0)
    pub action_confidence: f32,
    /// Confidence score for data type classification (0.0 to 1.0)
    pub data_type_confidence: f32,
}

impl Intent {
    /// Returns true if the classification is ambiguous (low confidence)
    ///
    /// An ambiguous intent means the engine isn't sure what the agent wants.
    /// In a cloud/stateless environment, this should trigger a clarification request.
    pub fn is_ambiguous(&self) -> bool {
        self.action_confidence < 0.15 || self.data_type_confidence < 0.15
    }

    /// Returns the overall confidence (minimum of both confidences)
    pub fn overall_confidence(&self) -> f32 {
        self.action_confidence.min(self.data_type_confidence)
    }

    /// Returns a clarification message if the intent is ambiguous
    ///
    /// This is the key for cloud-ready stateless design: instead of maintaining
    /// context, we return a message telling the agent what we need.
    pub fn clarification_message(&self) -> Option<String> {
        if !self.is_ambiguous() {
            return None;
        }

        let mut issues = Vec::new();

        if self.action_confidence < 0.15 {
            issues.push(format!(
                "I'm not sure if you want to store this information or search for something. \
                 Try rephrasing with 'remember that...' to store, or 'what is...' to search. \
                 (action confidence: {:.1}%)",
                self.action_confidence * 100.0
            ));
        }

        if self.data_type_confidence < 0.15 {
            issues.push(format!(
                "I can't determine if this is a task (action item) or a memory (fact/preference). \
                 Try being more specific: 'remind me to...' for tasks, 'my favorite X is...' for memories. \
                 (type confidence: {:.1}%)",
                self.data_type_confidence * 100.0
            ));
        }

        Some(issues.join("\n\n"))
    }
}

impl Brain {
    /// Creates a new Brain instance by loading the MiniLM model
    ///
    /// First tries to load from local `models/minilm` directory, then falls back
    /// to downloading from HuggingFace Hub (cached in ~/.cache/huggingface).
    pub fn new() -> Result<Self> {
        // Try local models directory first
        let local_model_dir = std::path::Path::new("models/minilm");
        if local_model_dir.exists() {
            return Self::new_from_local(local_model_dir);
        }

        // Fall back to HuggingFace Hub download
        Self::new_from_huggingface()
    }

    /// Creates a new Brain in mock mode for testing
    ///
    /// Uses hash-based deterministic embeddings instead of the ML model.
    /// This allows running all the same logic (storage, retrieval, classification)
    /// without requiring the actual model files.
    ///
    /// The mock embeddings are semantically-aware based on keyword detection,
    /// providing reasonable classification behavior for testing.
    pub fn new_mock() -> Result<Self> {
        let device = Device::Cpu;

        // Create placeholder tensors for anchors
        let placeholder = Tensor::zeros((1, 384), DType::F32, &device)?;

        let mut brain = Self {
            model: None,
            tokenizer: None,
            device,
            mock_mode: true,
            anchor_task: placeholder.clone(),
            anchor_memory: placeholder.clone(),
            anchor_preference: placeholder.clone(),
            anchor_relationship: placeholder.clone(),
            anchor_event: placeholder.clone(),
            anchor_store: placeholder.clone(),
            anchor_query: placeholder,
            embedding_cache: HashMap::new(),
            cache_order: VecDeque::new(),
        };

        // Initialize anchor vectors using mock embeddings
        brain.anchor_task = brain.embed(
            "action item, todo, remind me to, deadline, schedule, need to do, task, must complete",
        )?;
        brain.anchor_memory = brain.embed(
            "fact, information, context, background, remember that, note, detail about, knowledge",
        )?;
        brain.anchor_preference = brain.embed(
            "I prefer, I like, my favorite, I enjoy, I want, preference, likes, dislikes, choose",
        )?;
        brain.anchor_relationship = brain.embed(
            "is my, works at, knows, colleague, friend, family, partner, relationship, connected to",
        )?;
        brain.anchor_event = brain.embed(
            "meeting, appointment, event, happening, tomorrow, next week, on date, scheduled for, calendar",
        )?;
        brain.anchor_store = brain.embed(
            "save this, remember, store, add, note that, record, keep track of, my name is, I like",
        )?;
        brain.anchor_query = brain.embed(
            "what is, who is, find, search, look up, tell me about, retrieve, get, show me, list",
        )?;

        Ok(brain)
    }

    /// Returns whether this Brain is running in mock mode
    pub fn is_mock(&self) -> bool {
        self.mock_mode
    }

    /// Creates a new Brain instance from local model files
    ///
    /// Expects the directory to contain: config.json, tokenizer.json, model.safetensors
    pub fn new_from_local(model_dir: &std::path::Path) -> Result<Self> {
        let device = Device::Cpu;

        let config_filename = model_dir.join("config.json");
        let tokenizer_filename = model_dir.join("tokenizer.json");
        let weights_filename = model_dir.join("model.safetensors");

        // Verify files exist
        if !config_filename.exists() {
            anyhow::bail!("config.json not found in {:?}", model_dir);
        }
        if !tokenizer_filename.exists() {
            anyhow::bail!("tokenizer.json not found in {:?}", model_dir);
        }
        if !weights_filename.exists() {
            anyhow::bail!("model.safetensors not found in {:?}", model_dir);
        }

        Self::load_model(device, &config_filename, &tokenizer_filename, &weights_filename)
    }

    /// Creates a new Brain instance by downloading from HuggingFace Hub
    ///
    /// Downloads are cached automatically in ~/.cache/huggingface
    pub fn new_from_huggingface() -> Result<Self> {
        let device = Device::Cpu;

        let api = Api::new().context("Failed to initialize HuggingFace API")?;
        let repo = api.repo(Repo::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            RepoType::Model,
        ));

        let config_filename = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let weights_filename = repo
            .get("model.safetensors")
            .context("Failed to download model.safetensors")?;

        Self::load_model(device, &config_filename, &tokenizer_filename, &weights_filename)
    }

    /// Internal method to load model from file paths
    fn load_model(
        device: Device,
        config_filename: &std::path::Path,
        tokenizer_filename: &std::path::Path,
        weights_filename: &std::path::Path,
    ) -> Result<Self> {

        // Parse model configuration
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_filename)
                .context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_filename)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights using memory-mapped safetensors for efficiency
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)
                .context("Failed to load model weights")?
        };
        let model = BertModel::load(vb, &config).context("Failed to initialize BERT model")?;

        // Create placeholder tensors for anchors (will be initialized below)
        let placeholder = Tensor::zeros((1, 384), DType::F32, &device)?;

        let mut brain = Self {
            model: Some(model),
            tokenizer: Some(tokenizer),
            device,
            mock_mode: false,
            anchor_task: placeholder.clone(),
            anchor_memory: placeholder.clone(),
            anchor_preference: placeholder.clone(),
            anchor_relationship: placeholder.clone(),
            anchor_event: placeholder.clone(),
            anchor_store: placeholder.clone(),
            anchor_query: placeholder,
            embedding_cache: HashMap::new(),
            cache_order: VecDeque::new(),
        };

        // 2. Initialize Anchor Vectors (Zero-Shot Classification Logic)
        // Data type anchors: what kind of information is this?
        brain.anchor_task = brain.embed(
            "action item, todo, remind me to, deadline, schedule, need to do, task, must complete",
        )?;
        brain.anchor_memory = brain.embed(
            "fact, information, context, background, remember that, note, detail about, knowledge",
        )?;
        brain.anchor_preference = brain.embed(
            "I prefer, I like, my favorite, I enjoy, I want, preference, likes, dislikes, choose",
        )?;
        brain.anchor_relationship = brain.embed(
            "is my, works at, knows, colleague, friend, family, partner, relationship, connected to",
        )?;
        brain.anchor_event = brain.embed(
            "meeting, appointment, event, happening, tomorrow, next week, on date, scheduled for, calendar",
        )?;

        // Action anchors: what does the agent want to do?
        brain.anchor_store = brain.embed(
            "save this, remember, store, add, note that, record, keep track of, my name is, I like",
        )?;
        brain.anchor_query = brain.embed(
            "what is, who is, find, search, look up, tell me about, retrieve, get, show me, list",
        )?;

        Ok(brain)
    }

    /// Converts text into a normalized 384-dimensional embedding vector
    ///
    /// Uses mean pooling over all token embeddings followed by L2 normalization.
    /// The resulting vector is suitable for cosine similarity comparisons.
    /// Results are cached for performance.
    pub fn embed(&mut self, text: &str) -> Result<Tensor> {
        // Check cache first
        if let Some(cached) = self.embedding_cache.get(text) {
            return Ok(Tensor::new(cached.as_slice(), &self.device)?.unsqueeze(0)?);
        }

        // Compute embedding
        let embedding = self.compute_embedding(text)?;

        // Extract values for caching
        let values: Vec<f32> = embedding.squeeze(0)?.to_vec1()?;

        // Cache with LRU eviction (O(1) removal from front with VecDeque)
        if self.cache_order.len() >= EMBEDDING_CACHE_SIZE {
            if let Some(oldest) = self.cache_order.pop_front() {
                self.embedding_cache.remove(&oldest);
            }
        }
        self.embedding_cache.insert(text.to_string(), values);
        self.cache_order.push_back(text.to_string());

        Ok(embedding)
    }

    /// Internal: compute embedding without caching (used by embed())
    fn compute_embedding(&self, text: &str) -> Result<Tensor> {
        if self.mock_mode {
            return self.compute_mock_embedding(text);
        }

        // Tokenize input text
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not available"))?;
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids = tokens.get_ids();
        let token_type_ids = vec![0u32; token_ids.len()];
        let attention_mask = tokens.get_attention_mask();

        // Convert to tensors with batch dimension
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(token_type_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Run model inference
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not available"))?;
        let embeddings = model
            .forward(&token_ids_tensor, &token_type_ids_tensor, None)
            .context("Model forward pass failed")?;

        // Mean pooling: average all token embeddings (excluding padding)
        let (_batch_size, _n_tokens, _hidden_size) = embeddings.dims3()?;

        // Create attention mask tensor for proper pooling
        let attention_mask_tensor =
            Tensor::new(attention_mask, &self.device)?.unsqueeze(0)?.unsqueeze(2)?;
        let attention_mask_f32 = attention_mask_tensor.to_dtype(DType::F32)?;

        // Masked mean pooling
        let masked_embeddings = embeddings.broadcast_mul(&attention_mask_f32)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let mask_sum = attention_mask_f32.sum(1)?.clamp(1e-9, f64::MAX)?;
        let mean_embeddings = sum_embeddings.broadcast_div(&mask_sum)?;

        // L2 normalization (critical for cosine similarity)
        let norm = mean_embeddings
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        let normalized = mean_embeddings.broadcast_div(&norm)?;

        Ok(normalized)
    }

    /// Compute mock embedding using semantic-aware hash-based approach
    ///
    /// Creates deterministic 384-dimensional vectors that preserve semantic
    /// properties needed for classification and similarity search.
    fn compute_mock_embedding(&self, text: &str) -> Result<Tensor> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let text_lower = text.to_lowercase();
        let mut embedding = vec![0.0f32; 384];

        // Base hash for random-like distribution
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let base_hash = hasher.finish();

        // Fill with pseudo-random values based on text hash
        for i in 0..384 {
            let mut h = DefaultHasher::new();
            (base_hash, i).hash(&mut h);
            let val = h.finish();
            embedding[i] = ((val % 10000) as f32 / 10000.0) * 2.0 - 1.0;
        }

        // Semantic boosting: adjust specific dimensions based on keywords
        // This makes similar texts have similar embeddings
        // Use strong boosting (1.5) to create clear semantic clusters for mock mode

        // Task-related keywords (boost dimensions 0-50)
        let task_keywords = ["remind", "todo", "task", "need to", "schedule", "deadline", "call", "buy", "complete", "don't forget", "pick up", "book", "renew"];
        let task_score: f32 = task_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 1.5;
        for i in 0..50 {
            embedding[i] += task_score;
        }

        // Memory/fact keywords (boost dimensions 50-100)
        let memory_keywords = ["is", "name", "number", "fact", "information", "capital", "phone", "password", "email", "birthday", "founded", "runs on", "api", "key", "deadline"];
        let memory_score: f32 = memory_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 1.5;
        for i in 50..100 {
            embedding[i] += memory_score;
        }

        // Preference keywords (boost dimensions 100-150)
        let pref_keywords = ["prefer", "favorite", "like", "enjoy", "love", "hate", "want", "dark mode", "early", "video call"];
        let pref_score: f32 = pref_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 1.5;
        for i in 100..150 {
            embedding[i] += pref_score;
        }

        // Relationship keywords (boost dimensions 150-200)
        let rel_keywords = ["colleague", "friend", "family", "works at", "knows", "is my", "partner", "team lead", "manager", "mentor", "cto", "department"];
        let rel_score: f32 = rel_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 1.5;
        for i in 150..200 {
            embedding[i] += rel_score;
        }

        // Event keywords (boost dimensions 200-250)
        let event_keywords = ["meeting", "appointment", "event", "tomorrow", "next week", "calendar", "party", "every monday", "all-hands", "review", "launch", "holiday"];
        let event_score: f32 = event_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 1.5;
        for i in 200..250 {
            embedding[i] += event_score;
        }

        // Query keywords (boost dimensions 250-300) - strong boost for clear queries
        let query_keywords = ["what", "who", "where", "when", "how", "find", "search", "show", "list", "?", "tell me", "about"];
        let query_score: f32 = query_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 2.0;  // Extra strong for queries
        for i in 250..300 {
            embedding[i] += query_score;
        }

        // Store keywords (boost dimensions 300-350) - declarative statements default to store
        let store_keywords = ["remember", "save", "store", "add", "note", "record", "my name is", "i am", "my", "i prefer", "i like", "'s"];
        let store_score: f32 = store_keywords.iter()
            .filter(|kw| text_lower.contains(*kw))
            .count() as f32 * 2.0;  // Extra strong for stores
        for i in 300..350 {
            embedding[i] += store_score;
        }

        // Additional heuristic: statements without query markers default to store
        let has_query_marker = query_keywords.iter().any(|kw| text_lower.contains(*kw));
        if !has_query_marker && text_lower.len() > 10 {
            // Boost store dimensions for declarative statements
            for i in 300..350 {
                embedding[i] += 1.0;
            }
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in &mut embedding {
            *v /= norm;
        }

        Ok(Tensor::new(&embedding[..], &self.device)?.unsqueeze(0)?)
    }

    /// Returns cache statistics (hits can be inferred by caller)
    pub fn cache_size(&self) -> usize {
        self.embedding_cache.len()
    }

    /// Clears the embedding cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
        self.cache_order.clear();
    }

    /// Classifies the full intent of an input embedding vector
    ///
    /// Returns both the action (Store/Query) and data type (Task/Memory/Preference/Relationship/Event)
    /// by comparing against pre-computed anchor vectors.
    /// Also returns confidence scores for cloud-ready stateless operation.
    pub fn classify(&self, input_vec: &Tensor) -> Result<Intent> {
        // Determine action: Store or Query?
        let score_store = input_vec
            .mul(&self.anchor_store)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let score_query = input_vec
            .mul(&self.anchor_query)?
            .sum_all()?
            .to_scalar::<f32>()?;

        let (action, action_confidence) = if score_query > score_store {
            (Action::Query, (score_query - score_store).abs())
        } else {
            (Action::Store, (score_store - score_query).abs())
        };

        // Determine data type by comparing against all category anchors
        let scores = [
            (DataType::Task, input_vec.mul(&self.anchor_task)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Memory, input_vec.mul(&self.anchor_memory)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Preference, input_vec.mul(&self.anchor_preference)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Relationship, input_vec.mul(&self.anchor_relationship)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Event, input_vec.mul(&self.anchor_event)?.sum_all()?.to_scalar::<f32>()?),
        ];

        // Find the highest scoring data type
        let (best_type, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(dt, s)| (*dt, *s))
            .unwrap_or((DataType::Memory, 0.0));

        // Find the second highest score for confidence calculation
        let second_best_score = scores
            .iter()
            .filter(|(dt, _)| *dt != best_type)
            .map(|(_, s)| *s)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let data_type_confidence = (best_score - second_best_score).abs();

        Ok(Intent {
            action,
            data_type: best_type,
            action_confidence,
            data_type_confidence,
        })
    }

    /// Classifies only the data type - returns the best matching category
    pub fn classify_data_type(&self, input_vec: &Tensor) -> Result<DataType> {
        let scores = [
            (DataType::Task, input_vec.mul(&self.anchor_task)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Memory, input_vec.mul(&self.anchor_memory)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Preference, input_vec.mul(&self.anchor_preference)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Relationship, input_vec.mul(&self.anchor_relationship)?.sum_all()?.to_scalar::<f32>()?),
            (DataType::Event, input_vec.mul(&self.anchor_event)?.sum_all()?.to_scalar::<f32>()?),
        ];

        let (best_type, _) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(dt, s)| (*dt, *s))
            .unwrap_or((DataType::Memory, 0.0));

        Ok(best_type)
    }

    /// Classifies only the action (Store/Query)
    pub fn classify_action(&self, input_vec: &Tensor) -> Result<Action> {
        let score_store = input_vec
            .mul(&self.anchor_store)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let score_query = input_vec
            .mul(&self.anchor_query)?
            .sum_all()?
            .to_scalar::<f32>()?;

        if score_query > score_store {
            Ok(Action::Query)
        } else {
            Ok(Action::Store)
        }
    }

    /// Returns the embedding dimension (384 for MiniLM)
    pub fn embedding_dim(&self) -> usize {
        384
    }

    /// Generates embeddings for multiple texts in batch
    ///
    /// More efficient than calling embed() multiple times for bulk operations.
    /// Returns vectors in the same order as input texts.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.embed(text)?;
            let values: Vec<f32> = embedding.squeeze(0)?.to_vec1()?;
            results.push(values);
        }

        Ok(results)
    }

    /// Generates embeddings and returns as raw f32 vectors (skips tensor creation for results)
    pub fn embed_to_vec(&mut self, text: &str) -> Result<Vec<f32>> {
        let embedding = self.embed(text)?;
        let values: Vec<f32> = embedding.squeeze(0)?.to_vec1()?;
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_brain_initialization() {
        let brain = Brain::new().expect("Brain should initialize");
        assert_eq!(brain.embedding_dim(), 384);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedding_shape() {
        let mut brain = Brain::new().expect("Brain should initialize");
        let embedding = brain.embed("Hello world").expect("Embedding should succeed");
        let dims = embedding.dims();
        assert_eq!(dims, &[1, 384]);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_store_task_classification() {
        let mut brain = Brain::new().expect("Brain should initialize");
        let embedding = brain
            .embed("Remind me to buy groceries tomorrow")
            .expect("Embedding should succeed");
        let intent = brain.classify(&embedding).expect("Classification should succeed");
        assert_eq!(intent.action, Action::Store);
        assert_eq!(intent.data_type, DataType::Task);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_store_memory_classification() {
        let mut brain = Brain::new().expect("Brain should initialize");
        let embedding = brain
            .embed("My favorite programming language is Rust")
            .expect("Embedding should succeed");
        let intent = brain.classify(&embedding).expect("Classification should succeed");
        assert_eq!(intent.action, Action::Store);
        assert_eq!(intent.data_type, DataType::Memory);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_query_classification() {
        let mut brain = Brain::new().expect("Brain should initialize");
        let embedding = brain
            .embed("What is my favorite color?")
            .expect("Embedding should succeed");
        let intent = brain.classify(&embedding).expect("Classification should succeed");
        assert_eq!(intent.action, Action::Query);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_search_classification() {
        let mut brain = Brain::new().expect("Brain should initialize");
        let embedding = brain
            .embed("Find all my tasks for today")
            .expect("Embedding should succeed");
        let intent = brain.classify(&embedding).expect("Classification should succeed");
        assert_eq!(intent.action, Action::Query);
    }
}
