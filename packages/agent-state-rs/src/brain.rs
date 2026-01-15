//! Intelligence Layer - Handles AI model loading, embedding generation, and intent classification
//!
//! This module implements zero-shot intent routing using sentence transformer models.
//! It classifies input text by action (Store vs Query) and by data type (Task vs Memory)
//! by comparing embeddings against pre-computed anchor vectors.
//!
//! ## Supported Models
//!
//! | Model | Dimensions | MTEB Score | Notes |
//! |-------|------------|------------|-------|
//! | `MiniLM-L6` | 384 | 56.3 | Fast, decent accuracy (default) |
//! | `MiniLM-L12` | 384 | 59.8 | Better accuracy, same dims |
//! | `BGE-Small` | 384 | 62.2 | Best small model |
//! | `BGE-Base` | 768 | 64.2 | Best accuracy/size ratio |
//!
//! ## Backends
//!
//! - **Candle** (default): Pure Rust, no external dependencies
//! - **ONNX Runtime**: Faster inference, better CPU optimization

use anyhow::{Context, Result};
use std::collections::{HashMap, VecDeque};
use tokenizers::Tokenizer;

#[cfg(feature = "candle-backend")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle-backend")]
use candle_nn::VarBuilder;
#[cfg(feature = "candle-backend")]
use candle_transformers::models::bert::{BertModel, Config};

#[cfg(feature = "onnx-backend")]
use ort::{GraphOptimizationLevel, Session};

use hf_hub::{api::sync::Api, Repo, RepoType};

/// Maximum number of embeddings to cache (LRU eviction when exceeded)
const EMBEDDING_CACHE_SIZE: usize = 1000;

/// Available embedding models with their HuggingFace repo names
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModel {
    /// MiniLM-L6-v2: Fast, 384 dimensions, MTEB 56.3
    MiniLmL6,
    /// MiniLM-L12-v2: Better accuracy, 384 dimensions, MTEB 59.8
    MiniLmL12,
    /// BGE-Small-EN-v1.5: Best small model, 384 dimensions, MTEB 62.2
    BgeSmall,
    /// BGE-Base-EN-v1.5: Best accuracy/size, 768 dimensions, MTEB 64.2
    BgeBase,
    /// E5-Small-v2: Microsoft model, 384 dimensions, MTEB 61.5
    E5Small,
    /// GTE-Small: Alibaba model, 384 dimensions, MTEB 61.4
    GteSmall,
}

impl EmbeddingModel {
    /// Returns the HuggingFace repository name for this model
    pub fn hf_repo(&self) -> &'static str {
        match self {
            EmbeddingModel::MiniLmL6 => "sentence-transformers/all-MiniLM-L6-v2",
            EmbeddingModel::MiniLmL12 => "sentence-transformers/all-MiniLM-L12-v2",
            EmbeddingModel::BgeSmall => "BAAI/bge-small-en-v1.5",
            EmbeddingModel::BgeBase => "BAAI/bge-base-en-v1.5",
            EmbeddingModel::E5Small => "intfloat/e5-small-v2",
            EmbeddingModel::GteSmall => "thenlper/gte-small",
        }
    }

    /// Returns the embedding dimension for this model
    pub fn embedding_dim(&self) -> usize {
        match self {
            EmbeddingModel::MiniLmL6 => 384,
            EmbeddingModel::MiniLmL12 => 384,
            EmbeddingModel::BgeSmall => 384,
            EmbeddingModel::BgeBase => 768,
            EmbeddingModel::E5Small => 384,
            EmbeddingModel::GteSmall => 384,
        }
    }

    /// Returns the ONNX model filename if available
    pub fn onnx_filename(&self) -> &'static str {
        "model.onnx"
    }

    /// Returns true if this model requires query prefixing (BGE, E5 models)
    pub fn needs_query_prefix(&self) -> bool {
        matches!(self, EmbeddingModel::BgeSmall | EmbeddingModel::BgeBase | EmbeddingModel::E5Small)
    }

    /// Returns the query prefix for models that need it
    pub fn query_prefix(&self) -> &'static str {
        match self {
            EmbeddingModel::BgeSmall | EmbeddingModel::BgeBase => "Represent this sentence for searching relevant passages: ",
            EmbeddingModel::E5Small => "query: ",
            _ => "",
        }
    }

    /// Returns the MTEB score for this model (higher is better)
    pub fn mteb_score(&self) -> f32 {
        match self {
            EmbeddingModel::MiniLmL6 => 56.3,
            EmbeddingModel::MiniLmL12 => 59.8,
            EmbeddingModel::BgeSmall => 62.2,
            EmbeddingModel::BgeBase => 64.2,
            EmbeddingModel::E5Small => 61.5,
            EmbeddingModel::GteSmall => 61.4,
        }
    }
}

impl Default for EmbeddingModel {
    fn default() -> Self {
        // Default to BGE-Small for better accuracy with same dimensions
        EmbeddingModel::BgeSmall
    }
}

/// Backend for embedding computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Candle (pure Rust, default)
    Candle,
    /// ONNX Runtime (faster, better CPU optimization)
    Onnx,
    /// Mock mode (for testing without models)
    Mock,
}

impl Default for Backend {
    fn default() -> Self {
        #[cfg(feature = "onnx-backend")]
        return Backend::Onnx;
        #[cfg(not(feature = "onnx-backend"))]
        return Backend::Candle;
    }
}

/// Configuration for Brain initialization
#[derive(Debug, Clone)]
pub struct BrainConfig {
    /// Which embedding model to use
    pub model: EmbeddingModel,
    /// Which backend to use for inference
    pub backend: Backend,
    /// Optional local model directory (bypasses HuggingFace download)
    pub local_model_dir: Option<std::path::PathBuf>,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::default(),
            backend: Backend::default(),
            local_model_dir: None,
        }
    }
}

impl BrainConfig {
    /// Create config for the fastest model (MiniLM-L6)
    pub fn fast() -> Self {
        Self {
            model: EmbeddingModel::MiniLmL6,
            ..Default::default()
        }
    }

    /// Create config for the most accurate model (BGE-Base)
    pub fn accurate() -> Self {
        Self {
            model: EmbeddingModel::BgeBase,
            ..Default::default()
        }
    }

    /// Create config for balanced performance (BGE-Small)
    pub fn balanced() -> Self {
        Self {
            model: EmbeddingModel::BgeSmall,
            ..Default::default()
        }
    }

    /// Create config for testing (mock mode)
    pub fn mock() -> Self {
        Self {
            model: EmbeddingModel::MiniLmL6, // doesn't matter for mock
            backend: Backend::Mock,
            local_model_dir: None,
        }
    }
}

/// The Brain handles all AI-related operations including embedding generation and intent classification
pub struct Brain {
    /// Configuration used to create this brain
    config: BrainConfig,
    /// Tokenizer (shared between backends)
    tokenizer: Option<Tokenizer>,
    /// Candle model (if using Candle backend)
    #[cfg(feature = "candle-backend")]
    candle_model: Option<BertModel>,
    #[cfg(feature = "candle-backend")]
    candle_device: Device,
    /// ONNX session (if using ONNX backend)
    #[cfg(feature = "onnx-backend")]
    onnx_session: Option<Session>,
    /// Anchor vectors for classification (stored as Vec<f32> for flexibility)
    anchor_task: Vec<f32>,
    anchor_memory: Vec<f32>,
    anchor_preference: Vec<f32>,
    anchor_relationship: Vec<f32>,
    anchor_event: Vec<f32>,
    anchor_store: Vec<f32>,
    anchor_query: Vec<f32>,
    /// Cache for computed embeddings (text -> embedding vector as f32 array)
    embedding_cache: HashMap<String, Vec<f32>>,
    /// Order of cache insertions for LRU eviction
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    /// Action items, reminders, todos
    Task,
    /// General facts and information
    Memory,
    /// User preferences and likes/dislikes
    Preference,
    /// Relationships between entities
    Relationship,
    /// Time-based events and appointments
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
    pub fn is_ambiguous(&self) -> bool {
        self.action_confidence < 0.15 || self.data_type_confidence < 0.15
    }

    /// Returns the overall confidence (minimum of both confidences)
    pub fn overall_confidence(&self) -> f32 {
        self.action_confidence.min(self.data_type_confidence)
    }

    /// Returns a clarification message if the intent is ambiguous
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
    /// Creates a new Brain instance with default configuration
    ///
    /// Uses BGE-Small model with the best available backend.
    pub fn new() -> Result<Self> {
        Self::with_config(BrainConfig::default())
    }

    /// Creates a new Brain instance with custom configuration
    pub fn with_config(config: BrainConfig) -> Result<Self> {
        match config.backend {
            Backend::Mock => Self::new_mock_internal(config),
            #[cfg(feature = "candle-backend")]
            Backend::Candle => Self::new_candle(config),
            #[cfg(feature = "onnx-backend")]
            Backend::Onnx => Self::new_onnx(config),
            #[cfg(not(feature = "candle-backend"))]
            Backend::Candle => anyhow::bail!("Candle backend not enabled. Compile with --features candle-backend"),
            #[cfg(not(feature = "onnx-backend"))]
            Backend::Onnx => anyhow::bail!("ONNX backend not enabled. Compile with --features onnx-backend"),
        }
    }

    /// Creates a new Brain in mock mode for testing
    pub fn new_mock() -> Result<Self> {
        Self::with_config(BrainConfig::mock())
    }

    fn new_mock_internal(config: BrainConfig) -> Result<Self> {
        let dim = config.model.embedding_dim();
        let placeholder = vec![0.0f32; dim];

        let mut brain = Self {
            config,
            tokenizer: None,
            #[cfg(feature = "candle-backend")]
            candle_model: None,
            #[cfg(feature = "candle-backend")]
            candle_device: Device::Cpu,
            #[cfg(feature = "onnx-backend")]
            onnx_session: None,
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

        // Initialize anchors with mock embeddings
        brain.initialize_anchors()?;
        Ok(brain)
    }

    #[cfg(feature = "candle-backend")]
    fn new_candle(config: BrainConfig) -> Result<Self> {
        let device = Device::Cpu;
        let dim = config.model.embedding_dim();

        // Get model files
        let (config_path, tokenizer_path, weights_path) = if let Some(ref local_dir) = config.local_model_dir {
            (
                local_dir.join("config.json"),
                local_dir.join("tokenizer.json"),
                local_dir.join("model.safetensors"),
            )
        } else {
            // Try local models directory first
            let local_model_dir = std::path::Path::new("models").join(match config.model {
                EmbeddingModel::MiniLmL6 => "minilm-l6",
                EmbeddingModel::MiniLmL12 => "minilm-l12",
                EmbeddingModel::BgeSmall => "bge-small",
                EmbeddingModel::BgeBase => "bge-base",
                EmbeddingModel::E5Small => "e5-small",
                EmbeddingModel::GteSmall => "gte-small",
            });

            if local_model_dir.exists() {
                (
                    local_model_dir.join("config.json"),
                    local_model_dir.join("tokenizer.json"),
                    local_model_dir.join("model.safetensors"),
                )
            } else {
                // Download from HuggingFace
                let api = Api::new().context("Failed to initialize HuggingFace API")?;
                let repo = api.repo(Repo::new(config.model.hf_repo().to_string(), RepoType::Model));

                (
                    repo.get("config.json").context("Failed to download config.json")?,
                    repo.get("tokenizer.json").context("Failed to download tokenizer.json")?,
                    repo.get("model.safetensors").context("Failed to download model.safetensors")?,
                )
            }
        };

        // Parse config
        let bert_config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path).context("Failed to read config.json")?,
        )
        .context("Failed to parse config.json")?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
                .context("Failed to load model weights")?
        };
        let model = BertModel::load(vb, &bert_config).context("Failed to initialize BERT model")?;

        let placeholder = vec![0.0f32; dim];

        let mut brain = Self {
            config,
            tokenizer: Some(tokenizer),
            candle_model: Some(model),
            candle_device: device,
            #[cfg(feature = "onnx-backend")]
            onnx_session: None,
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

        brain.initialize_anchors()?;
        Ok(brain)
    }

    #[cfg(feature = "onnx-backend")]
    fn new_onnx(config: BrainConfig) -> Result<Self> {
        let dim = config.model.embedding_dim();

        // Get model files
        let (tokenizer_path, onnx_path) = if let Some(ref local_dir) = config.local_model_dir {
            (
                local_dir.join("tokenizer.json"),
                local_dir.join("model.onnx"),
            )
        } else {
            // Try local models directory first
            let local_model_dir = std::path::Path::new("models").join(match config.model {
                EmbeddingModel::MiniLmL6 => "minilm-l6",
                EmbeddingModel::MiniLmL12 => "minilm-l12",
                EmbeddingModel::BgeSmall => "bge-small",
                EmbeddingModel::BgeBase => "bge-base",
                EmbeddingModel::E5Small => "e5-small",
                EmbeddingModel::GteSmall => "gte-small",
            });

            if local_model_dir.join("model.onnx").exists() {
                (
                    local_model_dir.join("tokenizer.json"),
                    local_model_dir.join("model.onnx"),
                )
            } else {
                // Download from HuggingFace
                let api = Api::new().context("Failed to initialize HuggingFace API")?;
                let repo = api.repo(Repo::new(config.model.hf_repo().to_string(), RepoType::Model));

                // Try to get ONNX model, fall back to regular model
                let onnx_path = repo.get("onnx/model.onnx")
                    .or_else(|_| repo.get("model.onnx"))
                    .context("Failed to download ONNX model. Model may not have ONNX export.")?;

                (
                    repo.get("tokenizer.json").context("Failed to download tokenizer.json")?,
                    onnx_path,
                )
            }
        };

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Initialize ONNX Runtime session with optimizations
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&onnx_path)
            .context("Failed to load ONNX model")?;

        let placeholder = vec![0.0f32; dim];

        let mut brain = Self {
            config,
            tokenizer: Some(tokenizer),
            #[cfg(feature = "candle-backend")]
            candle_model: None,
            #[cfg(feature = "candle-backend")]
            candle_device: Device::Cpu,
            onnx_session: Some(session),
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

        brain.initialize_anchors()?;
        Ok(brain)
    }

    /// Initialize anchor vectors for classification
    fn initialize_anchors(&mut self) -> Result<()> {
        // Data type anchors
        self.anchor_task = self.embed_to_vec(
            "action item, todo, remind me to, deadline, schedule, need to do, task, must complete",
        )?;
        self.anchor_memory = self.embed_to_vec(
            "fact, information, context, background, remember that, note, detail about, knowledge",
        )?;
        self.anchor_preference = self.embed_to_vec(
            "I prefer, I like, my favorite, I enjoy, I want, preference, likes, dislikes, choose",
        )?;
        self.anchor_relationship = self.embed_to_vec(
            "is my, works at, knows, colleague, friend, family, partner, relationship, connected to",
        )?;
        self.anchor_event = self.embed_to_vec(
            "meeting, appointment, event, happening, tomorrow, next week, on date, scheduled for, calendar",
        )?;

        // Action anchors
        self.anchor_store = self.embed_to_vec(
            "save this, remember, store, add, note that, record, keep track of, my name is, I like",
        )?;
        self.anchor_query = self.embed_to_vec(
            "what is, who is, find, search, look up, tell me about, retrieve, get, show me, list",
        )?;

        Ok(())
    }

    /// Returns whether this Brain is running in mock mode
    pub fn is_mock(&self) -> bool {
        self.config.backend == Backend::Mock
    }

    /// Returns the configured model
    pub fn model(&self) -> EmbeddingModel {
        self.config.model
    }

    /// Returns the configured backend
    pub fn backend(&self) -> Backend {
        self.config.backend
    }

    /// Returns the embedding dimension for the configured model
    pub fn embedding_dim(&self) -> usize {
        self.config.model.embedding_dim()
    }

    /// Converts text into a normalized embedding vector
    pub fn embed_to_vec(&mut self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        if let Some(cached) = self.embedding_cache.get(text) {
            return Ok(cached.clone());
        }

        // Compute embedding based on backend
        let embedding = self.compute_embedding(text)?;

        // Cache with LRU eviction
        if self.cache_order.len() >= EMBEDDING_CACHE_SIZE {
            if let Some(oldest) = self.cache_order.pop_front() {
                self.embedding_cache.remove(&oldest);
            }
        }
        self.embedding_cache.insert(text.to_string(), embedding.clone());
        self.cache_order.push_back(text.to_string());

        Ok(embedding)
    }

    /// Converts text into a Candle tensor (for backward compatibility)
    #[cfg(feature = "candle-backend")]
    pub fn embed(&mut self, text: &str) -> Result<Tensor> {
        let vec = self.embed_to_vec(text)?;
        let tensor = Tensor::new(&vec[..], &self.candle_device)?.unsqueeze(0)?;
        Ok(tensor)
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed_to_vec(text)
    }

    fn compute_embedding(&self, text: &str) -> Result<Vec<f32>> {
        match self.config.backend {
            Backend::Mock => self.compute_mock_embedding(text),
            #[cfg(feature = "candle-backend")]
            Backend::Candle => self.compute_candle_embedding(text),
            #[cfg(feature = "onnx-backend")]
            Backend::Onnx => self.compute_onnx_embedding(text),
            #[cfg(not(feature = "candle-backend"))]
            Backend::Candle => anyhow::bail!("Candle backend not enabled"),
            #[cfg(not(feature = "onnx-backend"))]
            Backend::Onnx => anyhow::bail!("ONNX backend not enabled"),
        }
    }

    #[cfg(feature = "candle-backend")]
    fn compute_candle_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not available"))?;
        let model = self.candle_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not available"))?;

        // Tokenize input text
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids = tokens.get_ids();
        let token_type_ids = vec![0u32; token_ids.len()];
        let attention_mask = tokens.get_attention_mask();

        // Convert to tensors
        let token_ids_tensor = Tensor::new(token_ids, &self.candle_device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(token_type_ids.as_slice(), &self.candle_device)?.unsqueeze(0)?;

        // Run inference
        let embeddings = model
            .forward(&token_ids_tensor, &token_type_ids_tensor, None)
            .context("Model forward pass failed")?;

        // Mean pooling with attention mask
        let attention_mask_tensor =
            Tensor::new(attention_mask, &self.candle_device)?.unsqueeze(0)?.unsqueeze(2)?;
        let attention_mask_f32 = attention_mask_tensor.to_dtype(DType::F32)?;

        let masked_embeddings = embeddings.broadcast_mul(&attention_mask_f32)?;
        let sum_embeddings = masked_embeddings.sum(1)?;
        let mask_sum = attention_mask_f32.sum(1)?.clamp(1e-9, f64::MAX)?;
        let mean_embeddings = sum_embeddings.broadcast_div(&mask_sum)?;

        // L2 normalize
        let norm = mean_embeddings
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::MAX)?;
        let normalized = mean_embeddings.broadcast_div(&norm)?;

        // Convert to Vec<f32>
        let values: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
        Ok(values)
    }

    #[cfg(feature = "onnx-backend")]
    fn compute_onnx_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Tokenizer not available"))?;
        let session = self.onnx_session.as_ref()
            .ok_or_else(|| anyhow::anyhow!("ONNX session not available"))?;

        // Tokenize
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids: Vec<i64> = tokens.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = tokens.get_attention_mask().iter().map(|&x| x as i64).collect();
        let token_type_ids: Vec<i64> = vec![0i64; token_ids.len()];

        let seq_len = token_ids.len();

        // Create ONNX input arrays
        let input_ids = ndarray::Array2::from_shape_vec((1, seq_len), token_ids)?;
        let attention = ndarray::Array2::from_shape_vec((1, seq_len), attention_mask.clone())?;
        let type_ids = ndarray::Array2::from_shape_vec((1, seq_len), token_type_ids)?;

        // Run inference
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention,
            "token_type_ids" => type_ids,
        ]?)?;

        // Get output tensor - try different common output names
        let output = outputs.get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .or_else(|| outputs.get("sentence_embedding"))
            .ok_or_else(|| anyhow::anyhow!("Could not find output tensor"))?;

        let output_tensor = output.try_extract_tensor::<f32>()?;
        let output_view = output_tensor.view();

        // Mean pooling
        let dim = self.config.model.embedding_dim();
        let mut pooled = vec![0.0f32; dim];
        let mut count = 0.0f32;

        for i in 0..seq_len {
            if attention_mask[i] == 1 {
                for j in 0..dim {
                    pooled[j] += output_view[[0, i, j]];
                }
                count += 1.0;
            }
        }

        if count > 0.0 {
            for v in &mut pooled {
                *v /= count;
            }
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in &mut pooled {
            *v /= norm;
        }

        Ok(pooled)
    }

    /// Compute mock embedding using semantic-aware hash-based approach
    fn compute_mock_embedding(&self, text: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let dim = self.config.model.embedding_dim();
        let text_lower = text.to_lowercase();
        let mut embedding = vec![0.0f32; dim];

        // Base hash for random-like distribution
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let base_hash = hasher.finish();

        // Fill with pseudo-random values
        for i in 0..dim {
            let mut h = DefaultHasher::new();
            (base_hash, i).hash(&mut h);
            let val = h.finish();
            embedding[i] = ((val % 10000) as f32 / 10000.0) * 2.0 - 1.0;
        }

        // Semantic boosting based on dimension ranges (scaled for different dim sizes)
        let scale = dim as f32 / 384.0;
        let range_size = (50.0 * scale) as usize;

        // Task keywords
        let task_keywords = ["remind", "todo", "task", "need to", "schedule", "deadline", "call", "buy", "complete"];
        let task_score: f32 = task_keywords.iter().filter(|kw| text_lower.contains(*kw)).count() as f32 * 1.5;
        for i in 0..range_size.min(dim) {
            embedding[i] += task_score;
        }

        // Memory keywords
        let memory_keywords = ["is", "name", "number", "fact", "information", "capital", "phone", "birthday"];
        let memory_score: f32 = memory_keywords.iter().filter(|kw| text_lower.contains(*kw)).count() as f32 * 1.5;
        for i in range_size..(2 * range_size).min(dim) {
            embedding[i] += memory_score;
        }

        // Preference keywords
        let pref_keywords = ["prefer", "favorite", "like", "enjoy", "love", "hate", "want"];
        let pref_score: f32 = pref_keywords.iter().filter(|kw| text_lower.contains(*kw)).count() as f32 * 1.5;
        for i in (2 * range_size)..(3 * range_size).min(dim) {
            embedding[i] += pref_score;
        }

        // Query keywords
        let query_keywords = ["what", "who", "where", "when", "how", "find", "search", "show", "list", "?"];
        let query_score: f32 = query_keywords.iter().filter(|kw| text_lower.contains(*kw)).count() as f32 * 2.0;
        for i in (5 * range_size)..(6 * range_size).min(dim) {
            embedding[i] += query_score;
        }

        // Store keywords
        let store_keywords = ["remember", "save", "store", "add", "note", "record", "my name is", "i am", "my"];
        let store_score: f32 = store_keywords.iter().filter(|kw| text_lower.contains(*kw)).count() as f32 * 2.0;
        for i in (6 * range_size)..(7 * range_size).min(dim) {
            embedding[i] += store_score;
        }

        // Declarative statements default to store
        let has_query_marker = query_keywords.iter().any(|kw| text_lower.contains(*kw));
        if !has_query_marker && text_lower.len() > 10 {
            for i in (6 * range_size)..(7 * range_size).min(dim) {
                embedding[i] += 1.0;
            }
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for v in &mut embedding {
            *v /= norm;
        }

        Ok(embedding)
    }

    /// Returns cache statistics
    pub fn cache_size(&self) -> usize {
        self.embedding_cache.len()
    }

    /// Clears the embedding cache
    pub fn clear_cache(&mut self) {
        self.embedding_cache.clear();
        self.cache_order.clear();
    }

    /// Cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Classifies the full intent of an input text
    pub fn classify_text(&mut self, text: &str) -> Result<Intent> {
        let input_vec = self.embed_to_vec(text)?;
        self.classify_vec(&input_vec)
    }

    /// Classifies the full intent from a pre-computed embedding vector
    pub fn classify_vec(&self, input_vec: &[f32]) -> Result<Intent> {
        // Determine action
        let score_store = Self::cosine_similarity(input_vec, &self.anchor_store);
        let score_query = Self::cosine_similarity(input_vec, &self.anchor_query);

        let (action, action_confidence) = if score_query > score_store {
            (Action::Query, (score_query - score_store).abs())
        } else {
            (Action::Store, (score_store - score_query).abs())
        };

        // Determine data type
        let scores = [
            (DataType::Task, Self::cosine_similarity(input_vec, &self.anchor_task)),
            (DataType::Memory, Self::cosine_similarity(input_vec, &self.anchor_memory)),
            (DataType::Preference, Self::cosine_similarity(input_vec, &self.anchor_preference)),
            (DataType::Relationship, Self::cosine_similarity(input_vec, &self.anchor_relationship)),
            (DataType::Event, Self::cosine_similarity(input_vec, &self.anchor_event)),
        ];

        let (best_type, best_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(dt, s)| (*dt, *s))
            .unwrap_or((DataType::Memory, 0.0));

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

    /// Backward compatibility: classify from tensor
    #[cfg(feature = "candle-backend")]
    pub fn classify(&self, input_tensor: &Tensor) -> Result<Intent> {
        let values: Vec<f32> = input_tensor.squeeze(0)?.to_vec1()?;
        self.classify_vec(&values)
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn classify(&self, input_vec: &[f32]) -> Result<Intent> {
        self.classify_vec(input_vec)
    }

    /// Classifies only the data type from text
    pub fn classify_data_type_text(&mut self, text: &str) -> Result<DataType> {
        let vec = self.embed_to_vec(text)?;
        self.classify_data_type_vec(&vec)
    }

    /// Classifies only the data type from a vector
    pub fn classify_data_type_vec(&self, input_vec: &[f32]) -> Result<DataType> {
        let scores = [
            (DataType::Task, Self::cosine_similarity(input_vec, &self.anchor_task)),
            (DataType::Memory, Self::cosine_similarity(input_vec, &self.anchor_memory)),
            (DataType::Preference, Self::cosine_similarity(input_vec, &self.anchor_preference)),
            (DataType::Relationship, Self::cosine_similarity(input_vec, &self.anchor_relationship)),
            (DataType::Event, Self::cosine_similarity(input_vec, &self.anchor_event)),
        ];

        let (best_type, _) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(dt, s)| (*dt, *s))
            .unwrap_or((DataType::Memory, 0.0));

        Ok(best_type)
    }

    /// Backward compatibility
    #[cfg(feature = "candle-backend")]
    pub fn classify_data_type(&self, input_tensor: &Tensor) -> Result<DataType> {
        let values: Vec<f32> = input_tensor.squeeze(0)?.to_vec1()?;
        self.classify_data_type_vec(&values)
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn classify_data_type(&self, input_vec: &[f32]) -> Result<DataType> {
        self.classify_data_type_vec(input_vec)
    }

    /// Classifies only the action from text
    pub fn classify_action_text(&mut self, text: &str) -> Result<Action> {
        let vec = self.embed_to_vec(text)?;
        self.classify_action_vec(&vec)
    }

    /// Classifies only the action from a vector
    pub fn classify_action_vec(&self, input_vec: &[f32]) -> Result<Action> {
        let score_store = Self::cosine_similarity(input_vec, &self.anchor_store);
        let score_query = Self::cosine_similarity(input_vec, &self.anchor_query);

        if score_query > score_store {
            Ok(Action::Query)
        } else {
            Ok(Action::Store)
        }
    }

    /// Backward compatibility
    #[cfg(feature = "candle-backend")]
    pub fn classify_action(&self, input_tensor: &Tensor) -> Result<Action> {
        let values: Vec<f32> = input_tensor.squeeze(0)?.to_vec1()?;
        self.classify_action_vec(&values)
    }

    #[cfg(not(feature = "candle-backend"))]
    pub fn classify_action(&self, input_vec: &[f32]) -> Result<Action> {
        self.classify_action_vec(input_vec)
    }

    /// Generates embeddings for multiple texts in batch
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed_to_vec(text)?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_brain() {
        let mut brain = Brain::new_mock().expect("Mock brain should initialize");
        assert!(brain.is_mock());
        assert_eq!(brain.embedding_dim(), 384);
    }

    #[test]
    fn test_mock_embedding() {
        let mut brain = Brain::new_mock().expect("Mock brain should initialize");
        let embedding = brain.embed_to_vec("Hello world").expect("Embedding should succeed");
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_mock_classification() {
        let mut brain = Brain::new_mock().expect("Mock brain should initialize");

        // Test query detection
        let intent = brain.classify_text("What is my name?").expect("Classification should succeed");
        assert_eq!(intent.action, Action::Query);

        // Test store detection
        let intent = brain.classify_text("My name is Alice").expect("Classification should succeed");
        assert_eq!(intent.action, Action::Store);
    }

    #[test]
    fn test_embedding_cache() {
        let mut brain = Brain::new_mock().expect("Mock brain should initialize");

        // First call computes
        let _ = brain.embed_to_vec("test text").expect("Should work");
        assert_eq!(brain.cache_size(), 8); // 7 anchors + 1 new

        // Second call uses cache
        let _ = brain.embed_to_vec("test text").expect("Should work");
        assert_eq!(brain.cache_size(), 8); // Still 8, cache hit
    }

    #[test]
    fn test_model_configs() {
        assert_eq!(EmbeddingModel::MiniLmL6.embedding_dim(), 384);
        assert_eq!(EmbeddingModel::BgeBase.embedding_dim(), 768);
        assert!(EmbeddingModel::BgeBase.mteb_score() > EmbeddingModel::MiniLmL6.mteb_score());
    }
}
