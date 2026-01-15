//! Intelligence Layer - Handles AI model loading, embedding generation, and intent classification
//!
//! This module implements zero-shot intent routing using a MiniLM sentence transformer model.
//! It classifies input text as either a "Task" (action item) or "Memory" (contextual information)
//! by comparing embeddings against pre-computed anchor vectors.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

/// The Brain handles all AI-related operations including embedding generation and intent classification
pub struct Brain {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    /// Anchor vector representing "Task" intent
    anchor_task: Tensor,
    /// Anchor vector representing "Memory" intent
    anchor_memory: Tensor,
}

/// Represents the classified intent of user input
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    /// Action items, reminders, todos - things that need to be done
    Task,
    /// Facts, preferences, context - information to remember
    Memory,
}

impl Brain {
    /// Creates a new Brain instance by loading the MiniLM model from HuggingFace
    ///
    /// This will automatically download and cache the model on first run.
    /// Subsequent runs will use the cached version.
    pub fn new() -> Result<Self> {
        let device = Device::Cpu; // MiniLM is small enough for CPU inference

        // 1. Load Model from HuggingFace Hub (Cached automatically in ~/.cache/huggingface)
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
            model,
            tokenizer,
            device,
            anchor_task: placeholder.clone(),
            anchor_memory: placeholder,
        };

        // 2. Initialize Anchor Vectors (Zero-Shot Classification Logic)
        // These anchor phrases define what "Task" and "Memory" look like in embedding space
        brain.anchor_task = brain.embed(
            "I need to do this, add to todo list, remind me, deadline, schedule, action item",
        )?;
        brain.anchor_memory = brain.embed(
            "This is a fact, user preference, historical context, background info, remember this",
        )?;

        Ok(brain)
    }

    /// Converts text into a normalized 384-dimensional embedding vector
    ///
    /// Uses mean pooling over all token embeddings followed by L2 normalization.
    /// The resulting vector is suitable for cosine similarity comparisons.
    pub fn embed(&self, text: &str) -> Result<Tensor> {
        // Tokenize input text
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let token_ids = tokens.get_ids();
        let token_type_ids = vec![0u32; token_ids.len()];
        let attention_mask = tokens.get_attention_mask();

        // Convert to tensors with batch dimension
        let token_ids_tensor = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let token_type_ids_tensor = Tensor::new(token_type_ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // Run model inference
        let embeddings = self
            .model
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

    /// Classifies the intent of an input embedding vector
    ///
    /// Uses cosine similarity (via dot product of normalized vectors) to compare
    /// the input against pre-computed Task and Memory anchor vectors.
    pub fn classify(&self, input_vec: &Tensor) -> Result<Intent> {
        // Compute dot product (cosine similarity since vectors are normalized)
        let score_task = input_vec
            .mul(&self.anchor_task)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let score_memory = input_vec
            .mul(&self.anchor_memory)?
            .sum_all()?
            .to_scalar::<f32>()?;

        if score_task > score_memory {
            Ok(Intent::Task)
        } else {
            Ok(Intent::Memory)
        }
    }

    /// Returns the embedding dimension (384 for MiniLM)
    pub fn embedding_dim(&self) -> usize {
        384
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
        let brain = Brain::new().expect("Brain should initialize");
        let embedding = brain.embed("Hello world").expect("Embedding should succeed");
        let dims = embedding.dims();
        assert_eq!(dims, &[1, 384]);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_task_classification() {
        let brain = Brain::new().expect("Brain should initialize");
        let embedding = brain
            .embed("Remind me to buy groceries tomorrow")
            .expect("Embedding should succeed");
        let intent = brain.classify(&embedding).expect("Classification should succeed");
        assert_eq!(intent, Intent::Task);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_memory_classification() {
        let brain = Brain::new().expect("Brain should initialize");
        let embedding = brain
            .embed("My favorite programming language is Rust")
            .expect("Embedding should succeed");
        let intent = brain.classify(&embedding).expect("Classification should succeed");
        assert_eq!(intent, Intent::Memory);
    }
}
