//! Nemotron streaming speech recognition engine.
//!
//! This module wraps the [`parakeet-rs`](https://crates.io/crates/parakeet-rs)
//! `Nemotron` model to implement [`StreamingTranscriptionEngine`] for
//! cache-aware, chunk-based transcription.
//!
//! # Requirements
//!
//! The model directory must contain:
//! - `encoder.onnx` + `encoder.onnx.data` — encoder weights
//! - `decoder_joint.onnx` — decoder/joint network
//! - `tokenizer.model` — SentencePiece vocabulary
//!
//! # Examples
//!
//! ```rust,no_run
//! use transcribe_rs::{StreamingTranscriptionEngine, engines::nemotron_streaming::NemotronStreamingEngine};
//! use std::path::PathBuf;
//!
//! let mut engine = NemotronStreamingEngine::new();
//! engine.load_model(&PathBuf::from("models/nemotron-speech-streaming-en-0.6b"))?;
//!
//! // Push 560ms chunks of 16kHz mono f32 audio
//! # let audio_chunks: Vec<Vec<f32>> = vec![];
//! for chunk in &audio_chunks {
//!     let segments = engine.push_samples(chunk)?;
//!     for seg in &segments {
//!         print!("{}", seg.text);
//!         if seg.is_endpoint {
//!             println!(); // newline after each sentence
//!         }
//!     }
//! }
//! println!("\nFinal: {}", engine.get_transcript());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use parakeet_rs::Nemotron;
use std::path::{Path, PathBuf};

use crate::{split_at_sentence_boundaries, StreamingSegment, StreamingTranscriptionEngine};

/// Recommended chunk size in samples (560ms at 16kHz).
pub const CHUNK_SIZE: usize = 8960;

/// Errors produced by [`NemotronStreamingEngine`].
#[derive(thiserror::Error, Debug)]
pub enum NemotronStreamingError {
    #[error("Model not loaded. Call load_model() first.")]
    ModelNotLoaded,
    #[error("Invalid model path: {0}")]
    InvalidPath(String),
    #[error("Parakeet error: {0}")]
    Parakeet(String),
}

/// Streaming transcription engine backed by NVIDIA Nemotron (0.6B).
///
/// Wraps [`parakeet_rs::Nemotron`] and implements [`StreamingTranscriptionEngine`].
/// Audio is fed in fixed-size chunks via [`push_samples`](StreamingTranscriptionEngine::push_samples);
/// the engine maintains internal encoder cache and decoder state across chunks.
pub struct NemotronStreamingEngine {
    model: Option<Nemotron>,
    loaded_model_path: Option<PathBuf>,
}

impl NemotronStreamingEngine {
    /// Create a new engine with no model loaded.
    pub fn new() -> Self {
        Self {
            model: None,
            loaded_model_path: None,
        }
    }
}

impl Default for NemotronStreamingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NemotronStreamingEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl StreamingTranscriptionEngine for NemotronStreamingEngine {
    type ModelParams = ();

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        _params: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path_str = model_path.to_str().ok_or_else(|| {
            NemotronStreamingError::InvalidPath(format!("{}", model_path.display()))
        })?;
        let model = Nemotron::from_pretrained(path_str, None)
            .map_err(|e| NemotronStreamingError::Parakeet(format!("{e}")))?;
        self.model = Some(model);
        self.loaded_model_path = Some(model_path.to_path_buf());
        Ok(())
    }

    fn unload_model(&mut self) {
        self.model = None;
        self.loaded_model_path = None;
    }

    fn push_samples(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<StreamingSegment>, Box<dyn std::error::Error>> {
        let model = self
            .model
            .as_mut()
            .ok_or(NemotronStreamingError::ModelNotLoaded)?;
        let text = model
            .transcribe_chunk(samples)
            .map_err(|e| NemotronStreamingError::Parakeet(format!("{e}")))?;
        if text.is_empty() {
            Ok(vec![])
        } else {
            Ok(split_at_sentence_boundaries(&text))
        }
    }

    fn get_transcript(&self) -> String {
        self.model
            .as_ref()
            .map(|m| m.get_transcript())
            .unwrap_or_default()
    }

    fn reset(&mut self) {
        if let Some(model) = self.model.as_mut() {
            model.reset();
        }
    }
}
