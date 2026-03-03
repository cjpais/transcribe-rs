//! Qwen3-ASR speech recognition engine implementation.
//!
//! This module provides a Qwen3-ASR-based transcription engine using the
//! [`qwen-asr`](https://crates.io/crates/qwen-asr) crate for CPU-only
//! speech-to-text conversion. Qwen3-ASR models are provided as directories
//! containing SafeTensors weights and a vocabulary file.
//!
//! # Model Format
//!
//! Qwen3-ASR expects a model directory containing:
//! - `model*.safetensors` - Model weight files
//! - `vocab.json` - Tokenizer vocabulary
//!
//! Available models:
//! - `Qwen/Qwen3-ASR-0.6B` (smaller, faster)
//! - `Qwen/Qwen3-ASR-1.7B` (larger, more accurate)
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use transcribe_rs::{TranscriptionEngine, engines::qwen_asr::QwenAsrEngine};
//! use std::path::PathBuf;
//!
//! let mut engine = QwenAsrEngine::new();
//! engine.load_model(&PathBuf::from("models/qwen3-asr-0.6b"))?;
//!
//! let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## With Custom Parameters
//!
//! ```rust,no_run
//! use transcribe_rs::{TranscriptionEngine, engines::qwen_asr::{QwenAsrEngine, QwenAsrInferenceParams}};
//! use std::path::PathBuf;
//!
//! let mut engine = QwenAsrEngine::new();
//! engine.load_model(&PathBuf::from("models/qwen3-asr-0.6b"))?;
//!
//! let params = QwenAsrInferenceParams {
//!     language: Some("Chinese".to_string()),
//!     prompt: Some("This is a technical discussion.".to_string()),
//! };
//!
//! let result = engine.transcribe_file(&PathBuf::from("audio.wav"), Some(params))?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{TranscriptionEngine, TranscriptionResult};
use log::{debug, info};
use std::path::{Path, PathBuf};

/// Errors specific to the Qwen ASR engine.
#[derive(thiserror::Error, Debug)]
pub enum QwenAsrError {
    #[error("Failed to load Qwen ASR model from {0}")]
    ModelLoadFailed(PathBuf),
    #[error("Model not loaded. Call load_model() first.")]
    ModelNotLoaded,
    #[error("Transcription failed")]
    TranscriptionFailed,
    #[error("Invalid language: {0}. See qwen_asr::config::SUPPORTED_LANGUAGES for valid options.")]
    InvalidLanguage(String),
}

/// Parameters for configuring Qwen ASR model loading.
///
/// These parameters control segmentation and silence detection behavior
/// which are set at model load time on the `QwenCtx`.
#[derive(Debug, Clone)]
pub struct QwenAsrModelParams {
    /// Segment duration in seconds for splitting long audio.
    /// 0.0 means no splitting (process entire audio at once).
    pub segment_sec: f32,
    /// Whether to skip silent spans before transcription.
    pub skip_silence: bool,
}

impl Default for QwenAsrModelParams {
    fn default() -> Self {
        Self {
            segment_sec: 0.0,
            skip_silence: false,
        }
    }
}

/// Parameters for configuring Qwen ASR inference behavior.
///
/// These parameters are applied per-transcription call.
#[derive(Debug, Clone, Default)]
pub struct QwenAsrInferenceParams {
    /// Force a specific language (e.g. `"Chinese"`, `"English"`).
    /// `None` means auto-detection.
    /// See `qwen_asr::config::SUPPORTED_LANGUAGES` for valid values.
    pub language: Option<String>,
    /// Optional text prompt to guide transcription.
    pub prompt: Option<String>,
}

/// Qwen3-ASR speech recognition engine.
///
/// This engine uses the `qwen-asr` crate for CPU-only speech recognition.
/// It supports Qwen3-ASR 0.6B and 1.7B model variants.
///
/// # Model Requirements
///
/// - **Format**: Directory with SafeTensors weights + `vocab.json`
/// - **Models**: 0.6B and 1.7B variants
/// - **Platform**: CPU-only (uses BLAS/vDSP acceleration)
pub struct QwenAsrEngine {
    ctx: Option<qwen_asr::context::QwenCtx>,
    loaded_model_path: Option<PathBuf>,
}

impl Default for QwenAsrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl QwenAsrEngine {
    /// Create a new Qwen ASR engine instance.
    ///
    /// The engine starts unloaded - you must call `load_model()` before
    /// performing transcription operations.
    pub fn new() -> Self {
        Self {
            ctx: None,
            loaded_model_path: None,
        }
    }
}

impl Drop for QwenAsrEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for QwenAsrEngine {
    type InferenceParams = QwenAsrInferenceParams;
    type ModelParams = QwenAsrModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let model_dir = model_path
            .to_str()
            .ok_or_else(|| QwenAsrError::ModelLoadFailed(model_path.to_path_buf()))?;

        info!("Loading Qwen ASR model from: {}", model_dir);

        let mut ctx = qwen_asr::context::QwenCtx::load(model_dir)
            .ok_or_else(|| QwenAsrError::ModelLoadFailed(model_path.to_path_buf()))?;

        ctx.segment_sec = params.segment_sec;
        ctx.skip_silence = params.skip_silence;

        debug!(
            "Qwen ASR model loaded (segment_sec={}, skip_silence={})",
            params.segment_sec, params.skip_silence
        );

        self.ctx = Some(ctx);
        self.loaded_model_path = Some(model_path.to_path_buf());
        Ok(())
    }

    fn unload_model(&mut self) {
        if self.ctx.is_some() {
            info!("Unloading Qwen ASR model");
        }
        self.ctx = None;
        self.loaded_model_path = None;
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let ctx = self.ctx.as_mut().ok_or(QwenAsrError::ModelNotLoaded)?;

        let inference_params = params.unwrap_or_default();

        // Apply language setting
        if let Some(ref language) = inference_params.language {
            ctx.set_force_language(language)
                .map_err(|()| QwenAsrError::InvalidLanguage(language.clone()))?;
        } else {
            // Reset to auto-detection
            let _ = ctx.set_force_language("");
        }

        // Apply prompt setting
        if let Some(ref prompt) = inference_params.prompt {
            let _ = ctx.set_prompt(prompt);
        } else {
            let _ = ctx.set_prompt("");
        }

        debug!("Transcribing {} samples with Qwen ASR", samples.len());

        let text = qwen_asr::transcribe::transcribe_audio(ctx, &samples)
            .ok_or(QwenAsrError::TranscriptionFailed)?;

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            segments: None,
        })
    }
}
