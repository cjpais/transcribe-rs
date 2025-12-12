//! Moonshine ONNX transcription engine.
//!
//! This module provides transcription using the Moonshine encoder-decoder transformer model
//! via ONNX Runtime. It supports multiple model variants for different languages.
//!
//! # Model Architecture
//!
//! Moonshine uses an encoder-decoder architecture with KV caching:
//! - **Encoder**: Processes audio into hidden states (run once per transcription)
//! - **Decoder**: Autoregressively generates tokens using cached key-value states
//!
//! # Model Format
//!
//! Expects a directory containing:
//! - `encoder_model.onnx` - Audio encoder
//! - `decoder_model_merged.onnx` - Merged decoder with cache support
//! - `tokenizer.json` - BPE tokenizer vocabulary
//!
//! # Supported Variants
//!
//! | Variant | Language | Token Rate |
//! |---------|----------|------------|
//! | Tiny | English | 6 |
//! | TinyAr | Arabic | 13 |
//! | TinyZh | Chinese | 13 |
//! | TinyJa | Japanese | 13 |
//! | TinyKo | Korean | 13 |
//! | TinyUk | Ukrainian | 8 |
//! | TinyVi | Vietnamese | 13 |
//! | Base | English | 6 |
//! | BaseEs | Spanish | 6 |
//!
//! # Audio Requirements
//!
//! - Sample rate: 16 kHz
//! - Format: Mono, 16-bit PCM
//! - Duration: 0.1s to 64s
//!
//! # Example
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use transcribe_rs::{TranscriptionEngine, engines::moonshine::{MoonshineEngine, MoonshineModelParams, ModelVariant}};
//!
//! let mut engine = MoonshineEngine::new();
//! engine.load_model_with_params(
//!     &PathBuf::from("models/moonshine-tiny"),
//!     MoonshineModelParams::variant(ModelVariant::Tiny),
//! )?;
//!
//! let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod cache;
pub mod engine;
pub mod model;
mod tokenizer;

pub use engine::{ModelVariant, MoonshineEngine, MoonshineInferenceParams, MoonshineModelParams};
