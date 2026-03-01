//! ONNX-based speech recognition engines.
//!
//! This module provides a unified interface for all ONNX-based transcription models
//! including SenseVoice, GigaAM, Parakeet, and Moonshine.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use transcribe_rs::onnx::{Engine, Model, InferenceParams, Language};
//!
//! let mut engine = Engine::new();
//! engine.load(&PathBuf::from("models/sense-voice"), Model::sense_voice_int8())?;
//!
//! let result = engine.transcribe_file(
//!     &PathBuf::from("audio.wav"),
//!     Some(InferenceParams {
//!         language: Some(Language::English),
//!         ..Default::default()
//!     }),
//! )?;
//! println!("{}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod engine;
pub mod features;
pub mod decode;
pub mod models;
pub mod session;
pub mod types;

pub use engine::{Engine, InferenceParams, Model};
pub use types::{
    Language, MoonshineVariant, Quantization, TimestampGranularity,
};
