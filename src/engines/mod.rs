//! Speech recognition engines for transcription.
//!
//! This module contains implementations of different speech recognition engines
//! that can be used for audio transcription. Each engine has its own requirements
//! for model formats and provides different capabilities.
//!
//! # Available Engines
//!
//! Enable engines via Cargo features:
//! - `whisper` - OpenAI's Whisper (GGML format)
//! - `parakeet` - NVIDIA NeMo Parakeet (ONNX format)
//! - `moonshine` - Moonshine lightweight models (ONNX format)
//! - `whisperfile` - Mozilla whisperfile server wrapper
//!
//! # Example
//!
//! ```toml
//! [dependencies]
//! transcribe-rs = { version = "0.2", features = ["parakeet", "whisper"] }
//! ```

#[cfg(feature = "moonshine")]
pub mod moonshine;
#[cfg(feature = "parakeet")]
pub mod parakeet;
#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(feature = "whisperfile")]
pub mod whisperfile;
