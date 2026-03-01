//! Speech recognition engines for transcription.
//!
//! This module contains implementations of non-ONNX speech recognition engines.
//! For ONNX-based models (SenseVoice, GigaAM, Parakeet, Moonshine), see the `onnx` module.
//!
//! # Available Engines
//!
//! Enable engines via Cargo features:
//! - `onnx` - All ONNX-based models (SenseVoice, GigaAM, Parakeet, Moonshine)
//! - `whisper` - OpenAI's Whisper (GGML format)
//! - `whisperfile` - Mozilla whisperfile server wrapper

#[cfg(feature = "whisper")]
pub mod whisper;
#[cfg(feature = "whisperfile")]
pub mod whisperfile;
