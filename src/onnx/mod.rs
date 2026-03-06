//! ONNX-based speech recognition engines.
//!
//! Each model is available as a top-level module (e.g. `onnx::sense_voice::SenseVoiceModel`)
//! and implements the `SpeechModel` trait for a unified transcription API.

pub mod session;

/// Quantization type for ONNX model loading.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Quantization {
    #[default]
    FP32,
    Int8,
}

pub mod sense_voice;
pub mod gigaam;
pub mod parakeet;
pub mod moonshine;
