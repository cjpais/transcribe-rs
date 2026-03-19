//! ONNX-based speech recognition engines.
//!
//! Each model is available as a top-level module (e.g. `onnx::sense_voice::SenseVoiceModel`)
//! and implements the `SpeechModel` trait for a unified transcription API.

pub mod session;

/// Preferred precision for ONNX model loading.
///
/// This selects which model file variant to load. If the requested
/// variant is not found on disk, falls back to FP32 with a warning.
/// ONNX quantization is baked into the model file — this enum controls
/// file selection, not runtime behavior.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Quantization {
    #[default]
    FP32,
    FP16,
    Int8,
}

pub mod canary;
pub mod gigaam;
pub mod moonshine;
pub mod parakeet;
pub mod sense_voice;

// qwen3 has a separate per-engine feature flag ("qwen3") rather than being enabled
// unconditionally when "onnx" is active, because its ONNX export format (three session
// files + embed_tokens.bin) and decode loop differ substantially from the other engines.
// Gating it separately keeps the default "onnx" build lean until the model artifacts are
// available.  The other engines share enough structure that separate flags add no value.
#[cfg(feature = "qwen3")]
pub mod qwen3;
