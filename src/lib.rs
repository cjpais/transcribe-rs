//! # transcribe-rs
//!
//! A Rust library providing unified transcription capabilities using multiple speech recognition engines.
//!
//! ## Features
//!
//! - **ONNX Models**: SenseVoice, GigaAM, Parakeet, Moonshine (requires `onnx` feature)
//! - **Whisper**: OpenAI Whisper via GGML (requires `whisper-cpp` feature)
//! - **Whisperfile**: Mozilla Whisperfile server (requires `whisperfile` feature)
//! - **Remote**: OpenAI API (requires `openai` feature)
//! - **Timestamped Results**: Detailed timing information for transcribed segments
//! - **Unified API**: `SpeechModel` trait for all engines
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! transcribe-rs = { version = "0.3", features = ["onnx"] }
//! ```
//!
//! ```ignore
//! use std::path::PathBuf;
//! use transcribe_rs::onnx::sense_voice::{SenseVoiceModel, SenseVoiceParams};
//! use transcribe_rs::onnx::Quantization;
//! use transcribe_rs::SpeechModel;
//!
//! let mut model = SenseVoiceModel::load(
//!     &PathBuf::from("models/sense-voice"),
//!     &Quantization::Int8,
//! )?;
//!
//! let result = model.transcribe(&samples, Some("en"))?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Audio Requirements
//!
//! Input audio files must be:
//! - WAV format
//! - 16 kHz sample rate
//! - 16-bit samples
//! - Mono (single channel)

pub mod audio;

#[cfg(feature = "onnx")]
pub mod features;
#[cfg(feature = "onnx")]
pub mod decode;
#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "whisper-cpp")]
pub mod whisper_cpp;
#[cfg(feature = "whisperfile")]
pub mod whisperfile;

#[cfg(feature = "openai")]
pub mod remote;
#[cfg(feature = "openai")]
pub use remote::RemoteTranscriptionEngine;

use std::path::Path;

/// Describes the capabilities of a speech model.
#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    /// Human-readable model name.
    pub name: &'static str,
    /// Languages supported (BCP-47 codes, e.g. "en", "zh"). Empty = any/unknown.
    pub languages: &'static [&'static str],
    /// Whether the model can produce word/segment timestamps.
    pub supports_timestamps: bool,
    /// Whether the model can translate to English.
    pub supports_translation: bool,
    /// Whether the model supports streaming inference.
    pub supports_streaming: bool,
}

/// Unified interface for speech-to-text models.
///
/// Each model implements this trait to provide a common transcription API.
/// Model-specific parameters are exposed via a separate `transcribe_with()` method
/// on the concrete type.
pub trait SpeechModel {
    /// Report this model's capabilities.
    fn capabilities(&self) -> ModelCapabilities;

    /// Transcribe audio samples (16 kHz, mono, f32 in [-1, 1]).
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>>;

    /// Transcribe a WAV file (16 kHz, 16-bit, mono).
    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        language: Option<&str>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let samples = audio::read_wav_samples(wav_path)?;
        self.transcribe(&samples, language)
    }
}

/// Interface for streaming speech-to-text models (future use).
pub trait StreamingSpeechModel: SpeechModel {
    /// Begin a new streaming session.
    fn begin(&mut self) -> Result<(), Box<dyn std::error::Error>>;

    /// Push a chunk of audio samples into the stream.
    fn push(&mut self, samples: &[f32]) -> Result<Option<TranscriptionResult>, Box<dyn std::error::Error>>;

    /// End the streaming session and return the final result.
    fn end(&mut self) -> Result<TranscriptionResult, Box<dyn std::error::Error>>;
}

/// The result of a transcription operation.
///
/// Contains both the full transcribed text and detailed timing information
/// for individual segments within the audio.
#[derive(Debug)]
pub struct TranscriptionResult {
    /// The complete transcribed text from the audio
    pub text: String,
    /// Individual segments with timing information
    pub segments: Option<Vec<TranscriptionSegment>>,
}

/// A single transcribed segment with timing information.
///
/// Represents a portion of the transcribed audio with start and end timestamps
/// and the corresponding text content.
#[derive(Debug)]
pub struct TranscriptionSegment {
    /// Start time of the segment in seconds
    pub start: f32,
    /// End time of the segment in seconds
    pub end: f32,
    /// The transcribed text for this segment
    pub text: String,
}
