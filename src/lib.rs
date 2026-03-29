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
//! - **Unified API**: `SpeechModel` trait for all local engines
//! - **Hardware Acceleration**: GPU support for ORT engines (`ort-cuda`, `ort-rocm`,
//!   `ort-directml`, `ort-coreml`, `ort-webgpu`) and whisper.cpp (Metal/Vulkan)
//!   via the [`accel`] module
//!
//! ## Backend Categories
//!
//! This crate provides two categories of transcription backend:
//!
//! - **Local models** implement [`SpeechModel`] and run inference in-process or via
//!   a local binary. This includes all ONNX models, Whisper (via whisper.cpp), and
//!   Whisperfile.
//! - **Remote services** implement [`RemoteTranscriptionEngine`] (requires `openai`
//!   feature) and make async network calls to external APIs. This includes OpenAI.
//!
//! These traits are intentionally separate because the execution model differs:
//! local models are synchronous and take audio samples directly, while remote
//! services are async and may only accept file uploads.
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
//! let result = model.transcribe(&samples, &transcribe_rs::TranscribeOptions::default())?;
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
//!
//! ## Migrating from 0.2.x to 0.3.0
//!
//! Version 0.3.0 is a breaking release. If you need the old API, pin to `version = "=0.2.9"`.
//!
//! **`SpeechModel::transcribe` signature changed:**
//!
//! ```rust,ignore
//! // Before (0.2.x):
//! model.transcribe(&samples, Some("en"))?;
//! model.transcribe_file(&path, None)?;
//!
//! // After (0.3.0):
//! use transcribe_rs::TranscribeOptions;
//! model.transcribe(&samples, &TranscribeOptions { language: Some("en".into()), ..Default::default() })?;
//! model.transcribe_file(&path, &TranscribeOptions::default())?;
//! ```
//!
//! **`SpeechModel` now requires `Send`**, enabling `Box<dyn SpeechModel + Send>` for
//! use across threads.
//!
//! **`TranscribeOptions` includes a `translate` field** (default `false`). Engines that
//! support translation (Whisper, Whisperfile) will translate to English when set to `true`.
//!
//! **Whisper capabilities are now dynamic.** `WhisperEngine::capabilities()` returns the
//! actual language support of the loaded model (English-only vs multilingual) rather than
//! always reporting all 99 languages.

pub mod accel;
pub mod audio;
pub mod error;
pub use accel::{
    get_ort_accelerator, get_whisper_accelerator, get_whisper_gpu_device, set_ort_accelerator,
    set_whisper_accelerator, set_whisper_gpu_device, OrtAccelerator, WhisperAccelerator,
    GPU_DEVICE_AUTO,
};
pub use error::TranscribeError;

#[cfg(feature = "audio-features")]
pub mod decode;
#[cfg(feature = "audio-features")]
pub mod features;
#[cfg(feature = "onnx")]
pub mod onnx;

pub mod transcriber;
pub mod vad;

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
    /// Machine-friendly engine identifier (e.g. "sense_voice", "whisper_cpp").
    pub engine_id: &'static str,
    /// Expected input sample rate in Hz (e.g. 16000).
    pub sample_rate: u32,
    /// Languages supported (BCP-47 codes, e.g. "en", "zh"). Empty = any/unknown.
    pub languages: &'static [&'static str],
    /// Whether the model can produce word/segment timestamps.
    pub supports_timestamps: bool,
    /// Whether the model can translate to English.
    pub supports_translation: bool,
    /// Whether the model supports streaming inference.
    pub supports_streaming: bool,
}

/// Options for transcription.
#[derive(Debug, Clone, Default)]
pub struct TranscribeOptions {
    /// Language hint (BCP-47 code, e.g. "en", "zh").
    /// Multilingual models use this as a hint; single-language models ignore it.
    pub language: Option<String>,
    /// Whether to translate the output to English (only supported by some engines).
    pub translate: bool,
    /// Leading silence padding in milliseconds prepended before audio.
    /// Some models (e.g. Parakeet) can drop the beginning of audio due to
    /// mel spectrogram windowing. Set to `Some(0)` to explicitly disable.
    /// When `None`, each engine applies its own default (e.g. Parakeet uses 250 ms).
    pub leading_silence_ms: Option<u32>,
    /// Trailing silence padding in milliseconds appended after audio.
    /// Set to `Some(0)` to explicitly disable.
    /// When `None`, each engine applies its own default (typically 0 ms).
    pub trailing_silence_ms: Option<u32>,
}

/// Unified interface for speech-to-text models.
///
/// Each model implements this trait to provide a common transcription API.
/// Model-specific parameters are exposed via a separate `transcribe_with()` method
/// on the concrete type.
///
/// Engines implement [`transcribe_raw`](SpeechModel::transcribe_raw) with their
/// inference logic. The default [`transcribe`](SpeechModel::transcribe) method
/// handles silence padding and timestamp adjustment automatically.
pub trait SpeechModel: Send {
    /// Report this model's capabilities.
    fn capabilities(&self) -> ModelCapabilities;

    /// Default leading silence in milliseconds for this engine.
    ///
    /// Override to set a non-zero default. For example, Parakeet returns 250
    /// because its mel spectrogram preprocessor attenuates the start of audio.
    fn default_leading_silence_ms(&self) -> u32 {
        0
    }

    /// Default trailing silence in milliseconds for this engine.
    fn default_trailing_silence_ms(&self) -> u32 {
        0
    }

    /// Raw transcription — engines implement this with their inference logic.
    ///
    /// Callers should prefer [`transcribe`](SpeechModel::transcribe) which
    /// handles silence padding automatically. Use this method directly only
    /// when managing padding yourself (e.g. chunked transcription).
    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError>;

    /// Transcribe audio samples (16 kHz, mono, f32 in [-1, 1]).
    ///
    /// Prepends/appends silence padding based on [`TranscribeOptions`] (or
    /// engine defaults), runs [`transcribe_raw`](SpeechModel::transcribe_raw),
    /// then adjusts segment timestamps to account for the leading padding.
    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let lead_ms = options
            .leading_silence_ms
            .unwrap_or_else(|| self.default_leading_silence_ms());
        let trail_ms = options
            .trailing_silence_ms
            .unwrap_or_else(|| self.default_trailing_silence_ms());

        // Fast path: no padding needed.
        if lead_ms == 0 && trail_ms == 0 {
            return self.transcribe_raw(samples, options);
        }

        let mut buf = if lead_ms > 0 {
            audio::prepend_silence(samples, lead_ms)
        } else {
            samples.to_vec()
        };
        if trail_ms > 0 {
            let trail_len = trail_ms as usize * audio::SAMPLES_PER_MS;
            buf.resize(buf.len() + trail_len, 0.0);
        }

        let mut result = self.transcribe_raw(&buf, options)?;

        if lead_ms > 0 {
            result.offset_timestamps(-(lead_ms as f32 / 1000.0));
        }

        Ok(result)
    }

    /// Transcribe a WAV file (16 kHz, 16-bit, mono).
    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let samples = audio::read_wav_samples(wav_path)?;
        self.transcribe(&samples, options)
    }
}

/// The result of a transcription operation.
///
/// Contains both the full transcribed text and detailed timing information
/// for individual segments within the audio.
#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    /// The complete transcribed text from the audio
    pub text: String,
    /// Individual segments with timing information
    pub segments: Option<Vec<TranscriptionSegment>>,
}

impl TranscriptionResult {
    /// Shift all segment timestamps by `offset_secs`, clamping to zero.
    ///
    /// Use a negative offset to compensate for leading silence padding,
    /// or a positive offset to place a chunk within a longer audio stream.
    pub fn offset_timestamps(&mut self, offset_secs: f32) {
        if let Some(segs) = &mut self.segments {
            for seg in segs {
                seg.start = (seg.start + offset_secs).max(0.0);
                seg.end = (seg.end + offset_secs).max(0.0);
            }
        }
    }
}

/// A single transcribed segment with timing information.
///
/// Represents a portion of the transcribed audio with start and end timestamps
/// and the corresponding text content.
#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Start time of the segment in seconds
    pub start: f32,
    /// End time of the segment in seconds
    pub end: f32,
    /// The transcribed text for this segment
    pub text: String,
}
