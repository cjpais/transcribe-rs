//! # transcribe-rs
//!
//! A Rust library providing unified transcription capabilities using multiple speech recognition engines.
//! Currently supports Whisper and Parakeet (NeMo) models for accurate speech-to-text transcription.
//!
//! ## Features
//!
//! - **Multiple Engines**: Support for both Whisper and Parakeet transcription engines
//! - **Flexible Model Loading**: Load models with custom parameters (quantization, etc.)
//! - **Timestamped Results**: Get detailed timing information for transcribed segments
//! - **Audio Processing**: Built-in WAV file processing with proper format validation
//! - **Unified API**: Common trait-based interface for all transcription engines
//!
//! ## Model Format Requirements
//!
//! - **Whisper**: Expects a single GGML format file (e.g., `whisper-medium-q4_1.bin`)
//! - **Parakeet**: Expects a directory containing the model files (e.g., `parakeet-v0.3/`)
//!
//! ## Quick Start
//!
//! ```toml
//! [dependencies]
//! transcribe-rs = { version = "0.2", features = ["whisper"] }
//! ```
//!
//! ```ignore
//! use std::path::PathBuf;
//! use transcribe_rs::{engines::whisper::WhisperEngine, TranscriptionEngine};
//!
//! let mut engine = WhisperEngine::new();
//! engine.load_model(&PathBuf::from("models/whisper-medium-q4_1.bin"))?;
//!
//! let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
//! println!("Transcription: {}", result.text);
//!
//! if let Some(segments) = result.segments {
//!     for segment in segments {
//!         println!(
//!             "[{:.2}s - {:.2}s]: {}",
//!             segment.start, segment.end, segment.text
//!         );
//!     }
//! }
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
pub mod engines;

#[cfg(feature = "openai")]
pub mod remote;
#[cfg(feature = "openai")]
pub use remote::RemoteTranscriptionEngine;

use std::path::Path;

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

/// A segment of text from a streaming transcription engine.
///
/// Each segment contains incremental text and a flag indicating whether the
/// segment ends at a detected sentence boundary (period, question mark, or
/// exclamation mark). This allows callers to flush output or trigger
/// downstream processing when a sentence completes.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingSegment {
    /// The incremental text for this segment.
    pub text: String,
    /// Whether this segment ends at a detected sentence boundary (`.` `?` `!`).
    pub is_endpoint: bool,
}

/// Common interface for speech transcription engines.
///
/// This trait defines the standard operations that all transcription engines must support.
/// Each engine may have different parameter types for model loading and inference configuration.
///
/// # Examples
///
/// ## Using Whisper Engine (requires `whisper` feature)
///
/// ```ignore
/// use std::path::PathBuf;
/// use transcribe_rs::{engines::whisper::WhisperEngine, TranscriptionEngine};
///
/// let mut engine = WhisperEngine::new();
/// engine.load_model(&PathBuf::from("models/whisper-medium-q4_1.bin"))?;
///
/// let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
/// println!("Transcription: {}", result.text);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Using Parakeet Engine (requires `parakeet` feature)
///
/// ```ignore
/// use std::path::PathBuf;
/// use transcribe_rs::{
///     engines::parakeet::{ParakeetEngine, ParakeetModelParams},
///     TranscriptionEngine,
/// };
///
/// let mut engine = ParakeetEngine::new();
/// engine.load_model_with_params(
///     &PathBuf::from("models/parakeet-v0.3"),
///     ParakeetModelParams::int8(),
/// )?;
///
/// let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
/// println!("Transcription: {}", result.text);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait TranscriptionEngine {
    /// Parameters for configuring inference behavior (language, timestamps, etc.)
    type InferenceParams;
    /// Parameters for configuring model loading (quantization, etc.)
    type ModelParams: Default;

    /// Load a model from the specified path using default parameters.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file or directory
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the model loads successfully, or an error if loading fails.
    fn load_model(&mut self, model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.load_model_with_params(model_path, Self::ModelParams::default())
    }

    /// Load a model from the specified path with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file or directory
    /// * `params` - Engine-specific model loading parameters
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the model loads successfully, or an error if loading fails.
    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Unload the currently loaded model and free associated resources.
    fn unload_model(&mut self);

    /// Transcribe audio samples directly.
    ///
    /// # Arguments
    ///
    /// * `samples` - Audio samples as f32 values (16kHz, mono)
    /// * `params` - Optional engine-specific inference parameters
    ///
    /// # Returns
    ///
    /// Returns transcription result with text and timing information.
    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>>;

    /// Transcribe audio from a WAV file.
    ///
    /// The WAV file must meet the following requirements:
    /// - 16 kHz sample rate
    /// - 16-bit samples
    /// - Mono (single channel)
    /// - PCM format
    ///
    /// # Arguments
    ///
    /// * `wav_path` - Path to the WAV file to transcribe
    /// * `params` - Optional engine-specific inference parameters
    ///
    /// # Returns
    ///
    /// Returns transcription result with text and timing information.
    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let samples = audio::read_wav_samples(wav_path)?;
        self.transcribe_samples(samples, params)
    }
}

/// Interface for streaming (chunk-based) transcription engines.
///
/// Unlike [`TranscriptionEngine`] which processes complete audio,
/// streaming engines accept audio incrementally via [`push_samples`](StreamingTranscriptionEngine::push_samples)
/// and emit text as it becomes available.
///
/// Input audio must be 16kHz mono f32, same as `TranscriptionEngine`.
///
/// # Examples
///
/// ```ignore
/// use transcribe_rs::{StreamingTranscriptionEngine, engines::nemotron_streaming::NemotronStreamingEngine};
/// use std::path::PathBuf;
///
/// let mut engine = NemotronStreamingEngine::new();
/// engine.load_model(&PathBuf::from("models/nemotron-speech-streaming-en-0.6b"))?;
///
/// // Feed audio in chunks (e.g. from a microphone)
/// for chunk in audio_chunks {
///     let segments = engine.push_samples(&chunk)?;
///     for seg in &segments {
///         print!("{}", seg.text);
///         if seg.is_endpoint {
///             println!(); // newline after each sentence
///         }
///     }
/// }
///
/// let full = engine.get_transcript();
/// println!("\nFinal: {}", full);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait StreamingTranscriptionEngine {
    /// Parameters for configuring model loading.
    type ModelParams: Default;

    /// Load a streaming model from the given path using default parameters.
    fn load_model(&mut self, model_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        self.load_model_with_params(model_path, Self::ModelParams::default())
    }

    /// Load a streaming model from the given path with custom parameters.
    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>>;

    /// Unload the model and free resources.
    fn unload_model(&mut self);

    /// Push audio samples (16kHz mono f32) and return newly emitted segments.
    ///
    /// Returns a vec of [`StreamingSegment`]s produced by this chunk, or an
    /// empty vec if no new tokens were produced. Each segment's `is_endpoint`
    /// flag indicates whether the text ends at a detected sentence boundary.
    ///
    /// The concatenation of all segment texts approximates
    /// [`get_transcript`](StreamingTranscriptionEngine::get_transcript),
    /// modulo tokenizer whitespace normalization.
    fn push_samples(
        &mut self,
        samples: &[f32],
    ) -> Result<Vec<StreamingSegment>, Box<dyn std::error::Error>>;

    /// Get the canonical accumulated transcript so far.
    ///
    /// This returns the full text produced by all `push_samples` calls since the
    /// last [`reset`](StreamingTranscriptionEngine::reset) (or engine creation).
    fn get_transcript(&self) -> String;

    /// Reset all decoder state for a new utterance.
    ///
    /// After this call, [`get_transcript`](StreamingTranscriptionEngine::get_transcript)
    /// returns an empty string. The model remains loaded and ready for new audio.
    fn reset(&mut self);
}

/// Split text at sentence boundaries (`. `, `? `, `! `, or sentence-ending
/// punctuation at end-of-string) and return [`StreamingSegment`]s.
///
/// Splitting happens *after* the punctuation character, so the punctuation
/// stays with the preceding segment. A trailing segment without sentence-ending
/// punctuation gets `is_endpoint: false`.
///
/// Returns an empty vec for empty input.
pub fn split_at_sentence_boundaries(text: &str) -> Vec<StreamingSegment> {
    if text.is_empty() {
        return vec![];
    }

    let sentence_end = ['.', '?', '!'];
    let mut segments = Vec::new();
    let mut start = 0;
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        if sentence_end.contains(&chars[i]) {
            // Include the punctuation character
            let end_byte = text
                .char_indices()
                .nth(i + 1)
                .map(|(idx, _)| idx)
                .unwrap_or(text.len());

            // Check if this is end-of-string or followed by a space
            let at_end = i + 1 >= len;
            let followed_by_space = !at_end && chars[i + 1] == ' ';

            if at_end || followed_by_space {
                let segment_text = &text[start..end_byte];
                segments.push(StreamingSegment {
                    text: segment_text.to_string(),
                    is_endpoint: true,
                });
                start = end_byte;
            }
        }
        i += 1;
    }

    // Remaining text after last boundary (if any)
    if start < text.len() {
        segments.push(StreamingSegment {
            text: text[start..].to_string(),
            is_endpoint: false,
        });
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_single_sentence_with_period() {
        let result = split_at_sentence_boundaries("Hello world.");
        assert_eq!(
            result,
            vec![StreamingSegment {
                text: "Hello world.".to_string(),
                is_endpoint: true,
            }]
        );
    }

    #[test]
    fn test_split_two_sentences() {
        let result = split_at_sentence_boundaries("Hello. World");
        assert_eq!(
            result,
            vec![
                StreamingSegment {
                    text: "Hello.".to_string(),
                    is_endpoint: true,
                },
                StreamingSegment {
                    text: " World".to_string(),
                    is_endpoint: false,
                },
            ]
        );
    }

    #[test]
    fn test_split_no_punctuation() {
        let result = split_at_sentence_boundaries("no punctuation");
        assert_eq!(
            result,
            vec![StreamingSegment {
                text: "no punctuation".to_string(),
                is_endpoint: false,
            }]
        );
    }

    #[test]
    fn test_split_empty() {
        let result = split_at_sentence_boundaries("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_multiple_sentence_types() {
        let result = split_at_sentence_boundaries("First? Second! Third.");
        assert_eq!(
            result,
            vec![
                StreamingSegment {
                    text: "First?".to_string(),
                    is_endpoint: true,
                },
                StreamingSegment {
                    text: " Second!".to_string(),
                    is_endpoint: true,
                },
                StreamingSegment {
                    text: " Third.".to_string(),
                    is_endpoint: true,
                },
            ]
        );
    }

    #[test]
    fn test_split_preserves_decimal_numbers() {
        // "3.14" has a period NOT followed by a space and NOT at end, so no split
        let result = split_at_sentence_boundaries("The value is 3.14 exactly");
        assert_eq!(
            result,
            vec![StreamingSegment {
                text: "The value is 3.14 exactly".to_string(),
                is_endpoint: false,
            }]
        );
    }

    #[test]
    fn test_split_sentence_ending_question() {
        let result = split_at_sentence_boundaries("Are you sure? Yes.");
        assert_eq!(
            result,
            vec![
                StreamingSegment {
                    text: "Are you sure?".to_string(),
                    is_endpoint: true,
                },
                StreamingSegment {
                    text: " Yes.".to_string(),
                    is_endpoint: true,
                },
            ]
        );
    }
}
