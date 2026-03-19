//! `SpeechModel` implementation for Qwen3-ASR (batch mode).
//!
//! Two-layer structure:
//! - `model.rs` — raw ONNX inference (encoder, decoder, KV-cache) returning plain Rust types.
//! - `engine.rs` — wraps `Qwen3AsrModel`, adapts it to the `SpeechModel` trait (capabilities,
//!   options, language-prefix stripping). Keeps `model.rs` free of library-level concerns.

use std::path::Path;

use crate::onnx::Quantization;
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

use super::model::Qwen3AsrModel;

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Qwen3-ASR",
    engine_id: "qwen3",
    sample_rate: 16000,
    languages: &[], // multilingual: supports many languages
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Batch transcription model backed by Qwen3-ASR.
///
/// Processes complete audio in a single encoder-decoder pass, suitable for
/// the standard record-then-transcribe workflow.
///
/// This model does not produce timestamp segments (`TranscriptionResult.segments`
/// is always `None`).
pub struct Qwen3Model {
    inner: Qwen3AsrModel,
}

impl Qwen3Model {
    /// Load a Qwen3-ASR model from `model_dir` with the given quantization.
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let inner = Qwen3AsrModel::load(model_dir, quantization)?;
        log::info!("Loaded Qwen3-ASR model from {:?}", model_dir);
        Ok(Self { inner })
    }

    /// Transcribe with explicit per-call parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &Qwen3Params,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let raw = self.inner.transcribe(samples, params.max_tokens)?;
        let text = strip_language_prefix(&raw);
        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }
}

/// Per-call parameters for Qwen3-ASR transcription.
#[derive(Debug, Clone)]
pub struct Qwen3Params {
    /// Maximum number of decoder tokens to generate.
    ///
    /// 512 tokens is sufficient for approximately 60 s of typical English audio.
    pub max_tokens: usize,
}

impl Default for Qwen3Params {
    fn default() -> Self {
        Self { max_tokens: 512 }
    }
}

impl SpeechModel for Qwen3Model {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        if let Some(ref lang) = options.language {
            log::warn!(
                "Qwen3-ASR: language hint {:?} is not supported; the model auto-detects language",
                lang
            );
        }
        if options.translate {
            log::warn!(
                "Qwen3-ASR: translate is not supported; the model produces transcription only"
            );
        }
        self.transcribe_with(samples, &Qwen3Params::default())
    }
}

/// Known language names emitted by Qwen3-ASR as the prefix tag.
const KNOWN_LANGUAGES: &[&str] = &[
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "Cantonese",
    "French",
    "Spanish",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Arabic",
    "Hindi",
    "Thai",
    "Vietnamese",
    "Indonesian",
    "Malay",
    "Turkish",
    "Dutch",
    "Polish",
    "Swedish",
    "Norwegian",
    "Danish",
    "Finnish",
    "Czech",
    "Romanian",
    "Hungarian",
    "Greek",
    "Hebrew",
    "Ukrainian",
    "Bulgarian",
    "Croatian",
    "Slovak",
    "Slovenian",
    "Lithuanian",
    "Latvian",
    "Estonian",
    "Serbian",
    "Tagalog",
    "Burmese",
    "Tibetan",
    "Uyghur",
    "Mongolian",
    "Amharic",
    "Swahili",
    "Kazakh",
    "Uzbek",
    "Azerbaijani",
    "Georgian",
    "Armenian",
    "Nepali",
    "Bengali",
    "Tamil",
    "Telugu",
    "Urdu",
    "Persian",
    "Lao",
    "Khmer",
    "Catalan",
    "Galician",
    "Basque",
    "Afrikaans",
];

/// Strip the language identification prefix that Qwen3-ASR generates.
///
/// The model outputs `language <Name>\n<transcription>` where `\n` is GPT-2
/// BPE token 198. Occasionally the newline is absent: `language <Name><text>`.
/// We try the newline split first (handles unknown languages automatically),
/// then fall back to known language name matching for the no-newline case.
///
/// Known limitation: if the transcribed text starts with a substring that
/// matches a language name (e.g. "Georgian" → "Georgia is..."), the boundary
/// detection is ambiguous. This is inherent to the model's output format
/// when no separator character is present.
fn strip_language_prefix(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("language ") {
        // Primary: split on newline (the model's standard delimiter)
        if let Some(newline_pos) = rest.find('\n') {
            return rest[newline_pos + 1..].to_string();
        }
        // Fallback: match known language name (no newline separator)
        for lang in KNOWN_LANGUAGES {
            if let Some(after) = rest.strip_prefix(lang) {
                return after.to_string();
            }
        }
        // Unknown language without newline: preserve text rather than drop it
        log::warn!(
            "Qwen3-ASR: unrecognised language prefix with no newline separator; raw output: {:?}",
            rest
        );
        return rest.to_string();
    }
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_language_prefix_with_newline() {
        assert_eq!(
            strip_language_prefix("language English\nHello world"),
            "Hello world"
        );
    }

    #[test]
    fn test_strip_language_prefix_without_newline() {
        assert_eq!(
            strip_language_prefix("language EnglishHello world"),
            "Hello world"
        );
    }

    #[test]
    fn test_strip_language_prefix_without_newline_starts_uppercase() {
        assert_eq!(
            strip_language_prefix("language EnglishAre there more features?"),
            "Are there more features?"
        );
    }

    #[test]
    fn test_strip_language_prefix_body_starts_with_language_name() {
        // Transcript begins with a language name — verify we strip only the prefix tag.
        assert_eq!(
            strip_language_prefix("language EnglishEnglish is a language"),
            "English is a language"
        );
    }

    #[test]
    fn test_strip_language_prefix_chinese_with_newline() {
        assert_eq!(
            strip_language_prefix("language Chinese\n你好世界"),
            "你好世界"
        );
    }

    #[test]
    fn test_strip_language_prefix_chinese_without_newline() {
        assert_eq!(
            strip_language_prefix("language Chinese你好世界"),
            "你好世界"
        );
    }

    #[test]
    fn test_strip_language_prefix_no_text() {
        assert_eq!(strip_language_prefix("language English"), "");
    }

    #[test]
    fn test_strip_language_prefix_no_prefix() {
        assert_eq!(strip_language_prefix("Hello world"), "Hello world");
    }

    #[test]
    fn test_strip_language_prefix_empty() {
        assert_eq!(strip_language_prefix(""), "");
    }

    #[test]
    fn test_strip_language_prefix_preserves_newlines_in_body() {
        assert_eq!(
            strip_language_prefix("language English\nline one\nline two"),
            "line one\nline two"
        );
    }

    #[test]
    fn test_strip_language_prefix_unknown_language_with_newline() {
        assert_eq!(strip_language_prefix("language Klingon\nQapla!"), "Qapla!");
    }

    #[test]
    fn test_strip_language_prefix_unknown_language_without_newline() {
        // Unknown language without newline: text preserved (not dropped)
        assert_eq!(strip_language_prefix("language Klingon"), "Klingon");
    }
}
