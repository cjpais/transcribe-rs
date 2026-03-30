//! `SpeechModel` implementation for Qwen3-ASR (batch mode).
//!
//! Two-layer structure:
//! - `model.rs` — raw ONNX inference (encoder, decoder, KV-cache) returning plain Rust types.
//! - `engine.rs` — wraps `Qwen3AsrModel`, adapts it to the `SpeechModel` trait (capabilities,
//!   options, language-prefix stripping). Keeps `model.rs` free of library-level concerns.

use std::collections::HashMap;
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
    /// Cache of language name → token IDs. Populated on first use of each
    /// language, then reused for all subsequent transcriptions.
    language_cache: HashMap<String, Vec<i64>>,
}

impl Qwen3Model {
    /// Load a Qwen3-ASR model from `model_dir` with the given quantization.
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let inner = Qwen3AsrModel::load(model_dir, quantization)?;
        log::info!("Loaded Qwen3-ASR model from {:?}", model_dir);
        Ok(Self {
            inner,
            language_cache: HashMap::new(),
        })
    }

    /// Transcribe with explicit per-call parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &Qwen3Params,
    ) -> Result<TranscriptionResult, TranscribeError> {
        // Encode language on first use, then borrow from cache.
        // Treat empty strings as None (no language conditioning).
        let lang_key = params.language.as_deref().filter(|s| !s.is_empty());
        if let Some(lang) = lang_key {
            self.ensure_language_cached(lang);
        }
        let lang_tokens = lang_key
            .and_then(|lang| self.language_cache.get(lang))
            .map(|v| v.as_slice());

        let raw = self
            .inner
            .transcribe(samples, params.max_tokens, lang_tokens)?;
        log::debug!("Qwen3-ASR raw decoder output: {:?}", raw);
        let text = strip_language_prefix(&raw);
        log::debug!("Qwen3-ASR after strip_language_prefix: {:?}", text);
        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    /// Ensure language token IDs are in the cache, encoding on first use.
    fn ensure_language_cached(&mut self, language: &str) {
        if self.language_cache.contains_key(language) {
            return;
        }
        let tokens = self.inner.encode_language(language);
        log::info!(
            "Qwen3-ASR: encoded language {:?} → {:?} (cached for reuse)",
            language,
            tokens
        );
        self.language_cache.insert(language.to_string(), tokens);
    }
}

/// Per-call parameters for Qwen3-ASR transcription.
#[derive(Debug, Clone)]
pub struct Qwen3Params {
    /// Maximum number of decoder tokens to generate.
    ///
    /// Default: 512, sufficient for approximately 60 s of typical English audio.
    pub max_tokens: usize,

    /// Language hint (e.g. "English", "Chinese", "en"). When set, the prompt
    /// includes "Please transcribe the above {language} audio." which conditions
    /// the decoder toward the specified language and reduces degenerate output
    /// on non-speech audio. The string is tokenized and embedded in the prompt
    /// as-is — full English names ("English") work best, but short codes ("en")
    /// are also accepted.
    pub language: Option<String>,
}

impl Default for Qwen3Params {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            language: None,
        }
    }
}

impl SpeechModel for Qwen3Model {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        if options.translate {
            log::warn!(
                "Qwen3-ASR: translate is not supported; the model produces transcription only"
            );
        }
        let params = Qwen3Params {
            language: options.language.clone(),
            ..Default::default()
        };
        self.transcribe_with(samples, &params)
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
