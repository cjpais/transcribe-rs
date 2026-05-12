//! `SpeechModel` implementation for Qwen3-ASR (batch mode).

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
    languages: &[],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

pub struct Qwen3Model {
    inner: Qwen3AsrModel,
    language_cache: HashMap<String, Vec<i64>>,
}

impl Qwen3Model {
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let inner = Qwen3AsrModel::load(model_dir, quantization)?;
        log::info!("Loaded Qwen3-ASR model from {:?}", model_dir);
        Ok(Self {
            inner,
            language_cache: HashMap::new(),
        })
    }

    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &Qwen3Params,
    ) -> Result<TranscriptionResult, TranscribeError> {
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

    fn ensure_language_cached(&mut self, language: &str) {
        if self.language_cache.contains_key(language) {
            return;
        }
        let normalized = normalize_language(language);
        let spaced = format!(" {normalized}");
        let tokens = self.inner.encode_language(&spaced);
        log::info!(
            "Qwen3-ASR: encoded language {:?} → {:?} → {:?} (cached for reuse)",
            language,
            normalized,
            tokens
        );
        self.language_cache.insert(language.to_string(), tokens);
    }
}

/// Per-call parameters for Qwen3-ASR transcription.
#[derive(Debug, Clone)]
pub struct Qwen3Params {
    pub max_tokens: usize,
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

fn normalize_language(code: &str) -> &str {
    match code {
        "en" => "English",
        "zh" | "zh-Hans" | "zh-Hant" => "Chinese",
        "ja" => "Japanese",
        "ko" => "Korean",
        "yue" => "Cantonese",
        "fr" => "French",
        "es" => "Spanish",
        "de" => "German",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ru" => "Russian",
        "ar" => "Arabic",
        "hi" => "Hindi",
        "th" => "Thai",
        "vi" => "Vietnamese",
        "id" => "Indonesian",
        "ms" => "Malay",
        "tr" => "Turkish",
        "nl" => "Dutch",
        "pl" => "Polish",
        "sv" => "Swedish",
        "no" => "Norwegian",
        "da" => "Danish",
        "fi" => "Finnish",
        "cs" => "Czech",
        "ro" => "Romanian",
        "hu" => "Hungarian",
        "el" => "Greek",
        "he" => "Hebrew",
        "uk" => "Ukrainian",
        "bg" => "Bulgarian",
        "hr" => "Croatian",
        "sk" => "Slovak",
        "sl" => "Slovenian",
        "lt" => "Lithuanian",
        "lv" => "Latvian",
        "et" => "Estonian",
        "sr" => "Serbian",
        "tl" => "Tagalog",
        "my" => "Burmese",
        "bo" => "Tibetan",
        "ug" => "Uyghur",
        "mn" => "Mongolian",
        "am" => "Amharic",
        "sw" => "Swahili",
        "kk" => "Kazakh",
        "uz" => "Uzbek",
        "az" => "Azerbaijani",
        "ka" => "Georgian",
        "hy" => "Armenian",
        "ne" => "Nepali",
        "bn" => "Bengali",
        "ta" => "Tamil",
        "te" => "Telugu",
        "ur" => "Urdu",
        "fa" => "Persian",
        "lo" => "Lao",
        "km" => "Khmer",
        "ca" => "Catalan",
        "gl" => "Galician",
        "eu" => "Basque",
        "af" => "Afrikaans",
        other => other,
    }
}

const KNOWN_LANGUAGES: &[&str] = &[
    "English", "Chinese", "Japanese", "Korean", "Cantonese", "French", "Spanish", "German",
    "Italian", "Portuguese", "Russian", "Arabic", "Hindi", "Thai", "Vietnamese", "Indonesian",
    "Malay", "Turkish", "Dutch", "Polish", "Swedish", "Norwegian", "Danish", "Finnish",
    "Czech", "Romanian", "Hungarian", "Greek", "Hebrew", "Ukrainian", "Bulgarian",
    "Croatian", "Slovak", "Slovenian", "Lithuanian", "Latvian", "Estonian", "Serbian",
    "Tagalog", "Burmese", "Tibetan", "Uyghur", "Mongolian", "Amharic", "Swahili",
    "Kazakh", "Uzbek", "Azerbaijani", "Georgian", "Armenian", "Nepali", "Bengali",
    "Tamil", "Telugu", "Urdu", "Persian", "Lao", "Khmer", "Catalan", "Galician",
    "Basque", "Afrikaans",
];

fn strip_language_prefix(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("language ") {
        if let Some(newline_pos) = rest.find('\n') {
            return rest[newline_pos + 1..].to_string();
        }
        for lang in KNOWN_LANGUAGES {
            if let Some(after) = rest.strip_prefix(lang) {
                return after.to_string();
            }
        }
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
    fn test_strip_language_prefix_chinese_with_newline() {
        assert_eq!(
            strip_language_prefix("language Chinese\n你好世界"),
            "你好世界"
        );
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
    fn test_normalize_language_maps_codes() {
        assert_eq!(normalize_language("en"), "English");
        assert_eq!(normalize_language("zh"), "Chinese");
        assert_eq!(normalize_language("ja"), "Japanese");
        assert_eq!(normalize_language("ko"), "Korean");
        assert_eq!(normalize_language("yue"), "Cantonese");
        assert_eq!(normalize_language("af"), "Afrikaans");
    }

    #[test]
    fn test_normalize_language_passthrough() {
        assert_eq!(normalize_language("English"), "English");
        assert_eq!(normalize_language("unknown"), "unknown");
    }
}
