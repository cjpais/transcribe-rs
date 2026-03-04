//! Batch (non-streaming) `TranscriptionEngine` implementation for Qwen3-ASR.

use std::path::{Path, PathBuf};

use crate::TranscriptionEngine;

use super::model::{Qwen3AsrModel, Qwen3Error, Qwen3ModelOptions};

/// Whether to use quantized (int8) model files.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum QuantizationType {
    #[default]
    FP32,
    Int8,
}

/// Parameters for loading a Qwen3-ASR model.
#[derive(Debug, Clone, Default)]
pub struct Qwen3ModelParams {
    pub quantization: QuantizationType,
}

impl Qwen3ModelParams {
    pub fn fp32() -> Self {
        Self {
            quantization: QuantizationType::FP32,
        }
    }

    pub fn int8() -> Self {
        Self {
            quantization: QuantizationType::Int8,
        }
    }
}

/// Parameters for configuring Qwen3-ASR inference.
#[derive(Debug, Clone, Default)]
pub struct Qwen3InferenceParams {
    /// Maximum number of tokens to decode. Default: 512.
    /// At ~8 tokens/second for English, 512 covers ~60s of audio.
    pub max_tokens: Option<usize>,
}

/// Batch transcription engine backed by Qwen3-ASR.
///
/// Processes complete audio in a single encoder-decoder pass, suitable for
/// the standard record-then-transcribe workflow.
///
/// This engine does not currently produce timestamp segments
/// (`TranscriptionResult.segments` is always `None`).
pub struct Qwen3Engine {
    model: Option<Qwen3AsrModel>,
    loaded_model_path: Option<PathBuf>,
}

impl Qwen3Engine {
    pub fn new() -> Self {
        Self {
            model: None,
            loaded_model_path: None,
        }
    }
}

impl Default for Qwen3Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Qwen3Engine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for Qwen3Engine {
    type InferenceParams = Qwen3InferenceParams;
    type ModelParams = Qwen3ModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.unload_model();

        let options = Qwen3ModelOptions {
            quantized: params.quantization == QuantizationType::Int8,
        };
        let model = Qwen3AsrModel::new(model_path, &options)?;
        self.model = Some(model);
        self.loaded_model_path = Some(model_path.to_path_buf());

        log::info!("Loaded Qwen3-ASR model from {:?}", model_path);
        Ok(())
    }

    fn unload_model(&mut self) {
        if self.model.is_some() {
            log::debug!("Unloading Qwen3-ASR model");
            self.model = None;
            self.loaded_model_path = None;
        }
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<crate::TranscriptionResult, Box<dyn std::error::Error>> {
        let model = self.model.as_mut().ok_or(Qwen3Error::ModelNotLoaded)?;

        let max_tokens = params.and_then(|p| p.max_tokens).unwrap_or(512);
        let raw = model.transcribe(&samples, max_tokens)?;
        let text = strip_language_prefix(&raw);

        Ok(crate::TranscriptionResult {
            text,
            segments: None,
        })
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
/// The model outputs `language <Name><transcription>` — the language name
/// (e.g. "English", "Chinese") immediately precedes the transcription text,
/// sometimes with a newline separator and sometimes without. We match against
/// known language names to find the exact boundary.
///
/// Known limitation: if the transcribed text starts with a substring that
/// matches a language name (e.g. "Georgian" → "Georgia is..."), the boundary
/// detection is ambiguous. This is inherent to the model's output format
/// when no separator character is present.
fn strip_language_prefix(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("language ") {
        // Try to match a known language name
        for lang in KNOWN_LANGUAGES {
            if let Some(after) = rest.strip_prefix(lang) {
                return after.trim_start_matches('\n').to_string();
            }
        }
        // Fallback: skip to first newline (original behaviour)
        if let Some(newline_pos) = rest.find('\n') {
            return rest[newline_pos + 1..].to_string();
        }
        return String::new();
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
}
