//! Whisper speech recognition via whisper-rs (whisper.cpp bindings).
//!
//! # Model Format
//!
//! Whisper expects a single model file in GGML format, typically with names like:
//! - `whisper-tiny.bin`
//! - `whisper-base.bin`
//! - `whisper-small.bin`
//! - `whisper-medium.bin`
//! - `whisper-large.bin`
//! - Quantized variants like `whisper-medium-q4_1.bin`
//!
//! # Examples
//!
//! ```rust,no_run
//! use transcribe_rs::whisper_cpp::{WhisperEngine, WhisperModelParams};
//! use transcribe_rs::SpeechModel;
//! use std::path::PathBuf;
//!
//! let mut engine = WhisperEngine::new();
//! engine.load_model(&PathBuf::from("models/whisper-medium-q4_1.bin"))?;
//!
//! let result = engine.transcribe(&[], Some("en"))?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{ModelCapabilities, SpeechModel, TranscribeError, TranscriptionResult, TranscriptionSegment};
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Whisper",
    languages: &["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"],
    supports_timestamps: true,
    supports_translation: true,
    supports_streaming: false,
};

/// Parameters for configuring Whisper model loading.
#[derive(Debug, Clone)]
pub struct WhisperModelParams {
    pub use_gpu: bool,
}

impl Default for WhisperModelParams {
    fn default() -> Self {
        Self { use_gpu: true }
    }
}

/// Parameters for configuring Whisper inference behavior.
#[derive(Debug, Clone)]
pub struct WhisperInferenceParams {
    /// Target language for transcription (e.g., "en", "es", "fr").
    /// If None, Whisper will auto-detect the language.
    pub language: Option<String>,

    /// Whether to translate the transcription to English.
    pub translate: bool,

    /// Whether to print special tokens in the output
    pub print_special: bool,

    /// Whether to print progress information during transcription
    pub print_progress: bool,

    /// Whether to print results in real-time as they're generated
    pub print_realtime: bool,

    /// Whether to include timestamp information in the output
    pub print_timestamps: bool,

    /// Whether to suppress blank/empty segments in the output
    pub suppress_blank: bool,

    /// Whether to suppress non-speech tokens
    pub suppress_non_speech_tokens: bool,

    /// Threshold for detecting silence/no-speech segments (0.0-1.0).
    pub no_speech_thold: f32,

    /// Initial prompt to provide context to the model.
    pub initial_prompt: Option<String>,
}

impl Default for WhisperInferenceParams {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            print_special: false,
            print_progress: false,
            print_realtime: false,
            print_timestamps: false,
            suppress_blank: true,
            suppress_non_speech_tokens: true,
            no_speech_thold: 0.2,
            initial_prompt: None,
        }
    }
}

/// Whisper speech recognition engine.
pub struct WhisperEngine {
    loaded_model_path: Option<PathBuf>,
    state: Option<whisper_rs::WhisperState>,
    context: Option<whisper_rs::WhisperContext>,
}

impl Default for WhisperEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperEngine {
    /// Create a new Whisper engine instance (unloaded).
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            state: None,
            context: None,
        }
    }

    /// Load a Whisper model with default parameters.
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), TranscribeError> {
        self.load_model_with_params(model_path, WhisperModelParams::default())
    }

    /// Load a Whisper model with custom parameters.
    pub fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: WhisperModelParams,
    ) -> Result<(), TranscribeError> {
        let mut context_params = WhisperContextParameters::default();
        context_params.use_gpu = params.use_gpu;
        let context = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            context_params,
        )
        .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let state = context
            .create_state()
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        self.context = Some(context);
        self.state = Some(state);
        self.loaded_model_path = Some(model_path.to_path_buf());
        Ok(())
    }

    /// Unload the current model.
    pub fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.state = None;
        self.context = None;
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &WhisperInferenceParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples.to_vec(), params)
    }

    fn infer(
        &mut self,
        samples: Vec<f32>,
        params: &WhisperInferenceParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| TranscribeError::Inference("Model not loaded. Call load_model() first.".to_string()))?;

        let mut full_params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 3,
            patience: -1.0,
        });
        full_params.set_language(params.language.as_deref());
        full_params.set_translate(params.translate);
        full_params.set_print_special(params.print_special);
        full_params.set_print_progress(params.print_progress);
        full_params.set_print_realtime(params.print_realtime);
        full_params.set_print_timestamps(params.print_timestamps);
        full_params.set_suppress_blank(params.suppress_blank);
        full_params.set_suppress_non_speech_tokens(params.suppress_non_speech_tokens);
        full_params.set_no_speech_thold(params.no_speech_thold);

        if let Some(ref prompt) = params.initial_prompt {
            full_params.set_initial_prompt(prompt);
        }

        state
            .full(full_params, &samples)
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let num_segments = state
            .full_n_segments()
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| TranscribeError::Inference(e.to_string()))?;
            let start = state
                .full_get_segment_t0(i)
                .map_err(|e| TranscribeError::Inference(e.to_string()))? as f32
                / 100.0;
            let end = state
                .full_get_segment_t1(i)
                .map_err(|e| TranscribeError::Inference(e.to_string()))? as f32
                / 100.0;

            segments.push(TranscriptionSegment {
                start,
                end,
                text: text.clone(),
            });
            full_text.push_str(&text);
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            segments: Some(segments),
        })
    }
}

impl Drop for WhisperEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl SpeechModel for WhisperEngine {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = WhisperInferenceParams {
            language: language.map(|s| s.to_string()),
            ..Default::default()
        };
        self.infer(samples.to_vec(), &params)
    }
}
