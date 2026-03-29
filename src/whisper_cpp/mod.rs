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
//! Quantization is baked into the model file — pick the right file.
//!
//! # Examples
//!
//! ```rust,no_run
//! use transcribe_rs::whisper_cpp::WhisperEngine;
//! use transcribe_rs::SpeechModel;
//! use std::path::PathBuf;
//!
//! let mut engine = WhisperEngine::load(&PathBuf::from("models/whisper-medium-q4_1.bin"))?;
//!
//! let result = engine.transcribe(&[], &transcribe_rs::TranscribeOptions::default())?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod gpu;

use crate::accel::{get_whisper_accelerator, get_whisper_gpu_device, GPU_DEVICE_AUTO};
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
    TranscriptionSegment,
};
use gpu::auto_select_gpu_device;
use log::info;
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const MULTILINGUAL_LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];
const ENGLISH_ONLY_LANGUAGES: &[&str] = &["en"];

/// Parameters for configuring Whisper model loading.
#[derive(Debug, Clone)]
pub struct WhisperLoadParams {
    pub use_gpu: bool,
    /// Enable flash attention for faster inference.
    /// Cannot be used with DTW token-level timestamps.
    pub flash_attn: bool,
    /// GPU device index.
    ///
    /// - [`GPU_DEVICE_AUTO`] (default): automatically select the best GPU
    ///   (prefers dedicated over integrated, then most VRAM).
    /// - `0, 1, 2, …`: use a specific device by backend index.
    pub gpu_device: i32,
}

impl Default for WhisperLoadParams {
    fn default() -> Self {
        Self {
            use_gpu: true,
            flash_attn: true,
            gpu_device: GPU_DEVICE_AUTO,
        }
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

    /// Number of CPU threads for decoding. 0 uses the whisper.cpp default (min(4, num_cores)).
    pub n_threads: i32,

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
            n_threads: 0,
            initial_prompt: None,
        }
    }
}

/// Whisper speech recognition engine.
pub struct WhisperEngine {
    state: whisper_rs::WhisperState,
    #[allow(dead_code)] // context must stay alive — it owns the C memory backing `state`
    context: whisper_rs::WhisperContext,
    is_multilingual: bool,
}

impl WhisperEngine {
    /// Load a Whisper model, respecting the global accelerator and GPU device preferences.
    ///
    /// Use [`load_with_params`](Self::load_with_params) for explicit control.
    pub fn load(model_path: &Path) -> Result<Self, TranscribeError> {
        let params = WhisperLoadParams {
            use_gpu: get_whisper_accelerator().use_gpu(),
            gpu_device: get_whisper_gpu_device(),
            ..Default::default()
        };
        Self::load_with_params(model_path, params)
    }

    /// Load a Whisper model with custom parameters.
    ///
    /// When `params.gpu_device` is [`GPU_DEVICE_AUTO`] and GPU is enabled,
    /// the best GPU is selected automatically (preferring dedicated over
    /// integrated, then most VRAM).
    pub fn load_with_params(
        model_path: &Path,
        params: WhisperLoadParams,
    ) -> Result<Self, TranscribeError> {
        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path.to_path_buf()));
        }

        let gpu_device = if !params.use_gpu {
            0
        } else if params.gpu_device == GPU_DEVICE_AUTO {
            auto_select_gpu_device()
        } else {
            info!("Using user-selected GPU device {}", params.gpu_device);
            params.gpu_device
        };

        let mut context_params = WhisperContextParameters::default();
        context_params.use_gpu = params.use_gpu;
        context_params.flash_attn = params.flash_attn;
        context_params.gpu_device = gpu_device;
        let context = WhisperContext::new_with_params(model_path.to_str().unwrap(), context_params)
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let is_multilingual = context.is_multilingual();

        let state = context
            .create_state()
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        Ok(Self {
            state,
            context,
            is_multilingual,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &WhisperInferenceParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples, params)
    }

    fn infer(
        &mut self,
        samples: &[f32],
        params: &WhisperInferenceParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
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
        full_params.set_suppress_nst(params.suppress_non_speech_tokens);
        full_params.set_no_speech_thold(params.no_speech_thold);
        if params.n_threads > 0 {
            full_params.set_n_threads(params.n_threads);
        }

        if let Some(ref prompt) = params.initial_prompt {
            full_params.set_initial_prompt(prompt);
        }

        self.state
            .full(full_params, samples)
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let num_segments = self.state.full_n_segments();

        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let segment = self
                .state
                .get_segment(i)
                .ok_or_else(|| TranscribeError::Inference(format!("segment {i} out of bounds")))?;
            let text = segment
                .to_str()
                .map_err(|e| TranscribeError::Inference(e.to_string()))?;
            let start = segment.start_timestamp() as f32 / 100.0;
            let end = segment.end_timestamp() as f32 / 100.0;

            segments.push(TranscriptionSegment {
                start,
                end,
                text: text.to_string(),
            });
            full_text.push_str(text);
        }

        Ok(TranscriptionResult {
            text: full_text.trim().to_string(),
            segments: Some(segments),
        })
    }
}

impl SpeechModel for WhisperEngine {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            name: "Whisper",
            engine_id: "whisper_cpp",
            sample_rate: 16000,
            languages: if self.is_multilingual {
                MULTILINGUAL_LANGUAGES
            } else {
                ENGLISH_ONLY_LANGUAGES
            },
            supports_timestamps: true,
            supports_translation: self.is_multilingual,
            supports_streaming: false,
        }
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = WhisperInferenceParams {
            language: options.language.clone(),
            translate: options.translate,
            ..Default::default()
        };
        self.infer(samples, &params)
    }
}
