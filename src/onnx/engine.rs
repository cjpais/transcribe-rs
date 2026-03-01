use std::path::Path;

use crate::{TranscriptionEngine, TranscriptionResult};

use super::models::gigaam::GigaAMModel;
use super::models::moonshine::{MoonshineModel, StreamingModel};
use super::models::parakeet::ParakeetModel;
use super::models::sense_voice::SenseVoiceModel;
use super::types::*;

/// ONNX model selection for loading.
///
/// Each variant configures load-time parameters. Use the helper constructors
/// for common configurations.
pub enum Model {
    SenseVoice { quantization: Quantization },
    GigaAM,
    Moonshine { variant: MoonshineVariant },
    MoonshineStreaming { num_threads: usize },
    Parakeet { quantization: Quantization },
}

impl Default for Model {
    fn default() -> Self {
        Model::SenseVoice {
            quantization: Quantization::default(),
        }
    }
}

impl Model {
    pub fn sense_voice() -> Self {
        Model::SenseVoice {
            quantization: Quantization::FP32,
        }
    }

    pub fn sense_voice_int8() -> Self {
        Model::SenseVoice {
            quantization: Quantization::Int8,
        }
    }

    pub fn gigaam() -> Self {
        Model::GigaAM
    }

    pub fn moonshine_tiny() -> Self {
        Model::Moonshine {
            variant: MoonshineVariant::Tiny,
        }
    }

    pub fn moonshine_base() -> Self {
        Model::Moonshine {
            variant: MoonshineVariant::Base,
        }
    }

    pub fn moonshine(variant: MoonshineVariant) -> Self {
        Model::Moonshine { variant }
    }

    pub fn moonshine_streaming() -> Self {
        Model::MoonshineStreaming { num_threads: 0 }
    }

    pub fn parakeet() -> Self {
        Model::Parakeet {
            quantization: Quantization::FP32,
        }
    }

    pub fn parakeet_int8() -> Self {
        Model::Parakeet {
            quantization: Quantization::Int8,
        }
    }
}

/// Per-call inference parameters.
///
/// Each model uses only the fields relevant to it and ignores the rest.
#[derive(Debug, Clone, Default)]
pub struct InferenceParams {
    /// SenseVoice: language for transcription
    pub language: Option<Language>,
    /// SenseVoice: whether to apply inverse text normalization
    pub use_itn: Option<bool>,
    /// Parakeet: timestamp granularity
    pub timestamp_granularity: Option<TimestampGranularity>,
    /// Moonshine: maximum number of tokens to generate
    pub max_length: Option<usize>,
}

enum ActiveModel {
    SenseVoice(SenseVoiceModel),
    GigaAM(GigaAMModel),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
}

/// Unified ONNX transcription engine.
///
/// Supports all ONNX-based models behind a single interface.
/// Load a model with `load()`, then call `transcribe_samples()` or `transcribe_file()`.
pub struct Engine {
    active_model: Option<ActiveModel>,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            active_model: None,
        }
    }

    /// Load a model from the specified path.
    pub fn load(&mut self, path: &Path, model: Model) -> Result<(), Box<dyn std::error::Error>> {
        self.unload();
        self.active_model = Some(match model {
            Model::SenseVoice { ref quantization } => {
                ActiveModel::SenseVoice(SenseVoiceModel::load(path, quantization)?)
            }
            Model::GigaAM => ActiveModel::GigaAM(GigaAMModel::load(path)?),
            Model::Parakeet { ref quantization } => {
                ActiveModel::Parakeet(ParakeetModel::load(path, quantization)?)
            }
            Model::Moonshine { variant } => {
                ActiveModel::Moonshine(MoonshineModel::load(path, variant)?)
            }
            Model::MoonshineStreaming { num_threads } => {
                ActiveModel::MoonshineStreaming(StreamingModel::load(path, num_threads)?)
            }
        });
        Ok(())
    }

    /// Unload the current model and free resources.
    pub fn unload(&mut self) {
        self.active_model = None;
    }

    /// Transcribe audio samples with optional parameters.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        params: Option<InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let params = params.unwrap_or_default();
        match self
            .active_model
            .as_mut()
            .ok_or("Model not loaded. Call load() first.")?
        {
            ActiveModel::SenseVoice(m) => Ok(m.transcribe(samples, &params)?),
            ActiveModel::GigaAM(m) => Ok(m.transcribe(samples, &params)?),
            ActiveModel::Parakeet(m) => Ok(m.transcribe(samples, &params)?),
            ActiveModel::Moonshine(m) => Ok(m.transcribe(samples, &params)?),
            ActiveModel::MoonshineStreaming(m) => Ok(m.transcribe(samples, &params)?),
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.unload();
    }
}

impl TranscriptionEngine for Engine {
    type InferenceParams = InferenceParams;
    type ModelParams = Model;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.load(model_path, params)
    }

    fn unload_model(&mut self) {
        self.unload();
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.transcribe(&samples, params)
    }
}
