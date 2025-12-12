use std::path::{Path, PathBuf};

use crate::{TranscriptionEngine, TranscriptionResult};

use super::model::MoonshineModel;

const SAMPLE_RATE: u32 = 16000;

/// Moonshine model variant.
///
/// Each variant has different parameters for number of layers, heads, and head dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// English model (6 layers, token_rate=6)
    Tiny,
    /// Arabic model (6 layers, token_rate=13)
    TinyAr,
    /// Chinese model (6 layers, token_rate=13)
    TinyZh,
    /// Japanese model (6 layers, token_rate=13)
    TinyJa,
    /// Korean model (6 layers, token_rate=13)
    TinyKo,
    /// Ukrainian model (6 layers, token_rate=8)
    TinyUk,
    /// Vietnamese model (6 layers, token_rate=13)
    TinyVi,
    /// English model (8 layers, token_rate=6)
    Base,
    /// Spanish model (8 layers, token_rate=6)
    BaseEs,
}

impl ModelVariant {
    /// Number of decoder layers for this variant.
    pub fn num_layers(&self) -> usize {
        match self {
            ModelVariant::Tiny
            | ModelVariant::TinyAr
            | ModelVariant::TinyZh
            | ModelVariant::TinyJa
            | ModelVariant::TinyKo
            | ModelVariant::TinyUk
            | ModelVariant::TinyVi => 6,
            ModelVariant::Base | ModelVariant::BaseEs => 8,
        }
    }

    /// Number of key-value heads for attention (same for all variants).
    pub fn num_key_value_heads(&self) -> usize {
        8
    }

    /// Head dimension for attention.
    pub fn head_dim(&self) -> usize {
        match self {
            ModelVariant::Tiny
            | ModelVariant::TinyAr
            | ModelVariant::TinyZh
            | ModelVariant::TinyJa
            | ModelVariant::TinyKo
            | ModelVariant::TinyUk
            | ModelVariant::TinyVi => 36,
            ModelVariant::Base | ModelVariant::BaseEs => 52,
        }
    }

    /// Token generation rate (tokens per second of audio).
    pub fn token_rate(&self) -> usize {
        match self {
            ModelVariant::Tiny | ModelVariant::Base | ModelVariant::BaseEs => 6,
            ModelVariant::TinyUk => 8,
            ModelVariant::TinyAr
            | ModelVariant::TinyZh
            | ModelVariant::TinyJa
            | ModelVariant::TinyKo
            | ModelVariant::TinyVi => 13,
        }
    }
}

impl Default for ModelVariant {
    fn default() -> Self {
        ModelVariant::Tiny
    }
}

/// Parameters for loading a Moonshine model.
#[derive(Debug, Clone, Default)]
pub struct MoonshineModelParams {
    /// The model variant to load.
    pub variant: ModelVariant,
}

impl MoonshineModelParams {
    /// Create params for the Tiny English model.
    pub fn tiny() -> Self {
        Self {
            variant: ModelVariant::Tiny,
        }
    }

    /// Create params for the Base English model.
    pub fn base() -> Self {
        Self {
            variant: ModelVariant::Base,
        }
    }

    /// Create params for a specific variant.
    pub fn variant(variant: ModelVariant) -> Self {
        Self { variant }
    }
}

/// Parameters for inference.
#[derive(Debug, Clone, Default)]
pub struct MoonshineInferenceParams {
    /// Maximum number of tokens to generate.
    /// If None, automatically calculated from audio duration and model's token_rate.
    pub max_length: Option<usize>,
}

/// Moonshine ONNX transcription engine.
///
/// Implements the `TranscriptionEngine` trait for Moonshine models.
pub struct MoonshineEngine {
    loaded_model_path: Option<PathBuf>,
    model: Option<MoonshineModel>,
    variant: ModelVariant,
}

impl MoonshineEngine {
    /// Create a new Moonshine engine (model not loaded).
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            model: None,
            variant: ModelVariant::default(),
        }
    }
}

impl Default for MoonshineEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MoonshineEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for MoonshineEngine {
    type InferenceParams = MoonshineInferenceParams;
    type ModelParams = MoonshineModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Unload any existing model
        self.unload_model();

        self.variant = params.variant;
        self.model = Some(MoonshineModel::new(model_path, params.variant)?);
        self.loaded_model_path = Some(model_path.to_path_buf());

        log::info!(
            "Loaded Moonshine {:?} model from {:?}",
            params.variant,
            model_path
        );

        Ok(())
    }

    fn unload_model(&mut self) {
        if self.model.is_some() {
            log::debug!("Unloading Moonshine model");
            self.model = None;
            self.loaded_model_path = None;
        }
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| super::model::MoonshineError::ModelNotLoaded)?;

        let params = params.unwrap_or_default();

        // Calculate max_length from audio duration if not provided
        let max_length = params.max_length.unwrap_or_else(|| {
            let audio_duration_sec = samples.len() as f32 / SAMPLE_RATE as f32;
            (audio_duration_sec * self.variant.token_rate() as f32).ceil() as usize
        });

        log::debug!(
            "Transcribing {} samples ({:.2}s), max_length={}",
            samples.len(),
            samples.len() as f32 / SAMPLE_RATE as f32,
            max_length
        );

        // Generate tokens
        let tokens = model.generate(&samples, max_length)?;

        // Decode tokens to text
        let text = model.decode_tokens(&tokens)?;

        Ok(TranscriptionResult {
            text,
            segments: None, // Moonshine doesn't provide timestamp segments
        })
    }
}
