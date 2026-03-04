use std::path::{Path, PathBuf};

use crate::{TranscriptionEngine, TranscriptionResult};

use super::model::ZipformerTransducerModel;

#[derive(Debug, Clone, Default, PartialEq)]
pub enum QuantizationType {
    FP32,
    #[default]
    Int8,
}

#[derive(Debug, Clone, Default)]
pub struct ZipformerTransducerModelParams {
    pub quantization: QuantizationType,
}

impl ZipformerTransducerModelParams {
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

#[derive(Debug, Clone, Default)]
pub struct ZipformerTransducerInferenceParams {}

pub struct ZipformerTransducerEngine {
    loaded_model_path: Option<PathBuf>,
    model: Option<ZipformerTransducerModel>,
}

impl ZipformerTransducerEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            model: None,
        }
    }
}

impl Default for ZipformerTransducerEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ZipformerTransducerEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for ZipformerTransducerEngine {
    type InferenceParams = ZipformerTransducerInferenceParams;
    type ModelParams = ZipformerTransducerModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.unload_model();

        let quantized = matches!(params.quantization, QuantizationType::Int8);
        let model = ZipformerTransducerModel::new(model_path, quantized)?;
        self.model = Some(model);
        self.loaded_model_path = Some(model_path.to_path_buf());

        log::info!("Loaded Zipformer Transducer model from {:?}", model_path);
        Ok(())
    }

    fn unload_model(&mut self) {
        if self.model.is_some() {
            log::debug!("Unloading Zipformer Transducer model");
            self.model = None;
            self.loaded_model_path = None;
        }
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let model = self
            .model
            .as_mut()
            .ok_or("Model not loaded. Call load_model() first.")?;

        let result = model.transcribe(&samples)?;
        log::debug!("Decoded {} zipformer transducer tokens", result.token_ids.len());

        Ok(TranscriptionResult {
            text: result.text,
            segments: None,
        })
    }
}
