use std::path::{Path, PathBuf};

use crate::{TranscriptionEngine, TranscriptionResult};

use super::model::{GigaAMError, GigaAMModel};

/// GigaAM v3 ONNX transcription engine.
///
/// Implements the `TranscriptionEngine` trait for GigaAM v3 e2e_ctc models.
/// Supports Russian speech recognition with punctuation and Latin characters.
pub struct GigaAMEngine {
    loaded_model_path: Option<PathBuf>,
    model: Option<GigaAMModel>,
}

impl GigaAMEngine {
    /// Create a new GigaAM engine (model not loaded).
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            model: None,
        }
    }
}

impl Default for GigaAMEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GigaAMEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for GigaAMEngine {
    type InferenceParams = ();
    type ModelParams = ();

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        _params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.unload_model();

        self.model = Some(GigaAMModel::new(model_path)?);
        self.loaded_model_path = Some(model_path.to_path_buf());

        log::info!("Loaded GigaAM model from {:?}", model_path);
        Ok(())
    }

    fn unload_model(&mut self) {
        if self.model.is_some() {
            log::debug!("Unloading GigaAM model");
            self.model = None;
            self.loaded_model_path = None;
        }
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        _params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let model = self.model.as_mut().ok_or(GigaAMError::ModelNotLoaded)?;

        log::debug!(
            "Transcribing {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / 16000.0,
        );

        let text = model.transcribe(&samples)?;

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }
}
