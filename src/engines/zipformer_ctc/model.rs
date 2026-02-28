use std::path::Path;

use ndarray::{Array2, ArrayView2};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use super::super::zipformer_common::*;

#[derive(thiserror::Error, Debug)]
pub enum ZipformerCtcError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Tokens file not found: {0}")]
    TokensNotFound(String),
    #[error("Invalid model: {0}")]
    InvalidModel(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
}

pub struct ZipformerCtcModel {
    session: Session,
    symbol_table: SymbolTable,
    blank_id: i32,
    x_input_name: String,
    x_lens_input_name: String,
    log_probs_output_name: String,
    log_probs_len_output_name: String,
}

impl Drop for ZipformerCtcModel {
    fn drop(&mut self) {
        log::debug!("Dropping ZipformerCtcModel");
    }
}

impl ZipformerCtcModel {
    pub fn new(model_dir: &Path, quantized: bool) -> Result<Self, ZipformerCtcError> {
        let model_path = Self::find_model_file(model_dir, quantized)?;
        let tokens_path = model_dir.join("tokens.txt");

        if !tokens_path.exists() {
            return Err(ZipformerCtcError::TokensNotFound(
                tokens_path.display().to_string(),
            ));
        }

        log::info!(
            "Loading Zipformer CTC model from {:?}...",
            model_path
        );
        let session = Self::init_session(&model_path)?;
        let symbol_table = SymbolTable::load(&tokens_path)?;

        let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

        if input_names.is_empty() || output_names.is_empty() {
            return Err(ZipformerCtcError::InvalidModel(
                "Model has no inputs or outputs".to_string(),
            ));
        }

        // Detect streaming models: they have cached_* state inputs (>2 inputs)
        // and fixed time dimension. These require a dedicated streaming engine.
        let has_cached_inputs = input_names.iter().any(|n| n.starts_with("cached_"));
        if has_cached_inputs {
            return Err(ZipformerCtcError::InvalidModel(
                "This is a streaming model (has cached_* state inputs). \
                 Streaming models require fixed-size chunk input and state management. \
                 Please use an offline (non-streaming) model instead."
                    .to_string(),
            ));
        }

        // Model inputs: x [N, T, 80], x_lens [N]
        let x_input_name = input_names
            .iter()
            .find(|n| n.as_str() == "x")
            .cloned()
            .unwrap_or_else(|| input_names[0].clone());

        let x_lens_input_name = input_names
            .iter()
            .find(|n| n.as_str() == "x_lens")
            .or_else(|| input_names.iter().find(|n| n.contains("lens")))
            .cloned()
            .unwrap_or_else(|| {
                if input_names.len() > 1 {
                    input_names[1].clone()
                } else {
                    input_names[0].clone()
                }
            });

        // Model outputs: log_probs [N, T, vocab_size], log_probs_len [N]
        let log_probs_output_name = output_names
            .iter()
            .find(|n| n.as_str() == "log_probs")
            .cloned()
            .unwrap_or_else(|| output_names[0].clone());

        let log_probs_len_output_name = output_names
            .iter()
            .find(|n| n.as_str() == "log_probs_len")
            .cloned()
            .unwrap_or_else(|| {
                if output_names.len() > 1 {
                    output_names[1].clone()
                } else {
                    output_names[0].clone()
                }
            });

        log::info!(
            "Zipformer CTC I/O: x='{}', x_lens='{}', log_probs='{}', log_probs_len='{}'",
            x_input_name,
            x_lens_input_name,
            log_probs_output_name,
            log_probs_len_output_name
        );

        // CTC blank_id is typically 0
        let blank_id = 0;

        Ok(Self {
            session,
            symbol_table,
            blank_id,
            x_input_name,
            x_lens_input_name,
            log_probs_output_name,
            log_probs_len_output_name,
        })
    }

    /// Find the CTC model ONNX file, trying exact names first then globbing.
    fn find_model_file(
        model_dir: &Path,
        quantized: bool,
    ) -> Result<std::path::PathBuf, ZipformerCtcError> {
        // Try exact names first
        if quantized {
            let int8_path = model_dir.join("model.int8.onnx");
            if int8_path.exists() {
                return Ok(int8_path);
            }
        }
        let fp32_path = model_dir.join("model.onnx");
        if fp32_path.exists() {
            return Ok(fp32_path);
        }

        // Fallback: scan directory for any .onnx file
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            let onnx_files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "onnx")
                        .unwrap_or(false)
                })
                .collect();

            // Prefer int8 variant when quantized, non-int8 otherwise
            let preferred = onnx_files.iter().find(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if quantized {
                    name.contains("int8")
                } else {
                    !name.contains("int8")
                }
            });

            if let Some(entry) = preferred.or_else(|| onnx_files.first()) {
                let path = entry.path();
                log::info!("Found model file via fallback: {:?}", path);
                return Ok(path);
            }
        }

        Err(ZipformerCtcError::ModelNotFound(format!(
            "No .onnx model found in {}",
            model_dir.display()
        )))
    }

    fn init_session(path: &Path) -> Result<Session, ZipformerCtcError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .with_parallel_execution(true)?
            .commit_from_file(path)?;

        for input in &session.inputs {
            log::info!(
                "Model input: name={}, type={:?}",
                input.name,
                input.input_type
            );
        }
        for output in &session.outputs {
            log::info!(
                "Model output: name={}, type={:?}",
                output.name,
                output.output_type
            );
        }
        Ok(session)
    }

    fn compute_fbank(&self, samples: &[f32]) -> Array2<f32> {
        compute_fbank_kaldi(samples, &FbankConfig::default())
    }

    fn forward(
        &mut self,
        features: &ArrayView2<f32>,
    ) -> Result<(Array2<f32>, i32), ZipformerCtcError> {
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, features.nrows(), features.ncols()))?;
        let x_len = features.nrows() as i32;

        let feat_dyn = feat_3d.into_dyn();
        let len_dyn = ndarray::arr1(&[x_len as i64]).into_dyn();

        let inputs = inputs![
            self.x_input_name.as_str() => TensorRef::from_array_view(feat_dyn.view())?,
            self.x_lens_input_name.as_str() => TensorRef::from_array_view(len_dyn.view())?,
        ];

        let outputs = self.session.run(inputs)?;

        let log_probs = outputs
            .get(self.log_probs_output_name.as_str())
            .ok_or_else(|| {
                ZipformerCtcError::OutputNotFound(self.log_probs_output_name.clone())
            })?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()?;

        // Get output length and vocab size before consuming log_probs
        let time_steps = log_probs.shape()[1] as i32;
        let vocab_size = log_probs.shape()[2];
        let output_len = if let Some(v) = outputs.get(self.log_probs_len_output_name.as_str()) {
            if let Ok(arr) = v.try_extract_array::<i64>() {
                arr.as_slice().and_then(|s| s.first().copied()).map(|v| v as i32).unwrap_or(time_steps)
            } else {
                time_steps
            }
        } else {
            time_steps
        };

        // Convert log_probs from [1, T, V] to [T, V]
        let log_probs_2d = log_probs.into_shape_with_order((time_steps as usize, vocab_size))?;

        Ok((log_probs_2d, output_len))
    }

    fn decode_ctc(&self, log_probs: &Array2<f32>, output_len: i32) -> Vec<i32> {
        let seq_len = log_probs.shape()[0].min(output_len as usize);
        let vocab_size = log_probs.shape()[1];
        let mut token_ids = Vec::new();
        let mut prev_id: Option<i32> = None;

        for t in 0..seq_len {
            // Greedy decoding: find argmax
            let mut max_id = 0i32;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = log_probs[[t, v]];
                if val > max_val {
                    max_val = val;
                    max_id = v as i32;
                }
            }

            // CTC collapse: skip blanks and repeated tokens
            if max_id != self.blank_id && Some(max_id) != prev_id {
                token_ids.push(max_id);
            }
            prev_id = Some(max_id);
        }

        token_ids
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<ZipformerCtcResult, ZipformerCtcError> {
        let features = self.compute_fbank(samples);
        if features.nrows() == 0 {
            return Ok(ZipformerCtcResult {
                text: String::new(),
                token_ids: Vec::new(),
            });
        }

        log::debug!("Computed {} frames, {} dims", features.nrows(), features.ncols());

        let (log_probs, output_len) = self.forward(&features.view())?;
        log::debug!(
            "Forward pass done, output_len={}, log_probs shape=[{}, {}]",
            output_len,
            log_probs.shape()[0],
            log_probs.shape()[1]
        );

        let token_ids = self.decode_ctc(&log_probs, output_len);
        let text = self.symbol_table.decode(&token_ids);
        log::debug!("Decoded {} tokens: {}", token_ids.len(), text);

        Ok(ZipformerCtcResult { text, token_ids })
    }
}

pub struct ZipformerCtcResult {
    pub text: String,
    pub token_ids: Vec<i32>,
}
