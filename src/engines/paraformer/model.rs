use std::path::Path;

use ndarray::{Array1, Array2, Array3, ArrayView2};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use super::features::{apply_lfr, apply_mean_cmvn, compute_fbank, FbankConfig};
use super::tokens::SymbolTable;

#[derive(thiserror::Error, Debug)]
pub enum ParaformerError {
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

#[derive(Debug, Clone)]
pub struct ParaformerMetadata {
    pub lfr_window_size: usize,
    pub lfr_window_shift: usize,
    pub blank_id: i32,
    pub sos_id: i32,
    pub eos_id: i32,
}

impl Default for ParaformerMetadata {
    fn default() -> Self {
        Self {
            lfr_window_size: 7,
            lfr_window_shift: 6,
            blank_id: 0,
            sos_id: 1,
            eos_id: 2,
        }
    }
}

pub struct ParaformerModel {
    session: Session,
    symbol_table: SymbolTable,
    metadata: ParaformerMetadata,
    cmvn_mean: Option<Array1<f32>>,
    speech_input_name: String,
    speech_lengths_input_name: String,
    logits_output_name: String,
    token_num_output_name: Option<String>,
}

impl Drop for ParaformerModel {
    fn drop(&mut self) {
        log::debug!("Dropping ParaformerModel");
    }
}

impl ParaformerModel {
    pub fn new(model_dir: &Path, quantized: bool) -> Result<Self, ParaformerError> {
        let model_path = if quantized {
            let int8_path = model_dir.join("model.int8.onnx");
            if int8_path.exists() {
                int8_path
            } else {
                log::warn!("Quantized model not found, falling back to model.onnx");
                model_dir.join("model.onnx")
            }
        } else {
            model_dir.join("model.onnx")
        };
        let tokens_path = model_dir.join("tokens.txt");
        let cmvn_path = model_dir.join("am.mvn");

        if !model_path.exists() {
            return Err(ParaformerError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }
        if !tokens_path.exists() {
            return Err(ParaformerError::TokensNotFound(
                tokens_path.display().to_string(),
            ));
        }

        log::info!("Loading Paraformer model from {:?}...", model_path);
        let session = Self::init_session(&model_path)?;
        let symbol_table = SymbolTable::load(&tokens_path)?;
        let metadata = Self::parse_metadata(&session)?;

        let input_names: Vec<String> = session.inputs.iter().map(|i| i.name.clone()).collect();
        let output_names: Vec<String> = session.outputs.iter().map(|o| o.name.clone()).collect();

        if input_names.is_empty() || output_names.is_empty() {
            return Err(ParaformerError::InvalidModel(
                "Model has no inputs or outputs".to_string(),
            ));
        }

        let speech_input_name = input_names
            .iter()
            .find(|n| n.as_str() == "speech")
            .or_else(|| input_names.iter().find(|n| n.contains("speech")))
            .cloned()
            .unwrap_or_else(|| input_names[0].clone());

        let speech_lengths_input_name = input_names
            .iter()
            .find(|n| n.as_str() == "speech_lengths")
            .or_else(|| input_names.iter().find(|n| n.contains("length")))
            .cloned()
            .unwrap_or_else(|| {
                if input_names.len() > 1 {
                    input_names[1].clone()
                } else {
                    input_names[0].clone()
                }
            });

        let logits_output_name = output_names
            .iter()
            .find(|n| n.as_str() == "logits")
            .or_else(|| output_names.iter().find(|n| n.contains("logit")))
            .cloned()
            .unwrap_or_else(|| output_names[0].clone());

        let token_num_output_name = output_names
            .iter()
            .find(|n| n.as_str() == "token_num")
            .or_else(|| output_names.iter().find(|n| n.contains("token_num")))
            .cloned();

        log::info!(
            "Paraformer I/O: speech='{}', speech_lengths='{}', logits='{}', token_num={:?}",
            speech_input_name,
            speech_lengths_input_name,
            logits_output_name,
            token_num_output_name
        );
        log::info!(
            "Metadata: lfr_window_size={}, lfr_window_shift={}, blank_id={}, sos_id={}, eos_id={}",
            metadata.lfr_window_size,
            metadata.lfr_window_shift,
            metadata.blank_id,
            metadata.sos_id,
            metadata.eos_id
        );

        let expected_dim = 80 * metadata.lfr_window_size;
        let cmvn_mean = match Self::read_meta_float_vec(&session, "neg_mean") {
            Ok(Some(v)) if v.len() >= expected_dim => {
                let mean = Array1::from_vec(v.into_iter().take(expected_dim).collect());
                log::info!("Loaded CMVN mean from ONNX metadata, dims={}", mean.len());
                Some(mean)
            }
            Ok(Some(v)) => {
                log::warn!(
                    "ONNX metadata neg_mean dims={} < expected {}, fallback to am.mvn",
                    v.len(),
                    expected_dim
                );
                Self::load_cmvn_from_file(&cmvn_path, expected_dim)
            }
            Ok(None) | Err(_) => Self::load_cmvn_from_file(&cmvn_path, expected_dim),
        };

        Ok(Self {
            session,
            symbol_table,
            metadata,
            cmvn_mean,
            speech_input_name,
            speech_lengths_input_name,
            logits_output_name,
            token_num_output_name,
        })
    }

    fn init_session(path: &Path) -> Result<Session, ParaformerError> {
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

    fn read_meta_i32(
        session: &Session,
        key: &str,
        default: Option<i32>,
    ) -> Result<i32, ParaformerError> {
        let meta = session.metadata()?;
        let value = meta.custom(key)?;
        match value {
            Some(v) => v.parse::<i32>().map_err(|e| {
                ParaformerError::InvalidModel(format!("Failed to parse metadata '{}' : {}", key, e))
            }),
            None => default.ok_or_else(|| {
                ParaformerError::InvalidModel(format!("Missing required metadata key: {}", key))
            }),
        }
    }

    fn read_meta_float_vec(
        session: &Session,
        key: &str,
    ) -> Result<Option<Vec<f32>>, ParaformerError> {
        let meta = session.metadata()?;
        let value = meta.custom(key)?;
        let Some(v) = value else {
            return Ok(None);
        };

        let mut out = Vec::new();
        for item in v.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            out.push(item.parse::<f32>().map_err(|e| {
                ParaformerError::InvalidModel(format!(
                    "Failed to parse metadata float '{}' in key '{}': {}",
                    item, key, e
                ))
            })?);
        }
        Ok(Some(out))
    }

    fn parse_metadata(session: &Session) -> Result<ParaformerMetadata, ParaformerError> {
        // Most public paraformer exports do not include these keys. Keep defaults.
        let mut meta = ParaformerMetadata::default();

        if let Ok(v) = Self::read_meta_i32(
            session,
            "lfr_window_size",
            Some(meta.lfr_window_size as i32),
        ) {
            meta.lfr_window_size = v.max(1) as usize;
        }
        if let Ok(v) = Self::read_meta_i32(
            session,
            "lfr_window_shift",
            Some(meta.lfr_window_shift as i32),
        ) {
            meta.lfr_window_shift = v.max(1) as usize;
        }
        if let Ok(v) = Self::read_meta_i32(session, "blank_id", Some(meta.blank_id)) {
            meta.blank_id = v;
        }
        if let Ok(v) = Self::read_meta_i32(session, "sos_id", Some(meta.sos_id)) {
            meta.sos_id = v;
        }
        if let Ok(v) = Self::read_meta_i32(session, "eos_id", Some(meta.eos_id)) {
            meta.eos_id = v;
        }

        Ok(meta)
    }

    fn load_cmvn_mean(
        path: &Path,
        target_dim: usize,
    ) -> Result<Option<Array1<f32>>, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let Some(start_idx) = content.find("<LearnRateCoef>") else {
            return Ok(None);
        };
        let rest = &content[start_idx..];
        let Some(lb_rel) = rest.find('[') else {
            return Ok(None);
        };
        let Some(rb_rel) = rest.find(']') else {
            return Ok(None);
        };
        if rb_rel <= lb_rel {
            return Ok(None);
        }

        let body = &rest[lb_rel + 1..rb_rel];
        let mut values = Vec::new();
        for tok in body.split_whitespace() {
            if let Ok(v) = tok.parse::<f32>() {
                values.push(v);
            }
        }

        if values.len() < target_dim {
            return Ok(None);
        }

        Ok(Some(Array1::from_vec(
            values.into_iter().take(target_dim).collect(),
        )))
    }

    fn load_cmvn_from_file(cmvn_path: &Path, expected_dim: usize) -> Option<Array1<f32>> {
        if cmvn_path.exists() {
            match Self::load_cmvn_mean(cmvn_path, expected_dim) {
                Ok(Some(mean)) => {
                    log::info!("Loaded CMVN mean from am.mvn, dims={}", mean.len());
                    Some(mean)
                }
                Ok(None) => {
                    log::warn!("CMVN file exists but no usable mean parsed");
                    None
                }
                Err(e) => {
                    log::warn!("Failed to parse CMVN from {:?}: {}", cmvn_path, e);
                    None
                }
            }
        } else {
            log::warn!("CMVN file not found: {:?}", cmvn_path);
            None
        }
    }

    fn compute_features(&self, samples: &[f32]) -> Array2<f32> {
        let fbank = compute_fbank(samples, &FbankConfig::default());
        let mut features = apply_lfr(
            &fbank,
            self.metadata.lfr_window_size,
            self.metadata.lfr_window_shift,
        );

        if let Some(mean) = &self.cmvn_mean {
            apply_mean_cmvn(&mut features, mean);
        }

        features
    }

    fn forward(
        &mut self,
        features: &ArrayView2<f32>,
    ) -> Result<(Array3<f32>, Option<i32>), ParaformerError> {
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, features.nrows(), features.ncols()))?;
        let speech_len = ndarray::arr1(&[features.nrows() as i32]);

        let feat_dyn = feat_3d.into_dyn();
        let len_dyn = speech_len.into_dyn();

        let inputs = inputs![
            self.speech_input_name.as_str() => TensorRef::from_array_view(feat_dyn.view())?,
            self.speech_lengths_input_name.as_str() => TensorRef::from_array_view(len_dyn.view())?,
        ];

        let outputs = self.session.run(inputs)?;

        let logits = outputs
            .get(self.logits_output_name.as_str())
            .ok_or_else(|| ParaformerError::OutputNotFound(self.logits_output_name.clone()))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()?;

        let token_num = if let Some(name) = &self.token_num_output_name {
            if let Some(v) = outputs.get(name.as_str()) {
                let arr = v.try_extract_array::<i32>()?;
                arr.as_slice().and_then(|s| s.first().copied())
            } else {
                None
            }
        } else {
            None
        };

        Ok((logits, token_num))
    }

    fn decode_logits(&self, logits: &Array3<f32>, token_num: Option<i32>) -> Vec<i32> {
        let seq_len = logits.shape()[1];
        let vocab_size = logits.shape()[2];
        let max_steps = token_num.unwrap_or(seq_len as i32).max(0) as usize;
        let mut token_ids = Vec::with_capacity(max_steps.min(seq_len));

        for t in 0..max_steps.min(seq_len) {
            let mut max_id = 0i32;
            let mut max_val = f32::NEG_INFINITY;
            for v in 0..vocab_size {
                let val = logits[[0, t, v]];
                if val > max_val {
                    max_val = val;
                    max_id = v as i32;
                }
            }

            if max_id == self.metadata.eos_id {
                break;
            }
            if max_id == self.metadata.blank_id || max_id == self.metadata.sos_id {
                continue;
            }
            token_ids.push(max_id);
        }

        token_ids
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<ParaformerResult, ParaformerError> {
        let features = self.compute_features(samples);
        if features.nrows() == 0 {
            return Ok(ParaformerResult {
                text: String::new(),
                token_ids: Vec::new(),
            });
        }

        let (logits, token_num) = self.forward(&features.view())?;
        let token_ids = self.decode_logits(&logits, token_num);
        let text = self.symbol_table.decode(&token_ids);

        Ok(ParaformerResult { text, token_ids })
    }
}

pub struct ParaformerResult {
    pub text: String,
    pub token_ids: Vec<i32>,
}
