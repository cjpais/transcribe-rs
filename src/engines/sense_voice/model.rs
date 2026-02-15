use ndarray::{Array1, Array3, ArrayView2};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::collections::HashMap;
use std::path::Path;

use super::decoder::{ctc_greedy_decode, CtcDecoderResult};
use super::features::{apply_cmvn, apply_lfr, compute_fbank, FbankConfig};
use super::tokens::SymbolTable;

#[derive(thiserror::Error, Debug)]
pub enum SenseVoiceError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Tokens file not found: {0}")]
    TokensNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Metadata error: {0}")]
    Metadata(String),
    #[error("Unknown language: {0}")]
    UnknownLanguage(String),
}

/// Metadata parsed from the ONNX model's custom properties.
pub struct SenseVoiceMetadata {
    pub vocab_size: i32,
    pub blank_id: i32,
    pub lfr_window_size: usize,
    pub lfr_window_shift: usize,
    pub normalize_samples: bool,
    pub with_itn_id: i32,
    pub without_itn_id: i32,
    pub lang2id: HashMap<String, i32>,
    pub neg_mean: Array1<f32>,
    pub inv_stddev: Array1<f32>,
    pub is_funasr_nano: bool,
}

/// The loaded SenseVoice ONNX model.
pub struct SenseVoiceModel {
    session: Session,
    pub metadata: SenseVoiceMetadata,
    pub symbol_table: SymbolTable,
    input_names: Vec<String>,
}

impl Drop for SenseVoiceModel {
    fn drop(&mut self) {
        log::debug!("Dropping SenseVoiceModel");
    }
}

impl SenseVoiceModel {
    /// Load SenseVoice model from a directory containing model.onnx and tokens.txt.
    ///
    /// If `quantized` is true, loads `model.int8.onnx` (falls back to `model.onnx`).
    /// If `quantized` is false, loads `model.onnx`.
    pub fn new(model_dir: &Path, quantized: bool) -> Result<Self, SenseVoiceError> {
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

        if !model_path.exists() {
            return Err(SenseVoiceError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }
        if !tokens_path.exists() {
            return Err(SenseVoiceError::TokensNotFound(
                tokens_path.display().to_string(),
            ));
        }

        log::info!("Loading SenseVoice model from {:?}...", model_path);
        let session = Self::init_session(&model_path)?;

        let input_names: Vec<String> = session.inputs().iter().map(|i| i.name.clone()).collect();
        log::debug!("Model inputs: {:?}", input_names);

        let metadata = Self::parse_metadata(&session)?;
        log::info!(
            "Model metadata: vocab_size={}, lfr_window_size={}, lfr_window_shift={}, is_nano={}",
            metadata.vocab_size,
            metadata.lfr_window_size,
            metadata.lfr_window_shift,
            metadata.is_funasr_nano,
        );

        let mut symbol_table = SymbolTable::load(&tokens_path)?;
        if metadata.is_funasr_nano {
            log::info!("FunASR Nano model detected, applying base64 decode to tokens");
            symbol_table.apply_base64_decode();
        }

        Ok(Self {
            session,
            metadata,
            symbol_table,
            input_names,
        })
    }

    fn init_session(path: &Path) -> Result<Session, SenseVoiceError> {
        let providers = vec![CPUExecutionProvider::default().build()];

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(path)?;

        for input in session.inputs() {
            log::info!("Model input: name={}, type={:?}", input.name(), input.dtype());
        }
        for output in session.outputs() {
            log::info!("Model output: name={}, type={:?}", output.name(), output.dtype());
        }

        Ok(session)
    }

    /// Read a custom metadata string value from the ONNX model.
    fn read_meta_str(session: &Session, key: &str) -> Result<Option<String>, SenseVoiceError> {
        let meta = session.metadata()?;
        Ok(meta.custom(key)?)
    }

    /// Read a custom metadata i32 value, with optional default.
    fn read_meta_i32(
        session: &Session,
        key: &str,
        default: Option<i32>,
    ) -> Result<i32, SenseVoiceError> {
        match Self::read_meta_str(session, key)? {
            Some(v) => v.parse::<i32>().map_err(|e| {
                SenseVoiceError::Metadata(format!("Failed to parse '{}': {}", key, e))
            }),
            None => default.ok_or_else(|| {
                SenseVoiceError::Metadata(format!("Missing required metadata key: {}", key))
            }),
        }
    }

    /// Read a comma-separated float vector from metadata.
    fn read_meta_float_vec(session: &Session, key: &str) -> Result<Vec<f32>, SenseVoiceError> {
        match Self::read_meta_str(session, key)? {
            Some(v) => v
                .split(',')
                .map(|s| {
                    s.trim().parse::<f32>().map_err(|e| {
                        SenseVoiceError::Metadata(format!(
                            "Failed to parse float in '{}': {}",
                            key, e
                        ))
                    })
                })
                .collect(),
            None => Ok(Vec::new()),
        }
    }

    fn parse_metadata(session: &Session) -> Result<SenseVoiceMetadata, SenseVoiceError> {
        // Check if this is a FunASR Nano model
        let comment = Self::read_meta_str(session, "comment")?.unwrap_or_default();
        let is_funasr_nano = comment.contains("Nano");

        let vocab_size = Self::read_meta_i32(session, "vocab_size", None)?;
        let blank_id = Self::read_meta_i32(session, "blank_id", Some(0))?;
        let lfr_window_size = Self::read_meta_i32(session, "lfr_window_size", Some(7))? as usize;
        let lfr_window_shift = Self::read_meta_i32(session, "lfr_window_shift", Some(6))? as usize;
        let normalize_samples_int = Self::read_meta_i32(session, "normalize_samples", Some(0))?;

        let (with_itn_id, without_itn_id, lang2id, neg_mean_vec, inv_stddev_vec) = if is_funasr_nano
        {
            (14, 15, HashMap::new(), Vec::new(), Vec::new())
        } else {
            let with_itn_id = Self::read_meta_i32(session, "with_itn", Some(14))?;
            let without_itn_id = Self::read_meta_i32(session, "without_itn", Some(15))?;

            let mut lang2id = HashMap::new();
            for (lang, key) in [
                ("auto", "lang_auto"),
                ("zh", "lang_zh"),
                ("en", "lang_en"),
                ("ja", "lang_ja"),
                ("ko", "lang_ko"),
                ("yue", "lang_yue"),
            ] {
                if let Ok(id) = Self::read_meta_i32(session, key, None) {
                    lang2id.insert(lang.to_string(), id);
                }
            }
            // Use defaults if not found in metadata
            if lang2id.is_empty() {
                lang2id = HashMap::from([
                    ("auto".to_string(), 0),
                    ("zh".to_string(), 3),
                    ("en".to_string(), 4),
                    ("yue".to_string(), 7),
                    ("ja".to_string(), 11),
                    ("ko".to_string(), 12),
                ]);
            }

            let neg_mean_vec = Self::read_meta_float_vec(session, "neg_mean")?;
            let inv_stddev_vec = Self::read_meta_float_vec(session, "inv_stddev")?;

            (
                with_itn_id,
                without_itn_id,
                lang2id,
                neg_mean_vec,
                inv_stddev_vec,
            )
        };

        Ok(SenseVoiceMetadata {
            vocab_size,
            blank_id,
            lfr_window_size,
            lfr_window_shift,
            normalize_samples: normalize_samples_int != 0,
            with_itn_id,
            without_itn_id,
            lang2id,
            neg_mean: Array1::from_vec(neg_mean_vec),
            inv_stddev: Array1::from_vec(inv_stddev_vec),
            is_funasr_nano,
        })
    }

    /// Run the full transcription pipeline: features → LFR → CMVN → forward → CTC decode.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: &str,
        use_itn: bool,
    ) -> Result<SenseVoiceResult, SenseVoiceError> {
        // Copy metadata values we need to avoid borrow conflicts with &mut self
        let normalize_samples = self.metadata.normalize_samples;
        let lfr_window_size = self.metadata.lfr_window_size;
        let lfr_window_shift = self.metadata.lfr_window_shift;
        let is_funasr_nano = self.metadata.is_funasr_nano;
        let blank_id = self.metadata.blank_id as i64;
        let has_cmvn = !is_funasr_nano && !self.metadata.neg_mean.is_empty();
        let neg_mean = self.metadata.neg_mean.clone();
        let inv_stddev = self.metadata.inv_stddev.clone();

        // 1. Compute FBANK features
        let fbank_config = FbankConfig::default();
        let features = compute_fbank(samples, &fbank_config, normalize_samples);

        log::debug!(
            "FBANK features: [{}, {}]",
            features.nrows(),
            features.ncols()
        );

        // 2. Apply LFR
        let features = apply_lfr(&features, lfr_window_size, lfr_window_shift);

        log::debug!("After LFR: [{}, {}]", features.nrows(), features.ncols());

        if features.nrows() == 0 {
            return Ok(SenseVoiceResult {
                text: String::new(),
                tokens: Vec::new(),
                timestamps: Vec::new(),
                language: None,
                emotion: None,
                event: None,
            });
        }

        // 3. Apply CMVN (not for FunASR Nano)
        let mut features = features;
        if has_cmvn {
            apply_cmvn(&mut features, &neg_mean, &inv_stddev);
        }

        let num_feature_frames = features.nrows();

        // 4. Run ONNX forward pass
        let logits = if is_funasr_nano {
            self.forward_nano(&features.view())?
        } else {
            self.forward(&features.view(), language, use_itn)?
        };

        log::debug!("Logits shape: {:?}", logits.shape());

        // 5. CTC greedy decode
        let num_frames = if is_funasr_nano {
            logits.shape()[1] as i64
        } else {
            num_feature_frames as i64 + 4 // +4 for prepended special tokens
        };
        let logits_lengths = vec![num_frames];
        let logits_view = logits.view();
        let decoder_results = ctc_greedy_decode(&logits_view, &logits_lengths, blank_id);

        // 6. Convert result
        let result = self.convert_result(&decoder_results[0]);
        Ok(result)
    }

    /// Forward pass for full SenseVoice model (4 inputs).
    fn forward(
        &mut self,
        features: &ArrayView2<f32>,
        language: &str,
        use_itn: bool,
    ) -> Result<Array3<f32>, SenseVoiceError> {
        let meta = &self.metadata;
        let num_frames = features.nrows() as i32;

        // Reshape features to [1, T, feat_dim]
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, features.nrows(), features.ncols()))?;

        let x_length = ndarray::arr1(&[num_frames]);

        // Resolve language ID
        let lang_id = if language.is_empty() {
            0i32 // auto
        } else {
            *meta
                .lang2id
                .get(language)
                .ok_or_else(|| SenseVoiceError::UnknownLanguage(language.to_string()))?
        };
        let language_arr = ndarray::arr1(&[lang_id]);

        let text_norm_id = if use_itn {
            meta.with_itn_id
        } else {
            meta.without_itn_id
        };
        let text_norm_arr = ndarray::arr1(&[text_norm_id]);

        let feat_dyn = feat_3d.into_dyn();
        let x_length_dyn = x_length.into_dyn();
        let language_dyn = language_arr.into_dyn();
        let text_norm_dyn = text_norm_arr.into_dyn();

        let inputs = inputs![
            self.input_names[0].as_str() => TensorRef::from_array_view(feat_dyn.view())?,
            self.input_names[1].as_str() => TensorRef::from_array_view(x_length_dyn.view())?,
            self.input_names[2].as_str() => TensorRef::from_array_view(language_dyn.view())?,
            self.input_names[3].as_str() => TensorRef::from_array_view(text_norm_dyn.view())?,
        ];

        let outputs = self.session.run(inputs)?;
        let logits = outputs[0].try_extract_array::<f32>()?;
        let logits_owned = logits.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        Ok(logits_owned)
    }

    /// Forward pass for FunASR Nano model (1 input).
    fn forward_nano(&mut self, features: &ArrayView2<f32>) -> Result<Array3<f32>, SenseVoiceError> {
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, features.nrows(), features.ncols()))?;

        let feat_dyn = feat_3d.into_dyn();

        let inputs = inputs![
            self.input_names[0].as_str() => TensorRef::from_array_view(feat_dyn.view())?,
        ];

        let outputs = self.session.run(inputs)?;
        let logits = outputs[0].try_extract_array::<f32>()?;
        let logits_owned = logits.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        Ok(logits_owned)
    }

    /// Convert CTC decoder output to a SenseVoiceResult, stripping special prefix tokens.
    fn convert_result(&self, decoder_result: &CtcDecoderResult) -> SenseVoiceResult {
        let meta = &self.metadata;
        let tokens = &decoder_result.tokens;
        let timestamps = &decoder_result.timestamps;

        let (start, language, emotion, event) = if meta.is_funasr_nano {
            (0, None, None, None)
        } else {
            let lang = tokens
                .first()
                .and_then(|&id| self.symbol_table.get(id))
                .map(|s| s.to_string());
            let emo = tokens
                .get(1)
                .and_then(|&id| self.symbol_table.get(id))
                .map(|s| s.to_string());
            let evt = tokens
                .get(2)
                .and_then(|&id| self.symbol_table.get(id))
                .map(|s| s.to_string());
            (4usize, lang, emo, evt)
        };

        // Build text from remaining tokens
        // Replace SentencePiece word boundary marker ▁ (\u{2581}) with space
        let mut text = String::new();
        let mut result_tokens = Vec::new();
        for &id in tokens.iter().skip(start) {
            let sym = self.symbol_table.get_or_empty(id);
            text.push_str(&sym.replace('\u{2581}', " "));
            result_tokens.push(sym.to_string());
        }
        // Clean up text:
        // - Trim leading/trailing whitespace
        // - Remove spaces before apostrophes/contractions (e.g. "can 't" → "can't")
        let text = text.trim().to_string();
        let text = text.replace(" '", "'").replace(" \u{2581}'", "'");

        // Calculate timestamps in seconds
        let frame_shift_s = 0.01 * meta.lfr_window_shift as f32; // 10ms * window_shift
        let result_timestamps: Vec<f32> = timestamps
            .iter()
            .skip(start)
            .map(|&t| frame_shift_s * (t - start as i32) as f32)
            .collect();

        SenseVoiceResult {
            text,
            tokens: result_tokens,
            timestamps: result_timestamps,
            language,
            emotion,
            event,
        }
    }
}

/// Result of SenseVoice transcription.
pub struct SenseVoiceResult {
    /// The transcribed text.
    pub text: String,
    /// Individual tokens.
    pub tokens: Vec<String>,
    /// Timestamp in seconds for each token.
    pub timestamps: Vec<f32>,
    /// Detected language (full model only).
    pub language: Option<String>,
    /// Detected emotion (full model only).
    pub emotion: Option<String>,
    /// Detected event type (full model only).
    pub event: Option<String>,
}
