//! Zipformer CTC ONNX speech recognition engine.
//!
//! Supports sherpa-onnx Zipformer CTC models (e.g. from Icefall) for
//! Chinese and English transcription. Streaming models (with `cached_*`
//! inputs) are rejected at load time.

use ndarray::Array2;
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::{Path, PathBuf};

use super::session;
use super::Quantization;
use crate::decode::{ctc_greedy_decode, BbpeSymbolTable};
use crate::features::{compute_kaldi_fbank, KaldiFbankConfig};
use crate::TranscribeError;
use crate::{ModelCapabilities, SpeechModel, TranscribeOptions, TranscriptionResult};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Zipformer CTC",
    engine_id: "zipformer_ctc",
    sample_rate: 16000,
    languages: &["zh", "en"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Per-model inference parameters for Zipformer CTC.
#[derive(Debug, Clone, Default)]
pub struct ZipformerCtcParams {
    /// Language hint (currently unused; the model handles zh/en automatically).
    pub language: Option<String>,
}

// ---- Model ----

pub struct ZipformerCtcModel {
    session: Session,
    symbol_table: BbpeSymbolTable,
    blank_id: i64,
    x_input_name: String,
    x_lens_input_name: String,
    #[allow(dead_code)]
    log_probs_output_name: String,
    /// Index of the output that contains output lengths, if present.
    log_probs_len_output_idx: Option<usize>,
}

impl ZipformerCtcModel {
    /// Load a Zipformer CTC model from `model_dir`.
    ///
    /// Attempts `session::resolve_model_path(dir, "model", quantization)` first,
    /// then falls back to scanning the directory for any `.onnx` file (for
    /// sherpa-onnx models with names like `model-epoch-34-avg-19.int8.onnx`).
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let model_path = Self::find_model_file(model_dir, quantization)?;
        let tokens_path = model_dir.join("tokens.txt");

        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading Zipformer CTC model from {:?}...", model_path);
        let session = session::create_session(&model_path)?;

        // Reject streaming models — they have cached_* inputs
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        if input_names.iter().any(|n| n.starts_with("cached_")) {
            return Err(TranscribeError::Config(format!(
                "Streaming Zipformer models are not supported (found cached_* inputs in {:?}). \
                 Use a non-streaming (offline) model.",
                model_path
            )));
        }

        log::debug!("Model inputs: {:?}", input_names);

        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        log::debug!("Model outputs: {:?}", output_names);

        // Detect input names
        let x_input_name = input_names
            .iter()
            .find(|n| n.as_str() == "x" || n.contains("feat") || n.contains("input"))
            .cloned()
            .unwrap_or_else(|| {
                input_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "x".to_string())
            });

        let x_lens_input_name = input_names
            .iter()
            .find(|n| n.contains("len") || n.contains("length"))
            .cloned()
            .unwrap_or_else(|| {
                input_names
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| "x_lens".to_string())
            });

        // Detect output names
        let log_probs_output_name = output_names
            .iter()
            .find(|n| n.contains("log_prob") || n.contains("logit") || n.contains("prob"))
            .cloned()
            .unwrap_or_else(|| {
                output_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "log_probs".to_string())
            });

        let log_probs_len_output_idx = output_names
            .iter()
            .position(|n| n.contains("len") || n.contains("length"));

        log::debug!(
            "I/O mapping: x={}, x_lens={}, log_probs={}, log_probs_len_idx={:?}",
            x_input_name,
            x_lens_input_name,
            log_probs_output_name,
            log_probs_len_output_idx,
        );

        // Load BBPE symbol table (auto-detects BBPE vs BPE encoding)
        let symbol_table = BbpeSymbolTable::load(&tokens_path)?;

        // blank_id is always 0 for sherpa-onnx Zipformer CTC models
        let blank_id = 0i64;

        Ok(Self {
            session,
            symbol_table,
            blank_id,
            x_input_name,
            x_lens_input_name,
            log_probs_output_name,
            log_probs_len_output_idx,
        })
    }

    /// Find the ONNX model file in `model_dir`.
    ///
    /// Priority:
    /// 1. `session::resolve_model_path(dir, "model", quantization)` — standard naming
    /// 2. Scan directory for `*.int8.onnx` (when Int8 requested) or `*.onnx`
    fn find_model_file(
        model_dir: &Path,
        quantization: &Quantization,
    ) -> Result<PathBuf, TranscribeError> {
        // Try standard path first
        let standard_path = session::resolve_model_path(model_dir, "model", quantization);
        if standard_path.exists() {
            return Ok(standard_path);
        }

        // Fallback: scan directory for onnx files
        let prefer_int8 = *quantization == Quantization::Int8;

        let read_dir = std::fs::read_dir(model_dir).map_err(|e| {
            TranscribeError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read model directory {:?}: {}", model_dir, e),
            ))
        })?;

        let mut int8_candidates: Vec<PathBuf> = Vec::new();
        let mut fp32_candidates: Vec<PathBuf> = Vec::new();

        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if name.contains("int8") || name.contains("int4") {
                    int8_candidates.push(path);
                } else {
                    fp32_candidates.push(path);
                }
            }
        }

        // Sort for determinism
        int8_candidates.sort();
        fp32_candidates.sort();

        if prefer_int8 {
            if let Some(p) = int8_candidates.into_iter().next() {
                log::info!("Found int8 model by directory scan: {:?}", p);
                return Ok(p);
            }
        }

        if let Some(p) = fp32_candidates.into_iter().next() {
            log::info!("Found model by directory scan: {:?}", p);
            return Ok(p);
        }

        // Last resort: return the int8 candidate even if fp32 preferred
        Err(TranscribeError::ModelNotFound(model_dir.join("model.onnx")))
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &ZipformerCtcParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        // 1. Compute Kaldi FBank features [frames, 80]
        let features = compute_kaldi_fbank(samples, &KaldiFbankConfig::default());

        if features.nrows() == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        log::debug!("Kaldi fbank: [{}, {}]", features.nrows(), features.ncols());

        // 2. Run ONNX forward pass → log_probs [1, T', vocab]
        let (log_probs, output_len) = self.forward(&features)?;

        log::debug!(
            "log_probs shape: {:?}, output_len={}",
            log_probs.shape(),
            output_len
        );

        // 3. CTC greedy decode (expects [batch, time, vocab])
        let logits_lengths = vec![output_len];
        let results = ctc_greedy_decode(&log_probs.view(), &logits_lengths, self.blank_id);

        // 4. Convert token IDs (i64 → i32) and decode to text
        let token_ids: Vec<i32> = results[0].tokens.iter().map(|&t| t as i32).collect();
        let text = self.symbol_table.decode(&token_ids);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    /// Run ONNX forward pass.
    ///
    /// Returns `(log_probs [1, T, vocab], output_len)` where `output_len` is
    /// the valid frame count for batch element 0.
    fn forward(
        &mut self,
        features: &Array2<f32>,
    ) -> Result<(ndarray::Array3<f32>, i64), TranscribeError> {
        let num_frames = features.nrows() as i64;

        // Shape: [1, T, 80]
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, features.nrows(), features.ncols()))?;
        let x_lens = ndarray::arr1(&[num_frames]);

        let feat_dyn = feat_3d.into_dyn();
        let lens_dyn = x_lens.into_dyn();

        let t_feat = TensorRef::from_array_view(feat_dyn.view())?;
        let t_lens = TensorRef::from_array_view(lens_dyn.view())?;

        let ort_inputs = inputs![
            self.x_input_name.as_str() => t_feat,
            self.x_lens_input_name.as_str() => t_lens,
        ];

        let outputs = self.session.run(ort_inputs)?;

        // Extract log_probs — always the first output, shape [1, T', vocab]
        let log_probs = outputs[0].try_extract_array::<f32>()?;
        let log_probs = log_probs.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        // Determine output length: use the length output if available, else T'
        let output_len = if let Some(len_idx) = self.log_probs_len_output_idx {
            if len_idx < outputs.len() {
                let lens = outputs[len_idx].try_extract_array::<i64>()?;
                lens.first().copied().unwrap_or(log_probs.shape()[1] as i64)
            } else {
                log_probs.shape()[1] as i64
            }
        } else {
            log_probs.shape()[1] as i64
        };

        Ok((log_probs, output_len))
    }
}

impl SpeechModel for ZipformerCtcModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }
}
