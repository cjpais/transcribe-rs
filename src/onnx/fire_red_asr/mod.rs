use ndarray::{Array2, ArrayD, IxDyn};
use ort::inputs;
use ort::session::Session;
use ort::value::{DynValue, Tensor, TensorRef};
use std::path::Path;

use super::session;
use super::Quantization;
use crate::decode::{sentencepiece_to_text, SymbolTable};
use crate::features::{compute_mel, MelConfig, WindowType};
use crate::TranscribeError;
use crate::{ModelCapabilities, SpeechModel, TranscribeOptions, TranscriptionResult};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "FireRedASR",
    engine_id: "fire_red_asr",
    sample_rate: 16000,
    languages: &["zh", "en"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Per-model inference parameters for FireRedASR.
#[derive(Debug, Clone, Default)]
pub struct FireRedAsrParams {
    /// Currently unused; FireRedASR model handles multilingual decoding implicitly.
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
struct FireRedAsrMetadata {
    num_decoder_layers: i32,
    num_head: i32,
    head_dim: i32,
    sos_id: i32,
    eos_id: i32,
    max_len: i32,
    cmvn_mean: Vec<f32>,
    cmvn_inv_stddev: Vec<f32>,
}

pub struct FireRedAsrModel {
    encoder: Session,
    decoder: Session,
    symbol_table: SymbolTable,
    metadata: FireRedAsrMetadata,
    encoder_input_names: Vec<String>,
    encoder_output_names: Vec<String>,
    decoder_input_names: Vec<String>,
    decoder_output_names: Vec<String>,
}

impl FireRedAsrModel {
    /// Load a FireRedASR AED model from `model_dir`.
    ///
    /// Expected directory contents:
    /// - `encoder.onnx` / `encoder.int8.onnx` / `encoder.fp16.onnx`
    /// - `decoder.onnx` / `decoder.int8.onnx` / `decoder.fp16.onnx`
    /// - `tokens.txt`
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let encoder_path = session::resolve_model_path(model_dir, "encoder", quantization);
        let decoder_path = session::resolve_model_path(model_dir, "decoder", quantization);
        let tokens_path = model_dir.join("tokens.txt");

        if !encoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(encoder_path));
        }
        if !decoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(decoder_path));
        }
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading FireRedASR encoder from {:?}...", encoder_path);
        let encoder = session::create_session(&encoder_path)?;
        log::info!("Loading FireRedASR decoder from {:?}...", decoder_path);
        let decoder = session::create_session(&decoder_path)?;

        let encoder_input_names: Vec<String> =
            encoder.inputs().iter().map(|i| i.name().to_string()).collect();
        let encoder_output_names: Vec<String> =
            encoder.outputs().iter().map(|o| o.name().to_string()).collect();
        let decoder_input_names: Vec<String> =
            decoder.inputs().iter().map(|i| i.name().to_string()).collect();
        let decoder_output_names: Vec<String> =
            decoder.outputs().iter().map(|o| o.name().to_string()).collect();

        if encoder_input_names.len() != 2 {
            return Err(TranscribeError::Config(format!(
                "FireRedASR encoder expected 2 inputs, got {}",
                encoder_input_names.len()
            )));
        }
        if encoder_output_names.len() != 2 {
            return Err(TranscribeError::Config(format!(
                "FireRedASR encoder expected 2 outputs, got {}",
                encoder_output_names.len()
            )));
        }
        if decoder_input_names.len() != 6 {
            return Err(TranscribeError::Config(format!(
                "FireRedASR decoder expected 6 inputs, got {}",
                decoder_input_names.len()
            )));
        }
        if decoder_output_names.len() != 3 {
            return Err(TranscribeError::Config(format!(
                "FireRedASR decoder expected 3 outputs, got {}",
                decoder_output_names.len()
            )));
        }

        let metadata = Self::parse_metadata(&encoder)?;
        let symbol_table = SymbolTable::load(&tokens_path)?;

        Ok(Self {
            encoder,
            decoder,
            symbol_table,
            metadata,
            encoder_input_names,
            encoder_output_names,
            decoder_input_names,
            decoder_output_names,
        })
    }

    fn parse_metadata(encoder: &Session) -> Result<FireRedAsrMetadata, TranscribeError> {
        let num_decoder_layers =
            session::read_metadata_i32(encoder, "num_decoder_layers", None)?.ok_or_else(|| {
                TranscribeError::Config("Missing required metadata key: num_decoder_layers".into())
            })?;
        let num_head = session::read_metadata_i32(encoder, "num_head", None)?.ok_or_else(|| {
            TranscribeError::Config("Missing required metadata key: num_head".into())
        })?;
        let head_dim = session::read_metadata_i32(encoder, "head_dim", None)?.ok_or_else(|| {
            TranscribeError::Config("Missing required metadata key: head_dim".into())
        })?;
        let sos_id = session::read_metadata_i32(encoder, "sos", None)?.ok_or_else(|| {
            TranscribeError::Config("Missing required metadata key: sos".into())
        })?;
        let eos_id = session::read_metadata_i32(encoder, "eos", None)?.ok_or_else(|| {
            TranscribeError::Config("Missing required metadata key: eos".into())
        })?;
        let max_len = session::read_metadata_i32(encoder, "max_len", None)?.ok_or_else(|| {
            TranscribeError::Config("Missing required metadata key: max_len".into())
        })?;
        let cmvn_mean =
            session::read_metadata_float_vec(encoder, "cmvn_mean")?.ok_or_else(|| {
                TranscribeError::Config("Missing required metadata key: cmvn_mean".into())
            })?;
        let cmvn_inv_stddev =
            session::read_metadata_float_vec(encoder, "cmvn_inv_stddev")?.ok_or_else(|| {
                TranscribeError::Config("Missing required metadata key: cmvn_inv_stddev".into())
            })?;

        Ok(FireRedAsrMetadata {
            num_decoder_layers,
            num_head,
            head_dim,
            sos_id,
            eos_id,
            max_len,
            cmvn_mean,
            cmvn_inv_stddev,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &FireRedAsrParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        // --- Step 1: FBANK features (match sherpa-onnx OfflineStream defaults for FireRedASR) ---
        let mel_config = MelConfig {
            sample_rate: 16000,
            num_mels: 80,
            n_fft: 400,
            hop_length: 160,
            window: WindowType::Povey,
            f_min: 20.0,
            f_max: None, // high_freq=0 in sherpa-onnx => Nyquist
            pre_emphasis: Some(0.97),
            snip_edges: true,
            remove_dc_offset: true,
            normalize_samples: false, // sherpa-onnx sets normalize_samples=false for FireRedASR
        };

        let mut features: Array2<f32> = compute_mel(samples, &mel_config);
        let num_frames = features.nrows();
        let feat_dim = features.ncols();

        if num_frames == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        // --- Step 2: Apply model-provided CMVN ---
        if self.metadata.cmvn_mean.len() != feat_dim || self.metadata.cmvn_inv_stddev.len() != feat_dim
        {
            return Err(TranscribeError::Config(format!(
                "CMVN vector size mismatch: feat_dim={}, mean={}, inv_stddev={}",
                feat_dim,
                self.metadata.cmvn_mean.len(),
                self.metadata.cmvn_inv_stddev.len()
            )));
        }
        for mut row in features.rows_mut() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = (*v - self.metadata.cmvn_mean[j]) * self.metadata.cmvn_inv_stddev[j];
            }
        }

        // --- Step 3: Encoder forward ---
        let feat_3d = features
            .into_shape_with_order((1, num_frames, feat_dim))?
            .into_dyn();
        let x_len = ndarray::arr1(&[num_frames as i64]).into_dyn();

        let t_feat = TensorRef::from_array_view(feat_3d.view())?;
        let t_len = TensorRef::from_array_view(x_len.view())?;

        let mut encoder_out = self.encoder.run(inputs![
            self.encoder_input_names[0].as_str() => t_feat,
            self.encoder_input_names[1].as_str() => t_len,
        ])?;

        let cross_k = encoder_out
            .remove(self.encoder_output_names[0].as_str())
            .ok_or_else(|| {
                TranscribeError::Inference("Missing encoder output 0".to_string())
            })?;
        let cross_v = encoder_out
            .remove(self.encoder_output_names[1].as_str())
            .ok_or_else(|| {
                TranscribeError::Inference("Missing encoder output 1".to_string())
            })?;

        // --- Step 4: Autoregressive greedy decoding (batch=1) ---
        let max_len = self.metadata.max_len.max(0) as usize;
        let num_layers = self.metadata.num_decoder_layers.max(0) as usize;
        let num_head = self.metadata.num_head.max(0) as usize;
        let head_dim = self.metadata.head_dim.max(0) as usize;

        if max_len == 0 || num_layers == 0 || num_head == 0 || head_dim == 0 {
            return Err(TranscribeError::Config(format!(
                "Invalid FireRedASR metadata: max_len={}, num_layers={}, num_head={}, head_dim={}",
                max_len, num_layers, num_head, head_dim
            )));
        }

        let cache_shape = IxDyn(&[num_layers, 1, max_len, num_head, head_dim]);
        let self_k_init = ArrayD::<f32>::zeros(cache_shape.clone());
        let self_v_init = ArrayD::<f32>::zeros(cache_shape);
        let mut self_k_cache: DynValue = Tensor::from_array(self_k_init)?.into_dyn();
        let mut self_v_cache: DynValue = Tensor::from_array(self_v_init)?.into_dyn();

        let eos_id = self.metadata.eos_id as i64;
        let mut token: i64 = self.metadata.sos_id as i64;
        let mut offset: i64 = 0;
        let mut decoded: Vec<i64> = Vec::new();

        let num_possible_tokens = ((num_frames as f32) / 100.0 * 6.0) as usize;
        let max_steps = num_possible_tokens.min(max_len / 2).max(1);

        for _ in 0..max_steps {
            let tokens_tensor =
                Tensor::from_array((vec![1i64, 1i64], vec![token].into_boxed_slice()))?;
            let offset_tensor =
                Tensor::from_array((vec![1i64], vec![offset].into_boxed_slice()))?;

            let mut decoder_out = self.decoder.run(inputs![
                self.decoder_input_names[0].as_str() => tokens_tensor,
                self.decoder_input_names[1].as_str() => &self_k_cache,
                self.decoder_input_names[2].as_str() => &self_v_cache,
                self.decoder_input_names[3].as_str() => &cross_k,
                self.decoder_input_names[4].as_str() => &cross_v,
                self.decoder_input_names[5].as_str() => offset_tensor,
            ])?;

            let logits = decoder_out
                .remove(self.decoder_output_names[0].as_str())
                .ok_or_else(|| TranscribeError::Inference("Missing decoder logits".to_string()))?;

            // Update caches (Arc clone internally; no data copy for many EPs)
            self_k_cache = decoder_out
                .remove(self.decoder_output_names[1].as_str())
                .ok_or_else(|| {
                    TranscribeError::Inference("Missing decoder self_k_cache".to_string())
                })?;
            self_v_cache = decoder_out
                .remove(self.decoder_output_names[2].as_str())
                .ok_or_else(|| {
                    TranscribeError::Inference("Missing decoder self_v_cache".to_string())
                })?;

            let next_token = argmax_logits(&logits)?;
            if next_token == eos_id {
                break;
            }

            decoded.push(next_token);
            token = next_token;
            offset += 1;
        }

        // --- Step 5: Token IDs -> text ---
        let mut token_strs: Vec<&str> = Vec::new();
        token_strs.reserve(decoded.len());
        for id in decoded {
            if let Some(s) = self.symbol_table.get(id) {
                token_strs.push(s);
            }
        }

        let text = sentencepiece_to_text(&token_strs);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }
}

impl SpeechModel for FireRedAsrModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }
}

fn argmax_logits(logits: &DynValue) -> Result<i64, TranscribeError> {
    let (shape, data) = logits
        .try_extract_tensor::<f32>()
        .map_err(|e| TranscribeError::Inference(format!("Failed to extract logits: {e}")))?;

    let vocab_size = shape
        .last()
        .copied()
        .ok_or_else(|| TranscribeError::Inference("Logits tensor has no shape".to_string()))?
        as usize;
    if vocab_size == 0 {
        return Err(TranscribeError::Inference(
            "Logits tensor has vocab_size=0".to_string(),
        ));
    }

    // Logits are expected to be [1, 1, vocab] (or compatible); take the last vocab slice.
    let start = data.len().saturating_sub(vocab_size);
    let slice = &data[start..start + vocab_size];

    let mut max_idx = 0usize;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    Ok(max_idx as i64)
}

