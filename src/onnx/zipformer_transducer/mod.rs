//! Zipformer Transducer (RNN-T) ONNX speech recognition engine.
//!
//! Supports sherpa-onnx Zipformer Transducer models with 3 ONNX sessions
//! (encoder, decoder, joiner) and RNN-T greedy search decoding. Streaming
//! models (with `cached_*` inputs) are rejected at load time.

use ndarray::{Array2, Array3, ArrayView1};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::{Path, PathBuf};

use super::session;
use super::Quantization;
use crate::decode::BbpeSymbolTable;
use crate::features::{compute_kaldi_fbank, KaldiFbankConfig};
use crate::TranscribeError;
use crate::{ModelCapabilities, SpeechModel, TranscribeOptions, TranscriptionResult};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Zipformer Transducer",
    engine_id: "zipformer_transducer",
    sample_rate: 16000,
    languages: &["zh", "en", "vi", "ru", "ko"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Per-model inference parameters for Zipformer Transducer.
#[derive(Debug, Clone, Default)]
pub struct ZipformerTransducerParams {
    /// Language hint (currently unused; the model handles languages automatically).
    pub language: Option<String>,
}

// ---- Model ----

pub struct ZipformerTransducerModel {
    encoder_session: Session,
    decoder_session: Session,
    joiner_session: Session,
    symbol_table: BbpeSymbolTable,
    blank_id: i32,
    context_size: usize, // always 2
    // Encoder I/O names
    enc_x_name: String,
    enc_x_lens_name: String,
    enc_out_name: String,
    enc_out_lens_name: String,
    // Decoder I/O names
    dec_y_name: String,
    dec_out_name: String,
    // Joiner I/O names
    join_enc_name: String,
    join_dec_name: String,
    join_logit_name: String,
}

impl ZipformerTransducerModel {
    /// Load a Zipformer Transducer model from `model_dir`.
    ///
    /// Expects three ONNX files (encoder, decoder, joiner) and a `tokens.txt`
    /// file in the model directory.
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let suffix = match quantization {
            Quantization::FP32 => "fp32",
            Quantization::FP16 => "fp16",
            Quantization::Int8 => "int8",
        };

        let encoder_path = Self::find_model_file(model_dir, "encoder", suffix)?;
        let decoder_path = Self::find_model_file(model_dir, "decoder", suffix)?;
        let joiner_path = Self::find_model_file(model_dir, "joiner", suffix)?;

        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!(
            "Loading Zipformer Transducer model from {:?} (encoder={:?}, decoder={:?}, joiner={:?})...",
            model_dir,
            encoder_path.file_name().unwrap_or_default(),
            decoder_path.file_name().unwrap_or_default(),
            joiner_path.file_name().unwrap_or_default(),
        );

        let encoder_session = session::create_session(&encoder_path)?;
        let decoder_session = session::create_session(&decoder_path)?;
        let joiner_session = session::create_session(&joiner_path)?;

        // Reject streaming models — they have cached_* inputs
        let enc_input_names: Vec<String> = encoder_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        if enc_input_names.iter().any(|n| n.starts_with("cached_")) {
            return Err(TranscribeError::Config(format!(
                "Streaming Zipformer Transducer models are not supported (found cached_* inputs in {:?}). \
                 Use a non-streaming (offline) model.",
                encoder_path
            )));
        }

        let enc_output_names: Vec<String> = encoder_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let dec_input_names: Vec<String> = decoder_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        let dec_output_names: Vec<String> = decoder_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let join_input_names: Vec<String> = joiner_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        let join_output_names: Vec<String> = joiner_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        log::debug!(
            "Encoder inputs: {:?}, outputs: {:?}",
            enc_input_names,
            enc_output_names
        );
        log::debug!(
            "Decoder inputs: {:?}, outputs: {:?}",
            dec_input_names,
            dec_output_names
        );
        log::debug!(
            "Joiner inputs: {:?}, outputs: {:?}",
            join_input_names,
            join_output_names
        );

        // Detect encoder I/O names
        let enc_x_name = Self::find_name(&enc_input_names, &["x", "features", "input"])
            .unwrap_or_else(|| {
                enc_input_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "x".to_string())
            });

        let enc_x_lens_name =
            Self::find_name(&enc_input_names, &["x_lens", "x_length", "input_lengths"])
                .unwrap_or_else(|| {
                    enc_input_names
                        .get(1)
                        .cloned()
                        .unwrap_or_else(|| "x_lens".to_string())
                });

        let enc_out_name = Self::find_name(
            &enc_output_names,
            &["encoder_out", "output", "encoder_output"],
        )
        .unwrap_or_else(|| {
            enc_output_names
                .first()
                .cloned()
                .unwrap_or_else(|| "encoder_out".to_string())
        });

        let enc_out_lens_name = Self::find_name(
            &enc_output_names,
            &["encoder_out_lens", "encoder_out_length", "output_lengths"],
        )
        .unwrap_or_else(|| {
            enc_output_names
                .get(1)
                .cloned()
                .unwrap_or_else(|| "encoder_out_lens".to_string())
        });

        // Detect decoder I/O names
        let dec_y_name = Self::find_name(&dec_input_names, &["y", "input", "decoder_input"])
            .unwrap_or_else(|| {
                dec_input_names
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "y".to_string())
            });

        let dec_out_name = Self::find_name(
            &dec_output_names,
            &["decoder_out", "output", "decoder_output"],
        )
        .unwrap_or_else(|| {
            dec_output_names
                .first()
                .cloned()
                .unwrap_or_else(|| "decoder_out".to_string())
        });

        // Detect joiner I/O names
        let join_enc_name = Self::find_name(
            &join_input_names,
            &["encoder_out", "enc_out", "encoder_input"],
        )
        .unwrap_or_else(|| {
            join_input_names
                .first()
                .cloned()
                .unwrap_or_else(|| "encoder_out".to_string())
        });

        let join_dec_name = Self::find_name(
            &join_input_names,
            &["decoder_out", "dec_out", "decoder_input"],
        )
        .unwrap_or_else(|| {
            join_input_names
                .get(1)
                .cloned()
                .unwrap_or_else(|| "decoder_out".to_string())
        });

        let join_logit_name =
            Self::find_name(&join_output_names, &["logit", "output", "joiner_output"])
                .unwrap_or_else(|| {
                    join_output_names
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "logit".to_string())
                });

        log::debug!(
            "I/O mapping: enc({}, {} -> {}, {}), dec({} -> {}), join({}, {} -> {})",
            enc_x_name,
            enc_x_lens_name,
            enc_out_name,
            enc_out_lens_name,
            dec_y_name,
            dec_out_name,
            join_enc_name,
            join_dec_name,
            join_logit_name,
        );

        // Load BBPE symbol table (auto-detects BBPE vs BPE encoding)
        let symbol_table = BbpeSymbolTable::load(&tokens_path)?;

        Ok(Self {
            encoder_session,
            decoder_session,
            joiner_session,
            symbol_table,
            blank_id: 0,
            context_size: 2,
            enc_x_name,
            enc_x_lens_name,
            enc_out_name,
            enc_out_lens_name,
            dec_y_name,
            dec_out_name,
            join_enc_name,
            join_dec_name,
            join_logit_name,
        })
    }

    /// Find an ONNX model file by component name, trying various naming conventions.
    ///
    /// Tries in order:
    /// 1. `{component}.{suffix}.onnx`
    /// 2. `{component}.onnx`
    /// 3. Any file starting with `{component}` ending with `.{suffix}.onnx`
    /// 4. Any file starting with `{component}` ending with `.onnx`
    fn find_model_file(
        model_dir: &Path,
        component: &str,
        suffix: &str,
    ) -> Result<PathBuf, TranscribeError> {
        // 1. Try exact: {component}.{suffix}.onnx
        let exact_suffixed = model_dir.join(format!("{}.{}.onnx", component, suffix));
        if exact_suffixed.exists() {
            return Ok(exact_suffixed);
        }

        // 2. Try exact: {component}.onnx
        let exact_plain = model_dir.join(format!("{}.onnx", component));
        if exact_plain.exists() {
            return Ok(exact_plain);
        }

        // 3/4. Scan directory for files matching the component prefix
        if let Ok(read_dir) = std::fs::read_dir(model_dir) {
            let mut suffixed_candidates: Vec<PathBuf> = Vec::new();
            let mut plain_candidates: Vec<PathBuf> = Vec::new();

            for entry in read_dir.flatten() {
                let path = entry.path();
                let file_name = match path.file_name().and_then(|n| n.to_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };

                if !file_name.starts_with(component) {
                    continue;
                }

                if file_name.ends_with(&format!(".{}.onnx", suffix)) {
                    suffixed_candidates.push(path);
                } else if file_name.ends_with(".onnx") {
                    plain_candidates.push(path);
                }
            }

            // Sort for determinism
            suffixed_candidates.sort();
            plain_candidates.sort();

            // 3. Prefer suffixed match
            if let Some(p) = suffixed_candidates.into_iter().next() {
                log::debug!(
                    "Found {} model by directory scan: {:?}",
                    component,
                    p.file_name().unwrap_or_default()
                );
                return Ok(p);
            }

            // 4. Fall back to any .onnx match
            if let Some(p) = plain_candidates.into_iter().next() {
                log::debug!(
                    "Found {} model by directory scan: {:?}",
                    component,
                    p.file_name().unwrap_or_default()
                );
                return Ok(p);
            }
        }

        Err(TranscribeError::ModelNotFound(exact_suffixed))
    }

    /// Find a name from a list of candidates, returning the first match.
    fn find_name(names: &[String], candidates: &[&str]) -> Option<String> {
        for &candidate in candidates {
            if let Some(n) = names.iter().find(|n| n.as_str() == candidate) {
                return Some(n.clone());
            }
        }
        None
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &ZipformerTransducerParams,
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

        // 2. RNN-T greedy search
        let token_ids = self.greedy_search(&features)?;

        // 3. Decode tokens to text
        let text = self.symbol_table.decode(&token_ids);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    /// Run encoder: features [1,T,80] + lens [1] -> encoder_out [1,T',D] + encoder_out_lens [1]
    fn run_encoder(
        &mut self,
        features: &Array2<f32>,
    ) -> Result<(Array3<f32>, i64), TranscribeError> {
        let num_frames = features.nrows();
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, num_frames, features.ncols()))?;
        let lens = ndarray::arr1(&[num_frames as i64]).into_dyn();

        let feat_dyn = feat_3d.into_dyn();
        let t_feat = TensorRef::from_array_view(feat_dyn.view())?;
        let t_lens = TensorRef::from_array_view(lens.view())?;

        let inputs = inputs![
            self.enc_x_name.as_str() => t_feat,
            self.enc_x_lens_name.as_str() => t_lens,
        ];
        let outputs = self.encoder_session.run(inputs)?;

        let encoder_out = outputs
            .get(self.enc_out_name.as_str())
            .ok_or_else(|| {
                TranscribeError::Inference(format!(
                    "encoder output '{}' not found",
                    self.enc_out_name
                ))
            })?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()?;

        let encoder_out_lens = outputs
            .get(self.enc_out_lens_name.as_str())
            .and_then(|v| v.try_extract_array::<i64>().ok())
            .and_then(|arr| arr.as_slice().and_then(|s| s.first().copied()))
            .unwrap_or(encoder_out.shape()[1] as i64);

        Ok((encoder_out, encoder_out_lens))
    }

    /// Run decoder: y [1, context_size] (i64) -> decoder_out [1, D]
    fn run_decoder(&mut self, context: &[i64]) -> Result<Array2<f32>, TranscribeError> {
        let y = ndarray::Array2::from_shape_vec((1, self.context_size), context.to_vec())?;
        let y_dyn = y.into_dyn();
        let t_y = TensorRef::from_array_view(y_dyn.view())?;

        let inputs = inputs![
            self.dec_y_name.as_str() => t_y,
        ];
        let outputs = self.decoder_session.run(inputs)?;

        let decoder_out = outputs
            .get(self.dec_out_name.as_str())
            .ok_or_else(|| {
                TranscribeError::Inference(format!(
                    "decoder output '{}' not found",
                    self.dec_out_name
                ))
            })?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok(decoder_out)
    }

    /// Run joiner: encoder_out_frame [1,D] + decoder_out [1,D] -> logit [1, vocab_size]
    fn run_joiner(
        &mut self,
        encoder_out_frame: &ArrayView1<f32>,
        decoder_out: &Array2<f32>,
    ) -> Result<Array2<f32>, TranscribeError> {
        let enc = encoder_out_frame
            .to_owned()
            .into_shape_with_order((1, encoder_out_frame.len()))?
            .into_dyn();
        let dec_dyn = decoder_out.clone().into_dyn();

        let t_enc = TensorRef::from_array_view(enc.view())?;
        let t_dec = TensorRef::from_array_view(dec_dyn.view())?;

        let inputs = inputs![
            self.join_enc_name.as_str() => t_enc,
            self.join_dec_name.as_str() => t_dec,
        ];
        let outputs = self.joiner_session.run(inputs)?;

        let logit = outputs
            .get(self.join_logit_name.as_str())
            .ok_or_else(|| {
                TranscribeError::Inference(format!(
                    "joiner output '{}' not found",
                    self.join_logit_name
                ))
            })?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok(logit)
    }

    /// Greedy search decoding for RNN-T (transducer).
    fn greedy_search(&mut self, features: &Array2<f32>) -> Result<Vec<i32>, TranscribeError> {
        let (encoder_out, encoder_out_lens) = self.run_encoder(features)?;
        let t_max = (encoder_out_lens as usize).min(encoder_out.shape()[1]);

        let mut context = vec![self.blank_id as i64; self.context_size];
        let mut decoder_out = self.run_decoder(&context)?;

        let mut tokens = Vec::new();

        for t in 0..t_max {
            let enc_frame = encoder_out.slice(ndarray::s![0, t, ..]);
            let logit = self.run_joiner(&enc_frame, &decoder_out)?;

            let logit_row = logit.row(0);
            let mut max_id = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in logit_row.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_id = i;
                }
            }

            if max_id as i32 != self.blank_id {
                tokens.push(max_id as i32);
                context.rotate_left(1);
                *context.last_mut().unwrap() = max_id as i64;
                decoder_out = self.run_decoder(&context)?;
            }
        }

        Ok(tokens)
    }
}

impl SpeechModel for ZipformerTransducerModel {
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
