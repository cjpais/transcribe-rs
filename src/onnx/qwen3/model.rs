//! Core ONNX inference for Qwen3-ASR: encoder, decoder prefill, and autoregressive decode.

use ndarray::{Array2, Array3, ArrayD, Ix3, IxDyn};
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use crate::TranscribeError;
use crate::onnx::{session, Quantization};

use super::config::Qwen3AsrConfig;
use super::mel::{self, log_mel_spectrogram};
use super::prompt::{build_prompt_ids, get_audio_pad_range, get_feat_extract_output_lengths};
use super::tokenizer::Qwen3Tokenizer;

/// Decoder session layout.
///
/// Unified (preferred): a single `decoder.onnx` handles both prefill (past_seq=0)
/// and autoregressive decode steps. Weights are stored once, halving decoder artifact size.
///
/// Split (legacy): separate `decoder_init.onnx` (prefill) and `decoder_step.onnx` (step).
/// Supported for backward compatibility with existing exported model directories.
enum DecoderSessions {
    /// Single session — handles both prefill (empty past KV) and decode steps.
    Unified(Session),
    /// Separate sessions for prefill and decode.
    Split { init: Session, step: Session },
}

pub struct Qwen3AsrModel {
    encoder: Session,
    decoder: DecoderSessions,
    embed_tokens: Array2<f32>,
    config: Qwen3AsrConfig,
    tokenizer: Qwen3Tokenizer,
}

impl Qwen3AsrModel {
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let config = Qwen3AsrConfig::load(model_dir)?;
        let tokenizer = Qwen3Tokenizer::new(model_dir)
            .map_err(|e| TranscribeError::Config(e.to_string()))?;

        {
            let st = &config.special_tokens;
            let valid = st.pad_token_id >= 0
                && st.im_start_token_id >= 0
                && st.im_end_token_id >= 0
                && st.audio_start_token_id >= 0
                && st.audio_end_token_id >= 0
                && st.audio_pad_token_id >= 0
                && !st.eos_token_ids.is_empty()
                && st.eos_token_ids.iter().all(|&id| id >= 0);
            if !valid {
                return Err(TranscribeError::Config(
                    "config.json: special_tokens contains negative or missing IDs".into(),
                ));
            }
        }

        if config.mel.n_mels != mel::N_MELS {
            return Err(TranscribeError::Config(format!(
                "Qwen3-ASR: expected n_mels={}, got {}",
                mel::N_MELS, config.mel.n_mels
            )));
        }
        if config.mel.n_fft != mel::N_FFT {
            return Err(TranscribeError::Config(format!(
                "Qwen3-ASR: expected n_fft={}, got {}",
                mel::N_FFT, config.mel.n_fft
            )));
        }
        if config.mel.hop_length != mel::HOP_LENGTH {
            return Err(TranscribeError::Config(format!(
                "Qwen3-ASR: expected hop_length={}, got {}",
                mel::HOP_LENGTH, config.mel.hop_length
            )));
        }

        let encoder_path = session::resolve_model_path(model_dir, "encoder", quantization);
        if !encoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(encoder_path));
        }

        // Encoder is a single-pass; use all available cores.
        // Decoder runs sequentially (one step per token); fewer threads reduces
        // synchronization overhead relative to useful work per step.
        let encoder_threads = 0; // 0 = ORT default (all cores)
        let decoder_threads = 6; // tuned: fewer threads for sequential per-token steps

        log::info!("Loading Qwen3-ASR encoder from {:?}", encoder_path);
        let encoder = session::create_session_with_threads(&encoder_path, encoder_threads)?;

        // Prefer unified decoder (decoder.onnx) — single file, no weight duplication.
        // Fall back to split (decoder_init.onnx + decoder_step.onnx) for existing exports.
        let unified_path = session::resolve_model_path(model_dir, "decoder", quantization);
        let decoder = if unified_path.exists() {
            log::info!("Loading Qwen3-ASR unified decoder from {:?}", unified_path);
            DecoderSessions::Unified(session::create_session_with_threads(&unified_path, decoder_threads)?)
        } else {
            let decoder_init_path =
                session::resolve_model_path(model_dir, "decoder_init", quantization);
            if !decoder_init_path.exists() {
                return Err(TranscribeError::ModelNotFound(decoder_init_path));
            }
            let decoder_step_path =
                session::resolve_model_path(model_dir, "decoder_step", quantization);
            if !decoder_step_path.exists() {
                return Err(TranscribeError::ModelNotFound(decoder_step_path));
            }
            log::info!("Loading Qwen3-ASR split decoder from {:?} + {:?}",
                decoder_init_path, decoder_step_path);
            DecoderSessions::Split {
                init: session::create_session_with_threads(&decoder_init_path, decoder_threads)?,
                step: session::create_session_with_threads(&decoder_step_path, decoder_threads)?,
            }
        };

        let embed_path = model_dir.join("embed_tokens.bin");
        if !embed_path.exists() {
            return Err(TranscribeError::ModelNotFound(embed_path));
        }
        log::info!("Loading embedding matrix from {:?}", embed_path);
        let embed_tokens = load_embed_tokens(&embed_path, &config)?;
        log::info!("Embedding matrix shape: {:?}", embed_tokens.shape());

        let sample = embed_tokens[[0, 0]];
        if sample.is_nan() || sample.is_infinite() {
            return Err(TranscribeError::Config(
                "embed_tokens.bin appears corrupt (NaN/Inf in first entry)".into(),
            ));
        }

        Ok(Self {
            encoder,
            decoder,
            embed_tokens,
            config,
            tokenizer,
        })
    }

    /// Run the encoder on a mel spectrogram.
    fn encode(&mut self, mel: &Array3<f32>) -> Result<Array3<f32>, TranscribeError> {
        let mel_dyn = mel.view().into_dyn();
        let inputs = ort::inputs![
            "mel" => TensorRef::from_array_view(mel_dyn.view())?,
        ];
        let outputs = self.encoder.run(inputs)?;

        let features = outputs
            .get("audio_features")
            .ok_or_else(|| TranscribeError::Inference("Missing 'audio_features' output".into()))?
            .try_extract_array::<f32>()?;

        Ok(features.to_owned().into_dimensionality::<Ix3>()?)
    }

    /// Build input embeddings: embed all prompt tokens, then replace audio_pad
    /// positions with encoder output features.
    fn build_input_embeds(
        &self,
        prompt_ids: &[i64],
        audio_features: &Array3<f32>,
    ) -> Result<Array3<f32>, TranscribeError> {
        let hidden_size = self.config.decoder.hidden_size;
        let seq_len = prompt_ids.len();

        let mut embeds = Array2::<f32>::zeros((seq_len, hidden_size));
        for (i, &id) in prompt_ids.iter().enumerate() {
            let id = id as usize;
            if id < self.embed_tokens.nrows() {
                embeds.row_mut(i).assign(&self.embed_tokens.row(id));
            } else {
                log::warn!(
                    "Prompt token ID {} exceeds embedding matrix rows ({})",
                    id,
                    self.embed_tokens.nrows()
                );
            }
        }

        let (audio_start, audio_end) =
            get_audio_pad_range(prompt_ids, self.config.special_tokens.audio_pad_token_id)
                .map_err(TranscribeError::Inference)?;
        let audio_len = audio_end - audio_start;
        for i in 0..audio_len {
            embeds
                .row_mut(audio_start + i)
                .assign(&audio_features.slice(ndarray::s![0, i, ..]));
        }

        Ok(embeds.into_shape_with_order((1, seq_len, hidden_size))?)
    }

    /// Run greedy decoding on audio features, returning decoded text.
    fn greedy_decode(
        &mut self,
        audio_features: &Array3<f32>,
        max_tokens: usize,
    ) -> Result<String, TranscribeError> {
        let audio_token_count = audio_features.shape()[1];
        let prompt_ids = build_prompt_ids(&self.config.special_tokens, audio_token_count);
        let input_embeds = self.build_input_embeds(&prompt_ids, audio_features)?;
        let seq_len = prompt_ids.len();

        let position_ids = Array2::<i64>::from_shape_fn((1, seq_len), |(_, j)| j as i64);
        let embeds_dyn = input_embeds.into_dyn();
        let pos_dyn = position_ids.into_dyn();

        let nl = self.config.decoder.num_layers;
        let nkv = self.config.decoder.num_key_value_heads;
        let hd = self.config.decoder.head_dim;

        // Prefill: run encoder output + prompt through the decoder.
        // Unified decoder accepts empty past KV (past_seq=0) for the prefill pass.
        // Split decoder_init takes only input_embeds + position_ids.
        //
        // The block scope ensures `init_outputs` (which borrows from the Session inside
        // `self.decoder`) is dropped before the step loop needs to borrow `self.decoder`
        // again, satisfying the borrow checker.
        let (mut current_token, mut keys, mut values) = {
            let (init_outputs, prefill_label) = match &mut self.decoder {
                DecoderSessions::Unified(s) => {
                    let empty_keys =
                        ArrayD::<f32>::zeros(IxDyn(&[nl, 1, nkv, 0, hd]));
                    let empty_values =
                        ArrayD::<f32>::zeros(IxDyn(&[nl, 1, nkv, 0, hd]));
                    let inputs = ort::inputs![
                        "input_embeds"  => TensorRef::from_array_view(embeds_dyn.view())?,
                        "position_ids"  => TensorRef::from_array_view(pos_dyn.view())?,
                        "past_keys"     => TensorRef::from_array_view(empty_keys.view())?,
                        "past_values"   => TensorRef::from_array_view(empty_values.view())?,
                    ];
                    (s.run(inputs)?, "decoder")
                }
                DecoderSessions::Split { init, .. } => {
                    let inputs = ort::inputs![
                        "input_embeds"  => TensorRef::from_array_view(embeds_dyn.view())?,
                        "position_ids"  => TensorRef::from_array_view(pos_dyn.view())?,
                    ];
                    (init.run(inputs)?, "decoder_init")
                }
            };

            let logits = init_outputs
                .get("logits")
                .ok_or_else(|| TranscribeError::Inference(
                    format!("Missing 'logits' from {prefill_label}")
                ))?
                .try_extract_array::<f32>()?;
            let present_keys = init_outputs
                .get("present_keys")
                .ok_or_else(|| TranscribeError::Inference(
                    format!("Missing 'present_keys' from {prefill_label}")
                ))?
                .try_extract_array::<f32>()?;
            let present_values = init_outputs
                .get("present_values")
                .ok_or_else(|| TranscribeError::Inference(
                    format!("Missing 'present_values' from {prefill_label}")
                ))?
                .try_extract_array::<f32>()?;

            let last_pos = logits.shape()[1].checked_sub(1).ok_or_else(|| {
                TranscribeError::Inference(
                    format!("{prefill_label} returned empty logits sequence")
                )
            })?;
            let token = argmax_slice(&logits, last_pos)?;
            // Clone KV cache into owned arrays before init_outputs drops.
            let keys = present_keys.to_owned().into_dyn();
            let values = present_values.to_owned().into_dyn();
            (token, keys, values)
            // init_outputs drops here, releasing the borrow on self.decoder.
        };

        let mut output_tokens = vec![current_token];

        if self.config.special_tokens.eos_token_ids.contains(&current_token) {
            return Ok(self.tokenizer.decode(&output_tokens));
        }

        let mut pos = seq_len as i64;

        for _ in 1..max_tokens {
            let hidden_size = self.config.decoder.hidden_size;

            let token_embed = {
                let id = current_token as usize;
                if id >= self.embed_tokens.nrows() {
                    return Err(TranscribeError::Inference(format!(
                        "token ID {} exceeds embedding matrix rows {}",
                        id,
                        self.embed_tokens.nrows()
                    )));
                }
                let row = self.embed_tokens.row(id);
                let mut arr = Array3::<f32>::zeros((1, 1, hidden_size));
                arr.slice_mut(ndarray::s![0, 0, ..]).assign(&row);
                arr.into_dyn()
            };

            let step_pos = ArrayD::<i64>::from_shape_vec(IxDyn(&[1, 1]), vec![pos])?;

            // Both unified and split step sessions share the same KV-cache interface.
            let step_inputs = ort::inputs![
                "input_embeds"  => TensorRef::from_array_view(token_embed.view())?,
                "position_ids"  => TensorRef::from_array_view(step_pos.view())?,
                "past_keys"     => TensorRef::from_array_view(keys.view())?,
                "past_values"   => TensorRef::from_array_view(values.view())?,
            ];
            let step_outputs = match &mut self.decoder {
                DecoderSessions::Unified(s) => s.run(step_inputs)?,
                DecoderSessions::Split { step, .. } => step.run(step_inputs)?,
            };

            let step_logits = step_outputs
                .get("logits")
                .ok_or_else(|| TranscribeError::Inference(
                    "Missing 'logits' from decoder_step".into()
                ))?
                .try_extract_array::<f32>()?;

            // Extract updated KV state before any early exit so the ordering is explicit.
            let next_keys = step_outputs
                .get("present_keys")
                .ok_or_else(|| TranscribeError::Inference(
                    "Missing 'present_keys' from step".into()
                ))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dyn();
            let next_values = step_outputs
                .get("present_values")
                .ok_or_else(|| TranscribeError::Inference(
                    "Missing 'present_values' from step".into()
                ))?
                .try_extract_array::<f32>()?
                .to_owned()
                .into_dyn();

            current_token = argmax_slice(&step_logits, 0)?;
            output_tokens.push(current_token);
            pos += 1;
            keys = next_keys;
            values = next_values;

            if self.config.special_tokens.eos_token_ids.contains(&current_token) {
                break;
            }
        }

        if !self.config.special_tokens.eos_token_ids.contains(&current_token) {
            log::warn!(
                "Qwen3-ASR: max_tokens ({}) reached without EOS token",
                max_tokens
            );
        }

        Ok(self.tokenizer.decode(&output_tokens))
    }

    /// Full transcription pipeline: audio samples → raw decoded text.
    ///
    /// Returns the raw model output including the language prefix. The caller
    /// (engine layer) is responsible for stripping the prefix.
    pub fn transcribe(&mut self, samples: &[f32], max_tokens: usize) -> Result<String, TranscribeError> {
        let mel = log_mel_spectrogram(samples);
        let mel_frames = mel.shape()[2];
        let audio_tokens = get_feat_extract_output_lengths(mel_frames);
        log::debug!(
            "Mel frames: {}, encoder output tokens: {}",
            mel_frames,
            audio_tokens
        );

        if audio_tokens == 0 {
            return Ok(String::new());
        }

        let audio_features = self.encode(&mel)?;
        log::debug!("Audio features shape: {:?}", audio_features.shape());

        let encoder_frames = audio_features.shape()[1];
        if encoder_frames != audio_tokens {
            return Err(TranscribeError::Inference(format!(
                "Qwen3-ASR encoder produced {} frames, expected {} from mel formula \
                 (mel_frames={}, audio_tokens={})",
                encoder_frames, audio_tokens, mel_frames, audio_tokens
            )));
        }

        self.greedy_decode(&audio_features, max_tokens)
    }
}

fn load_embed_tokens(path: &Path, config: &Qwen3AsrConfig) -> Result<Array2<f32>, TranscribeError> {
    if config.embed_tokens_dtype != "float32" {
        return Err(TranscribeError::Config(format!(
            "embed_tokens_dtype '{}' is not supported; only 'float32' is implemented",
            config.embed_tokens_dtype
        )));
    }

    let [vocab_size, hidden_size] = config.embed_tokens_shape;
    // Sanity-check sizes before multiplying to avoid overflow on 32-bit targets
    // and to catch obviously corrupt config values.
    const MAX_DIM: usize = 10_000_000;
    if vocab_size > MAX_DIM || hidden_size > MAX_DIM {
        return Err(TranscribeError::Config(format!(
            "embed_tokens_shape [{}, {}] exceeds sanity limit of {}",
            vocab_size, hidden_size, MAX_DIM
        )));
    }
    let expected_bytes = vocab_size
        .checked_mul(hidden_size)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| {
            TranscribeError::Config(format!(
                "embed_tokens_shape [{}, {}] overflows usize",
                vocab_size, hidden_size
            ))
        })?;

    // Check file size before allocating to fail fast on corrupt config values.
    let file_size_u64 = path.metadata()?.len();
    let file_size = usize::try_from(file_size_u64).map_err(|_| {
        TranscribeError::Config(format!(
            "embed_tokens.bin size {} bytes exceeds platform address space",
            file_size_u64
        ))
    })?;
    if file_size != expected_bytes {
        return Err(TranscribeError::Config(format!(
            "embed_tokens.bin size {} != expected {} ({}x{}x4)",
            file_size, expected_bytes, vocab_size, hidden_size
        )));
    }

    let data = std::fs::read(path)?;

    let float_data: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Array2::from_shape_vec(
        (vocab_size, hidden_size),
        float_data,
    )?)
}

/// Argmax over a 3D logits array at `[0, pos, :]`.
///
/// Returns `TranscribeError::Inference` if the array does not have shape `[1, T, vocab]`
/// or if `pos >= T`.
fn argmax_slice(logits: &ndarray::ArrayViewD<'_, f32>, pos: usize) -> Result<i64, TranscribeError> {
    let shape = logits.shape();
    if logits.ndim() != 3 || shape[0] != 1 {
        return Err(TranscribeError::Inference(format!(
            "argmax_slice: expected shape [1, T, vocab], got {:?}",
            shape
        )));
    }
    if pos >= shape[1] {
        return Err(TranscribeError::Inference(format!(
            "argmax_slice: pos {} out of range for sequence length {}",
            pos, shape[1]
        )));
    }

    let vocab_size = shape[2];
    let mut best_idx = 0i64;
    let mut best_val = f32::NEG_INFINITY;

    for v in 0..vocab_size {
        let val = logits[[0, pos, v]];
        if val > best_val {
            best_val = val;
            best_idx = v as i64;
        }
    }
    Ok(best_idx)
}
