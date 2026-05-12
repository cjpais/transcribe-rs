//! Core ONNX inference for Qwen3-ASR: encoder, decoder prefill, and autoregressive decode.

use ndarray::{Array1, Array2, Array3, ArrayD, Ix3, IxDyn};
use ort::session::Session;
use ort::value::{DynValue, TensorRef};
use std::path::Path;

use crate::onnx::{session, Quantization};
use crate::TranscribeError;

use super::config::Qwen3AsrConfig;
use super::mel::{self, log_mel_spectrogram};
use super::prompt::{
    build_prompt_ids_with_language, get_audio_pad_range, get_feat_extract_output_lengths,
};
use super::tokenizer::Qwen3Tokenizer;

pub struct Qwen3AsrModel {
    encoder: Session,
    decoder_init: Session,
    decoder_step: Session,
    /// Embedding matrix cached for fast per-step lookup. Loaded from
    /// `embed_tokens.bin` (FP32 cache).
    embed_tokens: Array2<f32>,
    config: Qwen3AsrConfig,
    tokenizer: Qwen3Tokenizer,
}

impl Qwen3AsrModel {
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let config = Qwen3AsrConfig::load(model_dir)?;
        let tokenizer =
            Qwen3Tokenizer::new(model_dir).map_err(|e| TranscribeError::Config(e.to_string()))?;

        {
            let st = &config.special_tokens;
            let valid = st.pad_token_id >= 0
                && st.im_start_token_id >= 0
                && st.im_end_token_id >= 0
                && st.audio_start_token_id >= 0
                && st.audio_end_token_id >= 0
                && st.audio_pad_token_id >= 0
                && st.asr_text_token_id >= 0
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
                mel::N_MELS,
                config.mel.n_mels
            )));
        }
        if config.mel.n_fft != mel::N_FFT {
            return Err(TranscribeError::Config(format!(
                "Qwen3-ASR: expected n_fft={}, got {}",
                mel::N_FFT,
                config.mel.n_fft
            )));
        }
        if config.mel.hop_length != mel::HOP_LENGTH {
            return Err(TranscribeError::Config(format!(
                "Qwen3-ASR: expected hop_length={}, got {}",
                mel::HOP_LENGTH,
                config.mel.hop_length
            )));
        }

        let encoder_path = session::resolve_model_path(model_dir, "encoder", quantization);
        if !encoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(encoder_path));
        }

        let decoder_threads: usize = 0;

        log::info!("Loading Qwen3-ASR encoder from {:?}", encoder_path);
        let encoder = session::create_session(&encoder_path)?;

        let decoder_init_path =
            session::resolve_model_path(model_dir, "decoder_init", quantization);
        let decoder_step_path =
            session::resolve_model_path(model_dir, "decoder_step", quantization);
        if !decoder_init_path.exists() {
            return Err(TranscribeError::ModelNotFound(decoder_init_path));
        }
        if !decoder_step_path.exists() {
            return Err(TranscribeError::ModelNotFound(decoder_step_path));
        }
        log::info!(
            "Loading Qwen3-ASR decoder from {:?} + {:?}",
            decoder_init_path,
            decoder_step_path
        );
        let decoder_init = session::create_decoder_session(&decoder_init_path, decoder_threads)?;
        let decoder_step = session::create_decoder_session(&decoder_step_path, decoder_threads)?;

        let embed_path = model_dir.join("embed_tokens.bin");
        if !embed_path.exists() {
            return Err(TranscribeError::ModelNotFound(embed_path));
        }
        log::info!("Loading embedding cache from {:?}", embed_path);
        let embed_tokens = load_embed_cache(&embed_path, &config)?;
        log::info!("Embedding cache: {:?}", embed_tokens.shape());

        Ok(Self {
            encoder,
            decoder_init,
            decoder_step,
            embed_tokens,
            config,
            tokenizer,
        })
    }

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

    fn greedy_decode(
        &mut self,
        audio_features: &Array3<f32>,
        max_tokens: usize,
        language_token_ids: Option<&[i64]>,
    ) -> Result<String, TranscribeError> {
        let max_tokens = max_tokens.min(4096);
        let audio_token_count = audio_features.shape()[1];
        let prompt_ids = build_prompt_ids_with_language(
            &self.config.special_tokens,
            audio_token_count,
            language_token_ids,
        );
        let seq_len = prompt_ids.len();

        let (audio_start, _) =
            get_audio_pad_range(&prompt_ids, self.config.special_tokens.audio_pad_token_id)
                .map_err(TranscribeError::Inference)?;

        let input_ids = Array2::<i64>::from_shape_vec((1, seq_len), prompt_ids.clone())?;
        let position_ids = Array2::<i64>::from_shape_fn((1, seq_len), |(_, j)| j as i64);
        let audio_offset = Array1::<i64>::from_elem(1, audio_start as i64);
        let ids_dyn = input_ids.into_dyn();
        let pos_dyn = position_ids.into_dyn();
        let audio_dyn = audio_features.view().into_dyn();
        let offset_dyn = audio_offset.into_dyn();

        let (mut current_token, mut keys, mut values) = {
            let inputs = ort::inputs![
                "input_ids"        => TensorRef::from_array_view(ids_dyn.view())?,
                "position_ids"     => TensorRef::from_array_view(pos_dyn.view())?,
                "audio_features"   => TensorRef::from_array_view(audio_dyn.view())?,
                "audio_offset"     => TensorRef::from_array_view(offset_dyn.view())?,
            ];
            let mut init_outputs = self.decoder_init.run(inputs)?;

            let logits = init_outputs
                .get("logits")
                .ok_or_else(|| {
                    TranscribeError::Inference("Missing 'logits' from decoder_init".into())
                })?
                .try_extract_array::<f32>()?;
            let last_pos = logits.shape()[1].checked_sub(1).ok_or_else(|| {
                TranscribeError::Inference("decoder_init returned empty logits sequence".into())
            })?;
            let token = argmax_slice(&logits, last_pos)?;
            drop(logits);
            let keys: DynValue = init_outputs.remove("present_keys").ok_or_else(|| {
                TranscribeError::Inference("Missing 'present_keys' from decoder_init".into())
            })?;
            let values: DynValue = init_outputs.remove("present_values").ok_or_else(|| {
                TranscribeError::Inference("Missing 'present_values' from decoder_init".into())
            })?;
            (token, keys, values)
        };

        let mut output_tokens = vec![current_token];

        if self
            .config
            .special_tokens
            .eos_token_ids
            .contains(&current_token)
        {
            return Ok(self.tokenizer.decode(&output_tokens));
        }

        let hidden_size = self.config.decoder.hidden_size;
        let mut pos = seq_len as i64;

        for _ in 1..max_tokens {
            let token_embed = {
                if current_token < 0 {
                    return Err(TranscribeError::Inference(format!(
                        "decoder produced negative token ID: {}",
                        current_token
                    )));
                }
                let id = current_token as usize;
                if id >= self.embed_tokens.nrows() {
                    return Err(TranscribeError::Inference(format!(
                        "token ID {} exceeds embedding rows {}",
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

            let step_inputs = ort::inputs![
                "input_embeds"  => TensorRef::from_array_view(token_embed.view())?,
                "position_ids"  => TensorRef::from_array_view(step_pos.view())?,
                "past_keys"     => keys,
                "past_values"   => values,
            ];
            let mut step_outputs = self.decoder_step.run(step_inputs)?;

            let step_logits = step_outputs
                .get("logits")
                .ok_or_else(|| {
                    TranscribeError::Inference("Missing 'logits' from decoder_step".into())
                })?
                .try_extract_array::<f32>()?;

            current_token = argmax_slice(&step_logits, 0)?;
            output_tokens.push(current_token);
            pos += 1;
            drop(step_logits);
            keys = step_outputs.remove("present_keys").ok_or_else(|| {
                TranscribeError::Inference("Missing 'present_keys' from step".into())
            })?;
            values = step_outputs.remove("present_values").ok_or_else(|| {
                TranscribeError::Inference("Missing 'present_values' from step".into())
            })?;

            if self
                .config
                .special_tokens
                .eos_token_ids
                .contains(&current_token)
            {
                break;
            }
        }

        let eos_reached = self
            .config
            .special_tokens
            .eos_token_ids
            .contains(&current_token);
        if !eos_reached {
            log::warn!(
                "Qwen3-ASR: max_tokens ({}) reached without EOS token",
                max_tokens
            );
        }

        let asr_text_id = self.config.special_tokens.asr_text_token_id;
        if language_token_ids.is_none() && eos_reached && !output_tokens.contains(&asr_text_id) {
            let preview: Vec<_> = output_tokens.iter().take(20).collect();
            log::warn!(
                "Qwen3-ASR: no <asr_text> token in output ({} tokens, first 20: {:?}); \
                 returning empty transcription",
                output_tokens.len(),
                preview,
            );
            return Ok(String::new());
        }

        Ok(self.tokenizer.decode(&output_tokens))
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
        max_tokens: usize,
        language_token_ids: Option<&[i64]>,
    ) -> Result<String, TranscribeError> {
        const MAX_SAMPLES: usize = 60 * 16_000;
        if samples.len() > MAX_SAMPLES {
            return Err(TranscribeError::Inference(format!(
                "audio too long: {} samples ({:.1} s); maximum is 60 s",
                samples.len(),
                samples.len() as f32 / 16_000.0,
            )));
        }

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

        self.greedy_decode(&audio_features, max_tokens, language_token_ids)
    }

    /// Encode a language name to token IDs using the tokenizer.
    pub fn encode_language(&self, language: &str) -> Vec<i64> {
        self.tokenizer.encode(language)
    }
}

/// Load the embedding cache from a raw binary file.
fn load_embed_cache(path: &Path, config: &Qwen3AsrConfig) -> Result<Array2<f32>, TranscribeError> {
    use super::config::EmbedDtype;

    let vocab_size = config.decoder.vocab_size;
    let hidden_size = config.decoder.hidden_size;
    let n_elements = vocab_size * hidden_size;
    let bpe = config.embed_tokens_dtype.bytes_per_element();
    let expected_bytes = n_elements * bpe;

    let file_size = path.metadata()?.len() as usize;
    if file_size != expected_bytes {
        return Err(TranscribeError::Config(format!(
            "embed_tokens.bin size {} != expected {} ({}x{}x{}, dtype={:?})",
            file_size, expected_bytes, vocab_size, hidden_size, bpe, config.embed_tokens_dtype
        )));
    }

    let data = std::fs::read(path)?;

    let float_data: Vec<f32> = match config.embed_tokens_dtype {
        EmbedDtype::Float16 => {
            log::info!(
                "Loading FP16 embedding cache ({} MB)",
                file_size / (1024 * 1024)
            );
            data.chunks_exact(2)
                .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect()
        }
        EmbedDtype::Float32 => {
            log::info!(
                "Loading FP32 embedding cache ({} MB)",
                file_size / (1024 * 1024)
            );
            data.chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        }
    };

    Ok(Array2::from_shape_vec(
        (vocab_size, hidden_size),
        float_data,
    )?)
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = exp;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e += 1;
            }
            f &= 0x3FF;
            let exp32 = 127 - 15 - e + 1;
            f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | (0xFF << 23) | (frac << 13))
    } else {
        let exp32 = exp + 127 - 15;
        f32::from_bits((sign << 31) | (exp32 << 23) | (frac << 13))
    }
}

/// Argmax over a 3D logits array at `[0, pos, :]`.
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

    let row = logits.slice(ndarray::s![0, pos, ..]);
    if let Some(slice) = row.as_slice() {
        let best_idx = slice
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            })
            .0;
        return Ok(best_idx as i64);
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

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn f16_positive_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0f32);
        assert!(f16_to_f32(0x0000).is_sign_positive());
    }

    #[test]
    fn f16_negative_zero() {
        assert_eq!(f16_to_f32(0x8000), -0.0f32);
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn f16_one() {
        assert_eq!(f16_to_f32(0x3C00), 1.0f32);
    }

    #[test]
    fn f16_negative_one() {
        assert_eq!(f16_to_f32(0xBC00), -1.0f32);
    }

    #[test]
    fn f16_positive_infinity() {
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    }

    #[test]
    fn f16_negative_infinity() {
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    }

    #[test]
    fn f16_nan() {
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_smallest_subnormal() {
        let val = f16_to_f32(0x0001);
        assert!(val > 0.0);
        assert!((val - 5.960_464_5e-8).abs() < 1e-12);
    }

    #[test]
    fn f16_largest_subnormal() {
        let val = f16_to_f32(0x03FF);
        assert!(val > 0.0);
        assert!((val - 6.097_555e-5).abs() < 1e-9);
    }

    #[test]
    fn f16_max_finite() {
        assert_eq!(f16_to_f32(0x7BFF), 65504.0f32);
    }

    #[test]
    fn f16_common_values() {
        assert_eq!(f16_to_f32(0x4000), 2.0f32);
        assert_eq!(f16_to_f32(0x3800), 0.5f32);
        assert!((f16_to_f32(0x3555) - 0.3333).abs() < 0.001);
    }
}
