mod decoder;
mod vocab;

use std::path::Path;
use std::time::Instant;

use ndarray::Array2;
use ort::session::Session;
use ort::value::{DynValue, Tensor};

use self::decoder::decode_autoregressive;
use self::vocab::Vocab;
use super::Quantization;
use crate::features::{compute_mel, MelConfig, WindowType};
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

const LANGUAGES: &[&str] = &[
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko",
];

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Cohere Transcribe",
    engine_id: "cohere_transcribe",
    sample_rate: 16000,
    languages: LANGUAGES,
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Mel spectrogram configuration matching the Cohere Transcribe preprocessor.
///
/// Parameters from config.json:
///   sample_rate=16000, features=128, n_fft=512,
///   window_size=0.025 (400 samples), window_stride=0.01 (160 samples),
///   window=hann, normalize=per_feature, dither=1e-5, log=true
fn mel_config() -> MelConfig {
    MelConfig {
        sample_rate: 16000,
        num_mels: 128,
        n_fft: 512,
        hop_length: 160,
        window: WindowType::Hann,
        f_min: 0.0,
        f_max: None,
        pre_emphasis: Some(0.97),
        snip_edges: false,
        normalize_samples: true,
    }
}

/// Per-model inference parameters for Cohere Transcribe.
#[derive(Debug, Clone, Default)]
pub struct CohereTranscribeParams {
    /// Source language (ISO-639-1, e.g. "en"). Defaults to "en".
    pub language: Option<String>,
    /// Maximum number of new tokens to generate. Defaults to 256.
    pub max_new_tokens: Option<usize>,
}

/// Cohere Transcribe speech model backed by ONNX sessions.
///
/// Architecture:
///   4 chained encoder splits (conformer) -> cross_kv projection -> autoregressive decoder
///
/// Expected directory contents:
///   - `encoder-0.onnx` through `encoder-3.onnx`
///   - `cross_kv.onnx`
///   - `decoder.onnx`
///   - `vocab.txt`
pub struct CohereTranscribeModel {
    encoders: Vec<Session>,
    cross_kv: Session,
    decoder: Session,
    vocab: Vocab,
}

impl CohereTranscribeModel {
    /// Load a Cohere Transcribe model from `model_dir`.
    ///
    /// The `quantization` parameter is accepted for API consistency but currently
    /// ignored since no quantized variants of this model are available.
    pub fn load(model_dir: &Path, _quantization: &Quantization) -> Result<Self, TranscribeError> {
        if !model_dir.exists() {
            return Err(TranscribeError::ModelNotFound(model_dir.to_path_buf()));
        }

        let load_start = Instant::now();

        // Load 4 encoder splits
        let mut encoders = Vec::with_capacity(4);
        for i in 0..4 {
            let path = model_dir.join(format!("encoder-{i}.onnx"));
            log::info!("Loading Cohere encoder-{i} from {:?}...", path);
            if !path.exists() {
                return Err(TranscribeError::ModelNotFound(path));
            }
            encoders.push(super::session::create_session(&path)?);
        }

        // Load cross_kv
        let cross_kv_path = model_dir.join("cross_kv.onnx");
        log::info!("Loading Cohere cross_kv from {:?}...", cross_kv_path);
        if !cross_kv_path.exists() {
            return Err(TranscribeError::ModelNotFound(cross_kv_path));
        }
        let cross_kv = super::session::create_session(&cross_kv_path)?;

        // Load decoder
        let decoder_path = model_dir.join("decoder.onnx");
        log::info!("Loading Cohere decoder from {:?}...", decoder_path);
        if !decoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(decoder_path));
        }
        let decoder = super::session::create_session(&decoder_path)?;

        // Load vocabulary
        let vocab_path = model_dir.join("vocab.txt");
        let vocab = Vocab::load(&vocab_path)?;

        log::info!(
            "Cohere Transcribe model loaded in {:.2?}",
            load_start.elapsed()
        );

        Ok(Self {
            encoders,
            cross_kv,
            decoder,
            vocab,
        })
    }

    /// Transcribe with model-specific parameters.
    ///
    /// The upstream config specifies `max_audio_clip_s = 35`, but the ONNX encoder
    /// accepts longer audio. The autoregressive decoder is the practical limit:
    /// it stops at EOS or `max_new_tokens` (default 256), so very long audio may
    /// be truncated. For long-form transcription, use a chunked transcriber
    /// (e.g. `VadChunkedTranscriber` or `EnergyAdaptiveTranscriber`).
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &CohereTranscribeParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let language = params.language.as_deref().unwrap_or("en");
        let max_new_tokens = params.max_new_tokens.unwrap_or(256);
        let total_start = Instant::now();

        // Step 1: Compute mel features
        let mel_start = Instant::now();
        let mel_features = self.compute_features(samples)?;
        log::debug!(
            "Mel features computed in {:.2?}: shape [{}, {}]",
            mel_start.elapsed(),
            mel_features.shape()[0],
            mel_features.shape()[1]
        );

        // Transpose to [B=1, mels, frames] for the encoder
        let num_frames = mel_features.shape()[0];
        let num_mels = mel_features.shape()[1];
        let mel_t = mel_features.t();
        let mel_3d_data: Vec<f32> = mel_t.iter().cloned().collect();
        let input_features = Tensor::from_array((
            [1usize, num_mels, num_frames],
            mel_3d_data.into_boxed_slice(),
        ))?;
        let length = Tensor::from_array(([1usize], vec![num_frames as i64].into_boxed_slice()))?;

        // Step 2: Run chained encoder
        let encode_start = Instant::now();
        let (encoder_out, encoder_lengths) = self.run_encoder(input_features, length)?;
        let enc_len_data = encoder_lengths.try_extract_tensor::<i64>().map_err(|e| {
            TranscribeError::Inference(format!("Failed to extract encoder lengths: {e}"))
        })?;
        let enc_len = enc_len_data.1[0];
        log::debug!(
            "Encoder completed in {:.2?}, output length={}",
            encode_start.elapsed(),
            enc_len
        );

        // Step 3: Compute cross-attention KV
        let cross_start = Instant::now();
        let cross_kv_outputs = self.run_cross_kv(&encoder_out)?;
        let src_len = cross_kv_outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                TranscribeError::Inference(format!("Failed to extract cross_kv shape: {e}"))
            })?
            .0[2] as usize;
        log::debug!(
            "Cross-KV computed in {:.2?}, src_len={}",
            cross_start.elapsed(),
            src_len
        );

        // Step 4: Build prompt and decode
        let decode_start = Instant::now();
        let prompt_tokens = self.vocab.build_prompt(language)?;
        log::debug!(
            "Prompt tokens ({}): {:?}",
            prompt_tokens.len(),
            prompt_tokens
        );

        let text = decode_autoregressive(
            &mut self.decoder,
            &cross_kv_outputs,
            enc_len,
            src_len,
            prompt_tokens,
            &self.vocab,
            max_new_tokens,
        )?;

        log::debug!("Decoding completed in {:.2?}", decode_start.elapsed());
        log::info!(
            "Transcription completed in {:.2?}: \"{}\"",
            total_start.elapsed(),
            text
        );

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    /// Compute mel spectrogram features with per-feature normalization.
    fn compute_features(&self, samples: &[f32]) -> Result<Array2<f32>, TranscribeError> {
        let config = mel_config();

        // compute_mel returns [num_frames, num_mels]
        let mut features = compute_mel(samples, &config);

        if features.shape()[0] == 0 {
            return Err(TranscribeError::Audio(
                "Audio too short to produce mel features".to_string(),
            ));
        }

        // Per-feature normalization: z-score per mel bin across time
        let num_frames = features.shape()[0];
        let num_mels = features.shape()[1];

        for mel in 0..num_mels {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;

            for frame in 0..num_frames {
                let v = features[[frame, mel]] as f64;
                sum += v;
                sum_sq += v * v;
            }

            let mean = sum / num_frames as f64;
            let variance = (sum_sq / num_frames as f64) - mean * mean;
            let std = variance.max(0.0).sqrt();
            let inv_std = if std > 1e-10 { 1.0 / std } else { 0.0 };

            for frame in 0..num_frames {
                let v = features[[frame, mel]] as f64;
                features[[frame, mel]] = ((v - mean) * inv_std) as f32;
            }
        }

        Ok(features)
    }

    /// Run the chained encoder (4 splits), returning (encoder_out, encoder_lengths).
    fn run_encoder(
        &mut self,
        input_features: Tensor<f32>,
        length: Tensor<i64>,
    ) -> Result<(DynValue, DynValue), TranscribeError> {
        // Intermediate output names (middle splits)
        const INTERMEDIATE_OUTPUTS: &[&str] = &[
            "hidden_states_out",
            "pos_emb_out",
            "att_mask_out",
            "pad_mask_out",
            "length_out",
        ];

        // First encoder split takes (input_features, length).
        // Collect intermediate values into owned DynValues to avoid borrow conflicts.
        let mut prev_values: Vec<DynValue> = {
            let mut outputs = self.encoders[0].run(ort::inputs![
                "input_features" => input_features,
                "length" => length
            ])?;
            INTERMEDIATE_OUTPUTS
                .iter()
                .map(|name| {
                    outputs.remove(*name).ok_or_else(|| {
                        TranscribeError::Inference(format!("Missing encoder-0 output: {name}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Middle splits (1 and 2): take 5 intermediates, produce 5 intermediates
        for i in 1..3 {
            // Read input names from the session metadata before running
            let input_names: Vec<String> = self.encoders[i]
                .inputs()
                .iter()
                .map(|inp| inp.name().to_string())
                .collect();

            let feeds: Vec<(std::borrow::Cow<'_, str>, DynValue)> = input_names
                .into_iter()
                .zip(prev_values.drain(..))
                .map(|(name, val)| (std::borrow::Cow::Owned(name), val))
                .collect();

            let mut outputs = self.encoders[i].run(feeds)?;
            prev_values = INTERMEDIATE_OUTPUTS
                .iter()
                .map(|name| {
                    outputs.remove(*name).ok_or_else(|| {
                        TranscribeError::Inference(format!("Missing encoder-{i} output: {name}"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
        }

        // Last split (3): takes 5 intermediates, produces encoder_out + encoder_lengths
        {
            let input_names: Vec<String> = self.encoders[3]
                .inputs()
                .iter()
                .map(|inp| inp.name().to_string())
                .collect();

            let feeds: Vec<(std::borrow::Cow<'_, str>, DynValue)> = input_names
                .into_iter()
                .zip(prev_values.drain(..))
                .map(|(name, val)| (std::borrow::Cow::Owned(name), val))
                .collect();

            let mut outputs = self.encoders[3].run(feeds)?;

            let encoder_out = outputs
                .remove("encoder_out")
                .ok_or_else(|| TranscribeError::Inference("Missing encoder_out".to_string()))?;
            let encoder_lengths = outputs
                .remove("encoder_lengths")
                .ok_or_else(|| TranscribeError::Inference("Missing encoder_lengths".to_string()))?;

            Ok((encoder_out, encoder_lengths))
        }
    }

    /// Run the cross-KV projection, returning a list of DynValues
    /// [cross_k_0, cross_v_0, cross_k_1, cross_v_1, ..., cross_k_7, cross_v_7].
    fn run_cross_kv(&mut self, encoder_out: &DynValue) -> Result<Vec<DynValue>, TranscribeError> {
        let mut outputs = self.cross_kv.run(ort::inputs![
            "encoder_out" => encoder_out
        ])?;

        let mut cross_kv = Vec::with_capacity(16);
        for i in 0..8 {
            let k = outputs
                .remove(&format!("cross_k_{i}"))
                .ok_or_else(|| TranscribeError::Inference(format!("Missing cross_k_{i} output")))?;
            let v = outputs
                .remove(&format!("cross_v_{i}"))
                .ok_or_else(|| TranscribeError::Inference(format!("Missing cross_v_{i} output")))?;
            cross_kv.push(k);
            cross_kv.push(v);
        }

        Ok(cross_kv)
    }
}

impl SpeechModel for CohereTranscribeModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = CohereTranscribeParams {
            language: options.language.clone(),
            ..Default::default()
        };
        self.transcribe_with(samples, &params)
    }
}
