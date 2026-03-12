mod decoder;
pub(crate) mod vocab;

use std::path::Path;
use std::time::Instant;

use ort::session::Session;
use ort::value::Tensor;

use self::decoder::decode_autoregressive;
use self::vocab::Vocab;
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Canary 1B v2",
    engine_id: "canary",
    sample_rate: 16000,
    languages: &[
        "en", "de", "es", "fr", "hi", "ja", "ko", "pt", "zh", "ar", "cs", "da", "fi", "hu", "it",
        "lt", "lv", "nl", "no", "pl", "ro", "ru", "sk", "sv", "tr", "uk", "vi",
    ],
    supports_timestamps: false,
    supports_translation: true,
    supports_streaming: false,
};

/// Per-model inference parameters for Canary.
#[derive(Debug, Clone)]
pub struct CanaryParams {
    /// Source language hint (e.g. "en", "de"). Defaults to "en".
    pub language: Option<String>,
    /// Target language for translation (e.g. "en"). Defaults to source language.
    pub target_language: Option<String>,
    /// Whether to use punctuation and capitalization. Defaults to true.
    pub use_pnc: bool,
    /// Maximum number of tokens to generate. Defaults to 1024.
    pub max_sequence_length: usize,
}

impl Default for CanaryParams {
    fn default() -> Self {
        Self {
            language: None,
            target_language: None,
            use_pnc: true,
            max_sequence_length: 1024,
        }
    }
}

/// Canary 1B v2 speech model backed by three ONNX sessions (preprocessor, encoder, decoder).
pub struct CanaryModel {
    preprocessor: Session,
    encoder: Session,
    decoder: Session,
    vocab: Vocab,
}

impl CanaryModel {
    /// Load a Canary model from `model_dir`.
    ///
    /// Expected directory contents:
    /// - `nemo128.onnx` (preprocessor, always FP32)
    /// - `encoder-model[.int8|.fp16].onnx` (quantization-aware)
    /// - `decoder-model[.int8|.fp16].onnx` (quantization-aware)
    /// - `vocab.txt`
    pub fn load(
        model_dir: &Path,
        quantization: &super::Quantization,
    ) -> Result<Self, TranscribeError> {
        if !model_dir.exists() {
            return Err(TranscribeError::ModelNotFound(model_dir.to_path_buf()));
        }

        let load_start = Instant::now();

        // Preprocessor is always FP32
        let preprocessor_path = model_dir.join("nemo128.onnx");
        log::info!(
            "Loading Canary preprocessor from {:?}...",
            preprocessor_path
        );
        let preprocessor = super::session::create_session(&preprocessor_path)?;

        // Encoder and decoder respect quantization
        let encoder_path =
            super::session::resolve_model_path(model_dir, "encoder-model", quantization);
        log::info!("Loading Canary encoder from {:?}...", encoder_path);
        let encoder = super::session::create_session(&encoder_path)?;

        let decoder_path =
            super::session::resolve_model_path(model_dir, "decoder-model", quantization);
        log::info!("Loading Canary decoder from {:?}...", decoder_path);
        let decoder = super::session::create_session(&decoder_path)?;

        // Vocabulary
        let vocab_path = model_dir.join("vocab.txt");
        let vocab = Vocab::load(&vocab_path)?;

        log::info!("Canary model loaded in {:.2?}", load_start.elapsed());

        Ok(Self {
            preprocessor,
            encoder,
            decoder,
            vocab,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &CanaryParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let src_lang = params.language.as_deref().unwrap_or("en");
        let tgt_lang = params.target_language.as_deref().unwrap_or(src_lang);

        let total_start = Instant::now();

        // --- Step 1: Preprocess audio -> mel features ---
        let preprocess_start = Instant::now();
        let num_samples = samples.len();

        log::debug!("Preprocessor input: waveforms shape [1, {}]", num_samples);

        let waveforms = Tensor::from_array((
            vec![1i64, num_samples as i64],
            samples.to_vec().into_boxed_slice(),
        ))?;
        let waveforms_lens =
            Tensor::from_array((vec![1i64], vec![num_samples as i64].into_boxed_slice()))?;

        let preprocess_out = self.preprocessor.run(ort::inputs![
            "waveforms" => waveforms,
            "waveforms_lens" => waveforms_lens
        ])?;

        let (features_shape, features_data) = preprocess_out["features"]
            .try_extract_tensor::<f32>()
            .map_err(|e| TranscribeError::Inference(format!("Failed to extract features: {e}")))?;
        let (features_lens_shape, features_lens_data) = preprocess_out["features_lens"]
            .try_extract_tensor::<i64>()
            .map_err(|e| {
                TranscribeError::Inference(format!("Failed to extract features_lens: {e}"))
            })?;

        let features_shape_vec: Vec<i64> = features_shape.iter().copied().collect();
        let features_lens_shape_vec: Vec<i64> = features_lens_shape.iter().copied().collect();

        log::debug!(
            "Preprocessor output: features shape {:?}, lens {:?} ({:.2?})",
            features_shape_vec,
            features_lens_data,
            preprocess_start.elapsed()
        );

        let features_tensor = Tensor::from_array((
            features_shape_vec,
            features_data.to_vec().into_boxed_slice(),
        ))?;
        let features_lens_tensor = Tensor::from_array((
            features_lens_shape_vec,
            features_lens_data.to_vec().into_boxed_slice(),
        ))?;

        // --- Step 2: Encode mel features -> encoder embeddings ---
        let encode_start = Instant::now();

        let encoder_out = self.encoder.run(ort::inputs![
            "audio_signal" => features_tensor,
            "length" => features_lens_tensor
        ])?;

        let (enc_emb_shape, enc_emb_data) = encoder_out["encoder_embeddings"]
            .try_extract_tensor::<f32>()
            .map_err(|e| {
                TranscribeError::Inference(format!("Failed to extract encoder_embeddings: {e}"))
            })?;
        let (enc_mask_shape, enc_mask_data) = encoder_out["encoder_mask"]
            .try_extract_tensor::<i64>()
            .map_err(|e| {
                TranscribeError::Inference(format!("Failed to extract encoder_mask: {e}"))
            })?;

        let enc_emb_shape_vec: Vec<i64> = enc_emb_shape.iter().copied().collect();
        let enc_mask_shape_vec: Vec<i64> = enc_mask_shape.iter().copied().collect();

        log::debug!(
            "Encoder output: embeddings shape {:?}, mask shape {:?} ({:.2?})",
            enc_emb_shape_vec,
            enc_mask_shape_vec,
            encode_start.elapsed()
        );

        let encoder_embeddings =
            Tensor::from_array((enc_emb_shape_vec, enc_emb_data.to_vec().into_boxed_slice()))?;
        let encoder_mask = Tensor::from_array((
            enc_mask_shape_vec,
            enc_mask_data.to_vec().into_boxed_slice(),
        ))?;

        // --- Step 3: Build prompt tokens ---
        let prompt_tokens = self
            .vocab
            .build_prompt(src_lang, tgt_lang, params.use_pnc)?;

        log::debug!(
            "Prompt tokens ({}): {:?}",
            prompt_tokens.len(),
            prompt_tokens
        );

        // --- Step 4: Autoregressive decoding ---
        let decode_start = Instant::now();

        let text = decode_autoregressive(
            &mut self.decoder,
            &encoder_embeddings,
            &encoder_mask,
            prompt_tokens,
            &self.vocab,
            params.max_sequence_length,
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
}

impl SpeechModel for CanaryModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let src_lang = options.language.as_deref().unwrap_or("en");
        let tgt_lang = if options.translate { "en" } else { src_lang };
        let params = CanaryParams {
            language: Some(src_lang.to_string()),
            target_language: Some(tgt_lang.to_string()),
            ..Default::default()
        };
        self.transcribe_with(samples, &params)
    }
}
