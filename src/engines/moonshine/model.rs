use ndarray::{Array2, ArrayD};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::cache::KVCache;
use super::engine::ModelVariant;
use super::tokenizer::MoonshineTokenizer;

const DECODER_START_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;
const SAMPLE_RATE: u32 = 16000;

#[derive(thiserror::Error, Debug)]
pub enum MoonshineError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ndarray shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Tokenizer file not found: {0}")]
    TokenizerNotFound(String),
    #[error("Model output not found: {0}")]
    OutputNotFound(String),
    #[error("Tokenization error: {0}")]
    Tokenization(String),
    #[error("Audio duration must be between 0.1s and 64s, got {0:.2}s")]
    AudioDuration(f32),
    #[error("Model not loaded")]
    ModelNotLoaded,
}

pub struct MoonshineModel {
    encoder: Session,
    decoder: Session,
    tokenizer: MoonshineTokenizer,
    variant: ModelVariant,
    encoder_input_names: Vec<String>,
    decoder_input_names: Vec<String>,
}

impl Drop for MoonshineModel {
    fn drop(&mut self) {
        log::debug!("Dropping MoonshineModel ({:?})", self.variant);
    }
}

impl MoonshineModel {
    pub fn new(model_dir: &Path, variant: ModelVariant) -> Result<Self, MoonshineError> {
        let encoder_path = model_dir.join("encoder_model.onnx");
        let decoder_path = model_dir.join("decoder_model_merged.onnx");

        if !encoder_path.exists() {
            return Err(MoonshineError::ModelNotFound(
                encoder_path.display().to_string(),
            ));
        }
        if !decoder_path.exists() {
            return Err(MoonshineError::ModelNotFound(
                decoder_path.display().to_string(),
            ));
        }

        log::info!("Loading Moonshine encoder from {:?}...", encoder_path);
        let encoder = Self::init_session(&encoder_path)?;

        log::info!("Loading Moonshine decoder from {:?}...", decoder_path);
        let decoder = Self::init_session(&decoder_path)?;

        let encoder_input_names: Vec<String> =
            encoder.inputs.iter().map(|i| i.name.clone()).collect();
        let decoder_input_names: Vec<String> =
            decoder.inputs.iter().map(|i| i.name.clone()).collect();

        log::debug!("Encoder inputs: {:?}", encoder_input_names);
        log::debug!("Decoder inputs: {:?}", decoder_input_names);

        let tokenizer = MoonshineTokenizer::new(model_dir)?;

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            variant,
            encoder_input_names,
            decoder_input_names,
        })
    }

    fn init_session(path: &Path) -> Result<Session, MoonshineError> {
        let providers = vec![CPUExecutionProvider::default().build()];

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(path)?;

        for input in &session.inputs {
            log::info!(
                "Model input: name={}, type={:?}",
                input.name,
                input.input_type
            );
        }

        Ok(session)
    }

    fn encode(&mut self, audio: &Array2<f32>) -> Result<ArrayD<f32>, MoonshineError> {
        let audio_dyn = audio.clone().into_dyn();

        // Check if encoder expects attention_mask
        let outputs = if self
            .encoder_input_names
            .contains(&"attention_mask".to_string())
        {
            let attention_mask =
                Array2::<i64>::ones((audio.shape()[0], audio.shape()[1])).into_dyn();
            let inputs = inputs![
                "input_values" => TensorRef::from_array_view(audio_dyn.view())?,
                "attention_mask" => TensorRef::from_array_view(attention_mask.view())?,
            ];
            self.encoder.run(inputs)?
        } else {
            let inputs = inputs![
                "input_values" => TensorRef::from_array_view(audio_dyn.view())?,
            ];
            self.encoder.run(inputs)?
        };

        let hidden_state = outputs
            .get("last_hidden_state")
            .ok_or_else(|| MoonshineError::OutputNotFound("last_hidden_state".to_string()))?
            .try_extract_array::<f32>()?;

        Ok(hidden_state.to_owned())
    }

    pub fn generate(
        &mut self,
        samples: &[f32],
        max_length: usize,
    ) -> Result<Vec<i64>, MoonshineError> {
        // Validate audio duration
        let audio_duration = samples.len() as f32 / SAMPLE_RATE as f32;
        if audio_duration < 0.1 || audio_duration > 64.0 {
            return Err(MoonshineError::AudioDuration(audio_duration));
        }

        // Prepare audio as [1, num_samples]
        let audio = Array2::from_shape_vec((1, samples.len()), samples.to_vec())?;
        let audio_attention_mask = Array2::<i64>::ones((1, samples.len()));

        // Run encoder once
        log::trace!("Running encoder...");
        let encoder_hidden_states = self.encode(&audio)?;
        log::trace!("Encoder output shape: {:?}", encoder_hidden_states.shape());

        // Initialize KV cache
        let mut cache = KVCache::new(&self.variant);

        // Start with decoder_start_token_id
        let mut tokens: Vec<i64> = vec![DECODER_START_TOKEN_ID];
        let mut input_ids = Array2::from_shape_vec((1, 1), vec![DECODER_START_TOKEN_ID])?;

        for i in 0..max_length {
            let use_cache_branch = i > 0;

            // Build decoder inputs
            let input_ids_dyn = input_ids.clone().into_dyn();
            let use_cache_branch_arr = ndarray::arr1(&[use_cache_branch]).into_dyn();

            // Prepare cache inputs
            let cache_inputs = cache.get_inputs();

            // Build inputs dynamically based on what decoder expects
            let mut ort_inputs: Vec<(std::borrow::Cow<'_, str>, ort::value::DynValue)> = vec![
                (
                    "input_ids".into(),
                    ort::value::Value::from_array(input_ids_dyn)?.into_dyn(),
                ),
                (
                    "encoder_hidden_states".into(),
                    ort::value::Value::from_array(encoder_hidden_states.clone())?.into_dyn(),
                ),
                (
                    "use_cache_branch".into(),
                    ort::value::Value::from_array(use_cache_branch_arr)?.into_dyn(),
                ),
            ];

            // Add encoder_attention_mask if expected
            if self
                .decoder_input_names
                .contains(&"encoder_attention_mask".to_string())
            {
                let mask_dyn = audio_attention_mask.clone().into_dyn();
                ort_inputs.push((
                    "encoder_attention_mask".into(),
                    ort::value::Value::from_array(mask_dyn)?.into_dyn(),
                ));
            }

            // Add all cache inputs
            for (name, arr) in cache_inputs {
                ort_inputs.push((name.into(), ort::value::Value::from_array(arr)?.into_dyn()));
            }

            // Run decoder
            let outputs = self.decoder.run(ort_inputs)?;

            // Extract logits [1, seq_len, vocab_size]
            let logits = outputs
                .get("logits")
                .ok_or_else(|| MoonshineError::OutputNotFound("logits".to_string()))?
                .try_extract_array::<f32>()?;

            // Greedy decode: argmax(logits[0, -1, :])
            let logits_shape = logits.shape();
            let last_pos = logits_shape[1] - 1;

            let last_logits = logits.slice(ndarray::s![0, last_pos, ..]);
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(EOS_TOKEN_ID);

            tokens.push(next_token);

            if next_token == EOS_TOKEN_ID {
                log::trace!("EOS token reached at position {}", i + 1);
                break;
            }

            // Update input_ids for next iteration
            input_ids = Array2::from_shape_vec((1, 1), vec![next_token])?;

            // Update cache from outputs
            cache.update_from_outputs(&outputs, use_cache_branch)?;
        }

        log::trace!("Generated {} tokens", tokens.len());
        Ok(tokens)
    }

    pub fn decode_tokens(&self, tokens: &[i64]) -> Result<String, MoonshineError> {
        self.tokenizer.decode(tokens)
    }
}
