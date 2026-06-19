//! ONNX session wrappers for the Nemotron encoder and decoder/joint graphs.
//!
//! The encoder is *cache-aware streaming*: each call consumes one chunk of
//! mel frames plus the running cache, and returns encoded frames together with
//! the updated cache to thread into the next chunk. The multilingual variant
//! additionally takes a `prompt_index` language id.
//!
//! Ported from parakeet-rs's `model_nemotron.rs` (MIT), using transcribe-rs's
//! ort idioms (`Vec<(Cow<str>, SessionInputValue)>` for the variable input set,
//! `try_extract_array().into_dimensionality()` for typed extraction).

use std::borrow::Cow;

use ndarray::{Array1, Array3, Array4, Ix1, Ix3, Ix4};
use ort::session::{Session, SessionInputValue};
use ort::value::Value;

use crate::TranscribeError;

/// Cache-aware encoder state threaded between chunks.
///
/// Shapes are read from the encoder graph at load (the multilingual 3.5 model
/// uses `left_context = 56`, the English-only 0.6B uses `70`), so always build
/// via [`EncoderCache::zeros`].
pub(super) struct EncoderCache {
    /// `[num_layers, 1, left_context, hidden]`
    pub last_channel: Array4<f32>,
    /// `[num_layers, 1, hidden, conv_context]`
    pub last_time: Array4<f32>,
    /// `[1]`, starts at 0 and grows as the stream advances.
    pub last_channel_len: Array1<i64>,
}

impl EncoderCache {
    pub(super) fn zeros(
        num_layers: usize,
        left_context: usize,
        hidden: usize,
        conv_context: usize,
    ) -> Self {
        Self {
            last_channel: Array4::zeros((num_layers, 1, left_context, hidden)),
            last_time: Array4::zeros((num_layers, 1, hidden, conv_context)),
            last_channel_len: Array1::from_vec(vec![0i64]),
        }
    }
}

fn missing(name: &str) -> TranscribeError {
    TranscribeError::Inference(format!("missing ONNX output: {name}"))
}

/// Run the streaming encoder over one mel chunk.
///
/// * `features` — `[1, n_mels, T]` log-mel chunk
/// * `length` — number of valid mel frames in the chunk
/// * `prompt_index` — `Some(idx)` for the multilingual variant, `None` for
///   English-only (a mismatch produces an ORT `InvalidArgument` error)
///
/// Returns `(encoded [1, hidden, T_out], encoded_len, next_cache)`.
pub(super) fn run_encoder(
    encoder: &mut Session,
    features: &Array3<f32>,
    length: i64,
    cache: &EncoderCache,
    prompt_index: Option<i64>,
) -> Result<(Array3<f32>, i64, EncoderCache), TranscribeError> {
    let mut inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
        (
            "processed_signal".into(),
            SessionInputValue::from(Value::from_array(features.clone())?),
        ),
        (
            "processed_signal_length".into(),
            SessionInputValue::from(Value::from_array(Array1::from_vec(vec![length]))?),
        ),
        (
            "cache_last_channel".into(),
            SessionInputValue::from(Value::from_array(cache.last_channel.clone())?),
        ),
        (
            "cache_last_time".into(),
            SessionInputValue::from(Value::from_array(cache.last_time.clone())?),
        ),
        (
            "cache_last_channel_len".into(),
            SessionInputValue::from(Value::from_array(cache.last_channel_len.clone())?),
        ),
    ];
    if let Some(idx) = prompt_index {
        inputs.push((
            "prompt_index".into(),
            SessionInputValue::from(Value::from_array(Array1::from_vec(vec![idx]))?),
        ));
    }

    let outputs = encoder.run(inputs)?;

    let encoded = outputs
        .get("encoded")
        .ok_or_else(|| missing("encoded"))?
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix3>()?
        .to_owned();

    let encoded_len = outputs
        .get("encoded_len")
        .ok_or_else(|| missing("encoded_len"))?
        .try_extract_array::<i64>()?
        .iter()
        .next()
        .copied()
        .ok_or_else(|| missing("encoded_len (empty)"))?;

    let next_cache = EncoderCache {
        last_channel: outputs
            .get("cache_last_channel_next")
            .ok_or_else(|| missing("cache_last_channel_next"))?
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned(),
        last_time: outputs
            .get("cache_last_time_next")
            .ok_or_else(|| missing("cache_last_time_next"))?
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix4>()?
            .to_owned(),
        last_channel_len: outputs
            .get("cache_last_channel_len_next")
            .ok_or_else(|| missing("cache_last_channel_len_next"))?
            .try_extract_array::<i64>()?
            .into_dimensionality::<Ix1>()?
            .to_owned(),
    };

    Ok((encoded, encoded_len, next_cache))
}

/// Run one decoder/joint step.
///
/// * `encoder_frame` — `[1, hidden, 1]` single encoded frame
/// * `target_token` — previously emitted token (or blank to start)
/// * `state_1` / `state_2` — `[lstm_layers, 1, lstm_dim]` decoder LSTM state
///
/// Returns `(logits [vocab + 1], next_state_1, next_state_2)`.
pub(super) fn run_decoder(
    decoder: &mut Session,
    encoder_frame: &Array3<f32>,
    target_token: i32,
    state_1: &Array3<f32>,
    state_2: &Array3<f32>,
) -> Result<(Vec<f32>, Array3<f32>, Array3<f32>), TranscribeError> {
    let targets = ndarray::Array2::<i32>::from_shape_vec((1, 1), vec![target_token])?;
    let target_length = Array1::<i32>::from_vec(vec![1]);

    let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
        (
            "encoder_outputs".into(),
            SessionInputValue::from(Value::from_array(encoder_frame.clone())?),
        ),
        (
            "targets".into(),
            SessionInputValue::from(Value::from_array(targets)?),
        ),
        (
            "target_length".into(),
            SessionInputValue::from(Value::from_array(target_length)?),
        ),
        (
            "input_states_1".into(),
            SessionInputValue::from(Value::from_array(state_1.clone())?),
        ),
        (
            "input_states_2".into(),
            SessionInputValue::from(Value::from_array(state_2.clone())?),
        ),
    ];

    let outputs = decoder.run(inputs)?;

    let logits: Vec<f32> = outputs
        .get("outputs")
        .ok_or_else(|| missing("outputs"))?
        .try_extract_array::<f32>()?
        .iter()
        .copied()
        .collect();

    let next_state_1 = outputs
        .get("output_states_1")
        .ok_or_else(|| missing("output_states_1"))?
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix3>()?
        .to_owned();
    let next_state_2 = outputs
        .get("output_states_2")
        .ok_or_else(|| missing("output_states_2"))?
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix3>()?
        .to_owned();

    Ok((logits, next_state_1, next_state_2))
}
