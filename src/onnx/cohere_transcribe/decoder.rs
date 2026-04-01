use std::borrow::Cow;

use ndarray::Array4;
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::{DynValue, Tensor};

use super::vocab::Vocab;
use crate::TranscribeError;

/// Number of decoder transformer layers.
const NUM_DECODER_LAYERS: usize = 8;
/// Number of attention heads in the decoder.
const DECODER_HEADS: usize = 8;
/// Dimension per attention head.
const HEAD_DIM: usize = 128;

/// Run autoregressive decoding using the Cohere Transcribe decoder ONNX session.
///
/// Follows the same algorithm as the Python reference:
/// 1. Feed prompt tokens one at a time, building up the KV cache
/// 2. Take argmax of the final prompt step's logits to get the first generated token
/// 3. Continue autoregressive generation until EOS or max_new_tokens
pub fn decode_autoregressive(
    decoder: &mut Session,
    cross_kv: &[DynValue],
    encoder_lengths: i64,
    src_len: usize,
    prompt_tokens: Vec<i64>,
    vocab: &Vocab,
    max_new_tokens: usize,
) -> Result<String, TranscribeError> {
    let eos_id = vocab.eos_token_id();

    // Cross-attention mask: [B=1, 1, 1, src_len]
    let mut cross_mask_data = vec![0.0f32; src_len];
    for i in (encoder_lengths as usize)..src_len {
        cross_mask_data[i] = -1e9;
    }
    let cross_mask =
        Tensor::from_array(([1usize, 1, 1, src_len], cross_mask_data.into_boxed_slice()))?;

    // Initialize empty self-attention KV cache: [B=1, heads, 0, head_dim]
    let mut self_kv: Vec<DynValue> = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
    for _ in 0..NUM_DECODER_LAYERS {
        let empty_k = Array4::<f32>::zeros((1, DECODER_HEADS, 0, HEAD_DIM));
        let empty_v = Array4::<f32>::zeros((1, DECODER_HEADS, 0, HEAD_DIM));
        self_kv.push(Tensor::from_array(empty_k)?.into_dyn());
        self_kv.push(Tensor::from_array(empty_v)?.into_dyn());
    }

    // Phase 1: Feed prompt tokens one at a time to build up the KV cache.
    // We need the logits from the last prompt step.
    let mut last_logits: Option<DynValue> = None;
    for (step, &tok) in prompt_tokens.iter().enumerate() {
        let (logits, new_self_kv) =
            run_decoder_step(decoder, tok, step as i64, &cross_mask, &self_kv, cross_kv)?;
        self_kv = new_self_kv;
        last_logits = Some(logits);
    }

    // Phase 2: Generate tokens autoregressively.
    // The first token comes from the logits of the last prompt step.
    let mut generated = Vec::new();
    let prompt_len = prompt_tokens.len();

    for gen_step in 0..max_new_tokens {
        let logits = last_logits
            .as_ref()
            .ok_or_else(|| TranscribeError::Inference("No logits produced".to_string()))?;

        let next_id = extract_argmax(logits)?;
        log::debug!("Step {}: predicted token ID {}", gen_step, next_id);

        if next_id == eos_id {
            log::debug!("EOS token reached at generation step {}", gen_step);
            break;
        }

        generated.push(next_id);

        // Feed the generated token at position prompt_len + gen_step
        let (logits, new_self_kv) = run_decoder_step(
            decoder,
            next_id,
            (prompt_len + gen_step) as i64,
            &cross_mask,
            &self_kv,
            cross_kv,
        )?;
        self_kv = new_self_kv;
        last_logits = Some(logits);
    }

    let text = vocab.decode_tokens(&generated);
    Ok(text)
}

/// Run a single decoder step, returning (logits, updated_self_kv).
fn run_decoder_step(
    decoder: &mut Session,
    token_id: i64,
    position: i64,
    cross_mask: &Tensor<f32>,
    self_kv: &[DynValue],
    cross_kv: &[DynValue],
) -> Result<(DynValue, Vec<DynValue>), TranscribeError> {
    let input_ids = Tensor::from_array(([1usize, 1], vec![token_id].into_boxed_slice()))?;
    let positions = Tensor::from_array(([1usize, 1], vec![position].into_boxed_slice()))?;

    let mut feeds: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::new();

    feeds.push(("input_ids".into(), (&input_ids).into()));
    feeds.push(("positions".into(), (&positions).into()));
    feeds.push(("cross_attention_mask".into(), cross_mask.into()));

    // Self-attention KV cache inputs (by reference)
    for i in 0..NUM_DECODER_LAYERS {
        feeds.push((format!("self_k_in_{i}").into(), (&self_kv[i * 2]).into()));
        feeds.push((
            format!("self_v_in_{i}").into(),
            (&self_kv[i * 2 + 1]).into(),
        ));
    }

    // Cross-attention KV (by reference, reused every step)
    for i in 0..NUM_DECODER_LAYERS {
        feeds.push((format!("cross_k_in_{i}").into(), (&cross_kv[i * 2]).into()));
        feeds.push((
            format!("cross_v_in_{i}").into(),
            (&cross_kv[i * 2 + 1]).into(),
        ));
    }

    let mut outputs = decoder.run(feeds)?;

    // Extract logits (owned, so caller can inspect them)
    let logits = outputs
        .remove("logits")
        .ok_or_else(|| TranscribeError::Inference("Missing logits output".to_string()))?;

    // Extract updated self-attention KV cache
    let mut new_self_kv = Vec::with_capacity(NUM_DECODER_LAYERS * 2);
    for i in 0..NUM_DECODER_LAYERS {
        let k = outputs
            .remove(&format!("self_k_out_{i}"))
            .ok_or_else(|| TranscribeError::Inference(format!("Missing self_k_out_{i} output")))?;
        let v = outputs
            .remove(&format!("self_v_out_{i}"))
            .ok_or_else(|| TranscribeError::Inference(format!("Missing self_v_out_{i} output")))?;
        new_self_kv.push(k);
        new_self_kv.push(v);
    }

    Ok((logits, new_self_kv))
}

/// Extract the argmax token from the last position of the logits tensor [B, seq_len, vocab_size].
fn extract_argmax(logits: &DynValue) -> Result<i64, TranscribeError> {
    let (shape, data) = logits
        .try_extract_tensor::<f32>()
        .map_err(|e| TranscribeError::Inference(format!("Failed to extract logits: {e}")))?;

    let seq_len = shape[1] as usize;
    let vocab_size = shape[2] as usize;
    let offset = (seq_len - 1) * vocab_size;
    let last_logits = &data[offset..offset + vocab_size];

    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in last_logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }

    Ok(max_idx as i64)
}
