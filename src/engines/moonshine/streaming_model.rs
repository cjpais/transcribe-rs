use ndarray::{ArrayD, ArrayViewD, IxDyn};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::bin_tokenizer::BinTokenizer;
use super::model::MoonshineError;
use super::streaming_config::StreamingConfig;
use super::streaming_state::StreamingState;

const SAMPLE_RATE: u32 = 16000;
const CHUNK_SIZE: usize = 1280; // 80ms at 16kHz

/// Streaming model with 5 ONNX sessions matching C++ `MoonshineStreamingModel`.
pub struct StreamingModel {
    frontend: Session,
    encoder: Session,
    adapter: Session,
    cross_kv: Session,
    decoder_kv: Session,
    tokenizer: BinTokenizer,
    pub config: StreamingConfig,
}

impl StreamingModel {
    /// Load all 5 ONNX sessions and the binary tokenizer from the model directory.
    ///
    /// `num_threads` controls intra-op parallelism. 0 = let ORT decide (typically num cores).
    pub fn new(model_dir: &Path, num_threads: usize) -> Result<Self, MoonshineError> {
        let config = StreamingConfig::load(model_dir)?;

        let frontend = Self::load_session(model_dir, "frontend", num_threads)?;
        let encoder = Self::load_session(model_dir, "encoder", num_threads)?;
        let adapter = Self::load_session(model_dir, "adapter", num_threads)?;
        let cross_kv = Self::load_session(model_dir, "cross_kv", num_threads)?;
        let decoder_kv = Self::load_session(model_dir, "decoder_kv", num_threads)?;

        let tokenizer = BinTokenizer::new(model_dir)?;

        log::info!("Loaded streaming model from {:?}", model_dir);

        Ok(Self {
            frontend,
            encoder,
            adapter,
            cross_kv,
            decoder_kv,
            tokenizer,
            config,
        })
    }

    fn load_session(model_dir: &Path, name: &str, num_threads: usize) -> Result<Session, MoonshineError> {
        // Try .ort first, fall back to .onnx
        let ort_path = model_dir.join(format!("{}.ort", name));
        let onnx_path = model_dir.join(format!("{}.onnx", name));

        let path = if ort_path.exists() {
            ort_path
        } else if onnx_path.exists() {
            onnx_path
        } else {
            return Err(MoonshineError::ModelNotFound(format!(
                "{}.ort or {}.onnx not found in {}",
                name,
                name,
                model_dir.display()
            )));
        };

        log::info!("Loading session '{}' from {:?} (threads={})", name, path, num_threads);

        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;

        if num_threads > 0 {
            builder = builder.with_intra_threads(num_threads)?;
        }

        // Streaming uses CPU-only: small per-chunk sessions where GPU launch
        // overhead exceeds any compute benefit, and the 5 sessions would
        // compete for GPU memory with the batch model.
        let session = builder
            .with_execution_providers(crate::ort_providers::cpu_execution_providers())?
            .commit_from_file(&path)?;

        Ok(session)
    }

    /// Create a fresh streaming state for this model's config.
    pub fn create_state(&self) -> StreamingState {
        StreamingState::new(&self.config)
    }

    /// Process an audio chunk through the frontend session.
    ///
    /// Accumulates extracted features into `state.accumulated_features`.
    /// Returns the number of new feature frames produced.
    pub fn process_audio_chunk(
        &mut self,
        state: &mut StreamingState,
        audio_chunk: &[f32],
    ) -> Result<i32, MoonshineError> {
        if audio_chunk.is_empty() {
            return Ok(0);
        }

        let chunk_len = audio_chunk.len();

        // Build input tensors — small buffers, clone is fine here
        let audio_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1, chunk_len]),
            audio_chunk.to_vec(),
        )?;

        let sample_buffer_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1, 79]),
            state.sample_buffer.clone(),
        )?;

        let sample_len_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1]),
            vec![state.sample_len],
        )?;

        let conv1_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1, self.config.d_model_frontend, 4]),
            state.conv1_buffer.clone(),
        )?;

        let conv2_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1, self.config.c1, 4]),
            state.conv2_buffer.clone(),
        )?;

        let frame_count_dyn = ArrayD::from_shape_vec(
            IxDyn(&[1]),
            vec![state.frame_count],
        )?;

        let run_inputs = inputs![
            "audio_chunk" => TensorRef::from_array_view(audio_dyn.view())?,
            "sample_buffer" => TensorRef::from_array_view(sample_buffer_dyn.view())?,
            "sample_len" => TensorRef::from_array_view(sample_len_dyn.view())?,
            "conv1_buffer" => TensorRef::from_array_view(conv1_dyn.view())?,
            "conv2_buffer" => TensorRef::from_array_view(conv2_dyn.view())?,
            "frame_count" => TensorRef::from_array_view(frame_count_dyn.view())?,
        ];

        let outputs = self.frontend.run(run_inputs)?;

        // Extract features [1, T, encoder_dim]
        let features = outputs
            .get("features")
            .ok_or_else(|| MoonshineError::OutputNotFound("features".to_string()))?
            .try_extract_array::<f32>()?;

        let feat_shape = features.shape();
        let num_features = feat_shape[1] as i32;

        if num_features > 0 {
            let feat_data = features.as_slice().ok_or_else(|| {
                MoonshineError::OutputNotFound("features not contiguous".to_string())
            })?;
            let feat_size = feat_shape[1] * feat_shape[2];
            state
                .accumulated_features
                .extend_from_slice(&feat_data[..feat_size]);
            state.accumulated_feature_count += num_features;
        }

        // Update frontend state from outputs
        let sample_buffer_out = outputs
            .get("sample_buffer_out")
            .ok_or_else(|| MoonshineError::OutputNotFound("sample_buffer_out".to_string()))?
            .try_extract_array::<f32>()?;
        state.sample_buffer = sample_buffer_out.as_slice().unwrap()[..79].to_vec();

        let sample_len_out = outputs
            .get("sample_len_out")
            .ok_or_else(|| MoonshineError::OutputNotFound("sample_len_out".to_string()))?
            .try_extract_array::<i64>()?;
        state.sample_len = sample_len_out.as_slice().unwrap()[0];

        let conv1_out = outputs
            .get("conv1_buffer_out")
            .ok_or_else(|| MoonshineError::OutputNotFound("conv1_buffer_out".to_string()))?
            .try_extract_array::<f32>()?;
        let conv1_data = conv1_out.as_slice().unwrap();
        let conv1_expected = self.config.d_model_frontend * 4;
        if conv1_data.len() >= conv1_expected {
            state.conv1_buffer = conv1_data[..conv1_expected].to_vec();
        } else {
            // Frontend returned fewer elements than expected (e.g. short/tail chunk).
            // Pad with zeros to maintain the expected buffer size for the next chunk.
            state.conv1_buffer = vec![0.0; conv1_expected];
            state.conv1_buffer[..conv1_data.len()].copy_from_slice(conv1_data);
        }

        let conv2_out = outputs
            .get("conv2_buffer_out")
            .ok_or_else(|| MoonshineError::OutputNotFound("conv2_buffer_out".to_string()))?
            .try_extract_array::<f32>()?;
        let conv2_data = conv2_out.as_slice().unwrap();
        let conv2_expected = self.config.c1 * 4;
        if conv2_data.len() >= conv2_expected {
            state.conv2_buffer = conv2_data[..conv2_expected].to_vec();
        } else {
            state.conv2_buffer = vec![0.0; conv2_expected];
            state.conv2_buffer[..conv2_data.len()].copy_from_slice(conv2_data);
        }

        let frame_count_out = outputs
            .get("frame_count_out")
            .ok_or_else(|| MoonshineError::OutputNotFound("frame_count_out".to_string()))?
            .try_extract_array::<i64>()?;
        state.frame_count = frame_count_out.as_slice().unwrap()[0];

        Ok(num_features)
    }

    /// Run encoder + adapter on accumulated features.
    ///
    /// Calculates stable frames (accounting for lookahead), runs encoder with
    /// sliding window context, then adapter to produce memory frames.
    /// Returns the number of new memory frames added.
    pub fn encode(&mut self, state: &mut StreamingState, is_final: bool) -> Result<i32, MoonshineError> {
        let total_features = state.accumulated_feature_count;
        if total_features == 0 {
            return Ok(0);
        }

        let stable_count = if is_final {
            total_features
        } else {
            (total_features - self.config.total_lookahead as i32).max(0)
        };

        let new_frames = stable_count - state.encoder_frames_emitted;
        if new_frames <= 0 {
            return Ok(0);
        }

        // Encoder sliding window with left context
        let left_context_frames = (16 * self.config.depth) as i32;
        let window_start = (state.encoder_frames_emitted - left_context_frames).max(0);
        let window_size = total_features - window_start;

        log::trace!(
            "encode: total={}, stable={}, new={}, window_start={}, window_size={}",
            total_features,
            stable_count,
            new_frames,
            window_start,
            window_size
        );

        // Slice accumulated features for the window — borrow, don't clone
        let start_idx = (window_start as usize) * self.config.encoder_dim;
        let end_idx = start_idx + (window_size as usize) * self.config.encoder_dim;
        let window_features = &state.accumulated_features[start_idx..end_idx];

        let features_view = ArrayViewD::from_shape(
            IxDyn(&[1, window_size as usize, self.config.encoder_dim]),
            window_features,
        )?;

        let enc_inputs = inputs![
            "features" => TensorRef::from_array_view(features_view)?,
        ];

        let enc_outputs = self.encoder.run(enc_inputs)?;

        let encoded = enc_outputs
            .get("encoded")
            .ok_or_else(|| MoonshineError::OutputNotFound("encoded".to_string()))?
            .try_extract_array::<f32>()?;

        let enc_shape = encoded.shape();
        let total_encoded = enc_shape[1] as i32;
        let encoded_data = encoded.as_slice().ok_or_else(|| {
            MoonshineError::OutputNotFound("encoded not contiguous".to_string())
        })?;

        // Slice new frames from encoder output
        let slice_start = (state.encoder_frames_emitted - window_start) as usize;
        if slice_start + new_frames as usize > total_encoded as usize {
            return Err(MoonshineError::InvalidState(format!(
                "Encoder window misaligned: start={}, new_frames={}, total={}",
                slice_start, new_frames, total_encoded
            )));
        }

        let new_encoded: Vec<f32> = (0..new_frames as usize)
            .flat_map(|i| {
                let base = (slice_start + i) * self.config.encoder_dim;
                encoded_data[base..base + self.config.encoder_dim].iter().copied()
            })
            .collect();

        // Run adapter
        let enc_slice_view = ArrayViewD::from_shape(
            IxDyn(&[1, new_frames as usize, self.config.encoder_dim]),
            &new_encoded,
        )?;

        let pos_offset_val = [state.adapter_pos_offset];
        let pos_offset_view = ArrayViewD::from_shape(
            IxDyn(&[1]),
            &pos_offset_val,
        )?;

        let adapter_inputs = inputs![
            "encoded" => TensorRef::from_array_view(enc_slice_view)?,
            "pos_offset" => TensorRef::from_array_view(pos_offset_view)?,
        ];

        let adapter_outputs = self.adapter.run(adapter_inputs)?;

        let memory_out = adapter_outputs
            .get("memory")
            .ok_or_else(|| MoonshineError::OutputNotFound("memory".to_string()))?
            .try_extract_array::<f32>()?;

        let mem_data = memory_out.as_slice().ok_or_else(|| {
            MoonshineError::OutputNotFound("memory not contiguous".to_string())
        })?;
        let mem_size = new_frames as usize * self.config.decoder_dim;
        state.memory.extend_from_slice(&mem_data[..mem_size]);
        state.memory_len += new_frames;

        // Invalidate cross KV cache since memory changed
        state.cross_kv_valid = false;
        log::trace!("encode: cross KV invalidated, memory_len={}", state.memory_len);

        // Update tracking
        state.encoder_frames_emitted = stable_count;
        state.adapter_pos_offset += new_frames as i64;

        Ok(new_frames)
    }

    /// Compute cross-attention KV cache from memory.
    ///
    /// Input: memory `[1, mem_len, decoder_dim]`
    /// Output: k_cross, v_cross `[depth, 1, nheads, cross_len, head_dim]`
    pub fn compute_cross_kv(&mut self, state: &mut StreamingState) -> Result<(), MoonshineError> {
        if state.memory_len == 0 {
            return Err(MoonshineError::InvalidState(
                "Memory is empty, cannot compute cross K/V".to_string(),
            ));
        }

        // Borrow memory directly — no clone
        let memory_view = ArrayViewD::from_shape(
            IxDyn(&[1, state.memory_len as usize, self.config.decoder_dim]),
            &state.memory,
        )?;

        let run_inputs = inputs![
            "memory" => TensorRef::from_array_view(memory_view)?,
        ];

        let outputs = self.cross_kv.run(run_inputs)?;

        let k_cross = outputs
            .get("k_cross")
            .ok_or_else(|| MoonshineError::OutputNotFound("k_cross".to_string()))?
            .try_extract_array::<f32>()?;

        let v_cross = outputs
            .get("v_cross")
            .ok_or_else(|| MoonshineError::OutputNotFound("v_cross".to_string()))?
            .try_extract_array::<f32>()?;

        let k_shape = k_cross.shape();
        if k_shape.len() != 5 {
            return Err(MoonshineError::InvalidState(format!(
                "Expected 5D cross KV tensor, got {}D",
                k_shape.len()
            )));
        }

        let cross_len = k_shape[3] as i32;
        let kv_size =
            self.config.depth * self.config.nheads * cross_len as usize * self.config.head_dim;

        state.k_cross = k_cross.as_slice().unwrap()[..kv_size].to_vec();
        state.v_cross = v_cross.as_slice().unwrap()[..kv_size].to_vec();
        state.cross_len = cross_len;
        state.cross_kv_valid = true;

        log::trace!("compute_cross_kv: cross_len={}", cross_len);

        Ok(())
    }

    /// Core decoder execution: runs decoder_kv session and updates KV cache.
    ///
    /// Returns the raw ORT session outputs. Caller is responsible for
    /// extracting logits. This avoids copying logits when only argmax is needed.
    fn run_decoder(
        &mut self,
        state: &mut StreamingState,
        token: i64,
    ) -> Result<ort::session::SessionOutputs<'_>, MoonshineError> {
        // Compute cross KV if not valid
        if !state.cross_kv_valid {
            self.compute_cross_kv(state)?;
        }

        let cache_len = state.cache_seq_len as usize;
        let kv_self_size =
            self.config.depth * self.config.nheads * cache_len * self.config.head_dim;

        // Ensure self-attention cache is correctly sized
        if state.k_self.len() != kv_self_size {
            state.k_self.resize(kv_self_size, 0.0f32);
            state.v_self.resize(kv_self_size, 0.0f32);
        }

        // Build views from state — no cloning
        let token_val = [token];
        let token_view = ArrayViewD::from_shape(IxDyn(&[1, 1]), &token_val)?;

        let kv_shape = &[self.config.depth, 1, self.config.nheads, cache_len, self.config.head_dim];
        let k_self_view = ArrayViewD::from_shape(IxDyn(kv_shape), &state.k_self)?;
        let v_self_view = ArrayViewD::from_shape(IxDyn(kv_shape), &state.v_self)?;

        let cross_len = state.cross_len as usize;
        let cross_shape = &[self.config.depth, 1, self.config.nheads, cross_len, self.config.head_dim];
        let k_cross_view = ArrayViewD::from_shape(IxDyn(cross_shape), &state.k_cross)?;
        let v_cross_view = ArrayViewD::from_shape(IxDyn(cross_shape), &state.v_cross)?;

        // Note: decoder_kv expects cross K/V as "out_k_cross" and "out_v_cross"
        let run_inputs = inputs![
            "token" => TensorRef::from_array_view(token_view)?,
            "k_self" => TensorRef::from_array_view(k_self_view)?,
            "v_self" => TensorRef::from_array_view(v_self_view)?,
            "out_k_cross" => TensorRef::from_array_view(k_cross_view)?,
            "out_v_cross" => TensorRef::from_array_view(v_cross_view)?,
        ];

        let outputs = self.decoder_kv.run(run_inputs)?;

        // Update self-attention KV cache — reuse buffer, avoid reallocation
        let k_self_out = outputs
            .get("out_k_self")
            .ok_or_else(|| MoonshineError::OutputNotFound("out_k_self".to_string()))?
            .try_extract_array::<f32>()?;

        let v_self_out = outputs
            .get("out_v_self")
            .ok_or_else(|| MoonshineError::OutputNotFound("out_v_self".to_string()))?
            .try_extract_array::<f32>()?;

        let new_cache_len = k_self_out.shape()[3] as i32;
        let new_cache_size =
            self.config.depth * self.config.nheads * new_cache_len as usize * self.config.head_dim;

        let k_src = &k_self_out.as_slice().unwrap()[..new_cache_size];
        let v_src = &v_self_out.as_slice().unwrap()[..new_cache_size];

        state.k_self.resize(new_cache_size, 0.0);
        state.k_self.copy_from_slice(k_src);
        state.v_self.resize(new_cache_size, 0.0);
        state.v_self.copy_from_slice(v_src);
        state.cache_seq_len = new_cache_len;

        Ok(outputs)
    }

    /// Run a single decoder step, returning the full logits vector.
    ///
    /// Use `decode_step_greedy` instead when you only need the argmax token.
    pub fn decode_step(
        &mut self,
        state: &mut StreamingState,
        token: i64,
    ) -> Result<Vec<f32>, MoonshineError> {
        let vocab_size = self.config.vocab_size;
        let outputs = self.run_decoder(state, token)?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| MoonshineError::OutputNotFound("logits".to_string()))?
            .try_extract_array::<f32>()?;

        let logits_data = logits.as_slice().unwrap();
        Ok(logits_data[..vocab_size].to_vec())
    }

    /// Run a single decoder step and return the greedy (argmax) token directly.
    ///
    /// Avoids copying the full logits vector — performs argmax on the ORT output buffer.
    fn decode_step_greedy(
        &mut self,
        state: &mut StreamingState,
        token: i64,
    ) -> Result<i64, MoonshineError> {
        let vocab_size = self.config.vocab_size;
        let outputs = self.run_decoder(state, token)?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| MoonshineError::OutputNotFound("logits".to_string()))?
            .try_extract_array::<f32>()?;

        let logits_data = logits.as_slice().unwrap();
        let vocab = &logits_data[..vocab_size];

        let mut best_idx = 0u32;
        let mut best_val = vocab[0];
        for (i, &v) in vocab.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }

        Ok(best_idx as i64)
    }

    /// Reset decoder self-attention KV cache, preserving cross KV.
    pub fn decoder_reset(&self, state: &mut StreamingState) {
        state.decoder_reset();
        log::trace!("decoder_reset: self-attn KV cleared");
    }

    /// Decode token IDs to text using the binary tokenizer.
    pub fn decode_tokens(&self, tokens: &[i64]) -> Result<String, MoonshineError> {
        self.tokenizer.decode(tokens)
    }

    /// High-level offline transcription: process all audio and decode.
    ///
    /// 1. Process all audio in 1280-sample chunks through frontend (including partial tail)
    /// 2. Encode with is_final=true to flush all frames
    /// 3. Compute cross KV
    /// 4. Autoregressive decoding: BOS → greedy decode → until EOS or max tokens
    pub fn generate(
        &mut self,
        samples: &[f32],
        max_tokens_per_second: f32,
        max_tokens_override: Option<usize>,
    ) -> Result<Vec<i64>, MoonshineError> {
        let mut state = self.create_state();

        // Process all audio including partial tail chunk
        for chunk in samples.chunks(CHUNK_SIZE) {
            self.process_audio_chunk(&mut state, chunk)?;
        }

        // Encode with is_final=true to emit all frames including lookahead
        self.encode(&mut state, true)?;

        if state.memory_len == 0 {
            return Ok(Vec::new());
        }

        // Compute cross KV
        self.compute_cross_kv(&mut state)?;

        // Calculate max tokens
        let max_tokens = match max_tokens_override {
            Some(m) => m.min(self.config.max_seq_len),
            None => {
                let duration_sec = samples.len() as f32 / SAMPLE_RATE as f32;
                ((duration_sec * max_tokens_per_second).ceil() as usize)
                    .min(self.config.max_seq_len)
            }
        };

        log::debug!(
            "generate: {:.2}s audio, memory_len={}, max_tokens={}",
            samples.len() as f32 / SAMPLE_RATE as f32,
            state.memory_len,
            max_tokens
        );

        // Autoregressive decoding — use greedy path to avoid logits copy
        let mut tokens: Vec<i64> = Vec::new();
        let mut current_token = self.config.bos_id;

        for _step in 0..max_tokens {
            let next_token = self.decode_step_greedy(&mut state, current_token)?;

            if next_token == self.config.eos_id {
                log::trace!("EOS reached at step {}", _step);
                break;
            }

            tokens.push(next_token);
            current_token = next_token;
        }

        log::trace!("Generated {} tokens", tokens.len());
        Ok(tokens)
    }
}
