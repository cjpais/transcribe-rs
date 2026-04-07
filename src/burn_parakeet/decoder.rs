use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// LSTM Cell (single layer, single direction)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct LstmLayer<B: Backend> {
    pub ih: Linear<B>,
    pub hh: Linear<B>,
    pub hidden_size: usize,
}

impl<B: Backend> LstmLayer<B> {
    pub fn new(input_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            // ONNX LSTM stores [4*hidden, input] for input-hidden
            // and [4*hidden, hidden] for hidden-hidden
            ih: LinearConfig::new(input_size, 4 * hidden_size)
                .with_bias(false)
                .init(device),
            hh: LinearConfig::new(hidden_size, 4 * hidden_size)
                .with_bias(false)
                .init(device),
            hidden_size,
        }
    }

    /// Forward pass for one time step.
    /// - `x`: [B, input_size]
    /// - `h`: [B, hidden_size]
    /// - `c`: [B, hidden_size]
    /// - `bias`: [4*hidden_size] combined bias
    ///
    /// Returns (new_h, new_c).
    pub fn forward(
        &self,
        x: Tensor<B, 2>,
        h: Tensor<B, 2>,
        c: Tensor<B, 2>,
        bias: Tensor<B, 1>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hs = self.hidden_size;

        let gates = x.matmul(self.ih.weight.val()) + h.matmul(self.hh.weight.val())
            + bias.unsqueeze_dim::<2>(0);

        let [batch, _] = gates.dims();
        let i = burn::tensor::activation::sigmoid(gates.clone().slice([0..batch, 0..hs]));
        let f = burn::tensor::activation::sigmoid(gates.clone().slice([0..batch, hs..2 * hs]));
        let g = gates.clone().slice([0..batch, 2 * hs..3 * hs]).tanh();
        let o = burn::tensor::activation::sigmoid(gates.slice([0..batch, 3 * hs..4 * hs]));

        let new_c = f * c + i * g;
        let new_h = o * new_c.clone().tanh();

        (new_h, new_c)
    }
}

// ---------------------------------------------------------------------------
// TDT Predictor (embedding + 2-layer LSTM)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct TdtPredictor<B: Backend> {
    pub embed: Linear<B>,
    pub lstm0: LstmLayer<B>,
    pub lstm1: LstmLayer<B>,
    /// ONNX LSTM bias format: [8*hidden] = [4*hidden input_bias, 4*hidden hidden_bias]
    /// Combined into single [4*hidden] by addition.
    pub bias0: Param<Tensor<B, 1>>,
    pub bias1: Param<Tensor<B, 1>>,
    pub hidden_size: usize,
}

impl<B: Backend> TdtPredictor<B> {
    pub fn new(vocab_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        Self {
            // Embedding is implemented as a lookup; we use Linear without bias
            // and index into the weight matrix manually
            embed: LinearConfig::new(vocab_size, hidden_size)
                .with_bias(false)
                .init(device),
            lstm0: LstmLayer::new(hidden_size, hidden_size, device),
            lstm1: LstmLayer::new(hidden_size, hidden_size, device),
            bias0: Param::from_tensor(Tensor::zeros([4 * hidden_size], device)),
            bias1: Param::from_tensor(Tensor::zeros([4 * hidden_size], device)),
            hidden_size,
        }
    }

    /// Run predictor for a single token.
    /// - `token`: token ID (or None for blank)
    /// - `state`: (h0, c0, h1, c1) each [1, hidden_size]
    ///
    /// Returns (output [1, hidden_size], new_state).
    pub fn forward(
        &self,
        token: Option<usize>,
        state: LstmState<B>,
    ) -> (Tensor<B, 2>, LstmState<B>) {
        let device = state.h0.device();

        // Embedding lookup
        let emb = match token {
            Some(tok) => {
                // Index into embedding weight: [vocab, hidden] → [1, hidden]
                let weight = self.embed.weight.val();
                weight.slice([tok..tok + 1, 0..self.hidden_size])
            }
            None => Tensor::zeros([1, self.hidden_size], &device),
        };

        // LSTM layer 0
        let (h0, c0) = self.lstm0.forward(
            emb,
            state.h0,
            state.c0,
            self.bias0.val(),
        );

        // LSTM layer 1
        let (h1, c1) = self.lstm1.forward(
            h0.clone(),
            state.h1,
            state.c1,
            self.bias1.val(),
        );

        let out = h1.clone();
        let new_state = LstmState { h0, c0, h1, c1 };
        (out, new_state)
    }
}

#[derive(Debug, Clone)]
pub struct LstmState<B: Backend> {
    pub h0: Tensor<B, 2>,
    pub c0: Tensor<B, 2>,
    pub h1: Tensor<B, 2>,
    pub c1: Tensor<B, 2>,
}

impl<B: Backend> LstmState<B> {
    pub fn zeros(hidden_size: usize, device: &B::Device) -> Self {
        Self {
            h0: Tensor::zeros([1, hidden_size], device),
            c0: Tensor::zeros([1, hidden_size], device),
            h1: Tensor::zeros([1, hidden_size], device),
            c1: Tensor::zeros([1, hidden_size], device),
        }
    }
}

// ---------------------------------------------------------------------------
// Joint Network: encoder_proj + predictor_proj → ReLU → output
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct JointNetwork<B: Backend> {
    pub enc: Linear<B>,
    pub pred: Linear<B>,
    pub out: Linear<B>,
}

impl<B: Backend> JointNetwork<B> {
    pub fn new(
        encoder_dim: usize,
        pred_dim: usize,
        joint_dim: usize,
        output_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            enc: LinearConfig::new(encoder_dim, joint_dim).init(device),
            pred: LinearConfig::new(pred_dim, joint_dim).init(device),
            out: LinearConfig::new(joint_dim, output_dim).init(device),
        }
    }

    /// Compute joint logits.
    /// - `enc_out`: [1, encoder_dim] (single time step)
    /// - `pred_out`: [1, pred_dim]
    ///
    /// Returns [output_dim] logits (squeezed).
    pub fn forward(&self, enc_out: Tensor<B, 2>, pred_out: Tensor<B, 2>) -> Tensor<B, 1> {
        // Manual matmul to avoid cubecl autotune issues with M=1 matrices
        let enc_proj = enc_out.matmul(self.enc.weight.val());
        let enc_proj = match &self.enc.bias {
            Some(b) => enc_proj + b.val().unsqueeze_dim::<2>(0),
            None => enc_proj,
        };
        let pred_proj = pred_out.matmul(self.pred.weight.val());
        let pred_proj = match &self.pred.bias {
            Some(b) => pred_proj + b.val().unsqueeze_dim::<2>(0),
            None => pred_proj,
        };
        let x = enc_proj + pred_proj;
        let x = burn::tensor::activation::relu(x);
        let out = x.matmul(self.out.weight.val());
        let out = match &self.out.bias {
            Some(b) => out + b.val().unsqueeze_dim::<2>(0),
            None => out,
        };
        out.flatten(0, 1) // [output_dim]
    }
}

// ---------------------------------------------------------------------------
// TDT Greedy Decode
// ---------------------------------------------------------------------------

/// Result of TDT decoding.
pub struct DecodeResult {
    pub tokens: Vec<i32>,
    pub timestamps: Vec<usize>,
}

/// Greedy TDT decoding over encoder output.
///
/// - `encoder_out`: [1, T, encoder_dim]
/// - `predictor`: the LSTM predictor network
/// - `joint`: the joint network
/// - `blank_idx`: blank token index
/// - `vocab_size`: number of vocabulary tokens (not including blank)
/// - `durations`: list of possible frame durations [0, 1, 2, ...]
pub fn tdt_greedy_decode<B: Backend>(
    encoder_out: Tensor<B, 3>,
    predictor: &TdtPredictor<B>,
    joint: &JointNetwork<B>,
    blank_idx: usize,
    vocab_size: usize,
    durations: &[usize],
) -> DecodeResult {
    let [_b, seq_len, enc_dim] = encoder_out.dims();
    let device = encoder_out.device();

    // Pre-squeeze encoder output to [T, enc_dim] to avoid clone+slice+reshape per step
    let enc_2d: Tensor<B, 2> = encoder_out.reshape([seq_len, enc_dim]);

    let mut state = LstmState::zeros(predictor.hidden_size, &device);
    let mut tokens: Vec<i32> = Vec::new();
    let mut timestamps: Vec<usize> = Vec::new();

    let mut last_token: Option<usize> = None;
    let mut t: usize = 0;
    let max_tokens_per_step = 10;
    let num_logits = vocab_size + 1; // token logits including blank
    let n_dur = durations.len();

    log::debug!("TDT decode: seq_len={}, enc_dim={}", seq_len, enc_dim);

    while t < seq_len {
        let mut emitted = 0;

        loop {
            // Get encoder output at time t: [1, enc_dim] — no clone needed
            let enc_step = enc_2d.clone().slice([t..t + 1, 0..enc_dim]);

            // Run predictor
            let (pred_out, new_state) = predictor.forward(last_token, state.clone());

            // Joint network → logits
            let logits = joint.forward(enc_step, pred_out);

            // Argmax → scalar (works with both wgpu i32 and ndarray i64)
            let pred_token: usize = logits.clone().slice([0..num_logits])
                .argmax(0)
                .flatten::<1>(0, 0)
                .into_scalar()
                .elem::<i64>() as usize;

            let duration_idx: usize = if n_dur > 1 {
                logits.slice([num_logits..num_logits + n_dur])
                    .argmax(0)
                    .flatten::<1>(0, 0)
                    .into_scalar()
                    .elem::<i64>() as usize
            } else {
                0
            };

            let step_duration = durations.get(duration_idx).copied().unwrap_or(1);

            if pred_token != blank_idx {
                tokens.push(pred_token as i32);
                timestamps.push(t);
                last_token = Some(pred_token);
                state = new_state;
                emitted += 1;
            }

            // Advance time by duration, or if blank, or if max tokens per step
            if pred_token == blank_idx || emitted >= max_tokens_per_step {
                t += step_duration.max(1);
                break;
            }

            // For non-blank with duration > 0, advance and continue
            if step_duration > 0 {
                t += step_duration;
                break;
            }
            // duration=0 means emit token without advancing time
        }
    }

    DecodeResult { tokens, timestamps }
}
