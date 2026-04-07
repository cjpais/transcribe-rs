use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

/// Relative position multi-head attention.
///
/// Implements the attention mechanism from the FastConformer encoder with
/// learnable relative position biases (pos_bias_u, pos_bias_v).
#[derive(Module, Debug)]
pub struct RelPosAttention<B: Backend> {
    pub q: Linear<B>,
    pub k: Linear<B>,
    pub v: Linear<B>,
    pub pos_proj: Linear<B>,
    pub out: Linear<B>,
    pub pos_bias_u: Param<Tensor<B, 2>>,
    pub pos_bias_v: Param<Tensor<B, 2>>,
    pub n_heads: usize,
    pub head_dim: usize,
    pub scale: f64,
}

impl<B: Backend> RelPosAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let head_dim = d_model / n_heads;
        let linear = |dev: &B::Device| {
            LinearConfig::new(d_model, d_model)
                .with_bias(false)
                .init(dev)
        };
        Self {
            q: linear(device),
            k: linear(device),
            v: linear(device),
            pos_proj: linear(device),
            out: linear(device),
            pos_bias_u: Param::from_tensor(Tensor::zeros([n_heads, head_dim], device)),
            pos_bias_v: Param::from_tensor(Tensor::zeros([n_heads, head_dim], device)),
            n_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// Forward pass.
    ///
    /// - `x`: [B, T, d_model]
    /// - `pos_emb`: [1, 2T-1, d_model] relative positional encoding
    /// - `mask`: optional [B, 1, T, T] attention mask (0 = attend, -inf = ignore)
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _d] = x.dims();

        // Project Q, K, V: [B, T, d_model] → [B, T, n_heads, head_dim] → [B, n_heads, T, head_dim]
        let q = self.reshape_head(self.q.forward(x.clone()), batch, seq_len);
        let k = self.reshape_head(self.k.forward(x.clone()), batch, seq_len);
        let v = self.reshape_head(self.v.forward(x), batch, seq_len);

        // Project positional encoding
        let p = self.reshape_head(self.pos_proj.forward(pos_emb), 1, seq_len * 2 - 1);

        // Add position biases to query
        // pos_bias_u, pos_bias_v: [n_heads, head_dim] → [1, n_heads, 1, head_dim]
        let bias_u = self
            .pos_bias_u
            .val()
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);
        let bias_v = self
            .pos_bias_v
            .val()
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2);

        let q_u = q.clone() + bias_u; // [B, n_heads, T, head_dim]
        let q_v = q + bias_v;

        // Content-content attention: Q_u @ K^T
        log::debug!("attn q_u: {:?}, k: {:?}", q_u.dims(), k.dims());
        let matrix_ac = q_u.matmul(k.transpose()); // [B, n_heads, T, T]

        // Relative position attention: Q_v @ P^T, then rel_shift
        log::debug!("attn q_v: {:?}, p: {:?}", q_v.dims(), p.dims());
        let matrix_bd = q_v.matmul(p.transpose()); // [B, n_heads, T, 2T-1]
        log::debug!("matrix_bd pre-shift: {:?}", matrix_bd.dims());
        let matrix_bd = Self::rel_shift(matrix_bd, seq_len);
        log::debug!("matrix_bd post-shift: {:?}", matrix_bd.dims());

        // Combine and scale
        let mut scores = (matrix_ac + matrix_bd) * self.scale;

        // Apply mask
        if let Some(m) = mask {
            scores = scores + m;
        }

        // Softmax + weighted sum
        let attn = burn::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v); // [B, n_heads, T, head_dim]

        // Merge heads: [B, n_heads, T, head_dim] → [B, T, d_model]
        let out = out
            .swap_dims(1, 2) // [B, T, n_heads, head_dim]
            .reshape([batch, seq_len, self.n_heads * self.head_dim]);

        self.out.forward(out)
    }

    fn reshape_head(&self, x: Tensor<B, 3>, batch: usize, seq: usize) -> Tensor<B, 4> {
        x.reshape([batch, seq, self.n_heads, self.head_dim])
            .swap_dims(1, 2) // [B, n_heads, T, head_dim]
    }

    /// Relative shift operation.
    ///
    /// Converts relative position scores [B, H, T, 2T-1] into
    /// absolute position scores [B, H, T, T].
    ///
    /// Matches MLX: pad → reshape → slice_row0 → reshape_back → slice_cols.
    fn rel_shift(x: Tensor<B, 4>, seq_len: usize) -> Tensor<B, 4> {
        let [b, h, t, pos_len] = x.dims();
        // Pad left by 1: [B, H, T, pos_len] → [B, H, T, pos_len+1]
        let x = Tensor::cat(
            vec![Tensor::zeros([b, h, t, 1], &x.device()), x],
            3,
        );
        // Reshape to [B, H, pos_len+1, T]
        let x = x.reshape([b, h, pos_len + 1, t]);
        // Slice off first row: [B, H, pos_len, T]
        let x = x.slice([0..b, 0..h, 1..(pos_len + 1), 0..t]);
        // Reshape back to [B, H, T, pos_len]
        let x = x.reshape([b, h, t, pos_len]);
        // Take only the first seq_len columns (absolute positions)
        x.slice([0..b, 0..h, 0..t, 0..seq_len])
    }
}
