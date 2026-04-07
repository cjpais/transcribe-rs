use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;

use super::attention::RelPosAttention;

// ---------------------------------------------------------------------------
// Feed-Forward module: Linear → SiLU → Linear, applied with 0.5 residual weight
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(d_model: usize, expansion: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_model * expansion)
                .with_bias(false)
                .init(device),
            linear2: LinearConfig::new(d_model * expansion, d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = silu(x);
        self.linear2.forward(x)
    }
}

/// SiLU activation: x * sigmoid(x)
fn silu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.clone() * burn::tensor::activation::sigmoid(x)
}

// ---------------------------------------------------------------------------
// Convolution module: pointwise1 → GLU → depthwise → SiLU → pointwise2
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConvModule<B: Backend> {
    pub pw1: Conv1d<B>,
    pub dw: Conv1d<B>,
    pub pw2: Conv1d<B>,
    // BatchNorm parameters (applied manually since Burn's BatchNorm needs training state)
    pub bn_weight: Param<Tensor<B, 1>>,
    pub bn_bias: Param<Tensor<B, 1>>,
    pub bn_running_mean: Param<Tensor<B, 1>>,
    pub bn_running_var: Param<Tensor<B, 1>>,
    pub d_model: usize,
    /// When true, BN has been folded into the depthwise conv — skip BN in forward.
    pub bn_folded: bool,
}

impl<B: Backend> ConvModule<B> {
    pub fn new(d_model: usize, kernel_size: usize, device: &B::Device) -> Self {
        let padding = (kernel_size - 1) / 2;
        Self {
            pw1: Conv1dConfig::new(d_model, d_model * 2, 1).with_bias(false).init(device),
            dw: Conv1dConfig::new(d_model, d_model, kernel_size)
                .with_groups(d_model)
                .with_padding(burn::nn::PaddingConfig1d::Explicit(padding, padding))
                .with_bias(false)
                .init(device),
            pw2: Conv1dConfig::new(d_model, d_model, 1).with_bias(false).init(device),
            bn_weight: Param::from_tensor(Tensor::ones([d_model], device)),
            bn_bias: Param::from_tensor(Tensor::zeros([d_model], device)),
            bn_running_mean: Param::from_tensor(Tensor::zeros([d_model], device)),
            bn_running_var: Param::from_tensor(Tensor::ones([d_model], device)),
            d_model,
            bn_folded: false,
        }
    }

    /// Forward pass. Input: [B, T, d_model], output: [B, T, d_model].
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Conv1d expects [B, C, T], so transpose
        let x = x.swap_dims(1, 2); // [B, d_model, T]

        // Pointwise conv1: [B, d_model, T] → [B, 2*d_model, T]
        let x = self.pw1.forward(x);

        // GLU: split in half along channel dim, gate with sigmoid
        let [b, c2, t] = x.dims();
        let c = c2 / 2;
        let a = x.clone().slice([0..b, 0..c, 0..t]);
        let gate = x.slice([0..b, c..c2, 0..t]);
        let x = a * burn::tensor::activation::sigmoid(gate);

        // Depthwise conv: [B, d_model, T] → [B, d_model, T]
        let x = self.dw.forward(x);

        // BatchNorm — skipped when folded into depthwise conv weights at load time
        let x = if self.bn_folded {
            x
        } else {
            let eps = 1e-5;
            let mean = self.bn_running_mean.val().reshape([1, self.d_model, 1]);
            let var = self.bn_running_var.val().reshape([1, self.d_model, 1]);
            let w = self.bn_weight.val().reshape([1, self.d_model, 1]);
            let bi = self.bn_bias.val().reshape([1, self.d_model, 1]);
            (x - mean) / (var + eps).sqrt() * w + bi
        };

        let x = silu(x);

        // Pointwise conv2: [B, d_model, T] → [B, d_model, T]
        let x = self.pw2.forward(x);

        // Transpose back to [B, T, d_model]
        x.swap_dims(1, 2)
    }
}

// ---------------------------------------------------------------------------
// Conformer Layer
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConformerLayer<B: Backend> {
    pub norm_ff1: LayerNorm<B>,
    pub ff1: FeedForward<B>,
    pub norm_attn: LayerNorm<B>,
    pub attn: RelPosAttention<B>,
    pub norm_conv: LayerNorm<B>,
    pub conv: ConvModule<B>,
    pub norm_ff2: LayerNorm<B>,
    pub ff2: FeedForward<B>,
    pub norm_out: LayerNorm<B>,
}

impl<B: Backend> ConformerLayer<B> {
    pub fn new(d_model: usize, n_heads: usize, ff_expansion: usize, conv_kernel: usize, device: &B::Device) -> Self {
        let norm = |dev: &B::Device| LayerNormConfig::new(d_model).init(dev);
        Self {
            norm_ff1: norm(device),
            ff1: FeedForward::new(d_model, ff_expansion, device),
            norm_attn: norm(device),
            attn: RelPosAttention::new(d_model, n_heads, device),
            norm_conv: norm(device),
            conv: ConvModule::new(d_model, conv_kernel, device),
            norm_ff2: norm(device),
            ff2: FeedForward::new(d_model, ff_expansion, device),
            norm_out: norm(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        pos_emb: Tensor<B, 3>,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        // FF1 with 0.5 residual weight
        let out = self.ff1.forward(self.norm_ff1.forward(x.clone())) * 0.5;
        let x = x + out;

        // Self-attention
        let out = self.attn.forward(self.norm_attn.forward(x.clone()), pos_emb, mask);
        let x = x + out;

        // Convolution module
        let out = self.conv.forward(self.norm_conv.forward(x.clone()));
        let x = x + out;

        // FF2 with 0.5 residual weight
        let out = self.ff2.forward(self.norm_ff2.forward(x.clone())) * 0.5;
        let x = x + out;

        // Final layer norm
        self.norm_out.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Pre-encode: DW striding subsampling + linear projection
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct PreEncode<B: Backend> {
    pub conv0: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub conv5: Conv2d<B>,
    pub conv6: Conv2d<B>,
    pub out: Linear<B>,
}

impl<B: Backend> PreEncode<B> {
    pub fn new(feat_in: usize, d_model: usize, channels: usize, device: &B::Device) -> Self {
        let dw_conv = |c, dev: &B::Device| {
            Conv2dConfig::new([c, c], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .with_groups(c)
                .init(dev)
        };
        let pw_conv = |c, dev: &B::Device| {
            Conv2dConfig::new([c, c], [1, 1]).with_bias(false).init(dev)
        };

        // After 3 stride-2 convolutions on 128 mel bins:
        // 128 → 64 → 32 → 16, with `channels` output channels
        // Flatten: channels * 16 = projection input
        let proj_in = channels * (feat_in / 8); // 256 * 16 = 4096

        Self {
            conv0: Conv2dConfig::new([1, channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            conv2: dw_conv(channels, device),
            conv3: pw_conv(channels, device),
            conv5: dw_conv(channels, device),
            conv6: pw_conv(channels, device),
            out: LinearConfig::new(proj_in, d_model).init(device),
        }
    }

    /// Forward pass.
    /// Input: [B, T, feat_in] mel features (time-major).
    /// Output: [B, T/8, d_model].
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, feat] = x.dims();

        // Reshape to Conv2d format: [B, 1, T, feat_in]
        let x = x.unsqueeze_dim::<4>(1);

        // Subsampling convolutions with activations
        // MLX conv list: [Conv0, ReLU, DW, PW, ReLU, DW, PW, ReLU]
        // ReLU only after initial conv and after each pointwise conv, NOT after depthwise
        let x = relu(self.conv0.forward(x));
        let x = self.conv2.forward(x);
        let x = relu(self.conv3.forward(x));
        let x = self.conv5.forward(x);
        let x = relu(self.conv6.forward(x));

        // x is now [B, channels, T/8, feat_in/8]
        let [b, c, t, f] = x.dims();

        // Flatten channels and frequency: [B, T/8, channels * freq]
        let x = x.swap_dims(1, 2); // [B, T/8, channels, freq]
        let x = x.reshape([b, t, c * f]);

        // Linear projection to d_model
        self.out.forward(x)
    }
}

fn relu<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::relu(x)
}

// ---------------------------------------------------------------------------
// Full FastConformer Encoder
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct FastConformerEncoder<B: Backend> {
    pub pre_encode: PreEncode<B>,
    pub layers: Vec<ConformerLayer<B>>,
    pub d_model: usize,
}

impl<B: Backend> FastConformerEncoder<B> {
    pub fn new(
        feat_in: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        ff_expansion: usize,
        conv_kernel: usize,
        sub_channels: usize,
        device: &B::Device,
    ) -> Self {
        let layers = (0..n_layers)
            .map(|_| ConformerLayer::new(d_model, n_heads, ff_expansion, conv_kernel, device))
            .collect();

        Self {
            pre_encode: PreEncode::new(feat_in, d_model, sub_channels, device),
            layers,
            d_model,
        }
    }

    /// Forward pass.
    /// Input: [B, T, feat_in] mel features.
    /// Output: [B, T/8, d_model] encoded representations.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Subsampling
        let x = self.pre_encode.forward(x);
        let [_b, seq_len, _d] = x.dims();
        log::debug!("After pre_encode: seq_len={}, d_model={}", seq_len, _d);

        // Generate relative positional encoding
        let pos_emb = self.make_pos_encoding(seq_len, &x.device());
        log::debug!("pos_emb shape: {:?}", pos_emb.dims());

        // Run through conformer layers
        let mut x = x;
        for (i, layer) in self.layers.iter().enumerate() {
            log::debug!("Conformer layer {}/{}", i, self.layers.len());
            x = layer.forward(x, pos_emb.clone(), None);
        }

        x
    }

    /// Generate sinusoidal relative positional encoding.
    /// Returns [1, 2*seq_len-1, d_model].
    pub fn make_pos_encoding(&self, seq_len: usize, device: &B::Device) -> Tensor<B, 3> {
        let d = self.d_model;
        let pos_len = 2 * seq_len - 1;

        // Positions from -(seq_len-1) to +(seq_len-1)
        let positions: Vec<f32> = (0..pos_len)
            .map(|i| (seq_len as f32 - 1.0) - i as f32)
            .collect();

        // Dimension indices for sin/cos
        let div_term: Vec<f32> = (0..d)
            .step_by(2)
            .map(|i| (-(i as f32) * (10000.0_f32).ln() / d as f32).exp())
            .collect();

        let half_d = d / 2;
        let mut pe = vec![0.0f32; pos_len * d];

        for (t, &pos) in positions.iter().enumerate() {
            for (j, &div) in div_term.iter().enumerate() {
                let val = pos * div;
                pe[t * d + 2 * j] = val.sin();
                pe[t * d + 2 * j + 1] = val.cos();
            }
        }

        let flat: Tensor<B, 1> = Tensor::from_floats(&pe[..], device);
        flat.reshape([1, pos_len, d])
    }
}
