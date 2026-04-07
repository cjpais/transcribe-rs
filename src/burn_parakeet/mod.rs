pub mod attention;
pub mod decoder;
pub mod encoder;
pub mod preprocess;
pub mod weights;

use std::path::Path;

use burn::backend::{NdArray, Wgpu};
use burn::prelude::*;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
    TranscriptionSegment,
};

type B = Wgpu;
type CpuB = NdArray;

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Parakeet (Burn/wgpu)",
    engine_id: "burn_parakeet",
    sample_rate: 16000,
    languages: &["en"],
    supports_timestamps: true,
    supports_translation: false,
    supports_streaming: false,
};

// Model dimensions (Parakeet TDT 0.6B v3)
const D_MODEL: usize = 1024;
const N_HEADS: usize = 8;
const N_LAYERS: usize = 24;
const FF_EXPANSION: usize = 4;
const CONV_KERNEL: usize = 9;
const SUB_CHANNELS: usize = 256;
const FEAT_IN: usize = 128;
const PRED_HIDDEN: usize = 640;
const JOINT_DIM: usize = 640;
const NUM_CLASSES: usize = 8192; // vocab tokens (not including blank)
const EMBED_SIZE: usize = 8193;  // NUM_CLASSES + 1 (blank at index NUM_CLASSES)
const OUTPUT_DIM: usize = 8198;  // EMBED_SIZE + 5 duration outputs

const SUBSAMPLING_FACTOR: usize = 8;
const WINDOW_SIZE: f32 = 0.01; // 10ms
const TDT_DURATIONS: &[usize] = &[0, 1, 2, 3, 4];

static DECODE_SPACE_RE: Lazy<Result<Regex, regex::Error>> =
    Lazy::new(|| Regex::new(r"\A\s|\s\B|(\s)\b"));

pub struct BurnParakeetModel {
    pub encoder: encoder::FastConformerEncoder<B>,
    pub cpu_predictor: decoder::TdtPredictor<CpuB>,
    pub cpu_joint: decoder::JointNetwork<CpuB>,
    pub vocab: Vec<String>,
    pub blank_idx: i32,
    pub device: burn::backend::wgpu::WgpuDevice,
}

impl BurnParakeetModel {
    /// Load model from a directory containing:
    /// - `model.safetensors` (MLX format from mlx-community/parakeet-tdt-0.6b-v3)
    /// - `vocab.txt`
    pub fn load(model_dir: &Path) -> Result<Self, TranscribeError> {
        let device = burn::backend::wgpu::WgpuDevice::default();

        log::info!("Loading Burn Parakeet model from {:?}", model_dir);

        // Load vocab from config.json's joint.vocabulary (matching MLX reference)
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(TranscribeError::Io)?;
        let config: serde_json::Value = serde_json::from_str(&config_text)
            .map_err(|e| TranscribeError::Config(format!("Failed to parse config.json: {e}")))?;
        let vocab: Vec<String> = config["joint"]["vocabulary"]
            .as_array()
            .ok_or_else(|| TranscribeError::Config("Missing joint.vocabulary in config.json".into()))?
            .iter()
            .map(|v| v.as_str().unwrap_or("").replace('\u{2581}', " "))
            .collect();
        let blank_idx = vocab.len() as i32;
        log::info!("Loaded {} vocab tokens, blank_idx={}", vocab.len(), blank_idx);

        // Load safetensors via mmap — build modules directly from weights (no random init)
        let st_path = model_dir.join("model.safetensors");
        let cpu_dev = burn::backend::ndarray::NdArrayDevice::Cpu;
        let (encoder, cpu_predictor, cpu_joint) = if st_path.exists() {
            log::info!("Loading weights from {:?}", st_path);
            let mmap = weights::mmap_file(&st_path)?;
            let st = safetensors::SafeTensors::deserialize(&mmap).map_err(|e| {
                TranscribeError::Config(format!("Failed to parse safetensors: {e}"))
            })?;
            let enc = make_encoder(&st, &device);
            // Decoder on CPU — tiny matmuls ([1,640]) are faster without GPU dispatch overhead
            let pred = make_predictor_cpu(&st, &cpu_dev);
            let joint = make_joint_cpu(&st, &cpu_dev);
            log::info!("Weights loaded successfully");
            (enc, pred, joint)
        } else {
            log::warn!("No model.safetensors found — using random weights");
            (
                encoder::FastConformerEncoder::new(
                    FEAT_IN, D_MODEL, N_HEADS, N_LAYERS, FF_EXPANSION, CONV_KERNEL, SUB_CHANNELS, &device,
                ),
                decoder::TdtPredictor::new(EMBED_SIZE, PRED_HIDDEN, &cpu_dev),
                decoder::JointNetwork::new(D_MODEL, PRED_HIDDEN, JOINT_DIM, OUTPUT_DIM, &cpu_dev),
            )
        };

        Ok(Self {
            encoder,
            cpu_predictor,
            cpu_joint,
            vocab,
            blank_idx,
            device,
        })
    }

    fn transcribe_internal(
        &self,
        samples: &[f32],
    ) -> Result<TranscriptionResult, TranscribeError> {
        // 1. Preprocess: compute mel spectrogram
        let mel = preprocess::nemo_preprocess(samples);
        let [num_mels, num_frames] = [mel.nrows(), mel.ncols()];

        if num_frames == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        // 2. Convert to Burn tensor [1, T, num_mels] (batch, time, features)
        let mel_data: Vec<f32> = mel.t().iter().copied().collect(); // transpose to time-major
        let mel_flat: Tensor<B, 1> = Tensor::from_floats(&mel_data[..], &self.device);
        let mel_tensor: Tensor<B, 3> = mel_flat.reshape([1, num_frames, num_mels]);

        // 3. Encode (GPU)
        let t_enc = std::time::Instant::now();
        let encoder_out = self.encoder.forward(mel_tensor);

        // 4. Transfer encoder output to CPU for decoding
        let [_b, seq_len, enc_dim] = encoder_out.dims();
        let enc_data = encoder_out.to_data();
        let cpu_dev = burn::backend::ndarray::NdArrayDevice::Cpu;
        let cpu_encoder_out: Tensor<CpuB, 3> =
            Tensor::<CpuB, 3>::from_data(enc_data, &cpu_dev);
        let enc_ms = t_enc.elapsed().as_millis();
        log::info!("Encoder (GPU→CPU): {}ms", enc_ms);

        // 5. Decode on CPU — avoids GPU dispatch overhead for tiny per-step matmuls
        let t_dec = std::time::Instant::now();
        let decode_result = decoder::tdt_greedy_decode(
            cpu_encoder_out,
            &self.cpu_predictor,
            &self.cpu_joint,
            self.blank_idx as usize,
            NUM_CLASSES,
            TDT_DURATIONS,
        );

        let dec_ms = t_dec.elapsed().as_millis();
        log::info!("Decoder (CPU): {}ms ({} tokens)", dec_ms, decode_result.tokens.len());

        // 5. Convert tokens to text
        let token_strings: Vec<String> = decode_result
            .tokens
            .iter()
            .filter_map(|&id| self.vocab.get(id as usize).cloned())
            .collect();

        let raw_text = token_strings.join("");
        let text = if let Ok(re) = DECODE_SPACE_RE.as_ref() {
            re.replace_all(&raw_text, "$1").to_string()
        } else {
            raw_text
        };

        // 6. Build timestamp segments (word-level)
        let segments = self.build_word_segments(&decode_result, &token_strings);

        Ok(TranscriptionResult {
            text: text.trim().to_string(),
            segments: Some(segments),
        })
    }

    fn build_word_segments(
        &self,
        result: &decoder::DecodeResult,
        token_strings: &[String],
    ) -> Vec<TranscriptionSegment> {
        let time_ratio = SUBSAMPLING_FACTOR as f32 * WINDOW_SIZE;
        let mut segments: Vec<TranscriptionSegment> = Vec::new();
        let mut current_word = String::new();
        let mut word_start: Option<f32> = None;

        for (i, text) in token_strings.iter().enumerate() {
            let t = result.timestamps.get(i).copied().unwrap_or(0);
            let start_sec = t as f32 * time_ratio;

            // New word boundary: token starts with space/underscore
            let is_word_start = text.starts_with(' ') || text.starts_with('\u{2581}');

            if is_word_start && !current_word.is_empty() {
                // Emit previous word
                let end_sec = start_sec;
                segments.push(TranscriptionSegment {
                    start: word_start.unwrap_or(0.0),
                    end: end_sec,
                    text: current_word.trim().to_string(),
                });
                current_word.clear();
                word_start = Some(start_sec);
            }

            if word_start.is_none() {
                word_start = Some(start_sec);
            }
            current_word.push_str(text);
        }

        // Emit last word
        if !current_word.is_empty() {
            let last_t = result.timestamps.last().copied().unwrap_or(0);
            segments.push(TranscriptionSegment {
                start: word_start.unwrap_or(0.0),
                end: (last_t as f32 + 1.0) * SUBSAMPLING_FACTOR as f32 * WINDOW_SIZE,
                text: current_word.trim().to_string(),
            });
        }

        segments.retain(|s| !s.text.is_empty());
        segments
    }
}

// ---------------------------------------------------------------------------
// Weight loading from MLX safetensors (mlx-community/parakeet-tdt-0.6b-v3)
// ---------------------------------------------------------------------------

use burn::module::{Ignored, Param};
use burn::nn::conv::{Conv1d, Conv2d};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, PaddingConfig1d, PaddingConfig2d};
use safetensors::SafeTensors;

// ---------------------------------------------------------------------------
// make_* constructors: build Burn modules directly from mmap'd safetensors
// No random init — each GPU tensor is created exactly once with real weights.
// ---------------------------------------------------------------------------

fn get_f32<'a>(st: &'a SafeTensors<'a>, name: &str) -> std::borrow::Cow<'a, [f32]> {
    weights::get_f32(st, name).unwrap_or_else(|e| panic!("{e}"))
}

fn t1(st: &SafeTensors, name: &str, len: usize, dev: &<B as Backend>::Device) -> Tensor<B, 1> {
    let data = get_f32(st, name);
    assert_eq!(data.len(), len, "Shape mismatch for {name}: expected {len}, got {}", data.len());
    Tensor::<B, 1>::from_floats(&data[..], dev)
}

fn t2(st: &SafeTensors, name: &str, shape: [usize; 2], dev: &<B as Backend>::Device) -> Tensor<B, 2> {
    let data = get_f32(st, name);
    let expected = shape[0] * shape[1];
    assert_eq!(data.len(), expected, "Shape mismatch for {name}: expected {expected}, got {}", data.len());
    Tensor::<B, 1>::from_floats(&data[..], dev).reshape(shape)
}

/// CPU transpose MLX [out, in] → Burn [in, out], single GPU upload
fn make_linear(st: &SafeTensors, prefix: &str, in_f: usize, out_f: usize, dev: &<B as Backend>::Device) -> Linear<B> {
    let data = get_f32(st, &format!("{prefix}.weight"));
    let transposed = weights::cpu_transpose(&data, out_f, in_f);
    let weight = Tensor::<B, 1>::from_floats(&transposed[..], dev).reshape([in_f, out_f]);
    let bias_name = format!("{prefix}.bias");
    let bias = weights::get_f32(st, &bias_name)
        .ok()
        .map(|b| Param::from_tensor(Tensor::<B, 1>::from_floats(&b[..], dev)));
    Linear { weight: Param::from_tensor(weight), bias }
}

fn make_linear_no_bias(st: &SafeTensors, prefix: &str, in_f: usize, out_f: usize, dev: &<B as Backend>::Device) -> Linear<B> {
    let data = get_f32(st, &format!("{prefix}.weight"));
    let transposed = weights::cpu_transpose(&data, out_f, in_f);
    let weight = Tensor::<B, 1>::from_floats(&transposed[..], dev).reshape([in_f, out_f]);
    Linear { weight: Param::from_tensor(weight), bias: None }
}

/// LayerNorm — epsilon is private, so init from config then overwrite gamma/beta
fn make_norm(st: &SafeTensors, prefix: &str, size: usize, dev: &<B as Backend>::Device) -> LayerNorm<B> {
    let mut norm = LayerNormConfig::new(size).init(dev);
    norm.gamma = Param::from_tensor(t1(st, &format!("{prefix}.weight"), size, dev));
    norm.beta = Some(Param::from_tensor(t1(st, &format!("{prefix}.bias"), size, dev)));
    norm
}

/// Conv1d: CPU reorder MLX [out, kernel, in] → Burn [out, in/groups, kernel]
fn make_conv1d(
    st: &SafeTensors, prefix: &str,
    out_c: usize, in_c_per_group: usize, kernel: usize,
    groups: usize, padding: PaddingConfig1d,
    dev: &<B as Backend>::Device,
) -> Conv1d<B> {
    let data = get_f32(st, &format!("{prefix}.weight"));
    let reordered = weights::cpu_conv1d_reorder(&data, out_c, kernel, in_c_per_group);
    let weight = Tensor::<B, 1>::from_floats(&reordered[..], dev).reshape([out_c, in_c_per_group, kernel]);
    let bias = weights::get_f32(st, &format!("{prefix}.bias"))
        .ok()
        .map(|b| Param::from_tensor(Tensor::<B, 1>::from_floats(&b[..], dev)));
    Conv1d {
        weight: Param::from_tensor(weight),
        bias,
        stride: 1,
        kernel_size: kernel,
        dilation: 1,
        groups,
        padding: Ignored(padding),
    }
}

/// Conv2d: CPU reorder MLX [out, kH, kW, in] → Burn [out, in/groups, kH, kW]
fn make_conv2d(
    st: &SafeTensors, prefix: &str,
    out_c: usize, in_c_per_group: usize, kh: usize, kw: usize,
    stride: [usize; 2], groups: usize, padding: PaddingConfig2d,
    dev: &<B as Backend>::Device,
) -> Conv2d<B> {
    let data = get_f32(st, &format!("{prefix}.weight"));
    let reordered = weights::cpu_conv2d_reorder(&data, out_c, kh, kw, in_c_per_group);
    let weight = Tensor::<B, 1>::from_floats(&reordered[..], dev).reshape([out_c, in_c_per_group, kh, kw]);
    let bias = weights::get_f32(st, &format!("{prefix}.bias"))
        .ok()
        .map(|b| Param::from_tensor(Tensor::<B, 1>::from_floats(&b[..], dev)));
    Conv2d {
        weight: Param::from_tensor(weight),
        bias,
        stride,
        kernel_size: [kh, kw],
        dilation: [1, 1],
        groups,
        padding: Ignored(padding),
    }
}

fn make_ff(st: &SafeTensors, prefix: &str, d_model: usize, expansion: usize, dev: &<B as Backend>::Device) -> encoder::FeedForward<B> {
    encoder::FeedForward {
        linear1: make_linear_no_bias(st, &format!("{prefix}.linear1"), d_model, d_model * expansion, dev),
        linear2: make_linear_no_bias(st, &format!("{prefix}.linear2"), d_model * expansion, d_model, dev),
    }
}

fn make_attention(st: &SafeTensors, prefix: &str, d_model: usize, n_heads: usize, dev: &<B as Backend>::Device) -> attention::RelPosAttention<B> {
    let head_dim = d_model / n_heads;
    attention::RelPosAttention {
        q: make_linear_no_bias(st, &format!("{prefix}.linear_q"), d_model, d_model, dev),
        k: make_linear_no_bias(st, &format!("{prefix}.linear_k"), d_model, d_model, dev),
        v: make_linear_no_bias(st, &format!("{prefix}.linear_v"), d_model, d_model, dev),
        pos_proj: make_linear_no_bias(st, &format!("{prefix}.linear_pos"), d_model, d_model, dev),
        out: make_linear_no_bias(st, &format!("{prefix}.linear_out"), d_model, d_model, dev),
        pos_bias_u: Param::from_tensor(t2(st, &format!("{prefix}.pos_bias_u"), [n_heads, head_dim], dev)),
        pos_bias_v: Param::from_tensor(t2(st, &format!("{prefix}.pos_bias_v"), [n_heads, head_dim], dev)),
        n_heads,
        head_dim,
        scale: (head_dim as f64).powf(-0.5),
    }
}

fn make_conv_module(st: &SafeTensors, layer_prefix: &str, d_model: usize, kernel_size: usize, dev: &<B as Backend>::Device) -> encoder::ConvModule<B> {
    let padding = (kernel_size - 1) / 2;
    let p = format!("{layer_prefix}.conv");

    let pw1 = make_conv1d(st, &format!("{p}.pointwise_conv1"), d_model * 2, d_model, 1, 1, PaddingConfig1d::Valid, dev);
    // pw1 has no bias in config but safetensors might not have it — make_conv1d handles it
    let pw2 = make_conv1d(st, &format!("{p}.pointwise_conv2"), d_model, d_model, 1, 1, PaddingConfig1d::Valid, dev);
    let mut dw = make_conv1d(st, &format!("{p}.depthwise_conv"), d_model, 1, kernel_size, d_model, PaddingConfig1d::Explicit(padding, padding), dev);

    // Load BN params and fold into depthwise conv on GPU
    let gamma = t1(st, &format!("{p}.batch_norm.weight"), d_model, dev);
    let beta = t1(st, &format!("{p}.batch_norm.bias"), d_model, dev);
    let mean = t1(st, &format!("{p}.batch_norm.running_mean"), d_model, dev);
    let var = t1(st, &format!("{p}.batch_norm.running_var"), d_model, dev);
    let eps: f32 = 1e-5;
    let scale = gamma.clone() / (var + eps).sqrt();
    let scale_3d = scale.clone().reshape([d_model, 1, 1]);
    dw.weight = Param::from_tensor(dw.weight.val() * scale_3d);
    dw.bias = Some(Param::from_tensor(beta - mean * scale));

    encoder::ConvModule {
        pw1,
        dw,
        pw2,
        // BN params unused (folded) — use dummy 1-element tensors to avoid wasting GPU memory
        bn_weight: Param::from_tensor(Tensor::ones([1], dev)),
        bn_bias: Param::from_tensor(Tensor::zeros([1], dev)),
        bn_running_mean: Param::from_tensor(Tensor::zeros([1], dev)),
        bn_running_var: Param::from_tensor(Tensor::ones([1], dev)),
        d_model,
        bn_folded: true,
    }
}

fn make_conformer_layer(st: &SafeTensors, idx: usize, dev: &<B as Backend>::Device) -> encoder::ConformerLayer<B> {
    let p = format!("encoder.layers.{idx}");
    encoder::ConformerLayer {
        norm_ff1: make_norm(st, &format!("{p}.norm_feed_forward1"), D_MODEL, dev),
        ff1: make_ff(st, &format!("{p}.feed_forward1"), D_MODEL, FF_EXPANSION, dev),
        norm_attn: make_norm(st, &format!("{p}.norm_self_att"), D_MODEL, dev),
        attn: make_attention(st, &format!("{p}.self_attn"), D_MODEL, N_HEADS, dev),
        norm_conv: make_norm(st, &format!("{p}.norm_conv"), D_MODEL, dev),
        conv: make_conv_module(st, &p, D_MODEL, CONV_KERNEL, dev),
        norm_ff2: make_norm(st, &format!("{p}.norm_feed_forward2"), D_MODEL, dev),
        ff2: make_ff(st, &format!("{p}.feed_forward2"), D_MODEL, FF_EXPANSION, dev),
        norm_out: make_norm(st, &format!("{p}.norm_out"), D_MODEL, dev),
    }
}

fn make_pre_encode(st: &SafeTensors, dev: &<B as Backend>::Device) -> encoder::PreEncode<B> {
    let pad33 = PaddingConfig2d::Explicit(1, 1, 1, 1);
    encoder::PreEncode {
        conv0: make_conv2d(st, "encoder.pre_encode.conv.0", 256, 1, 3, 3, [2, 2], 1, pad33.clone(), dev),
        conv2: make_conv2d(st, "encoder.pre_encode.conv.2", 256, 1, 3, 3, [2, 2], 256, pad33.clone(), dev),
        conv3: make_conv2d(st, "encoder.pre_encode.conv.3", 256, 256, 1, 1, [1, 1], 1, PaddingConfig2d::Valid, dev),
        conv5: make_conv2d(st, "encoder.pre_encode.conv.5", 256, 1, 3, 3, [2, 2], 256, pad33, dev),
        conv6: make_conv2d(st, "encoder.pre_encode.conv.6", 256, 256, 1, 1, [1, 1], 1, PaddingConfig2d::Valid, dev),
        out: make_linear(st, "encoder.pre_encode.out", 4096, D_MODEL, dev),
    }
}

fn make_encoder(st: &SafeTensors, dev: &<B as Backend>::Device) -> encoder::FastConformerEncoder<B> {
    let pre_encode = make_pre_encode(st, dev);
    let layers = (0..N_LAYERS).map(|i| make_conformer_layer(st, i, dev)).collect();
    encoder::FastConformerEncoder {
        pre_encode,
        layers,
        d_model: D_MODEL,
    }
}

fn make_lstm(st: &SafeTensors, prefix: &str, hidden_size: usize, dev: &<B as Backend>::Device) -> (decoder::LstmLayer<B>, Param<Tensor<B, 1>>) {
    let wx_data = get_f32(st, &format!("{prefix}.Wx"));
    let wx_t = weights::cpu_transpose(&wx_data, 4 * hidden_size, hidden_size);
    let ih = Linear {
        weight: Param::from_tensor(Tensor::<B, 1>::from_floats(&wx_t[..], dev).reshape([hidden_size, 4 * hidden_size])),
        bias: None,
    };

    let wh_data = get_f32(st, &format!("{prefix}.Wh"));
    let wh_t = weights::cpu_transpose(&wh_data, 4 * hidden_size, hidden_size);
    let hh = Linear {
        weight: Param::from_tensor(Tensor::<B, 1>::from_floats(&wh_t[..], dev).reshape([hidden_size, 4 * hidden_size])),
        bias: None,
    };

    let bias = Param::from_tensor(t1(st, &format!("{prefix}.bias"), 4 * hidden_size, dev));
    (decoder::LstmLayer { ih, hh, hidden_size }, bias)
}

fn make_predictor(st: &SafeTensors, dev: &<B as Backend>::Device) -> decoder::TdtPredictor<B> {
    let emb = t2(st, "decoder.prediction.embed.weight", [EMBED_SIZE, PRED_HIDDEN], dev);
    let embed = Linear { weight: Param::from_tensor(emb), bias: None };

    let (lstm0, bias0) = make_lstm(st, "decoder.prediction.dec_rnn.lstm.0", PRED_HIDDEN, dev);
    let (lstm1, bias1) = make_lstm(st, "decoder.prediction.dec_rnn.lstm.1", PRED_HIDDEN, dev);

    decoder::TdtPredictor { embed, lstm0, lstm1, bias0, bias1, hidden_size: PRED_HIDDEN }
}

fn make_joint(st: &SafeTensors, dev: &<B as Backend>::Device) -> decoder::JointNetwork<B> {
    decoder::JointNetwork {
        enc: make_linear(st, "joint.enc", D_MODEL, JOINT_DIM, dev),
        pred: make_linear(st, "joint.pred", PRED_HIDDEN, JOINT_DIM, dev),
        out: make_linear(st, "joint.joint_net.2", JOINT_DIM, OUTPUT_DIM, dev),
    }
}

// ---------------------------------------------------------------------------
// CPU (NdArray) constructors for decoder — avoids GPU dispatch overhead
// ---------------------------------------------------------------------------

fn cpu_t1(st: &SafeTensors, name: &str, len: usize, dev: &<CpuB as Backend>::Device) -> Tensor<CpuB, 1> {
    let data = get_f32(st, name);
    assert_eq!(data.len(), len, "Shape mismatch for {name}");
    Tensor::<CpuB, 1>::from_floats(&data[..], dev)
}

fn cpu_t2(st: &SafeTensors, name: &str, shape: [usize; 2], dev: &<CpuB as Backend>::Device) -> Tensor<CpuB, 2> {
    let data = get_f32(st, name);
    assert_eq!(data.len(), shape[0] * shape[1], "Shape mismatch for {name}");
    Tensor::<CpuB, 1>::from_floats(&data[..], dev).reshape(shape)
}

fn cpu_make_linear(st: &SafeTensors, prefix: &str, in_f: usize, out_f: usize, dev: &<CpuB as Backend>::Device) -> Linear<CpuB> {
    let data = get_f32(st, &format!("{prefix}.weight"));
    let transposed = weights::cpu_transpose(&data, out_f, in_f);
    let weight = Tensor::<CpuB, 1>::from_floats(&transposed[..], dev).reshape([in_f, out_f]);
    let bias = weights::get_f32(st, &format!("{prefix}.bias"))
        .ok()
        .map(|b| Param::from_tensor(Tensor::<CpuB, 1>::from_floats(&b[..], dev)));
    Linear { weight: Param::from_tensor(weight), bias }
}

fn make_lstm_cpu(st: &SafeTensors, prefix: &str, hidden_size: usize, dev: &<CpuB as Backend>::Device) -> (decoder::LstmLayer<CpuB>, Param<Tensor<CpuB, 1>>) {
    let wx_data = get_f32(st, &format!("{prefix}.Wx"));
    let wx_t = weights::cpu_transpose(&wx_data, 4 * hidden_size, hidden_size);
    let ih = Linear {
        weight: Param::from_tensor(Tensor::<CpuB, 1>::from_floats(&wx_t[..], dev).reshape([hidden_size, 4 * hidden_size])),
        bias: None,
    };
    let wh_data = get_f32(st, &format!("{prefix}.Wh"));
    let wh_t = weights::cpu_transpose(&wh_data, 4 * hidden_size, hidden_size);
    let hh = Linear {
        weight: Param::from_tensor(Tensor::<CpuB, 1>::from_floats(&wh_t[..], dev).reshape([hidden_size, 4 * hidden_size])),
        bias: None,
    };
    let bias = Param::from_tensor(cpu_t1(st, &format!("{prefix}.bias"), 4 * hidden_size, dev));
    (decoder::LstmLayer { ih, hh, hidden_size }, bias)
}

fn make_predictor_cpu(st: &SafeTensors, dev: &<CpuB as Backend>::Device) -> decoder::TdtPredictor<CpuB> {
    let emb = cpu_t2(st, "decoder.prediction.embed.weight", [EMBED_SIZE, PRED_HIDDEN], dev);
    let embed = Linear { weight: Param::from_tensor(emb), bias: None };
    let (lstm0, bias0) = make_lstm_cpu(st, "decoder.prediction.dec_rnn.lstm.0", PRED_HIDDEN, dev);
    let (lstm1, bias1) = make_lstm_cpu(st, "decoder.prediction.dec_rnn.lstm.1", PRED_HIDDEN, dev);
    decoder::TdtPredictor { embed, lstm0, lstm1, bias0, bias1, hidden_size: PRED_HIDDEN }
}

fn make_joint_cpu(st: &SafeTensors, dev: &<CpuB as Backend>::Device) -> decoder::JointNetwork<CpuB> {
    decoder::JointNetwork {
        enc: cpu_make_linear(st, "joint.enc", D_MODEL, JOINT_DIM, dev),
        pred: cpu_make_linear(st, "joint.pred", PRED_HIDDEN, JOINT_DIM, dev),
        out: cpu_make_linear(st, "joint.joint_net.2", JOINT_DIM, OUTPUT_DIM, dev),
    }
}

// SAFETY: BurnParakeetModel is only used from a single thread.
// The wgpu backend's internal types are not Send due to GPU handle restrictions,
// but we ensure single-threaded access via &mut self in SpeechModel.
unsafe impl Send for BurnParakeetModel {}

impl SpeechModel for BurnParakeetModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn default_leading_silence_ms(&self) -> u32 {
        250
    }

    fn default_trailing_silence_ms(&self) -> u32 {
        0
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.transcribe_internal(samples)
    }
}
