use serde::Deserialize;
use std::fs;
use std::path::Path;

use crate::TranscribeError;

/// Top-level model configuration loaded from `config.json`.
// Fields are deserialized from config.json; not all are used at runtime.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Qwen3AsrConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub mel: Qwen3MelParams,
    pub special_tokens: SpecialTokens,
    /// Storage dtype of `embed_tokens.bin`.
    #[serde(default)]
    pub embed_tokens_dtype: EmbedDtype,
}

/// Storage dtype for `embed_tokens.bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbedDtype {
    /// 32-bit IEEE 754 float (4 bytes per element).
    Float32,
    /// 16-bit IEEE 754 half-precision float (2 bytes per element).
    Float16,
}

impl Default for EmbedDtype {
    fn default() -> Self {
        Self::Float32
    }
}

impl EmbedDtype {
    /// Bytes per element for this dtype.
    pub fn bytes_per_element(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Float16 => 2,
        }
    }
}

// Fields are deserialized from config.json; not all are used at runtime.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EncoderConfig {
    pub num_mel_bins: usize,
    pub output_dim: usize,
}

// Fields are deserialized from config.json; not all are used at runtime.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct DecoderConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

/// Mel spectrogram parameters from config.json, validated against the compiled-in constants
/// in `mel.rs` at model load time.
// Named `Qwen3MelParams` (not `MelConfig`) to avoid collision with `crate::features::MelConfig`.
// Fields are deserialized from config.json; not all are used at runtime.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Qwen3MelParams {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
}

// Fields are deserialized from config.json; not all are used at runtime.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct SpecialTokens {
    pub eos_token_ids: Vec<i64>,
    pub pad_token_id: i64,
    pub im_start_token_id: i64,
    pub im_end_token_id: i64,
    pub audio_start_token_id: i64,
    pub audio_end_token_id: i64,
    pub audio_pad_token_id: i64,
    /// Token that separates the language prefix from the transcription text.
    /// If absent from generated tokens, the model failed to produce a valid transcription.
    #[serde(default = "default_asr_text_token_id")]
    pub asr_text_token_id: i64,
}

fn default_asr_text_token_id() -> i64 {
    151704
}

impl Qwen3AsrConfig {
    pub fn load(model_dir: &Path) -> Result<Self, TranscribeError> {
        let config_path = model_dir.join("config.json");
        let data = fs::read_to_string(&config_path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_config() {
        let json = r#"{
            "model_type": "qwen3_asr",
            "encoder": {
                "num_layers": 18, "hidden_size": 896, "num_heads": 14,
                "ffn_dim": 3584, "conv_channels": 480, "output_dim": 1024,
                "downsample_factor": 8, "num_mel_bins": 128
            },
            "decoder": {
                "num_layers": 28, "hidden_size": 1024, "num_attention_heads": 16,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072,
                "vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-6,
                "tie_word_embeddings": true,
                "rope_scaling": { "mrope_section": [24, 20, 20], "interleaved": true }
            },
            "mel": {
                "sample_rate": 16000, "n_fft": 400, "hop_length": 160,
                "n_mels": 128, "fmin": 0, "fmax": 8000
            },
            "special_tokens": {
                "eos_token_ids": [151643, 151645],
                "pad_token_id": 151643,
                "im_start_token_id": 151644,
                "im_end_token_id": 151645,
                "audio_start_token_id": 151669,
                "audio_end_token_id": 151670,
                "audio_pad_token_id": 151676,
                "asr_text_token_id": 151704
            },
            "embed_tokens_shape": [151936, 1024],
            "embed_tokens_dtype": "float32"
        }"#;

        let config: Qwen3AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.decoder.num_layers, 28);
        assert_eq!(config.decoder.vocab_size, 151936);
        assert_eq!(config.mel.n_mels, 128);
        assert_eq!(config.special_tokens.eos_token_ids, vec![151643, 151645]);
        assert_eq!(config.embed_tokens_dtype, EmbedDtype::Float32);
    }

    #[test]
    fn test_embed_dtype_float16() {
        let json = r#"{
            "encoder": { "num_layers": 18, "hidden_size": 896, "num_heads": 14, "ffn_dim": 3584, "conv_channels": 480, "output_dim": 1024, "downsample_factor": 8, "num_mel_bins": 128 },
            "decoder": { "num_layers": 28, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "rope_scaling": { "mrope_section": [24, 20, 20], "interleaved": true } },
            "mel": { "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 128, "fmin": 0, "fmax": 8000 },
            "special_tokens": { "eos_token_ids": [151643, 151645], "pad_token_id": 151643, "im_start_token_id": 151644, "im_end_token_id": 151645, "audio_start_token_id": 151669, "audio_end_token_id": 151670, "audio_pad_token_id": 151676 },
            "embed_tokens_dtype": "float16"
        }"#;
        let config: Qwen3AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.embed_tokens_dtype, EmbedDtype::Float16);
        assert_eq!(config.embed_tokens_dtype.bytes_per_element(), 2);
    }

    #[test]
    fn test_embed_dtype_default_when_missing() {
        let json = r#"{
            "encoder": { "num_layers": 18, "hidden_size": 896, "num_heads": 14, "ffn_dim": 3584, "conv_channels": 480, "output_dim": 1024, "downsample_factor": 8, "num_mel_bins": 128 },
            "decoder": { "num_layers": 28, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "rope_scaling": { "mrope_section": [24, 20, 20], "interleaved": true } },
            "mel": { "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 128, "fmin": 0, "fmax": 8000 },
            "special_tokens": { "eos_token_ids": [151643, 151645], "pad_token_id": 151643, "im_start_token_id": 151644, "im_end_token_id": 151645, "audio_start_token_id": 151669, "audio_end_token_id": 151670, "audio_pad_token_id": 151676 }
        }"#;
        let config: Qwen3AsrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.embed_tokens_dtype, EmbedDtype::Float32);
    }

    #[test]
    fn test_embed_dtype_unknown_rejected() {
        let json = r#"{
            "encoder": { "num_layers": 18, "hidden_size": 896, "num_heads": 14, "ffn_dim": 3584, "conv_channels": 480, "output_dim": 1024, "downsample_factor": 8, "num_mel_bins": 128 },
            "decoder": { "num_layers": 28, "hidden_size": 1024, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 3072, "vocab_size": 151936, "rope_theta": 1000000, "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "rope_scaling": { "mrope_section": [24, 20, 20], "interleaved": true } },
            "mel": { "sample_rate": 16000, "n_fft": 400, "hop_length": 160, "n_mels": 128, "fmin": 0, "fmax": 8000 },
            "special_tokens": { "eos_token_ids": [151643, 151645], "pad_token_id": 151643, "im_start_token_id": 151644, "im_end_token_id": 151645, "audio_start_token_id": 151669, "audio_end_token_id": 151670, "audio_pad_token_id": 151676 },
            "embed_tokens_dtype": "bfloat16"
        }"#;
        let result = serde_json::from_str::<Qwen3AsrConfig>(json);
        assert!(result.is_err(), "bfloat16 should be rejected");
    }
}
