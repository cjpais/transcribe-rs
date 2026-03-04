use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Top-level model configuration loaded from `config.json`.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct Qwen3AsrConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub mel: MelConfig,
    pub special_tokens: SpecialTokens,
    pub embed_tokens_shape: [usize; 2],
    pub embed_tokens_dtype: String,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EncoderConfig {
    pub num_mel_bins: usize,
    pub output_dim: usize,
}

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

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
}

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
}

impl Qwen3AsrConfig {
    pub fn load(model_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
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
        assert_eq!(config.embed_tokens_shape, [151936, 1024]);
    }
}
