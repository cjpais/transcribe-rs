use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use super::model::MoonshineError;

/// A minimal tokenizer implementation for Moonshine that only supports decoding.
/// This replaces the `tokenizers` crate to avoid dependency conflicts on Windows.
pub struct MoonshineTokenizer {
    /// Maps token ID to token string
    vocab: HashMap<u32, String>,
    /// Set of special token IDs to skip during decoding
    special_token_ids: Vec<u32>,
}

impl MoonshineTokenizer {
    pub fn new(model_dir: &Path) -> Result<Self, MoonshineError> {
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(MoonshineError::TokenizerNotFound(
                tokenizer_path.display().to_string(),
            ));
        }

        log::info!("Loading tokenizer from {:?}...", tokenizer_path);

        let file = File::open(&tokenizer_path)
            .map_err(|e| MoonshineError::Tokenization(format!("Failed to open tokenizer: {}", e)))?;
        let reader = BufReader::new(file);
        let json: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|e| MoonshineError::Tokenization(format!("Failed to parse tokenizer JSON: {}", e)))?;

        // Build id → token vocabulary (inverse of the stored token → id mapping)
        let mut vocab = HashMap::new();
        if let Some(model) = json.get("model") {
            if let Some(v) = model.get("vocab").and_then(|v| v.as_object()) {
                for (token, id) in v {
                    if let Some(id) = id.as_u64() {
                        vocab.insert(id as u32, token.clone());
                    }
                }
            }
        }

        if vocab.is_empty() {
            return Err(MoonshineError::Tokenization(
                "No vocabulary found in tokenizer.json".to_string(),
            ));
        }

        log::info!("Loaded {} tokens from vocabulary", vocab.len());

        // Collect special token IDs from added_tokens
        let mut special_token_ids = Vec::new();
        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let is_special = token.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                if is_special {
                    if let Some(id) = token.get("id").and_then(|v| v.as_u64()) {
                        special_token_ids.push(id as u32);
                    }
                }
            }
        }

        log::debug!("Found {} special tokens", special_token_ids.len());

        Ok(Self {
            vocab,
            special_token_ids,
        })
    }

    pub fn decode(&self, token_ids: &[i64]) -> Result<String, MoonshineError> {
        // First pass: collect token strings, skipping special tokens
        let mut tokens: Vec<String> = Vec::with_capacity(token_ids.len());

        for &id in token_ids {
            let id = id as u32;

            // Skip special tokens
            if self.special_token_ids.contains(&id) {
                continue;
            }

            if let Some(token) = self.vocab.get(&id) {
                tokens.push(token.clone());
            }
            // Unknown tokens are silently skipped (same behavior as skip_special_tokens)
        }

        // Second pass: decode tokens to text
        // This implements the SentencePiece decoder pipeline:
        // 1. Replace ▁ with space
        // 2. Handle byte fallback tokens <0xNN>
        // 3. Strip leading space

        let mut bytes: Vec<u8> = Vec::new();

        for token in &tokens {
            if let Some(byte_val) = Self::parse_byte_token(token) {
                // Byte fallback token like <0x41>
                bytes.push(byte_val);
            } else {
                // Regular token - replace ▁ with space and convert to bytes
                let decoded = token.replace('▁', " ");
                bytes.extend(decoded.as_bytes());
            }
        }

        // Convert bytes to string, handling invalid UTF-8 gracefully
        let text = String::from_utf8_lossy(&bytes);

        // Strip leading space (from the SentencePiece decoder)
        let text = text.strip_prefix(' ').unwrap_or(&text);

        Ok(text.to_string())
    }

    /// Parse a byte fallback token like "<0x41>" and return the byte value
    fn parse_byte_token(token: &str) -> Option<u8> {
        if token.starts_with("<0x") && token.ends_with('>') && token.len() == 6 {
            let hex = &token[3..5];
            u8::from_str_radix(hex, 16).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_byte_token() {
        assert_eq!(MoonshineTokenizer::parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(MoonshineTokenizer::parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(MoonshineTokenizer::parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(MoonshineTokenizer::parse_byte_token("<0xff>"), Some(0xFF));
        assert_eq!(MoonshineTokenizer::parse_byte_token("hello"), None);
        assert_eq!(MoonshineTokenizer::parse_byte_token("<0x>"), None);
        assert_eq!(MoonshineTokenizer::parse_byte_token("<0x123>"), None);
    }
}
