//! Decode-only BPE tokenizer for Qwen3-ASR.
//!
//! Parses `tokenizer.json` (HuggingFace format) and decodes token IDs to text
//! using the GPT-2 byte-level BPE mapping. No dependency on the `tokenizers`
//! crate (follows Moonshine's pattern for Windows build compatibility).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::TranscribeError;

/// Decode-only tokenizer that maps token IDs to text.
pub struct Qwen3Tokenizer {
    /// Maps token ID to raw byte sequence (after GPT-2 unicode-to-byte decode).
    id_to_bytes: HashMap<u32, Vec<u8>>,
    /// Set of special token IDs to skip during decode.
    special_token_ids: HashSet<u32>,
}

impl Qwen3Tokenizer {
    pub fn new(model_dir: &Path) -> Result<Self, TranscribeError> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        // Reads the entire tokenizer.json into memory. Typical Qwen/GPT-2 tokenizer files
        // are ~2 MB; this is acceptable for a one-time model-load operation.
        let file = fs::File::open(&tokenizer_path)?;
        let reader = std::io::BufReader::new(file);
        let json: serde_json::Value = serde_json::from_reader(reader)?;

        let byte_decoder = build_gpt2_byte_decoder();

        // Build id → bytes vocabulary from model.vocab
        let mut id_to_bytes = HashMap::new();
        if let Some(model) = json.get("model") {
            if let Some(vocab) = model.get("vocab").and_then(|v| v.as_object()) {
                for (token_str, id_val) in vocab {
                    if let Some(id) = id_val.as_u64() {
                        let bytes = decode_token_string(token_str, &byte_decoder);
                        id_to_bytes.insert(id as u32, bytes);
                    }
                }
            }
        }

        if id_to_bytes.is_empty() {
            return Err(TranscribeError::Config(
                "No vocabulary found in tokenizer.json".into(),
            ));
        }

        log::info!(
            "Loaded {} tokens from Qwen3-ASR vocabulary",
            id_to_bytes.len()
        );

        // Collect special token IDs from added_tokens
        let mut special_token_ids = HashSet::new();
        if let Some(added_tokens) = json.get("added_tokens").and_then(|v| v.as_array()) {
            for token in added_tokens {
                let is_special = token
                    .get("special")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if is_special {
                    if let Some(id) = token.get("id").and_then(|v| v.as_u64()) {
                        special_token_ids.insert(id as u32);
                    }
                }
            }
        }

        Ok(Self {
            id_to_bytes,
            special_token_ids,
        })
    }

    /// Decode a sequence of token IDs to a string, skipping special tokens.
    pub fn decode(&self, token_ids: &[i64]) -> String {
        let mut bytes = Vec::new();
        for &id in token_ids {
            if id < 0 {
                log::warn!("Qwen3 tokenizer: skipping negative token ID {}", id);
                continue;
            }
            let id_u32 = id as u32;
            if self.special_token_ids.contains(&id_u32) {
                continue;
            }
            if let Some(token_bytes) = self.id_to_bytes.get(&id_u32) {
                bytes.extend_from_slice(token_bytes);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Build the GPT-2 unicode-to-byte decoder table.
///
/// GPT-2 BPE uses a mapping from byte values (0-255) to printable Unicode characters
/// to avoid control characters in the vocabulary. This builds the inverse mapping.
fn build_gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut byte_encoder: HashMap<u8, char> = HashMap::new();

    // Printable ASCII ranges that map to themselves
    // '!' (33) through '~' (126)
    for b in b'!'..=b'~' {
        byte_encoder.insert(b, b as char);
    }
    // Latin-1 supplement range: 161-172, 174-255
    for b in 0xa1u8..=0xacu8 {
        byte_encoder.insert(b, b as char);
    }
    for b in 0xaeu8..=0xffu8 {
        byte_encoder.insert(b, b as char);
    }

    // Remaining bytes get mapped to Unicode starting at 256
    let mut n = 256u32;
    for b in 0u8..=255u8 {
        if let std::collections::hash_map::Entry::Vacant(e) = byte_encoder.entry(b) {
            e.insert(char::from_u32(n).expect("BPE byte range always valid Unicode"));
            n += 1;
        }
    }

    // Invert: char → byte
    byte_encoder.into_iter().map(|(b, c)| (c, b)).collect()
}

/// Decode a GPT-2 BPE token string to raw bytes.
fn decode_token_string(token_str: &str, byte_decoder: &HashMap<char, u8>) -> Vec<u8> {
    token_str
        .chars()
        .filter_map(|c| byte_decoder.get(&c).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_byte_decoder_coverage() {
        let decoder = build_gpt2_byte_decoder();
        // Should have exactly 256 entries (one per byte value)
        assert_eq!(decoder.len(), 256);
    }

    #[test]
    fn test_gpt2_ascii_passthrough() {
        let decoder = build_gpt2_byte_decoder();
        // Printable ASCII (0x21-0x7E) maps to itself
        assert_eq!(decoder[&'A'], b'A');
        assert_eq!(decoder[&'z'], b'z');
        assert_eq!(decoder[&'0'], b'0');
        // Space (0x20) is NOT in the printable range, so it maps to a Unicode
        // char > 0xFF. Verify it round-trips correctly through encode → decode.
        let space_exists = decoder.values().any(|&b| b == 0x20);
        assert!(
            space_exists,
            "Space byte should be represented in the decoder"
        );
    }

    #[test]
    fn test_gpt2_space_mapping() {
        // GPT-2 encodes space (0x20) as 'Ġ' (U+0120).
        let decoder = build_gpt2_byte_decoder();
        assert_eq!(decoder[&'\u{0120}'], 0x20);
    }

    #[test]
    fn test_decode_token_string_simple() {
        let decoder = build_gpt2_byte_decoder();
        let bytes = decode_token_string("Hello", &decoder);
        assert_eq!(bytes, b"Hello");
    }
}
