use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::TranscribeError;

/// Vocabulary for Cohere Transcribe, loaded from a `vocab.txt` file.
///
/// The file format is `token id` per line (same as Canary).
pub struct Vocab {
    token_to_id: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    eos_id: i64,
}

impl Vocab {
    pub fn load(path: &Path) -> Result<Self, TranscribeError> {
        let content = fs::read_to_string(path)
            .map_err(|e| TranscribeError::Config(format!("Failed to read vocab file: {e}")))?;

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let last_space = line.rfind(' ').ok_or_else(|| {
                TranscribeError::Config(format!(
                    "Invalid vocab line {}: missing space separator",
                    line_num + 1
                ))
            })?;

            let token = &line[..last_space];
            let id_str = &line[last_space + 1..];

            let id: i64 = id_str.parse().map_err(|e| {
                TranscribeError::Config(format!("Invalid token ID on line {}: {e}", line_num + 1))
            })?;

            token_to_id.insert(token.to_string(), id);
            id_to_token.insert(id, token.to_string());
        }

        let eos_id = *token_to_id.get("<|endoftext|>").ok_or_else(|| {
            TranscribeError::Config("Vocabulary missing required <|endoftext|> token".to_string())
        })?;

        let size = token_to_id.len();
        log::info!(
            "Loaded vocabulary with {} tokens from {}",
            size,
            path.display()
        );

        Ok(Self {
            token_to_id,
            id_to_token,
            eos_id,
        })
    }

    pub fn eos_token_id(&self) -> i64 {
        self.eos_id
    }

    /// Build the prompt token sequence for Cohere Transcribe.
    ///
    /// Format: `<|startofcontext|><|startoftranscript|><|emo:undefined|><|lang|><|lang|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>`
    pub fn build_prompt(&self, language: &str) -> Result<Vec<i64>, TranscribeError> {
        let tokens = [
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            &format!("<|{language}|>"),
            &format!("<|{language}|>"),
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ];

        let mut ids = Vec::with_capacity(tokens.len());
        for token in &tokens {
            let id = self.token_to_id.get(*token).copied().ok_or_else(|| {
                TranscribeError::Config(format!("Prompt token not found in vocabulary: {token}"))
            })?;
            ids.push(id);
        }

        log::debug!("Built prompt ({} tokens): {:?}", ids.len(), ids);
        Ok(ids)
    }

    /// Decode token IDs to text, skipping special tokens.
    pub fn decode_tokens(&self, token_ids: &[i64]) -> String {
        let mut pieces: Vec<String> = Vec::new();

        for &id in token_ids {
            if let Some(token) = self.id_to_token.get(&id) {
                // Skip special tokens
                if token.starts_with("<|") || token == "<unk>" || token == "<pad>" {
                    continue;
                }
                // Handle SentencePiece ▁ marker (U+2581) as space
                let cleaned = token.replace('\u{2581}', " ");
                pieces.push(cleaned);
            }
        }

        pieces.join("").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_decode_tokens_filters_special() {
        let dir = std::env::temp_dir().join(format!("cohere_vocab_test_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let vocab_path = dir.join("vocab.txt");
        let mut f = fs::File::create(&vocab_path).unwrap();
        writeln!(f, "<|endoftext|> 3").unwrap();
        writeln!(f, "<|startoftranscript|> 4").unwrap();
        writeln!(f, "\u{2581}Hello 100").unwrap();
        writeln!(f, "\u{2581}world 200").unwrap();

        let vocab = Vocab::load(&vocab_path).unwrap();
        let text = vocab.decode_tokens(&[4, 100, 200, 3]);
        assert_eq!(text, "Hello world");

        let _ = fs::remove_dir_all(&dir);
    }
}
