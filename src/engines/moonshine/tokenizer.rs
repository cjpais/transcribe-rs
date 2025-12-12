use std::path::Path;
use tokenizers::Tokenizer;

use super::model::MoonshineError;

pub struct MoonshineTokenizer {
    tokenizer: Tokenizer,
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
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MoonshineError::Tokenization(e.to_string()))?;

        Ok(Self { tokenizer })
    }

    pub fn decode(&self, token_ids: &[i64]) -> Result<String, MoonshineError> {
        // Convert i64 to u32 for tokenizers crate
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        // skip_special_tokens=true to remove <s>, </s>, etc.
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| MoonshineError::Tokenization(e.to_string()))?;

        Ok(text)
    }
}
