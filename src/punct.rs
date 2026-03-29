//! Neural network-based Chinese/English punctuation restoration.
//!
//! Uses a CT-Transformer ONNX model to insert punctuation into raw ASR text.
//! The model operates at the character/token level and supports both Chinese
//! (full-width punctuation) and English (ASCII punctuation) contexts.
//!
//! # Feature gate
//!
//! This module requires the `punct` feature:
//!
//! ```toml
//! [dependencies]
//! transcribe-rs = { version = "0.3", features = ["punct"] }
//! ```
//!
//! # Model files
//!
//! You need the CT-Transformer model directory containing:
//! - `model.int8.onnx` (or `model.onnx` as fallback)
//! - `tokens.json` (vocabulary file as a JSON array of strings)
//!
//! # Usage
//!
//! ```ignore
//! use std::path::Path;
//! use transcribe_rs::punct::PunctModel;
//!
//! let mut model = PunctModel::new(Path::new("models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8/"))?;
//! let result = model.add_punctuation("今天天气很好我们去公园玩吧");
//! println!("{}", result);
//! // => "今天天气很好，我们去公园玩吧。"
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ndarray::{Array1, Array2};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::TranscribeError;

// ── Punctuation type enum ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PunctType {
    Underscore = 0,
    Comma = 2,
    Dot = 3,
    Quest = 4,
    Pause = 5,
}

impl PunctType {
    fn from_id(id: usize) -> Option<Self> {
        match id {
            0 => Some(PunctType::Underscore),
            2 => Some(PunctType::Comma),
            3 => Some(PunctType::Dot),
            4 => Some(PunctType::Quest),
            5 => Some(PunctType::Pause),
            _ => None,
        }
    }

    fn to_char(self) -> Option<char> {
        match self {
            PunctType::Underscore => None,
            PunctType::Comma => Some('，'),
            PunctType::Dot => Some('。'),
            PunctType::Quest => Some('？'),
            PunctType::Pause => Some('、'),
        }
    }

    fn to_ascii_char(self) -> Option<char> {
        match self {
            PunctType::Underscore => None,
            PunctType::Comma => Some(','),
            PunctType::Dot => Some('.'),
            PunctType::Quest => Some('?'),
            PunctType::Pause => Some(','),
        }
    }
}

// ── Token classification ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum TokenInfo {
    Word(String),
    Char(char),
    Space,
    Punct(char),
}

// ── PunctModel ────────────────────────────────────────────────────────────────

/// CT-Transformer ONNX-based punctuation restoration model.
///
/// Tokenizes input text, runs inference in 20-token windows with 2-token
/// overlap, and reconstructs text with intelligently placed punctuation marks.
pub struct PunctModel {
    session: Session,
    token2id: HashMap<String, i32>,
    unk_id: i32,
    input_name: String,
    length_name: String,
}

impl PunctModel {
    /// Load a PunctModel from a model directory.
    ///
    /// The directory must contain `model.int8.onnx` (or `model.onnx`) and
    /// `tokens.json`.
    pub fn new(model_dir: &Path) -> Result<Self, TranscribeError> {
        let model_path = model_dir.join("model.int8.onnx");
        let model_path = if !model_path.exists() {
            model_dir.join("model.onnx")
        } else {
            model_path
        };
        let tokens_path = model_dir.join("tokens.json");

        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path));
        }
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading punctuation model from {:?}...", model_path);

        let session = Session::builder()
            .map_err(|e| TranscribeError::Config(format!("ort session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TranscribeError::Config(format!("ort optimization level: {e}")))?
            .with_parallel_execution(true)
            .map_err(|e| TranscribeError::Config(format!("ort parallel execution: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| TranscribeError::Inference(format!("failed to load punct model: {e}")))?;

        let (token2id, unk_id) = Self::load_tokens(&tokens_path)?;

        // session.inputs() returns &[Outlet] — index directly
        let input_name = session.inputs()[0].name().to_string();
        let length_name = session.inputs()[1].name().to_string();

        log::info!(
            "Punct model input names: '{}' and '{}'",
            input_name,
            length_name
        );

        Ok(Self {
            session,
            token2id,
            unk_id,
            input_name,
            length_name,
        })
    }

    fn load_tokens(path: &Path) -> Result<(HashMap<String, i32>, i32), TranscribeError> {
        let file = File::open(path)?;
        let tokens: Vec<String> = serde_json::from_reader(file)?;
        let mut token2id = HashMap::new();
        for (id, token) in tokens.iter().enumerate() {
            token2id.insert(token.clone(), id as i32);
        }
        let unk_id = *token2id.get("<unk>").unwrap_or(&0);
        log::info!("Loaded {} tokens, unk_id={}", tokens.len(), unk_id);
        Ok((token2id, unk_id))
    }

    /// Tokenize input text into (token_ids, token_infos).
    ///
    /// - CJK characters are tokenized one character at a time.
    /// - ASCII words/digit runs are kept as a single token.
    /// - Whitespace is recorded as `TokenInfo::Space` (not submitted to model).
    /// - Existing punctuation is recorded as `TokenInfo::Punct` and skipped.
    fn tokenize(&self, text: &str) -> (Vec<i32>, Vec<TokenInfo>) {
        let mut ids: Vec<i32> = Vec::new();
        let mut infos: Vec<TokenInfo> = Vec::new();

        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            if c.is_whitespace() {
                infos.push(TokenInfo::Space);
                continue;
            }

            // Existing punctuation — preserve as-is, don't tokenize
            if is_existing_punct(c) {
                infos.push(TokenInfo::Punct(c));
                continue;
            }

            if is_cjk(c) {
                // Single CJK character
                let token = c.to_string();
                let id = *self.token2id.get(&token).unwrap_or(&self.unk_id);
                ids.push(id);
                infos.push(TokenInfo::Char(c));
            } else {
                // Collect a run of non-CJK, non-space, non-punct chars as a word
                let mut word = String::new();
                word.push(c);
                while let Some(&nc) = chars.peek() {
                    if nc.is_whitespace() || is_cjk(nc) || is_existing_punct(nc) {
                        break;
                    }
                    word.push(nc);
                    chars.next();
                }
                let lower = word.to_lowercase();
                let id = self
                    .token2id
                    .get(&lower)
                    .copied()
                    .unwrap_or_else(|| *self.token2id.get(&word).unwrap_or(&self.unk_id));
                ids.push(id);
                infos.push(TokenInfo::Word(word));
            }
        }

        (ids, infos)
    }

    /// Run punctuation inference on a batch of token IDs.
    ///
    /// Returns a Vec of punctuation class IDs, one per input token.
    fn run_inference(&mut self, token_ids: &[i32]) -> Result<Vec<usize>, TranscribeError> {
        let seq_len = token_ids.len();
        let input_array = Array2::from_shape_vec(
            (1, seq_len),
            token_ids.iter().map(|&x| x as i64).collect(),
        )
        .map_err(|e| TranscribeError::Inference(format!("shape error: {e}")))?;
        let length_array = Array1::from_vec(vec![seq_len as i64]);

        let input_tensor = TensorRef::from_array_view(input_array.view())
            .map_err(|e| TranscribeError::Inference(format!("input tensor: {e}")))?;
        let length_tensor = TensorRef::from_array_view(length_array.view())
            .map_err(|e| TranscribeError::Inference(format!("length tensor: {e}")))?;

        let outputs = self
            .session
            .run(inputs![
                self.input_name.as_str() => input_tensor,
                self.length_name.as_str() => length_tensor
            ])
            .map_err(|e| TranscribeError::Inference(format!("inference: {e}")))?;

        let output = outputs[0]
            .try_extract_array::<i64>()
            .map_err(|e| TranscribeError::Inference(format!("extract output: {e}")))?;
        let punct_ids: Vec<usize> = output.iter().map(|&x| x as usize).collect();
        Ok(punct_ids)
    }

    /// Add punctuation to raw ASR text.
    ///
    /// On inference error, logs a warning and returns the original text unchanged.
    pub fn add_punctuation(&mut self, text: &str) -> String {
        if text.is_empty() {
            return text.to_string();
        }

        let (token_ids, token_infos) = self.tokenize(text);

        if token_ids.is_empty() {
            return text.to_string();
        }

        // Process in windows of 20 tokens with 2-token overlap
        const WINDOW: usize = 20;
        const OVERLAP: usize = 2;

        let mut punctuations: Vec<usize> = vec![0usize; token_ids.len()];

        let mut start = 0;
        loop {
            let end = (start + WINDOW).min(token_ids.len());
            let chunk = &token_ids[start..end];

            match self.run_inference(chunk) {
                Ok(preds) => {
                    // Only take the non-overlapping part (except for the last chunk)
                    let take = if end < token_ids.len() {
                        preds.len().saturating_sub(OVERLAP)
                    } else {
                        preds.len()
                    };
                    for (i, &p) in preds[..take].iter().enumerate() {
                        punctuations[start + i] = p;
                    }
                }
                Err(e) => {
                    log::warn!("Punctuation inference failed: {}", e);
                    return text.to_string();
                }
            }

            if end >= token_ids.len() {
                break;
            }
            start += WINDOW - OVERLAP;
        }

        self.reconstruct_with_punctuation(&token_infos, &punctuations)
    }

    /// Reconstruct the output string by interleaving tokens with predicted punctuation.
    fn reconstruct_with_punctuation(
        &self,
        token_infos: &[TokenInfo],
        punctuations: &[usize],
    ) -> String {
        let mut result = String::new();
        let mut punct_iter = punctuations.iter();

        for info in token_infos {
            match info {
                TokenInfo::Space => {
                    // Spaces are absorbed; punctuation takes their place
                }
                TokenInfo::Punct(c) => {
                    result.push(*c);
                }
                TokenInfo::Char(c) => {
                    result.push(*c);
                    if let Some(&pid) = punct_iter.next() {
                        if let Some(pt) = PunctType::from_id(pid) {
                            // CJK character → use full-width punctuation
                            if let Some(pc) = pt.to_char() {
                                result.push(pc);
                            }
                        }
                    }
                }
                TokenInfo::Word(w) => {
                    result.push_str(w);
                    if let Some(&pid) = punct_iter.next() {
                        if let Some(pt) = PunctType::from_id(pid) {
                            let pc = choose_punct_char(pt, w, &result);
                            if let Some(pc) = pc {
                                result.push(pc);
                            }
                        }
                    }
                }
            }
        }

        result
    }
}

// ── Helper functions ─────────────────────────────────────────────────────────

/// Choose full-width or ASCII punctuation based on surrounding context.
fn choose_punct_char(pt: PunctType, current_word: &str, result_so_far: &str) -> Option<char> {
    // If the current word is an English/ASCII word, use ASCII punctuation.
    // If the preceding content ends in a CJK character, use full-width.
    let last_meaningful = result_so_far
        .chars()
        .rev()
        .find(|c| !c.is_whitespace());

    let use_ascii = is_english_token(current_word)
        || last_meaningful
            .map(|c| !is_cjk(c) && c.is_ascii_alphanumeric())
            .unwrap_or(false);

    if use_ascii {
        pt.to_ascii_char()
    } else {
        pt.to_char()
    }
}

/// Return true if the character is a punctuation mark that should be preserved.
fn is_existing_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(
            c,
            '，' | '。'
                | '？'
                | '！'
                | '、'
                | '；'
                | '：'
                | '…'
                | '\u{201C}' // "
                | '\u{201D}' // "
                | '\u{2018}' // '
                | '\u{2019}' // '
                | '—'
                | '【'
                | '】'
                | '《'
                | '》'
                | '（'
                | '）'
        )
}

/// Return true if the token looks like an English/ASCII word or number.
fn is_english_token(token: &str) -> bool {
    !token.is_empty()
        && token
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '\'')
}

/// Return true if the token consists entirely of CJK characters.
#[allow(dead_code)]
fn is_cjk_token(token: &str) -> bool {
    !token.is_empty() && token.chars().all(is_cjk)
}

/// Return true if the character is in a CJK Unicode block.
fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}'     // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}'   // CJK Extension A
        | '\u{20000}'..='\u{2A6DF}' // CJK Extension B
        | '\u{F900}'..='\u{FAFF}'   // CJK Compatibility Ideographs
        | '\u{2F800}'..='\u{2FA1F}' // CJK Compatibility Supplement
        | '\u{3000}'..='\u{303F}'   // CJK Symbols and Punctuation
        | '\u{31F0}'..='\u{31FF}'   // Katakana Phonetic Extensions
        | '\u{3200}'..='\u{32FF}'   // Enclosed CJK
        | '\u{3300}'..='\u{33FF}'   // CJK Compatibility
        | '\u{AC00}'..='\u{D7AF}'   // Hangul Syllables
    )
}
