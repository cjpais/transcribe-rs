//! Neural network-based Chinese punctuation restoration.
//!
//! Uses a CT-Transformer model to add punctuation to Chinese text.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use ndarray::{Array1, Array2};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

#[derive(thiserror::Error, Debug)]
pub enum PunctError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Tokens file not found: {0}")]
    TokensNotFound(String),
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Inference error: {0}")]
    Inference(String),
}

/// Punctuation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PunctType {
    Underscore = 0, // 无标点
    Comma = 2,      // ，
    Dot = 3,        // 。
    Quest = 4,      // ？
    Pause = 5,      // 、 (shuanghao)
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

/// Neural punctuation model
pub struct PunctModel {
    session: Session,
    token2id: HashMap<String, i32>,
    #[allow(dead_code)]
    id2token: Vec<String>,
    unk_id: i32,
    input_name: String,
    length_name: String,
}

/// Token information for reconstruction
#[derive(Debug, Clone)]
enum TokenInfo {
    Word(String), // English word or digit sequence
    Char(char),   // Chinese character
    Space,        // Space
    Punct(char),  // Punctuation
}

impl PunctModel {
    /// Load a punctuation model from a directory
    pub fn new(model_dir: &Path) -> Result<Self, PunctError> {
        let model_path = model_dir.join("model.int8.onnx");
        let model_path = if !model_path.exists() {
            model_dir.join("model.onnx")
        } else {
            model_path
        };

        let tokens_path = model_dir.join("tokens.json");

        if !model_path.exists() {
            return Err(PunctError::ModelNotFound(model_path.display().to_string()));
        }
        if !tokens_path.exists() {
            return Err(PunctError::TokensNotFound(
                tokens_path.display().to_string(),
            ));
        }

        log::info!("Loading punctuation model from {:?}...", model_path);

        let session = Self::init_session(&model_path)?;
        let (token2id, id2token, unk_id) = Self::load_tokens(&tokens_path)?;

        // Get input names
        let input_name = session.inputs[0].name.clone();
        let length_name = session.inputs[1].name.clone();

        log::info!(
            "Punct model input names: {} and {}",
            input_name,
            length_name
        );

        Ok(Self {
            session,
            token2id,
            id2token,
            unk_id,
            input_name,
            length_name,
        })
    }

    fn init_session(path: &Path) -> Result<Session, PunctError> {
        let providers = vec![CPUExecutionProvider::default().build()];

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(path)?;

        for input in &session.inputs {
            log::info!(
                "Punct model input: name={}, type={:?}",
                input.name,
                input.input_type
            );
        }
        for output in &session.outputs {
            log::info!(
                "Punct model output: name={}, type={:?}",
                output.name,
                output.output_type
            );
        }

        Ok(session)
    }

    fn load_tokens(path: &Path) -> Result<(HashMap<String, i32>, Vec<String>, i32), PunctError> {
        let file = File::open(path)?;
        let tokens: Vec<String> = serde_json::from_reader(file)?;

        let mut token2id = HashMap::new();
        for (id, token) in tokens.iter().enumerate() {
            token2id.insert(token.clone(), id as i32);
        }

        // Find unk token
        let unk_id = *token2id.get("<unk>").unwrap_or(&0);

        log::info!("Loaded {} tokens, unk_id={}", tokens.len(), unk_id);

        Ok((token2id, tokens, unk_id))
    }

    /// Tokenize input text
    /// Chinese characters are tokenized individually
    /// English words are tokenized using BPE
    /// Spaces are preserved as separate tokens
    fn tokenize(&self, text: &str) -> (Vec<i32>, Vec<TokenInfo>) {
        let mut ids = Vec::new();
        let mut token_infos = Vec::new();
        let mut current_word = String::new();

        fn is_cjk(c: char) -> bool {
            // CJK Unified Ideographs, CJK Extension A, etc.
            let code = c as u32;
            (0x4E00..=0x9FFF).contains(&code) ||  // CJK Unified Ideographs
            (0x3400..=0x4DBF).contains(&code) ||   // CJK Extension A
            (0x3000..=0x303F).contains(&code) // CJK Symbols
        }

        fn is_ascii_letter(c: char) -> bool {
            c.is_ascii_alphabetic()
        }

        fn is_ascii_digit(c: char) -> bool {
            c.is_ascii_digit()
        }

        // Collect all characters and process
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        let mut i = 0;
        while i < n {
            let c = chars[i];
            let is_c = is_cjk(c);
            let is_a = is_ascii_letter(c);
            let is_d = is_ascii_digit(c);

            if is_c {
                // Flush current word if any
                if !current_word.is_empty() {
                    if let Some(id) = self.token2id.get(&current_word) {
                        ids.push(*id);
                        token_infos.push(TokenInfo::Word(current_word.clone()));
                    } else {
                        ids.push(self.unk_id);
                        token_infos.push(TokenInfo::Word(current_word.clone()));
                    }
                    current_word.clear();
                }
                // Add Chinese character
                let char_str = c.to_string();
                if let Some(id) = self.token2id.get(&char_str) {
                    ids.push(*id);
                    token_infos.push(TokenInfo::Char(c));
                } else {
                    ids.push(self.unk_id);
                    token_infos.push(TokenInfo::Char(c));
                }
            } else if is_a || is_d {
                // Collect English word or digit
                current_word.push(c);
            } else if c == ' ' {
                // Flush current word if any
                if !current_word.is_empty() {
                    if let Some(id) = self.token2id.get(&current_word) {
                        ids.push(*id);
                        token_infos.push(TokenInfo::Word(current_word.clone()));
                    } else {
                        ids.push(self.unk_id);
                        token_infos.push(TokenInfo::Word(current_word.clone()));
                    }
                    current_word.clear();
                }
                // Add space as a token
                if let Some(id) = self.token2id.get(" ") {
                    ids.push(*id);
                    token_infos.push(TokenInfo::Space);
                }
            } else {
                // Flush current word if any
                if !current_word.is_empty() {
                    if let Some(id) = self.token2id.get(&current_word) {
                        ids.push(*id);
                        token_infos.push(TokenInfo::Word(current_word.clone()));
                    } else {
                        ids.push(self.unk_id);
                        token_infos.push(TokenInfo::Word(current_word.clone()));
                    }
                    current_word.clear();
                }
                // Handle punctuation - treat as separate token
                let char_str = c.to_string();
                if let Some(id) = self.token2id.get(&char_str) {
                    ids.push(*id);
                    token_infos.push(TokenInfo::Punct(c));
                }
            }
            i += 1;
        }

        // Flush remaining word
        if !current_word.is_empty() {
            if let Some(id) = self.token2id.get(&current_word) {
                ids.push(*id);
                token_infos.push(TokenInfo::Word(current_word.clone()));
            } else {
                ids.push(self.unk_id);
                token_infos.push(TokenInfo::Word(current_word.clone()));
            }
        }

        (ids, token_infos)
    }

    /// Add punctuation to text
    pub fn add_punctuation(&mut self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        let (ids, token_infos) = self.tokenize(text);
        if ids.is_empty() {
            return text.to_string();
        }

        let segment_size = 20;
        let max_len = 200;

        let num_segments = (ids.len() + segment_size - 1) / segment_size;

        let mut punctuations: Vec<usize> = Vec::new();
        let mut last_end = 0;

        for seg_idx in 0..num_segments {
            let this_start = if seg_idx == 0 { 0 } else { last_end };
            let this_end = std::cmp::min(this_start + segment_size, ids.len());

            let inputs_ids: Vec<i32> = ids[this_start..this_end].to_vec();

            // Create input tensors
            let input_array = Array2::from_shape_vec((1, inputs_ids.len()), inputs_ids.clone())
                .unwrap()
                .mapv(|v| v as i32);
            let length_array = Array1::from_vec(vec![inputs_ids.len() as i32]);

            // Run inference
            let inputs = inputs![
                self.input_name.as_str() => TensorRef::from_array_view(input_array.view()).unwrap(),
                self.length_name.as_str() => TensorRef::from_array_view(length_array.view()).unwrap(),
            ];

            let outputs = match self.session.run(inputs) {
                Ok(o) => o,
                Err(e) => {
                    log::warn!("Inference failed: {:?}", e);
                    // Fallback: no punctuation
                    for _ in 0..inputs_ids.len() {
                        punctuations.push(0);
                    }
                    continue;
                }
            };

            let logits = outputs[0].try_extract_array::<f32>().unwrap();

            // Get argmax along the last dimension (punctuation class)
            let shape = logits.shape();
            let mut seg_puncts = Vec::with_capacity(shape[1]);

            for t in 0..shape[1] {
                // Find argmax for each position
                let mut max_idx = 0;
                let mut max_val = f32::MIN;
                for c in 0..shape[2] {
                    let val = logits[[0, t, c]];
                    if val > max_val {
                        max_val = val;
                        max_idx = c;
                    }
                }
                seg_puncts.push(max_idx);
            }

            // Find the last punctuation position
            let mut dot_index = -1;
            let mut comma_index = -1;

            for k in (2..seg_puncts.len()).rev() {
                let p = seg_puncts[k];
                if p == 3 || p == 4 {
                    // Dot or Quest
                    dot_index = k as isize;
                    break;
                }
                if comma_index == -1 && p == 2 {
                    comma_index = k as isize;
                }
            }

            // Handle long segment
            if dot_index == -1 && inputs_ids.len() >= max_len && comma_index != -1 {
                dot_index = comma_index;
                seg_puncts[dot_index as usize] = 3; // dot
            }

            // Handle end of text
            if dot_index == -1 {
                if seg_idx == 0 {
                    last_end = this_start;
                }
                if seg_idx == num_segments - 1 {
                    dot_index = (inputs_ids.len() - 1) as isize;
                }
            } else {
                last_end = this_start + (dot_index as usize) + 1;
            }

            if dot_index != -1 {
                punctuations.extend_from_slice(&seg_puncts[..=dot_index as usize]);
            }
        }

        // Reconstruct text with punctuation, preserving spaces
        self.reconstruct_with_punctuation(&token_infos, &punctuations)
    }

    /// Reconstruct text with punctuation
    fn reconstruct_with_punctuation(
        &self,
        token_infos: &[TokenInfo],
        punctuations: &[usize],
    ) -> String {
        let mut result = String::new();

        for (i, info) in token_infos.iter().enumerate() {
            // Determine if we need space before this token
            if i > 0 {
                let need_space = match (&token_infos[i - 1], info) {
                    // Word to Word: add space (English words)
                    (TokenInfo::Word(_), TokenInfo::Word(_)) => true,
                    // Word to Char: add space (English to Chinese)
                    (TokenInfo::Word(_), TokenInfo::Char(_)) => true,
                    // Char to Word: add space (Chinese to English)
                    (TokenInfo::Char(_), TokenInfo::Word(_)) => true,
                    // After punctuation: add space
                    (TokenInfo::Punct(_), _) => true,
                    // After space: no extra space
                    (TokenInfo::Space, _) => false,
                    // Char to Char (Chinese): no space
                    (TokenInfo::Char(_), TokenInfo::Char(_)) => false,
                    _ => false,
                };

                if need_space && !result.ends_with(' ') {
                    result.push(' ');
                }
            }

            match info {
                TokenInfo::Word(w) => result.push_str(w),
                TokenInfo::Char(c) => result.push(*c),
                TokenInfo::Space => {
                    if !result.ends_with(' ') {
                        result.push(' ');
                    }
                }
                TokenInfo::Punct(c) => result.push(*c),
            }

            // Add punctuation after the token
            if i < punctuations.len() {
                let p = punctuations[i];
                let prev_info = Some(info);
                let next_info = token_infos.get(i + 1);
                if let Some(punct_char) =
                    PunctType::from_id(p).and_then(|pt| choose_punct_char(pt, prev_info, next_info))
                {
                    result.push(punct_char);
                }
            }
        }

        result
    }
}

fn choose_punct_char(
    pt: PunctType,
    prev: Option<&TokenInfo>,
    next: Option<&TokenInfo>,
) -> Option<char> {
    let prev_is_en = prev.map(is_english_token).unwrap_or(false);
    let next_is_en = next.map(is_english_token).unwrap_or(false);
    let prev_is_cjk = prev.map(is_cjk_token).unwrap_or(false);

    // Avoid awkward Chinese sentence break right before an English phrase.
    if matches!(pt, PunctType::Dot | PunctType::Comma | PunctType::Pause)
        && prev_is_cjk
        && next_is_en
    {
        return None;
    }

    if prev_is_en || next_is_en {
        pt.to_ascii_char()
    } else {
        pt.to_char()
    }
}

fn is_english_token(info: &TokenInfo) -> bool {
    match info {
        TokenInfo::Word(w) => {
            let w = w.trim();
            !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '\'' || c == '-' || c == '_')
        }
        _ => false,
    }
}

fn is_cjk_token(info: &TokenInfo) -> bool {
    match info {
        TokenInfo::Char(c) => is_cjk(*c),
        _ => false,
    }
}

fn is_cjk(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
}

/// Add punctuation to text using neural model
///
/// This is a convenience function that loads the model from the default path
/// and applies punctuation restoration.
///
/// Note: If the text already has punctuation (Chinese or English),
/// it will be returned as-is to avoid duplicate punctuation.
pub fn add_punctuation(text: &str) -> String {
    add_punctuation_with_model(
        text,
        Path::new("models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8"),
    )
}

/// Add punctuation to text using a specific punctuation model directory.
///
/// `model_dir` should contain `model.int8.onnx` (or `model.onnx`) and `tokens.json`.
pub fn add_punctuation_with_model(text: &str, model_dir: &Path) -> String {
    // Check if text already has punctuation
    if has_punctuation(text) {
        log::info!("Text already has punctuation, skipping");
        return text.to_string();
    }

    if let Ok(mut model) = PunctModel::new(model_dir) {
        // First add English spaces (for Sherpa output)
        let text_with_spaces = add_english_spaces(text);
        return model.add_punctuation(&text_with_spaces);
    }

    // Fallback to rule-based
    log::warn!("Failed to load neural punctuation model, using rule-based fallback");
    add_punctuation_fallback(text)
}

/// Rule-based fallback for punctuation
fn add_punctuation_fallback(text: &str) -> String {
    let mut result = text.to_string();

    if !result.is_empty() {
        let last_char = result.chars().last().unwrap();
        if !is_punctuation(last_char) {
            if is_question(&result) {
                result.push('?');
            } else {
                result.push('.');
            }
        }
    }

    result
}

fn is_punctuation(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | '?' | '!' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']'
    )
}

/// Check if text already has punctuation (Chinese or English)
fn has_punctuation(text: &str) -> bool {
    // Check for Chinese punctuation
    let chinese_punct = [
        '，', '。', '？', '！', '、', '；', '：', '"', '"', '\'', '\'',
    ];
    for c in text.chars() {
        if chinese_punct.contains(&c) {
            return true;
        }
    }

    // Check for English punctuation (at least one)
    let mut has_eng_punct = false;
    for c in text.chars() {
        if is_punctuation(c) {
            has_eng_punct = true;
            break;
        }
    }

    has_eng_punct
}

/// Add spaces between English words
/// E.g., "helloworld" -> "hello world"
/// This is a simple heuristic based on capital letters or known words
fn add_english_spaces(text: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    let mut i = 0;
    while i < n {
        let c = chars[i];

        if c.is_ascii_lowercase() {
            // Start of a lowercase sequence
            let mut word_end = i;
            while word_end < n && chars[word_end].is_ascii_lowercase() {
                word_end += 1;
            }

            // Check if next char is uppercase (likely new word like "hELLO")
            let has_uppercase_after = word_end < n && chars[word_end].is_ascii_uppercase();

            // Also check for digit patterns
            let has_digit_after = word_end < n && chars[word_end].is_ascii_digit();

            // Add the lowercase word
            for j in i..word_end {
                result.push(chars[j]);
            }

            // Add space if followed by uppercase or digit
            if has_uppercase_after || has_digit_after {
                result.push(' ');
            }

            i = word_end;
        } else if c.is_ascii_uppercase() {
            // Start of uppercase sequence (could be CamelCase or new word)
            if i > 0 && !result.is_empty() && !result.ends_with(' ') && !result.ends_with('(') {
                // Add space before uppercase (except after space or parenthesis)
                result.push(' ');
            }

            let mut word_end = i;
            while word_end < n
                && (chars[word_end].is_ascii_uppercase() || chars[word_end].is_ascii_lowercase())
            {
                word_end += 1;
            }

            for j in i..word_end {
                result.push(chars[j]);
            }

            // Add space after if followed by lowercase
            if word_end < n && chars[word_end].is_ascii_lowercase() {
                result.push(' ');
            }

            i = word_end;
        } else {
            result.push(c);
            i += 1;
        }
    }

    result
}

fn is_question(text: &str) -> bool {
    let question_indicators = [
        "吗",
        "呢",
        "吧",
        "啊",
        "?",
        "是不是",
        "有没有",
        "能不能",
        "会不会",
    ];
    for indicator in question_indicators {
        if text.contains(indicator) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_punct_model_load() {
        let model_dir = Path::new(
            "../models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
        );
        let model = PunctModel::new(model_dir);
        assert!(model.is_ok());
    }

    #[test]
    fn test_punct_model_inference() {
        let model_dir = Path::new(
            "../models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8",
        );
        let mut model = PunctModel::new(model_dir).unwrap();

        let text = "你好吗 how are you";
        let result = model.add_punctuation(text);
        println!("Input: {}", text);
        println!("Output: {}", result);
    }
}
