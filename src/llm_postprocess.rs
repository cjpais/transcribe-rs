//! LLM-based ASR post-processing using quantized Qwen2.5 (GGUF).
//!
//! Uses a quantized Qwen2.5-0.5B model via candle to add punctuation and
//! correct homophones in ASR output. All inference runs on CPU with no
//! dynamic library dependencies.
//!
//! # Feature gate
//!
//! This module requires the `llm-postprocess` feature:
//!
//! ```toml
//! [dependencies]
//! transcribe-rs = { version = "0.2", features = ["llm-postprocess"] }
//! ```
//!
//! # Model files
//!
//! You need a GGUF-quantized Qwen2.5 model and its tokenizer:
//!
//! - `qwen2.5-0.5b-instruct-q4_k_m.gguf` (~350 MB)
//! - `tokenizer.json` (from HuggingFace Qwen2.5-0.5B-Instruct)
//!
//! Place them in a single directory (e.g. `models/qwen2.5-0.5b/`).
//!
//! # Usage
//!
//! **Reusable processor** (recommended for multiple calls):
//!
//! ```ignore
//! use std::path::Path;
//! use transcribe_rs::llm_postprocess::LlmPostProcessor;
//!
//! let mut proc = LlmPostProcessor::new(Path::new("models/qwen2.5-0.5b/"))?;
//!
//! let corrected = proc.process("今天天气很好我们去公圆玩吧他说号的")?;
//! println!("{}", corrected);
//! // => "今天天气很好，我们去公园玩吧，他说好的。"
//! ```
//!
//! **One-shot convenience function**:
//!
//! ```ignore
//! use std::path::Path;
//! use transcribe_rs::llm_postprocess::llm_postprocess;
//!
//! let corrected = llm_postprocess(
//!     "今天天气很好我们去公圆玩吧",
//!     Path::new("models/qwen2.5-0.5b/"),
//! )?;
//! ```
//!
//! **Custom system prompt**:
//!
//! ```ignore
//! let corrected = proc.process_with_prompt(
//!     "the wether is grate today",
//!     "You are a post-processing assistant. Fix punctuation and spelling.",
//! )?;
//! ```
//!
//! # Pipeline integration
//!
//! Typical ASR post-processing pipeline:
//!
//! 1. **CT-Transformer** (`punct` feature) — fast punctuation restoration (~7 ms)
//! 2. **LLM post-process** (`llm-postprocess` feature) — deep correction (~1-3 s)
//!
//! ```ignore
//! // Step 1: fast punctuation
//! let mut punct = transcribe_rs::PunctModel::new(Path::new("models/punct/"))?;
//! let text = punct.add_punctuation(&raw_asr_text);
//!
//! // Step 2: LLM correction (optional, slower but more accurate)
//! let text = proc.process(&text)?;
//! ```

use std::path::Path;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use tokenizers::Tokenizer;

const MAX_TOKENS: usize = 256;
const EOS_TOKEN: &str = "<|im_end|>";
const DEFAULT_EOS_ID: u32 = 151645;

const DEFAULT_SYSTEM_PROMPT: &str = "你是语音识别后处理助手。用户输入是语音识别的原始输出，\
可能缺少标点、含有同音错别字。请添加正确的标点符号，并将同音错别字纠正为正确的字词。\
只输出纠正后的完整文本。";

#[derive(thiserror::Error, Debug)]
pub enum LlmPostProcessError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("model file not found: {0}")]
    ModelNotFound(String),
    #[error("tokenizer file not found: {0}")]
    TokenizerNotFound(String),
    #[error("no GGUF file found in directory: {0}")]
    NoGgufFile(String),
}

/// LLM-based post-processor holding a quantized Qwen2.5 model and tokenizer.
///
/// Reuse a single instance for multiple calls to avoid repeated model loading.
pub struct LlmPostProcessor {
    model: ModelWeights,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

impl LlmPostProcessor {
    /// Load from a model directory containing `*.gguf` and `tokenizer.json`.
    pub fn new(model_dir: &Path) -> Result<Self, LlmPostProcessError> {
        let gguf_path = find_gguf_file(model_dir)?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(LlmPostProcessError::TokenizerNotFound(
                tokenizer_path.display().to_string(),
            ));
        }
        Self::from_files(&gguf_path, &tokenizer_path)
    }

    /// Load from explicit file paths.
    pub fn from_files(
        gguf_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, LlmPostProcessError> {
        if !gguf_path.exists() {
            return Err(LlmPostProcessError::ModelNotFound(
                gguf_path.display().to_string(),
            ));
        }
        if !tokenizer_path.exists() {
            return Err(LlmPostProcessError::TokenizerNotFound(
                tokenizer_path.display().to_string(),
            ));
        }

        let device = Device::Cpu;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| LlmPostProcessError::Tokenizer(e.to_string()))?;

        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(content, &mut file, &device)?;

        let eos_token_id = tokenizer.token_to_id(EOS_TOKEN).unwrap_or(DEFAULT_EOS_ID);

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
        })
    }

    /// Process ASR text using the default system prompt.
    pub fn process(&mut self, text: &str) -> Result<String, LlmPostProcessError> {
        self.process_with_prompt(text, DEFAULT_SYSTEM_PROMPT)
    }

    /// Process ASR text using a custom system prompt.
    pub fn process_with_prompt(
        &mut self,
        text: &str,
        system_prompt: &str,
    ) -> Result<String, LlmPostProcessError> {
        let prompt = format!(
            "<|im_start|>system\n{system_prompt}<|im_end|>\n\
             <|im_start|>user\n\
             请纠正以下语音识别文本中的标点和错别字：\n\
             {text}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| LlmPostProcessError::Tokenizer(e.to_string()))?;
        let prompt_tokens = encoding.get_ids().to_vec();
        let prompt_len = prompt_tokens.len();

        // Feed prompt through the model
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let last_logits = extract_last_logits(&logits)?;
        let mut next_token = sample_greedy(&last_logits)?;

        let mut output_text = String::new();
        let mut generated_tokens: usize = 0;

        for _ in 0..MAX_TOKENS {
            if next_token == self.eos_token_id {
                break;
            }

            generated_tokens += 1;

            if let Ok(decoded) = self.tokenizer.decode(&[next_token], false) {
                output_text.push_str(&decoded);
            }

            // Forward pass for next token
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let pos = prompt_len + generated_tokens - 1;
            let logits = self.model.forward(&input, pos)?;
            let last_logits = extract_last_logits(&logits)?;
            next_token = sample_greedy(&last_logits)?;
        }

        Ok(output_text)
    }
}

/// Convenience function that loads the model and processes text in one call.
///
/// For repeated use, prefer creating an [`LlmPostProcessor`] instance directly.
pub fn llm_postprocess(text: &str, model_dir: &Path) -> Result<String, LlmPostProcessError> {
    let mut processor = LlmPostProcessor::new(model_dir)?;
    processor.process(text)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Find the first `*.gguf` file in a directory.
fn find_gguf_file(dir: &Path) -> Result<std::path::PathBuf, LlmPostProcessError> {
    if dir.is_file() && dir.extension().is_some_and(|e| e == "gguf") {
        return Ok(dir.to_path_buf());
    }

    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "gguf") {
            return Ok(path);
        }
    }

    Err(LlmPostProcessError::NoGgufFile(dir.display().to_string()))
}

/// Extract the last position's logits, handling 1D/2D/3D tensor shapes.
fn extract_last_logits(logits: &Tensor) -> Result<Tensor, candle_core::Error> {
    match logits.dims().len() {
        3 => {
            let logits = logits.squeeze(0)?;
            logits.get(logits.dim(0)? - 1)
        }
        2 => logits.get(logits.dim(0)? - 1),
        1 => Ok(logits.clone()),
        _ => Err(candle_core::Error::Msg(format!(
            "unexpected logits shape: {:?}",
            logits.dims()
        ))),
    }
}

/// Greedy (argmax) token sampling.
fn sample_greedy(logits: &Tensor) -> Result<u32, candle_core::Error> {
    logits.argmax(0)?.to_scalar::<u32>()
}
