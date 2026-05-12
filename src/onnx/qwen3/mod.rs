//! Qwen3-ASR speech recognition engine (batch mode).
//!
//! Implements [`SpeechModel`](crate::SpeechModel) for multilingual batch transcription.
//!
//! # Requirements
//!
//! The model directory must contain:
//! - `encoder.onnx` (+ `.data`) -- audio encoder
//! - `decoder_init.onnx` (+ `.data`) -- decoder prefill (embedding table in graph)
//! - `decoder_step.onnx` (+ `.data`) -- decoder autoregressive step (embedding table in graph)
//! - `config.json` -- model configuration
//! - `tokenizer.json` -- HuggingFace BPE tokenizer
//!
//! # Model output format
//!
//! The decoder produces a language identification prefix followed by the
//! transcription. The standard format uses a newline token (GPT-2 BPE token
//! ID 198, which decodes to `\n`) as the delimiter:
//!
//! ```text
//! language English\n<transcription text>
//! ```
//!
//! The engine layer ([`engine::strip_language_prefix`]) removes this prefix
//! before returning the final text. It tries the newline split first (handles
//! unknown languages automatically), then falls back to matching known
//! language names for the rare no-newline case.

mod config;
mod engine;
mod mel;
mod model;
mod prompt;
mod tokenizer;

pub use engine::{Qwen3Model, Qwen3Params};
