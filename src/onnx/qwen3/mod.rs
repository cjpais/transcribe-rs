//! Qwen3-ASR speech recognition engine (batch mode).
//!
//! Implements [`SpeechModel`](crate::SpeechModel) for multilingual batch transcription.
//!
//! # Requirements
//!
//! The model directory must contain:
//! - `encoder.onnx` (+ `.data`) -- audio encoder
//! - `decoder_init.onnx` (+ `.data`) -- decoder prefill
//! - `decoder_step.onnx` (+ `.data`) -- decoder autoregressive step
//! - `embed_tokens.bin` -- embedding matrix (raw f32)
//! - `config.json` -- model configuration
//! - `tokenizer.json` -- HuggingFace BPE tokenizer

mod config;
mod engine;
mod mel;
mod model;
mod prompt;
mod tokenizer;

pub use engine::{Qwen3Model, Qwen3Params};
