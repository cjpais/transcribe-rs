//! Qwen3-ASR speech recognition engine (batch mode).
//!
//! This module provides the core Qwen3-ASR inference components (encoder, decoder,
//! tokenizer, mel spectrogram) and a [`TranscriptionEngine`](crate::TranscriptionEngine)
//! implementation for standard record-then-transcribe workflows.
//!
//! For real-time streaming transcription, see the `qwen3-streaming` feature.
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

pub mod config;
mod engine;
pub mod mel;
pub mod model;
pub mod prompt;
mod tokenizer;

pub use engine::{QuantizationType, Qwen3Engine, Qwen3InferenceParams, Qwen3ModelParams};
pub use model::Qwen3Error;
