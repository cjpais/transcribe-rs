//! GigaAM v3 ONNX transcription engine.
//!
//! This module provides transcription using the GigaAM v3 e2e_ctc model via ONNX Runtime.
//! GigaAM is a CTC-based speech recognition model for Russian with punctuation support,
//! Latin character output, and BPE subword tokenization.
//!
//! # Model Architecture
//!
//! GigaAM v3 e2e_ctc uses a Conformer encoder with CTC decoder:
//! - Processes audio via mel spectrogram (n_fft=320, hop=160, 64 mels, HTK scale)
//! - BPE vocabulary with 257 tokens (Russian subwords, Latin chars, punctuation, digits)
//! - CTC greedy decoding with SentencePiece word boundary handling
//!
//! # Model Format
//!
//! Expects a single ONNX file (e.g., `v3_e2e_ctc.int8.onnx`).
//!
//! # Supported Languages
//!
//! Russian (with Latin character passthrough for loanwords).
//!
//! # Audio Requirements
//!
//! - Sample rate: 16 kHz
//! - Format: Mono, 16-bit PCM
//!
//! # Example
//!
//! ```rust,no_run
//! use std::path::PathBuf;
//! use transcribe_rs::{TranscriptionEngine, engines::gigaam::GigaAMEngine};
//!
//! let mut engine = GigaAMEngine::new();
//! engine.load_model(&PathBuf::from("models/v3_e2e_ctc.int8.onnx"))?;
//!
//! let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
//! println!("Transcription: {}", result.text);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod engine;
pub mod model;

pub use engine::GigaAMEngine;
pub use model::GigaAMError;
