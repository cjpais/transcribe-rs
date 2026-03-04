//! Zipformer Transducer (RNN-T) ONNX transcription engine.
//!
//! This module provides an ONNX implementation for Zipformer Transducer models
//! such as `sherpa-onnx-zipformer-zh-en-2023-11-22` (multilingual zh/en).
//! These models use 3 separate ONNX components: encoder, decoder, joiner.

mod engine;
mod model;

pub use engine::{
    ZipformerTransducerEngine, ZipformerTransducerInferenceParams,
    ZipformerTransducerModelParams, QuantizationType,
};
