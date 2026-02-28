//! Zipformer CTC ONNX transcription engine.
//!
//! This module provides an ONNX implementation for Zipformer CTC models
//! such as `sherpa-onnx-zipformer-ctc-small-zh-int8`.

mod engine;
mod model;

pub use engine::{
    ZipformerCtcEngine, ZipformerCtcInferenceParams, ZipformerCtcModelParams, QuantizationType,
};
