//! Paraformer ONNX transcription engine.
//!
//! This module provides a dedicated ONNX implementation for Paraformer models
//! such as `models/sherpa-paraformer`, without depending on the sherpa engine.

mod engine;
mod features;
mod model;
mod tokens;

pub use engine::{
    ParaformerEngine, ParaformerInferenceParams, ParaformerModelParams, QuantizationType,
};
