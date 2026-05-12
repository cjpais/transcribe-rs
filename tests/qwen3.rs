//! Qwen3-ASR unit tests — these run without model files present.
//!
//! Inference tests are gated behind `#[cfg(feature = "full-tests")]`
//! because they require a multi-GB model download.

use transcribe_rs::onnx::qwen3::Qwen3Params;

#[test]
fn test_qwen3_params_default() {
    let params = Qwen3Params::default();
    assert_eq!(params.max_tokens, 512);
    assert!(params.language.is_none());
}

#[test]
fn test_qwen3_params_with_language() {
    let params = Qwen3Params {
        language: Some("zh".into()),
        ..Default::default()
    };
    assert_eq!(params.language.as_deref(), Some("zh"));
    assert_eq!(params.max_tokens, 512);
}
