use std::path::PathBuf;

use transcribe_rs::{
    engines::qwen_asr::{QwenAsrEngine, QwenAsrInferenceParams, QwenAsrModelParams},
    TranscriptionEngine,
};

fn model_and_audio() -> Option<(PathBuf, PathBuf)> {
    let model_path = PathBuf::from("models/qwen3-asr-0.6b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return None;
    }
    if !wav_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", wav_path);
        return None;
    }
    Some((model_path, wav_path))
}

#[test]
fn test_qwen_asr_transcribe() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, wav_path) = match model_and_audio() {
        Some(v) => v,
        None => return Ok(()),
    };

    let mut engine = QwenAsrEngine::new();
    engine.load_model(&model_path)?;

    let result = engine.transcribe_file(&wav_path, None)?;
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    // JFK clip expected content
    assert!(
        result.text.contains("ask not what your country can do for you"),
        "Should transcribe JFK speech correctly, got: '{}'",
        result.text
    );

    engine.unload_model();
    Ok(())
}

#[test]
fn test_qwen_asr_load_unload() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, _) = match model_and_audio() {
        Some(v) => v,
        None => return Ok(()),
    };

    let mut engine = QwenAsrEngine::new();

    // Transcribing before loading should fail
    let err = engine.transcribe_file(&PathBuf::from("samples/jfk.wav"), None);
    assert!(err.is_err(), "Should fail when model not loaded");

    // Load and unload cycle
    engine.load_model(&model_path)?;
    engine.unload_model();

    // Transcribing after unload should fail
    let err = engine.transcribe_file(&PathBuf::from("samples/jfk.wav"), None);
    assert!(err.is_err(), "Should fail after model unloaded");

    Ok(())
}

#[test]
fn test_qwen_asr_with_model_params() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, wav_path) = match model_and_audio() {
        Some(v) => v,
        None => return Ok(()),
    };

    let mut engine = QwenAsrEngine::new();
    let params = QwenAsrModelParams {
        segment_sec: 30.0,
        skip_silence: true,
    };
    engine.load_model_with_params(&model_path, params)?;

    let result = engine.transcribe_file(&wav_path, None)?;
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    engine.unload_model();
    Ok(())
}

#[test]
fn test_qwen_asr_with_language() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, wav_path) = match model_and_audio() {
        Some(v) => v,
        None => return Ok(()),
    };

    let mut engine = QwenAsrEngine::new();
    engine.load_model(&model_path)?;

    let params = QwenAsrInferenceParams {
        language: Some("English".to_string()),
        prompt: None,
    };
    let result = engine.transcribe_file(&wav_path, Some(params))?;
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    engine.unload_model();
    Ok(())
}

#[test]
fn test_qwen_asr_invalid_language() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, wav_path) = match model_and_audio() {
        Some(v) => v,
        None => return Ok(()),
    };

    let mut engine = QwenAsrEngine::new();
    engine.load_model(&model_path)?;

    let params = QwenAsrInferenceParams {
        language: Some("Klingon".to_string()),
        prompt: None,
    };
    let result = engine.transcribe_file(&wav_path, Some(params));
    assert!(result.is_err(), "Should fail with invalid language");

    engine.unload_model();
    Ok(())
}
