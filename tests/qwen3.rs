mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::qwen3::{Qwen3Model, Qwen3Params};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_qwen3_transcribe() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let model_path = PathBuf::from("models/qwen3-asr-0.6b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return Ok(());
    }

    let mut model = Qwen3Model::load(&model_path, &Quantization::default())?;

    let result = model.transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())?;

    // INT8 produces comma; FP32 produces semicolon.
    // Both are acceptable transcriptions of the JFK quote.
    let acceptable = [
        "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.",
        "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    ];
    assert!(
        acceptable.contains(&result.text.trim()),
        "\nExpected one of: {:?}\nActual: '{}'",
        acceptable,
        result.text.trim()
    );

    Ok(())
}

#[test]
fn test_qwen3_1_7b_transcribe() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let model_path = PathBuf::from("models/qwen3-asr-1.7b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return Ok(());
    }

    let mut model = Qwen3Model::load(&model_path, &Quantization::default())?;
    let result = model.transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())?;

    // FP32 decoder produces ". Ask" (sentence break); INT8 decoder produces "; ask".
    // Both are acceptable transcriptions of the JFK quote.
    let acceptable = [
        "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.",
        "And so, my fellow Americans, ask not what your country can do for you; ask what you can do for your country.",
    ];
    assert!(
        acceptable.contains(&result.text.trim()),
        "\nExpected one of: {:?}\nActual: '{}'",
        acceptable,
        result.text.trim()
    );

    Ok(())
}

#[test]
fn test_qwen3_max_tokens_truncation() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let model_path = PathBuf::from("models/qwen3-asr-0.6b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return Ok(());
    }

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    let mut model = Qwen3Model::load(&model_path, &Quantization::default())?;
    let result = model.transcribe_with(&samples, &Qwen3Params { max_tokens: 5 })?;

    assert!(
        !result.text.is_empty(),
        "truncated result should be non-empty"
    );
    // With max_tokens=5, the output should be much shorter than the full transcription.
    assert!(
        result.text.trim().len() < 80,
        "5-token result should be shorter than full transcription, got: '{}'",
        result.text.trim()
    );

    Ok(())
}

#[test]
fn test_qwen3_empty_audio() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::try_init().ok();

    let model_path = PathBuf::from("models/qwen3-asr-0.6b");

    if !common::require_paths(&[&model_path]) {
        return Ok(());
    }

    let mut model = Qwen3Model::load(&model_path, &Quantization::default())?;
    let result = model.transcribe_with(&[], &Qwen3Params::default())?;
    assert_eq!(result.text, "", "empty audio should return empty string");

    Ok(())
}

#[test]
fn test_qwen3_audio_too_long() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = PathBuf::from("models/qwen3-asr-0.6b");
    if !common::require_paths(&[&model_path]) {
        return Ok(());
    }
    let mut model = Qwen3Model::load(&model_path, &Quantization::default())?;
    let too_long = vec![0.0f32; 60 * 16_000 + 1];
    let result = model.transcribe_with(&too_long, &Qwen3Params::default());
    assert!(result.is_err(), "expected error for audio > 60s");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("audio too long"), "unexpected error: {err}");
    Ok(())
}
