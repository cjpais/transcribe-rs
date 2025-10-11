use std::path::PathBuf;
use transcribe_rs::engines::whisper::{WhisperEngine, WhisperInferenceParams};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_jfk_transcription() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with default params
    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    let expected = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}

#[test]
fn test_initial_prompt_functionality() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Test with initial_prompt providing context
    let params = WhisperInferenceParams {
        initial_prompt: Some("This is a famous speech by President John F. Kennedy.".to_string()),
        ..Default::default()
    };

    // Transcribe with initial prompt
    let result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe with initial prompt");

    // The result should still be accurate (initial_prompt provides context but doesn't change the core transcription)
    assert!(!result.text.trim().is_empty(), "Transcription should not be empty");
    assert!(result.text.len() > 10, "Transcription should have reasonable length");
}

#[test]
fn test_empty_initial_prompt() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Test with empty initial_prompt
    let params = WhisperInferenceParams {
        initial_prompt: Some("".to_string()),
        ..Default::default()
    };

    // Transcribe with empty initial prompt
    let result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe with empty initial prompt");

    // Should still work normally
    assert!(!result.text.trim().is_empty(), "Transcription should not be empty");
}

#[test]
fn test_russian_translation() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the Russian audio file
    let audio_path = PathBuf::from("samples/russian.wav");

    // Set up inference params with translate enabled
    let params = WhisperInferenceParams {
        translate: true,
        ..Default::default()
    };

    // Transcribe and translate to English
    let result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe");

    let expected = "Check the connection.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}
