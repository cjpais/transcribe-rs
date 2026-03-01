use std::path::PathBuf;
use transcribe_rs::onnx::{Engine, Model};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_moonshine_base_jfk() {
    let mut engine = Engine::new();

    let model_path = PathBuf::from("models/moonshine-base");
    engine
        .load(&model_path, Model::moonshine_base())
        .expect("Failed to load model");

    let audio_path = PathBuf::from("samples/jfk.wav");

    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    println!("Transcription: {}", result.text);

    assert!(!result.text.is_empty(), "Transcription should not be empty");

    let expected = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}
