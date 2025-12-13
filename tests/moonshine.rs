use std::path::PathBuf;
use transcribe_rs::engines::moonshine::{ModelVariant, MoonshineEngine, MoonshineModelParams};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_moonshine_base_jfk() {
    let mut engine = MoonshineEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/moonshine-base");
    engine
        .load_model_with_params(
            &model_path,
            MoonshineModelParams::variant(ModelVariant::Base),
        )
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with default params
    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    println!("Transcription: {}", result.text);

    // Verify we got a non-empty transcription
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    let expected = "And so my fellow Americans ask not what your country can do for you ask what you can do for your country";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );

    // Check that it contains key words from the JFK speech
    // let text_lower = result.text.to_lowercase();
    // assert!(
    //     text_lower.contains("ask") && text_lower.contains("country"),
    //     "Transcription should contain 'ask' and 'country'. Got: '{}'",
    //     result.text
    // );
}
