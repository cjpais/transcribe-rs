mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::cohere::CohereModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_cohere_jfk() {
    let model_path = PathBuf::from("models/cohere-int4");
    let audio_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let mut model =
        CohereModel::load(&model_path, &Quantization::Int4).expect("Failed to load Cohere model");

    let result = model
        .transcribe_file(&audio_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe with Cohere model");

    println!("Transcription: {}", result.text);
    assert!(
        !result.text.trim().is_empty(),
        "Cohere transcription should not be empty"
    );
}

#[test]
fn test_cohere_german() {
    let model_path = PathBuf::from("models/cohere-int4");
    let audio_path = PathBuf::from("samples/german.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let mut model =
        CohereModel::load(&model_path, &Quantization::Int4).expect("Failed to load Cohere model");

    let result = model
        .transcribe_file(
            &audio_path,
            &transcribe_rs::TranscribeOptions {
                language: Some("de".into()),
                ..Default::default()
            },
        )
        .expect("Failed to transcribe German audio with Cohere model");

    println!("German transcription: {}", result.text);
    assert!(
        !result.text.trim().is_empty(),
        "Cohere German transcription should not be empty"
    );

    // Output should contain German words
    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("strand") || text_lower.contains("die") || text_lower.contains("der"),
        "German transcription should contain German words, got: '{}'",
        result.text
    );
}

#[test]
fn test_cohere_chinese() {
    let model_path = PathBuf::from("models/cohere-int4");
    let audio_path = PathBuf::from("samples/chinese.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let mut model =
        CohereModel::load(&model_path, &Quantization::Int4).expect("Failed to load Cohere model");

    let result = model
        .transcribe_file(
            &audio_path,
            &transcribe_rs::TranscribeOptions {
                language: Some("zh".into()),
                ..Default::default()
            },
        )
        .expect("Failed to transcribe Chinese audio with Cohere model");

    println!("Chinese transcription: {}", result.text);
    assert!(
        !result.text.trim().is_empty(),
        "Cohere Chinese transcription should not be empty"
    );

    // Output should contain actual Chinese characters, not byte tokens like <0xE5>
    assert!(
        !result.text.contains("<0x"),
        "Chinese transcription should not contain raw byte tokens, got: '{}'",
        result.text
    );
}
