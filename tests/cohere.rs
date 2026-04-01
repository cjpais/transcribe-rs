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
