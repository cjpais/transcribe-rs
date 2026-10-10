mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::nemotron::NemotronModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{SpeechModel, TranscribeOptions};

/// Smoke test against the JFK clip. Skips gracefully when the model isn't
/// present (CI / fresh checkouts won't have the ~2.4 GB FP32 export).
#[test]
fn test_jfk_transcription() {
    let model_path = PathBuf::from("models/nemotron-3.5-asr-streaming-0.6b-onnx");
    let audio_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let mut model =
        NemotronModel::load(&model_path, &Quantization::FP32).expect("Failed to load model");

    let result = model
        .transcribe_file(
            &audio_path,
            &TranscribeOptions {
                language: Some("en-US".to_string()),
                ..Default::default()
            },
        )
        .expect("Failed to transcribe");

    assert!(
        !result.text.trim().is_empty(),
        "Transcription should not be empty"
    );
    assert!(
        result.text.to_lowercase().contains("country"),
        "Expected the JFK line, got: '{}'",
        result.text
    );
}
