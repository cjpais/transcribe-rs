mod common;

use std::path::PathBuf;
use transcribe_rs::onnx::cohere_transcribe::CohereTranscribeModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_cohere_transcribe() {
    let _ = env_logger::try_init();

    let model_dir = PathBuf::from("models/cohere-transcribe");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_dir, &wav_path]) {
        return;
    }

    let mut model =
        CohereTranscribeModel::load(&model_dir, &Quantization::default()).expect("Failed to load model");

    let result = model
        .transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    assert!(
        result
            .text
            .to_lowercase()
            .contains("ask not what your country can do for you"),
        "Expected JFK quote, got: '{}'",
        result.text
    );
}
