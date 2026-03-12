mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::canary::{CanaryModel, CanaryParams};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_canary_transcribe() {
    env_logger::init();

    let model_dir = PathBuf::from("models/canary-1b-v2");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_dir, &wav_path]) {
        return;
    }

    let mut model =
        CanaryModel::load(&model_dir, &Quantization::Int8).expect("Failed to load model");

    let result = model
        .transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    assert!(!result.text.is_empty(), "Transcription should not be empty");
}

#[test]
fn test_canary_transcribe_with_params() {
    let model_dir = PathBuf::from("models/canary-1b-v2");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_dir, &wav_path]) {
        return;
    }

    let mut model =
        CanaryModel::load(&model_dir, &Quantization::Int8).expect("Failed to load model");

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).expect("Failed to read WAV");

    let params = CanaryParams {
        language: Some("en".to_string()),
        use_pnc: true,
        ..Default::default()
    };

    let result = model
        .transcribe_with(&samples, &params)
        .expect("Failed to transcribe");

    assert!(!result.text.is_empty(), "Transcription should not be empty");
}
