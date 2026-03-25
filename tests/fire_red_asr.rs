mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::fire_red_asr::FireRedAsrModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_fire_red_asr_transcribe() {
    env_logger::init();

    let model_dir = PathBuf::from("models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16");
    let wav_path = model_dir.join("test_wavs/0.wav");

    if !common::require_paths(&[&model_dir, &wav_path]) {
        return;
    }

    let mut model =
        FireRedAsrModel::load(&model_dir, &Quantization::Int8).expect("Failed to load model");
    let result = model
        .transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    assert!(!result.text.is_empty(), "Transcription should not be empty");
    println!("Transcription: {}", result.text);
}

