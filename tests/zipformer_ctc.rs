mod common;

use std::path::PathBuf;

use transcribe_rs::onnx::zipformer_ctc::ZipformerCtcModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn test_zipformer_ctc_transcribe() {
    env_logger::init();

    let model_dir = PathBuf::from("models/sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16");
    let wav_path = PathBuf::from("samples/zh.wav");

    if !common::require_paths(&[&model_dir, &wav_path]) {
        return;
    }

    let mut model =
        ZipformerCtcModel::load(&model_dir, &Quantization::Int8).expect("Failed to load model");

    let result = model
        .transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())
        .expect("Failed to transcribe");

    assert!(
        !result.text.is_empty(),
        "Transcription result should not be empty"
    );
}
