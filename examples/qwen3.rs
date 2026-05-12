//! Qwen3-ASR example — batch transcription.

use std::path::PathBuf;

use transcribe_rs::onnx::qwen3::{Qwen3Model, Qwen3Params};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::new().default_filter_or("info")).init();

    let model_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/qwen3-asr-0.6b"));

    let audio_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("tests/fixtures/test_audio.wav"));

    let mut model = Qwen3Model::load(&model_dir, &Quantization::FP32)?;
    let result = model.transcribe_file(&audio_path, &Default::default())?;
    println!("{}", result.text);

    Ok(())
}
