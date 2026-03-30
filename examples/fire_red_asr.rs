use std::path::PathBuf;

use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::onnx::fire_red_asr::{FireRedAsrModel, FireRedAsrParams};
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_dir = PathBuf::from("models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16");
    let wav_path = model_dir.join("test_wavs/0.wav");

    let mut model = FireRedAsrModel::load(&model_dir, &Quantization::Int8)?;
    let samples = read_wav_samples(&wav_path)?;

    let result = model.transcribe_with(&samples, &FireRedAsrParams::default())?;
    println!("{}", result.text);

    // Also demonstrate the generic trait API
    let result2 = model.transcribe(&samples, &transcribe_rs::TranscribeOptions::default())?;
    println!("{}", result2.text);

    Ok(())
}

