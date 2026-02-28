use std::path::PathBuf;

use transcribe_rs::audio;
use transcribe_rs::engines::zipformer_ctc::{ZipformerCtcEngine, ZipformerCtcModelParams};
use transcribe_rs::TranscriptionEngine;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    // Usage: zipformer_ctc [model_dir] [audio_file]
    let model_dir = PathBuf::from(
        args.get(1)
            .map(|s| s.as_str())
            .unwrap_or("models/sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16"),
    );
    let audio_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("test_wavs/0.wav"));

    // Auto-detect: use int8 if model.int8.onnx exists, otherwise fp32
    let params = if model_dir.join("model.int8.onnx").exists() {
        ZipformerCtcModelParams::int8()
    } else {
        ZipformerCtcModelParams::default()
    };

    println!("Loading model from {:?}...", model_dir);
    let mut engine = ZipformerCtcEngine::new();
    engine
        .load_model_with_params(&model_dir, params)
        .expect("Failed to load model");

    println!("Transcribing {:?}...", audio_path);
    let samples = audio::read_wav_samples(&audio_path).expect("Failed to read wav");
    println!("Loaded {} samples ({:.2}s)", samples.len(), samples.len() as f32 / 16000.0);

    let result = engine
        .transcribe_samples(samples, None)
        .expect("Failed to transcribe");
    println!("Result: {}", result.text);
}
