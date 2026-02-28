use std::path::PathBuf;

use transcribe_rs::audio;
use transcribe_rs::engines::zipformer_transducer::{
    ZipformerTransducerEngine, ZipformerTransducerModelParams,
};
use transcribe_rs::TranscriptionEngine;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    // Usage: zipformer_transducer [model_dir] [audio_file]
    let model_dir = PathBuf::from(
        args.get(1)
            .map(|s| s.as_str())
            .unwrap_or("models/sherpa-onnx-zipformer-zh-en-2023-11-22"),
    );
    let audio_path = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| model_dir.join("test_wavs/0.wav"));

    // Auto-detect: look for int8 encoder file, fall back to fp32
    let has_int8 = std::fs::read_dir(&model_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .any(|e| {
                    let name = e.file_name().to_string_lossy().to_string();
                    name.starts_with("encoder") && name.contains("int8") && name.ends_with(".onnx")
                })
        })
        .unwrap_or(false);

    let params = if has_int8 {
        ZipformerTransducerModelParams::int8()
    } else {
        ZipformerTransducerModelParams::fp32()
    };

    println!("Loading model from {:?}...", model_dir);
    let mut engine = ZipformerTransducerEngine::new();
    engine
        .load_model_with_params(&model_dir, params)
        .expect("Failed to load model");

    println!("Transcribing {:?}...", audio_path);
    let samples = audio::read_wav_samples(&audio_path).expect("Failed to read wav");
    println!(
        "Loaded {} samples ({:.2}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    let result = engine
        .transcribe_samples(samples, None)
        .expect("Failed to transcribe");
    println!("Result: {}", result.text);
}
