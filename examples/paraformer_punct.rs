use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::{
    engines::paraformer::{ParaformerEngine, ParaformerModelParams},
    punct::add_punctuation,
    TranscriptionEngine,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(
        args.get(1)
            .map(|s| s.as_str())
            .unwrap_or("models/sherpa-paraformer"),
    );
    let wav_path = PathBuf::from(
        args.get(2)
            .map(|s| s.as_str())
            .unwrap_or("/Users/zac/Downloads/test.wav"),
    );

    let mut engine = ParaformerEngine::new();
    let load_start = Instant::now();
    engine.load_model_with_params(&model_path, ParaformerModelParams::int8())?;
    let load_time = load_start.elapsed();

    let infer_start = Instant::now();
    let result = engine.transcribe_file(&wav_path, None)?;
    let infer_time = infer_start.elapsed();
    let transcribed_text = &result.text;
    engine.unload_model();

    let punctuated_text = add_punctuation(transcribed_text);

    println!("Model: {:?}", model_path);
    println!("Audio: {:?}", wav_path);
    println!("Load time: {:.2?}", load_time);
    println!("Infer time: {:.2?}", infer_time);
    println!();
    println!("Paraformer 识别: {}", transcribed_text);
    println!("添加标点后: {}", punctuated_text);

    Ok(())
}
