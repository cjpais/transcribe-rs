use std::env;
use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::qwen3::{Qwen3Model, Qwen3Params};
use transcribe_rs::onnx::Quantization;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_dir> <audio.wav> [fp32|fp16|int8|int4] [language] [chunk_seconds]",
            args[0]
        );
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let wav_path = PathBuf::from(&args[2]);
    let quantization = args
        .get(3)
        .map(|value| parse_quantization(value))
        .unwrap_or_else(|| infer_quantization_from_path(&model_dir));
    let language = args.get(4).filter(|value| !value.is_empty()).cloned();
    let chunk_seconds = args
        .get(5)
        .map(|value| parse_chunk_seconds(value))
        .unwrap_or(55.0);

    // Read audio
    let reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    let audio_duration = reader.duration() as f64 / spec.sample_rate as f64;
    println!("Audio: {:.2}s", audio_duration);

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;

    // Load model
    println!(
        "Loading Qwen3-ASR model from {:?} with {:?}",
        model_dir, quantization
    );
    let load_start = Instant::now();
    let mut model = Qwen3Model::load(&model_dir, &quantization)?;
    println!("Model loaded in {:.2?}", load_start.elapsed());
    println!(
        "Language hint: {}; chunk size: {:.2}s",
        language.as_deref().unwrap_or("auto"),
        chunk_seconds
    );

    // Transcribe
    let transcribe_start = Instant::now();
    let chunk_samples = (chunk_seconds * spec.sample_rate as f64).round() as usize;
    let mut texts = Vec::new();
    for (index, chunk) in samples.chunks(chunk_samples).enumerate() {
        let chunk_start = index as f64 * chunk_seconds;
        let chunk_duration = chunk.len() as f64 / spec.sample_rate as f64;
        println!(
            "Chunk {}: {:.2}s..{:.2}s ({:.2}s)",
            index + 1,
            chunk_start,
            chunk_start + chunk_duration,
            chunk_duration
        );
        let chunk_start_time = Instant::now();
        let params = Qwen3Params {
            language: language.clone(),
            ..Default::default()
        };
        let result = model.transcribe_with(chunk, &params)?;
        println!("Chunk {} transcription: {}", index + 1, result.text);
        println!(
            "Chunk {} completed in {:.2?}",
            index + 1,
            chunk_start_time.elapsed()
        );
        if !result.text.trim().is_empty() {
            texts.push(result.text);
        }
    }
    let transcribe_duration = transcribe_start.elapsed();
    let text = texts.join(" ");

    println!("Transcription: {}", text);
    println!("Completed in {:.2?}", transcribe_duration);
    let speedup = audio_duration / transcribe_duration.as_secs_f64();
    println!("Real-time factor: {:.2}x", speedup);

    Ok(())
}

fn parse_chunk_seconds(value: &str) -> f64 {
    let seconds = value.parse::<f64>().unwrap_or_else(|error| {
        eprintln!("Invalid chunk_seconds {value:?}: {error}");
        std::process::exit(2);
    });
    if !(0.0..=60.0).contains(&seconds) || seconds == 0.0 {
        eprintln!("chunk_seconds must be greater than 0 and at most 60; got {seconds}");
        std::process::exit(2);
    }
    seconds
}

fn parse_quantization(value: &str) -> Quantization {
    match value.to_ascii_lowercase().as_str() {
        "fp32" | "f32" => Quantization::FP32,
        "fp16" | "f16" => Quantization::FP16,
        "int8" | "i8" => Quantization::Int8,
        "int4" | "i4" => Quantization::Int4,
        other => {
            eprintln!("Unknown quantization {other:?}; expected fp32, fp16, int8, or int4");
            std::process::exit(2);
        }
    }
}

fn infer_quantization_from_path(model_dir: &std::path::Path) -> Quantization {
    let path = model_dir.to_string_lossy().to_ascii_lowercase();
    if path.contains("int4") || path.contains("i4") {
        Quantization::Int4
    } else if path.contains("int8") || path.contains("i8") {
        Quantization::Int8
    } else if path.contains("fp16") || path.contains("f16") {
        Quantization::FP16
    } else {
        Quantization::default()
    }
}
