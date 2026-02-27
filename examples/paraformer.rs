use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::{
    engines::paraformer::{ParaformerEngine, ParaformerInferenceParams, ParaformerModelParams},
    TranscriptionEngine,
};

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(duration)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let fp32 = args.iter().any(|a| a == "--fp32");
    let force_punct = args.iter().any(|a| a == "--punct");
    let disable_punct = args.iter().any(|a| a == "--no-punct");
    let punct_model_dir = args
        .iter()
        .find_map(|a| a.strip_prefix("--punct-model="))
        .map(PathBuf::from);
    let int8 = !fp32;
    let positional: Vec<&String> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .collect();

    let model_path = PathBuf::from(
        positional
            .first()
            .map(|s| s.as_str())
            .unwrap_or("models/sherpa-paraformer"),
    );
    let wav_path = PathBuf::from(
        positional
            .get(1)
            .map(|s| s.as_str())
            .unwrap_or("models/sherpa-paraformer/test_wavs/0.wav"),
    );

    let model_params = if int8 {
        ParaformerModelParams::int8()
    } else {
        ParaformerModelParams::fp32()
    };
    let auto_punct = if disable_punct {
        false
    } else if force_punct {
        true
    } else {
        ParaformerInferenceParams::default().auto_punctuation
    };

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);
    println!("Using Paraformer engine");
    println!(
        "Loading model: {:?} (quantization: {})",
        model_path,
        if int8 { "int8" } else { "fp32" }
    );
    println!("Auto punctuation: {}", auto_punct);
    if let Some(ref p) = punct_model_dir {
        println!("Punctuation model: {:?}", p);
    }

    let mut engine = ParaformerEngine::new();
    let load_start = Instant::now();
    engine.load_model_with_params(&model_path, model_params)?;
    let load_duration = load_start.elapsed();
    println!("Model loaded in {:.2?}", load_duration);

    println!("Transcribing file: {:?}", wav_path);
    let transcribe_start = Instant::now();
    let infer_params = ParaformerInferenceParams {
        auto_punctuation: auto_punct,
        punct_model_dir,
    };
    let result = engine.transcribe_file(&wav_path, Some(infer_params))?;
    let transcribe_duration = transcribe_start.elapsed();
    println!("Transcription completed in {:.2?}", transcribe_duration);

    let speedup_factor = audio_duration / transcribe_duration.as_secs_f64();
    println!(
        "Real-time speedup: {:.2}x faster than real-time",
        speedup_factor
    );
    println!("Transcription result:\n{}", result.text);

    engine.unload_model();
    Ok(())
}
