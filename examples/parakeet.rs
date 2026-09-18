use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity};
use transcribe_rs::onnx::Quantization;

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(duration)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut args = std::env::args_os().skip(1);
    let model_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/parakeet-tdt-0.6b-v3-int8"));
    let wav_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("samples/dots.wav"));
    if args.next().is_some() {
        return Err("usage: parakeet [MODEL_DIR] [AUDIO.wav]".into());
    }

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);

    println!("Using Parakeet engine");
    println!("Loading model: {:?}", model_path);

    let load_start = Instant::now();
    let mut model = ParakeetModel::load(&model_path, &Quantization::Int8)?;
    let load_duration = load_start.elapsed();
    println!("Model loaded in {:.2?}", load_duration);

    println!("Transcribing file: {:?}", wav_path);
    let transcribe_start = Instant::now();

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    let result = model.transcribe_with(
        &samples,
        &ParakeetParams {
            timestamp_granularity: Some(TimestampGranularity::Segment),
            ..Default::default()
        },
    )?;
    let transcribe_duration = transcribe_start.elapsed();
    println!("Transcription completed in {:.2?}", transcribe_duration);

    let speedup_factor = audio_duration / transcribe_duration.as_secs_f64();
    println!(
        "Real-time speedup: {:.2}x faster than real-time",
        speedup_factor
    );

    println!("Transcription result:");
    println!("{}", result.text);

    if let Some(segments) = result.segments {
        println!("\nSegments:");
        for segment in segments {
            println!(
                "[{:.2}s - {:.2}s]: {}",
                segment.start, segment.end, segment.text
            );
        }
    }

    Ok(())
}
