use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::cohere::CohereModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(duration)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let quant = args.get(1).map(|s| s.as_str()).unwrap_or("int8");

    let (model_path, quantization) = match quant {
        "int4" => ("models/cohere-int4", Quantization::Int4),
        "int8" => ("models/cohere-int8", Quantization::Int8),
        other => {
            eprintln!("Unknown quantization: {other}. Use 'int4' or 'int8'.");
            std::process::exit(1);
        }
    };

    let model_path = PathBuf::from(model_path);
    let wav_path = PathBuf::from("samples/dots.wav");

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);

    println!("Using Cohere ONNX engine ({quant})");
    println!("Loading model: {:?}", model_path);

    let load_start = Instant::now();
    let mut model = CohereModel::load(&model_path, &quantization)?;
    let load_duration = load_start.elapsed();
    println!("Model loaded in {:.2?}", load_duration);

    println!("Transcribing file: {:?}", wav_path);
    let transcribe_start = Instant::now();
    let result = model.transcribe_file(&wav_path, &transcribe_rs::TranscribeOptions::default())?;
    let transcribe_duration = transcribe_start.elapsed();
    println!("Transcription completed in {:.2?}", transcribe_duration);

    let speedup_factor = audio_duration / transcribe_duration.as_secs_f64();
    println!(
        "Real-time speedup: {:.2}x faster than real-time",
        speedup_factor
    );
    println!("Transcription result:\n{}", result.text);

    Ok(())
}
