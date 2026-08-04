use std::env;
use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::qwen3::Qwen3Model;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model_dir> <audio.wav>", args[0]);
        std::process::exit(1);
    }

    let model_dir = PathBuf::from(&args[1]);
    let wav_path = PathBuf::from(&args[2]);

    // Read audio
    let reader = hound::WavReader::open(&wav_path)?;
    let spec = reader.spec();
    let audio_duration = reader.duration() as f64 / spec.sample_rate as f64;
    println!("Audio: {:.2}s", audio_duration);

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;

    // Load model
    println!("Loading Qwen3-ASR model from {:?}", model_dir);
    let load_start = Instant::now();
    let mut model = Qwen3Model::load(&model_dir, &Quantization::default())?;
    println!("Model loaded in {:.2?}", load_start.elapsed());

    // Transcribe
    let transcribe_start = Instant::now();
    let result = model.transcribe(&samples, &transcribe_rs::TranscribeOptions::default())?;
    let transcribe_duration = transcribe_start.elapsed();

    println!("Transcription: {}", result.text);
    println!("Completed in {:.2?}", transcribe_duration);
    let speedup = audio_duration / transcribe_duration.as_secs_f64();
    println!("Real-time factor: {:.2}x", speedup);

    Ok(())
}
