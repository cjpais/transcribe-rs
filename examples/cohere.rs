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

    let model_path = PathBuf::from("models/cohere-int4-cstr");
    let wav_path = PathBuf::from("samples/dots.wav");

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);

    println!("Using Cohere ONNX engine");
    println!("Loading model: {:?}", model_path);

    let load_start = Instant::now();
    let mut model = CohereModel::load(&model_path, &Quantization::default())?;
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
