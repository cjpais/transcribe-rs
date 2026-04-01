use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::cohere_transcribe::{CohereTranscribeModel, CohereTranscribeParams};
use transcribe_rs::onnx::Quantization;

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(duration)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_path = PathBuf::from("models/cohere-transcribe");
    let wav_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("samples/jfk.wav"));

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);

    // Load
    let load_start = Instant::now();
    let mut model = CohereTranscribeModel::load(&model_path, &Quantization::default())?;
    println!("Model loaded in {:.2?}", load_start.elapsed());

    // Transcribe
    let transcribe_start = Instant::now();
    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    let result = model.transcribe_with(
        &samples,
        &CohereTranscribeParams {
            language: Some("en".to_string()),
            ..Default::default()
        },
    )?;
    let transcribe_duration = transcribe_start.elapsed();

    // Results
    println!("Transcription completed in {:.2?}", transcribe_duration);
    println!(
        "Real-time speedup: {:.2}x faster than real-time",
        audio_duration / transcribe_duration.as_secs_f64()
    );
    println!("Transcription result:");
    println!("{}", result.text);

    Ok(())
}
