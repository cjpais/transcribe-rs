use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::{
    engines::nemotron_streaming::{NemotronStreamingEngine, CHUNK_SIZE},
    StreamingTranscriptionEngine,
};

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(duration)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut engine = NemotronStreamingEngine::new();
    let model_path = PathBuf::from("models/nemotron-speech-streaming-en-0.6b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    let audio_duration = get_audio_duration(&wav_path)?;
    println!("Audio duration: {:.2}s", audio_duration);

    println!("Using Nemotron streaming engine");
    println!("Loading model: {:?}", model_path);

    let load_start = Instant::now();
    engine.load_model(&model_path)?;
    let load_duration = load_start.elapsed();
    println!("Model loaded in {:.2?}", load_duration);

    println!("Transcribing file: {:?}", wav_path);
    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;

    let transcribe_start = Instant::now();
    for chunk in samples.chunks(CHUNK_SIZE) {
        let text = engine.push_samples(chunk)?;
        if !text.is_empty() {
            print!("{}", text);
        }
    }
    let transcribe_duration = transcribe_start.elapsed();
    println!();

    println!("Transcription completed in {:.2?}", transcribe_duration);

    let speedup_factor = audio_duration / transcribe_duration.as_secs_f64();
    println!(
        "Real-time speedup: {:.2}x faster than real-time",
        speedup_factor
    );

    let transcript = engine.get_transcript();
    println!("Full transcript: {}", transcript);

    engine.unload_model();

    Ok(())
}
