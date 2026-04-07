use std::path::PathBuf;
use std::time::Instant;
use transcribe_rs::burn_parakeet::BurnParakeetModel;
use transcribe_rs::{SpeechModel, TranscribeOptions};

fn main() {
    env_logger::init();

    let model_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/parakeet-tdt-0.6b-v3"));

    let wav_path = std::env::args()
        .nth(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("samples/dots.wav"));

    let t0 = Instant::now();
    println!("Loading model from {:?}...", model_dir);
    let mut model = BurnParakeetModel::load(&model_dir).expect("Failed to load model");
    let load_time = t0.elapsed();
    println!(
        "Model loaded in {:.2}s: {}",
        load_time.as_secs_f64(),
        model.capabilities().name
    );

    println!("Transcribing {:?}...", wav_path);
    let t1 = Instant::now();
    let result = model
        .transcribe_file(&wav_path, &TranscribeOptions::default())
        .expect("Transcription failed");
    let transcribe_time = t1.elapsed();

    // Get audio duration for RTF calculation
    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).unwrap();
    let audio_duration = samples.len() as f64 / 16000.0;

    println!("\n--- Results ---");
    println!("Text: {}", result.text);
    if let Some(segments) = &result.segments {
        if !segments.is_empty() {
            println!("\nSegments:");
            for seg in segments {
                println!("  [{:.2}s - {:.2}s] {}", seg.start, seg.end, seg.text);
            }
        }
    }
    println!("\n--- Timing ---");
    println!("Audio duration:    {:.2}s", audio_duration);
    println!("Model load:        {:.2}s", load_time.as_secs_f64());
    println!("Transcription:     {:.2}s", transcribe_time.as_secs_f64());
    println!(
        "Real-time factor:  {:.3}x",
        transcribe_time.as_secs_f64() / audio_duration
    );
}
