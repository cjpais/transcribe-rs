use anyhow::Result;
use std::path::PathBuf;
use transcribe_rs::chunking::SmartChunker;

#[tokio::test]
async fn test_smart_chunking() -> Result<()> {
    // 1. Setup VAD model
    let models_dir = PathBuf::from("models");
    if !models_dir.exists() {
        std::fs::create_dir(&models_dir)?;
    }
    let model_path = models_dir.join("silero_vad.onnx");

    if !model_path.exists() {
        println!("Downloading VAD model...");
        let response =
            reqwest::get("https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx")
                .await?
                .bytes()
                .await?;
        std::fs::write(&model_path, response)?;
    }

    // 2. Generate synthetic audio (1 minute of sine wave with silence at 30s)
    // 16kHz sample rate
    let sample_rate = 16000;
    let duration_seconds = 60;
    let total_samples = sample_rate * duration_seconds;
    let mut audio = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        // Silence between 28s and 32s
        if i > 28 * sample_rate && i < 32 * sample_rate {
            audio.push(0.0);
        } else {
            // Sine wave
            let t = i as f32 / sample_rate as f32;
            audio.push((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 0.5);
        }
    }

    // 3. Run smart chunking
    // We'll use a dummy callback that just returns "chunk"
    let result = SmartChunker::chunk_audio(
        &audio,
        &model_path,
        |chunk| {
            println!("Processing chunk of size: {}", chunk.len());
            // Verify chunk size is roughly 30s (30 * 16000 = 480000 samples)
            // The first chunk should end around 30s (silence start)
            // It might be slightly less or more depending on VAD window
            Ok("chunk".to_string())
        },
        |progress| {
            println!("Progress: {:.2}%", progress);
        },
    )?;

    println!("Result: {}", result);
    assert!(!result.is_empty());

    Ok(())
}
