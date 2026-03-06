use std::path::PathBuf;

use transcribe_rs::{
    engines::qwen3::{Qwen3Engine, Qwen3ModelParams},
    TranscriptionEngine,
};

#[test]
fn test_qwen3_transcribe() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_path = PathBuf::from("models/qwen3-asr-0.6b");
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return Ok(());
    }
    if !wav_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", wav_path);
        return Ok(());
    }

    let mut engine = Qwen3Engine::new();
    engine.load_model_with_params(&model_path, Qwen3ModelParams::default())?;

    let result = engine.transcribe_file(&wav_path, None)?;

    assert!(!result.text.is_empty(), "Transcription should not be empty");
    println!("Transcription: {}", result.text);

    engine.unload_model();

    Ok(())
}
