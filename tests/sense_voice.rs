use std::path::PathBuf;

use transcribe_rs::onnx::{Engine, InferenceParams, Model};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_sense_voice_transcribe() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_path = PathBuf::from("models/sense-voice");
    let wav_path = PathBuf::from("samples/dots.wav");

    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return Ok(());
    }
    if !wav_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", wav_path);
        return Ok(());
    }

    let mut engine = Engine::new();
    engine.load(&model_path, Model::sense_voice())?;

    let params = InferenceParams::default();
    let result = engine.transcribe_file(&wav_path, Some(params))?;

    assert!(!result.text.is_empty(), "Transcription should not be empty");
    println!("Transcription: {}", result.text);

    engine.unload_model();

    Ok(())
}
