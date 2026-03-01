use std::path::PathBuf;

use transcribe_rs::onnx::{Engine, Model};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_gigaam_transcribe() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_path = PathBuf::from("models/v3_e2e_ctc.int8.onnx");
    let wav_path = PathBuf::from("samples/russian.wav");

    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return Ok(());
    }
    if !wav_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", wav_path);
        return Ok(());
    }

    let mut engine = Engine::new();
    engine.load(&model_path, Model::gigaam())?;

    let result = engine.transcribe_file(&wav_path, None)?;

    let expected = "Проверка связи.";
    assert_eq!(
        result.text, expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected, result.text
    );

    engine.unload_model();

    Ok(())
}
