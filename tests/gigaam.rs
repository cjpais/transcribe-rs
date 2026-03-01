use std::path::PathBuf;

use transcribe_rs::{engines::gigaam::GigaAMEngine, TranscriptionEngine};

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

    let mut engine = GigaAMEngine::new();
    engine.load_model(&model_path)?;

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
