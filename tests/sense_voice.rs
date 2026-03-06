use std::path::PathBuf;

use transcribe_rs::onnx::sense_voice::SenseVoiceModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

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

    let mut model = SenseVoiceModel::load(&model_path, &Quantization::FP32)?;

    let result = model.transcribe_file(&wav_path, None)?;

    assert!(!result.text.is_empty(), "Transcription should not be empty");
    println!("Transcription: {}", result.text);

    Ok(())
}
