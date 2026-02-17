use std::path::PathBuf;
use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::engines::nemotron_streaming::{NemotronStreamingEngine, CHUNK_SIZE};
use transcribe_rs::StreamingTranscriptionEngine;

const MODEL_PATH: &str = "models/nemotron-speech-streaming-en-0.6b";

#[test]
fn test_jfk_streaming_transcription() {
    let model_path = PathBuf::from(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return;
    }

    let audio_path = PathBuf::from("samples/jfk.wav");
    if !audio_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", audio_path);
        return;
    }

    let mut engine = NemotronStreamingEngine::new();
    engine
        .load_model(&model_path)
        .expect("Failed to load Nemotron streaming model");

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");

    // Feed audio in chunk-sized increments, matching real-time streaming usage
    for chunk in samples.chunks(CHUNK_SIZE) {
        engine
            .push_samples(chunk)
            .expect("push_samples should not fail");
    }

    let transcript = engine.get_transcript();
    let lower = transcript.to_lowercase();

    assert!(
        !transcript.is_empty(),
        "Streaming transcription should produce non-empty output"
    );
    assert!(
        lower.contains("my fellow americans"),
        "Transcript should contain 'my fellow americans', got: {}",
        transcript
    );
    assert!(
        lower.contains("ask not what your country can do for you"),
        "Transcript should contain 'ask not what your country can do for you', got: {}",
        transcript
    );

    engine.unload_model();
}

#[test]
fn test_reset_clears_state() {
    let model_path = PathBuf::from(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return;
    }

    let audio_path = PathBuf::from("samples/jfk.wav");
    if !audio_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", audio_path);
        return;
    }

    let mut engine = NemotronStreamingEngine::new();
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");

    // Feed some audio
    for chunk in samples.chunks(CHUNK_SIZE).take(5) {
        let _ = engine.push_samples(chunk);
    }

    let before_reset = engine.get_transcript();
    assert!(!before_reset.is_empty(), "Should have transcript before reset");

    engine.reset();
    let after_reset = engine.get_transcript();

    assert!(
        after_reset.is_empty(),
        "reset() should clear accumulated transcript. Before: {} chars, after: {} chars",
        before_reset.len(),
        after_reset.len()
    );
}

#[test]
fn test_push_samples_returns_incremental_text() {
    let model_path = PathBuf::from(MODEL_PATH);
    if !model_path.exists() {
        eprintln!("Skipping test: model not found at {:?}", model_path);
        return;
    }

    let audio_path = PathBuf::from("samples/jfk.wav");
    if !audio_path.exists() {
        eprintln!("Skipping test: audio not found at {:?}", audio_path);
        return;
    }

    let mut engine = NemotronStreamingEngine::new();
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");

    // Collect incremental text from push_samples
    let mut incremental_parts: Vec<String> = Vec::new();
    for chunk in samples.chunks(CHUNK_SIZE) {
        let text = engine.push_samples(chunk).expect("push_samples failed");
        if !text.is_empty() {
            incremental_parts.push(text);
        }
    }

    // Concatenation of incremental returns should match get_transcript()
    // after trimming — the sentencepiece tokenizer may emit a leading space
    // on the first token that get_transcript() strips.
    let concatenated: String = incremental_parts.concat();
    let full_transcript = engine.get_transcript();

    assert_eq!(
        concatenated.trim(), full_transcript.trim(),
        "Concatenated incremental text should match get_transcript()"
    );
}

#[test]
fn test_new_engine_empty_transcript() {
    let engine = NemotronStreamingEngine::new();
    assert!(
        engine.get_transcript().is_empty(),
        "Fresh engine should return empty transcript"
    );
}

#[test]
fn test_push_samples_without_model() {
    let mut engine = NemotronStreamingEngine::new();
    let result = engine.push_samples(&[0.0; 8960]);
    assert!(
        result.is_err(),
        "push_samples without a loaded model should return an error"
    );
}
