use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::Mutex;
use transcribe_rs::engines::whisper::{WhisperEngine, WhisperInferenceParams};
use transcribe_rs::TranscriptionEngine;

// Shared model loaded once for all tests
static MODEL_ENGINE: Lazy<Mutex<WhisperEngine>> = Lazy::new(|| {
    let mut engine = WhisperEngine::new();
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");
    Mutex::new(engine)
});

fn get_engine() -> std::sync::MutexGuard<'static, WhisperEngine> {
    MODEL_ENGINE.lock().expect("Failed to lock engine")
}

#[test]
fn test_jfk_transcription() {
    let mut engine = get_engine();

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with default params
    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    let expected = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}

#[test]
fn test_prompt_product_names() {
    let mut engine = get_engine();

    let audio_path = PathBuf::from("samples/product_names.wav");

    // Baseline transcription with no prompt
    let baseline_result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe without prompt");

    println!("\n=== Baseline Transcription (no prompt) ===");
    println!("{}", baseline_result.text);

    // With glossary prompt - should influence transcription to use specified spellings
    let glossary_prompt = "QuirkQuid Quill Inc, P3-Quattro, O3-Omni, B3-BondX, E3-Equity, W3-WrapZ, O2-Outlier, U3-UniFund, M3-Mover";
    let params = WhisperInferenceParams {
        initial_prompt: Some(glossary_prompt.to_string()),
        ..Default::default()
    };
    let prompted_result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe with prompt");

    println!("\n=== Transcription with Glossary Prompt ===");
    println!("{}", prompted_result.text);

    // The main assertion: prompting should produce different output
    assert_ne!(
        baseline_result.text, prompted_result.text,
        "Prompt should influence transcription output"
    );

    // Verify prompt influenced the output - should contain hyphenated product names from the glossary
    assert!(
        prompted_result.text.contains("P3-Quattro") || prompted_result.text.contains("O3-Omni"),
        "Prompted output should contain hyphenated product names from glossary"
    );

    // Baseline should NOT have the hyphenated format
    assert!(
        !baseline_result.text.contains("P3-Quattro"),
        "Baseline should not contain prompted spelling"
    );
}

#[test]
fn test_timestamps() {
    let mut engine = get_engine();

    let audio_path = PathBuf::from("samples/jfk.wav");

    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    // Verify segments are returned
    assert!(
        result.segments.is_some(),
        "Transcription should return segments"
    );

    let segments = result.segments.unwrap();
    assert!(!segments.is_empty(), "Segments should not be empty");

    // Verify timestamp properties
    for (i, segment) in segments.iter().enumerate() {
        // Start time should be non-negative
        assert!(
            segment.start >= 0.0,
            "Segment {} start time should be non-negative, got {}",
            i,
            segment.start
        );

        // End time should be greater than start time
        assert!(
            segment.end > segment.start,
            "Segment {} end time ({}) should be greater than start time ({})",
            i,
            segment.end,
            segment.start
        );

        // Segment should have text
        assert!(
            !segment.text.trim().is_empty(),
            "Segment {} should have non-empty text",
            i
        );
    }

    // Verify segments are in chronological order
    for i in 1..segments.len() {
        assert!(
            segments[i].start >= segments[i - 1].start,
            "Segments should be in chronological order"
        );
    }

    // Verify the audio duration is reasonable (JFK clip is ~11 seconds)
    let last_segment = segments.last().unwrap();
    assert!(
        last_segment.end > 10.0 && last_segment.end < 15.0,
        "Last segment end time should be around 11 seconds for JFK clip, got {}",
        last_segment.end
    );
}
