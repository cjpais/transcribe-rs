use std::path::PathBuf;
use transcribe_rs::engines::parakeet::{ParakeetEngine, ParakeetModelParams};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_jfk_transcription() {
    let mut engine = ParakeetEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/parakeet-tdt-0.6b-v3-int8");
    engine
        .load_model_with_params(&model_path, ParakeetModelParams::int8())
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with default params
    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    let expected = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}

#[test]
fn test_timestamps() {
    let mut engine = ParakeetEngine::new();

    let model_path = PathBuf::from("models/parakeet-tdt-0.6b-v3-int8");
    engine
        .load_model_with_params(&model_path, ParakeetModelParams::int8())
        .expect("Failed to load model");

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

    // Parakeet returns token-level segments, so we expect many segments
    assert!(
        segments.len() > 10,
        "Parakeet should return multiple token-level segments, got {}",
        segments.len()
    );

    // Verify timestamp properties
    for (i, segment) in segments.iter().enumerate() {
        // Start time should be non-negative
        assert!(
            segment.start >= 0.0,
            "Segment {} start time should be non-negative, got {}",
            i,
            segment.start
        );

        // End time should be >= start time (can be equal for very short tokens)
        assert!(
            segment.end >= segment.start,
            "Segment {} end time ({}) should be >= start time ({})",
            i,
            segment.end,
            segment.start
        );

        // Segment should have text
        assert!(
            !segment.text.is_empty(),
            "Segment {} should have non-empty text",
            i
        );
    }

    // Verify segments are in chronological order
    for i in 1..segments.len() {
        assert!(
            segments[i].start >= segments[i - 1].start,
            "Segments should be in chronological order: segment {} starts at {} but segment {} starts at {}",
            i,
            segments[i].start,
            i - 1,
            segments[i - 1].start
        );
    }

    // Verify the audio duration is reasonable (JFK clip is ~11 seconds)
    let last_segment = segments.last().unwrap();
    assert!(
        last_segment.end > 10.0 && last_segment.end < 15.0,
        "Last segment end time should be around 11 seconds for JFK clip, got {}",
        last_segment.end
    );

    // Verify first segment starts near the beginning
    let first_segment = segments.first().unwrap();
    assert!(
        first_segment.start < 1.0,
        "First segment should start near the beginning, got {}",
        first_segment.start
    );
}
