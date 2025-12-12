use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::Mutex;
use transcribe_rs::engines::whisperfile::{WhisperfileEngine, WhisperfileModelParams};
use transcribe_rs::TranscriptionEngine;

// Path to whisperfile binary - can be overridden with WHISPERFILE_BIN env var
fn binary_path() -> PathBuf {
    std::env::var("WHISPERFILE_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/whisperfile-0.9.3"))
}

// Path to model - can be overridden with WHISPERFILE_MODEL env var
fn model_path() -> PathBuf {
    std::env::var("WHISPERFILE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/ggml-small.bin"))
}

// Check if whisperfile binary exists
fn binary_available() -> bool {
    binary_path().exists()
}

// Shared engine for all tests (server started once)
static ENGINE: Lazy<Mutex<Option<WhisperfileEngine>>> = Lazy::new(|| {
    if !binary_available() {
        eprintln!(
            "Whisperfile binary not found at {:?}, skipping tests",
            binary_path()
        );
        return Mutex::new(None);
    }

    let model = model_path();
    if !model.exists() {
        eprintln!("Model not found at {:?}, skipping tests", model);
        return Mutex::new(None);
    }

    let mut engine = WhisperfileEngine::new(binary_path());

    // Use a specific port to avoid conflicts
    let params = WhisperfileModelParams {
        port: 18080,
        startup_timeout_secs: 60,
        ..Default::default()
    };

    match engine.load_model_with_params(&model, params) {
        Ok(_) => Mutex::new(Some(engine)),
        Err(e) => {
            eprintln!("Failed to start whisperfile server: {}", e);
            Mutex::new(None)
        }
    }
});

fn get_engine() -> Option<std::sync::MutexGuard<'static, Option<WhisperfileEngine>>> {
    // Use unwrap_or_else to recover from poisoned mutex (happens if a test panics)
    let guard = ENGINE.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        return None;
    }
    Some(guard)
}

macro_rules! skip_if_unavailable {
    () => {
        if !binary_available() || !model_path().exists() {
            eprintln!("Skipping test: whisperfile binary or model not available");
            return;
        }
    };
}

#[test]
fn test_jfk_transcription() {
    skip_if_unavailable!();

    let mut guard = match get_engine() {
        Some(g) => g,
        None => {
            eprintln!("Skipping test: engine not available");
            return;
        }
    };
    let engine = match guard.as_mut() {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: engine not initialized");
            return;
        }
    };

    let audio_path = PathBuf::from("samples/jfk.wav");

    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    // Whisperfile may produce slightly different output than whisper-rs
    // Normalize whitespace and check for key phrases
    let text_normalized: String = result
        .text
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");

    assert!(
        text_normalized.contains("my fellow americans"),
        "Should contain 'my fellow Americans', got: {}",
        result.text
    );
    assert!(
        text_normalized.contains("ask not what your country can do for you"),
        "Should contain 'ask not what your country can do for you', got: {}",
        result.text
    );
    assert!(
        text_normalized.contains("ask what you can do for your country"),
        "Should contain 'ask what you can do for your country', got: {}",
        result.text
    );
}

#[test]
fn test_timestamps() {
    skip_if_unavailable!();

    let mut guard = match get_engine() {
        Some(g) => g,
        None => {
            eprintln!("Skipping test: engine not available");
            return;
        }
    };
    let engine = match guard.as_mut() {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: engine not initialized");
            return;
        }
    };

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

        // End time should be greater than or equal to start time
        assert!(
            segment.end >= segment.start,
            "Segment {} end time ({}) should be >= start time ({})",
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

#[test]
fn test_transcribe_samples() {
    skip_if_unavailable!();

    let mut guard = match get_engine() {
        Some(g) => g,
        None => {
            eprintln!("Skipping test: engine not available");
            return;
        }
    };
    let engine = match guard.as_mut() {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: engine not initialized");
            return;
        }
    };

    // Read samples using the audio module
    let audio_path = PathBuf::from("samples/jfk.wav");
    let samples =
        transcribe_rs::audio::read_wav_samples(&audio_path).expect("Failed to read audio samples");

    let result = engine
        .transcribe_samples(samples, None)
        .expect("Failed to transcribe samples");

    // Verify we got a transcription
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("americans") || text_lower.contains("country"),
        "Should contain expected words from JFK speech, got: {}",
        result.text
    );
}

#[test]
fn test_language_parameter() {
    skip_if_unavailable!();

    let mut guard = match get_engine() {
        Some(g) => g,
        None => {
            eprintln!("Skipping test: engine not available");
            return;
        }
    };
    let engine = match guard.as_mut() {
        Some(e) => e,
        None => {
            eprintln!("Skipping test: engine not initialized");
            return;
        }
    };

    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with explicit English language
    let params = transcribe_rs::engines::whisperfile::WhisperfileInferenceParams {
        language: Some("en".to_string()),
        ..Default::default()
    };

    let result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe with language parameter");

    // Should still get valid transcription
    assert!(!result.text.is_empty(), "Transcription should not be empty");

    let text_lower = result.text.to_lowercase();
    assert!(
        text_lower.contains("country"),
        "Should transcribe English correctly, got: {}",
        result.text
    );
}
