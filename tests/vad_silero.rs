mod common;

use std::path::PathBuf;

use transcribe_rs::vad::{SileroVad, SmoothedVad, Vad};

#[test]
fn test_silero_vad_detects_speech_in_audio() {
    let model_path = PathBuf::from("models/silero_vad_v4.onnx");
    let wav_path = PathBuf::from("samples/dots.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return;
    }

    let mut vad = SileroVad::new(&model_path, 0.3).expect("Failed to load Silero VAD");
    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).expect("Failed to read WAV");

    let frame_size = vad.frame_size();
    assert_eq!(frame_size, 480);

    let mut speech_frames = 0;
    let mut total_frames = 0;

    for frame in samples.chunks_exact(frame_size) {
        total_frames += 1;
        if vad.is_speech(frame).unwrap() {
            speech_frames += 1;
        }
    }

    println!(
        "Silero VAD: {speech_frames}/{total_frames} frames detected as speech ({:.1}%)",
        speech_frames as f32 / total_frames as f32 * 100.0
    );

    // dots.wav has speech — should detect a meaningful amount
    assert!(speech_frames > 0, "Should detect some speech");
    assert!(
        speech_frames < total_frames,
        "Should not classify everything as speech"
    );
}

#[test]
fn test_silero_vad_silence_detection() {
    let model_path = PathBuf::from("models/silero_vad_v4.onnx");

    if !common::require_paths(&[&model_path]) {
        return;
    }

    let mut vad = SileroVad::new(&model_path, 0.3).expect("Failed to load Silero VAD");

    // Pure silence should not be detected as speech
    let silence = vec![0.0f32; 480];
    let is_speech = vad.is_speech(&silence).unwrap();
    assert!(!is_speech, "Silence should not be classified as speech");
}

#[test]
fn test_silero_vad_reset() {
    let model_path = PathBuf::from("models/silero_vad_v4.onnx");
    let wav_path = PathBuf::from("samples/dots.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return;
    }

    let mut vad = SileroVad::new(&model_path, 0.3).expect("Failed to load Silero VAD");
    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).expect("Failed to read WAV");

    // Feed some frames to build up LSTM state
    for frame in samples.chunks_exact(480).take(10) {
        let _ = vad.is_speech(frame);
    }

    // Reset should clear LSTM state
    vad.reset();

    // After reset, silence should still be silence
    let silence = vec![0.0f32; 480];
    assert!(!vad.is_speech(&silence).unwrap());
}

#[test]
fn test_smoothed_silero_vad() {
    let model_path = PathBuf::from("models/silero_vad_v4.onnx");
    let wav_path = PathBuf::from("samples/dots.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return;
    }

    let silero = SileroVad::new(&model_path, 0.3).expect("Failed to load Silero VAD");
    let mut vad = SmoothedVad::new(Box::new(silero), 15, 15, 2);

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).expect("Failed to read WAV");

    let mut speech_frames = 0;
    let mut total_frames = 0;

    for frame in samples.chunks_exact(480) {
        total_frames += 1;
        if vad.is_speech(frame).unwrap() {
            speech_frames += 1;
        }
    }

    println!(
        "SmoothedVad(Silero): {speech_frames}/{total_frames} frames detected as speech ({:.1}%)",
        speech_frames as f32 / total_frames as f32 * 100.0
    );

    // Smoothed should still detect speech, but with fewer transients
    assert!(speech_frames > 0, "Should detect some speech");
}

#[test]
fn test_smoothed_silero_vad_chunk_count() {
    let model_path = PathBuf::from("models/silero_vad_v4.onnx");
    let wav_path = PathBuf::from("samples/dots.wav");

    if !common::require_paths(&[&model_path, &wav_path]) {
        return;
    }

    let silero = SileroVad::new(&model_path, 0.3).expect("Failed to load Silero VAD");
    let mut vad = SmoothedVad::new(Box::new(silero), 15, 15, 2);

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path).expect("Failed to read WAV");

    let mut chunks = 0;
    let mut was_speech = false;

    for frame in samples.chunks_exact(480) {
        let is_speech = vad.is_speech(frame).unwrap();
        if was_speech && !is_speech {
            // speech → silence transition = end of a chunk
            chunks += 1;
        }
        was_speech = is_speech;
    }
    // If audio ends during speech, that's a final chunk
    if was_speech {
        chunks += 1;
    }

    println!("SmoothedVad(Silero) on dots.wav: {chunks} speech chunks");
    assert!(chunks >= 1, "Should detect at least one speech chunk");
}
