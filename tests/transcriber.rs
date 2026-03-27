mod common;

use std::path::PathBuf;

use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::onnx::parakeet::ParakeetModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::transcriber::{
    EnergyAdaptiveChunked, EnergyAdaptiveConfig, Transcriber, VadChunked, VadChunkedConfig,
};
use transcribe_rs::vad::{SileroVad, SmoothedVad};
use transcribe_rs::{SpeechModel, TranscribeOptions};

#[test]
fn compare_oneshot_vs_chunked() {
    let _ = env_logger::builder().is_test(true).try_init();

    let model_path = PathBuf::from("models/parakeet-tdt-0.6b-v3-int8");
    let vad_path = PathBuf::from("models/silero_vad_v4.onnx");
    let audio_path = PathBuf::from("samples/dots.wav");

    if !common::require_paths(&[&model_path, &vad_path, &audio_path]) {
        return;
    }

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");
    let options = TranscribeOptions::default();

    let mut model =
        ParakeetModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    // --- One-shot ---
    let oneshot = model
        .transcribe(&samples, &options)
        .expect("One-shot transcription failed");

    // --- EnergyAdaptiveChunked (10s target, 3s search window) ---
    let energy_config = EnergyAdaptiveConfig {
        target_chunk_secs: 10.0,
        search_window_secs: 3.0,
        padding_secs: 0.5,
        ..Default::default()
    };
    let energy = EnergyAdaptiveChunked::new(energy_config, options.clone())
        .transcribe(&mut model, &samples)
        .expect("EnergyAdaptiveChunked transcription failed");

    // --- VadChunked (SileroVad) ---
    let silero = SileroVad::new(&vad_path, 0.5).expect("Failed to load Silero VAD");
    let vad = SmoothedVad::new(Box::new(silero), 15, 20, 2);
    let vad_config = VadChunkedConfig {
        padding_secs: 0.2,
        ..Default::default()
    };
    let vad_result = VadChunked::new(Box::new(vad), vad_config, options.clone())
        .transcribe(&mut model, &samples)
        .expect("VadChunked transcription failed");

    println!("\n{}", "=".repeat(60));
    println!("ONE-SHOT ({} chars):", oneshot.text.len());
    println!("{}", oneshot.text);
    println!();
    println!("ENERGY ADAPTIVE 10s CHUNKS ({} chars):", energy.text.len());
    println!("{}", energy.text);
    println!();
    println!("VAD CHUNKED ({} chars):", vad_result.text.len());
    println!("{}", vad_result.text);
    println!("{}", "=".repeat(60));

    // All three should produce non-empty output
    assert!(!oneshot.text.is_empty(), "one-shot produced empty text");
    assert!(
        !energy.text.is_empty(),
        "energy adaptive produced empty text"
    );
    assert!(
        !vad_result.text.is_empty(),
        "vad chunked produced empty text"
    );
}

/// Simulate streaming by feeding audio in small increments (30ms frames)
/// with short chunk windows (3s). Tests EnergyAdaptiveChunked under
/// streaming-like conditions.
#[test]
fn compare_streaming_energy_adaptive() {
    let _ = env_logger::builder().is_test(true).try_init();

    let model_path = PathBuf::from("models/parakeet-tdt-0.6b-v3-int8");
    let audio_path = PathBuf::from("samples/dots.wav");

    if !common::require_paths(&[&model_path, &audio_path]) {
        return;
    }

    let samples = read_wav_samples(&audio_path).expect("Failed to read audio");
    let options = TranscribeOptions::default();

    let mut model =
        ParakeetModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    // --- One-shot baseline ---
    let oneshot = model
        .transcribe(&samples, &options)
        .expect("One-shot transcription failed");

    // --- EnergyAdaptiveChunked 3s target, 1s search, simulated streaming ---
    let energy_config = EnergyAdaptiveConfig {
        target_chunk_secs: 3.0,
        search_window_secs: 1.0,
        padding_secs: 0.3,
        ..Default::default()
    };
    let mut energy = EnergyAdaptiveChunked::new(energy_config, options.clone());

    let frame_size = 480; // 30ms at 16kHz
    let mut energy_intermediate: Vec<String> = Vec::new();
    for chunk in samples.chunks(frame_size) {
        let results = energy.feed(&mut model, chunk).expect("feed failed");
        for r in &results {
            energy_intermediate.push(r.text.clone());
        }
    }
    let energy_result = energy.finish(&mut model).expect("finish failed");

    println!("\n{}", "=".repeat(60));
    println!("SIMULATED STREAMING (30ms frames, 3s chunks)");
    println!("{}", "=".repeat(60));

    println!("\nONE-SHOT BASELINE ({} chars):", oneshot.text.len());
    println!("{}", oneshot.text);

    println!("\nENERGY ADAPTIVE 3s - intermediate chunks:");
    for (i, text) in energy_intermediate.iter().enumerate() {
        println!("  chunk {}: \"{}\"", i, text.trim());
    }
    println!(
        "ENERGY ADAPTIVE 3s MERGED ({} chars):",
        energy_result.text.len()
    );
    println!("{}", energy_result.text);

    println!("{}", "=".repeat(60));

    assert!(
        !energy_result.text.is_empty(),
        "energy adaptive produced empty text"
    );
}
