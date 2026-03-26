#![cfg(feature = "onnx")]

mod common;

use std::path::PathBuf;
use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::onnx::canary::CanaryModel;
use transcribe_rs::onnx::parakeet::ParakeetModel;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::SpeechModel;

#[test]
fn transcribe_fixed_chunk_002_parakeet() {
    let model_path = PathBuf::from("models/parakeet-tdt-0.6b-v3-int8");
    let chunk_path = PathBuf::from("target/debug_chunks/fixed/fixed_chunk_002.wav");

    if !common::require_paths(&[&model_path, &chunk_path]) {
        return;
    }

    let mut model =
        ParakeetModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    let samples = read_wav_samples(&chunk_path).expect("Failed to read chunk");
    println!(
        "fixed_chunk_002: {} samples, {:.2}s",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    let result = model
        .transcribe(&samples, &transcribe_rs::TranscribeOptions::default())
        .expect("Transcription failed");

    println!("parakeet result: \"{}\"", result.text);
}

#[test]
fn transcribe_fixed_chunk_002_canary() {
    let model_path = PathBuf::from("models/canary-1b-v2");
    let chunk_path = PathBuf::from("target/debug_chunks/fixed/fixed_chunk_002.wav");

    if !common::require_paths(&[&model_path, &chunk_path]) {
        return;
    }

    let mut model =
        CanaryModel::load(&model_path, &Quantization::Int8).expect("Failed to load model");

    let samples = read_wav_samples(&chunk_path).expect("Failed to read chunk");
    println!(
        "fixed_chunk_002: {} samples, {:.2}s",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    let result = model
        .transcribe(&samples, &transcribe_rs::TranscribeOptions::default())
        .expect("Transcription failed");

    println!("canary result: \"{}\"", result.text);
}
