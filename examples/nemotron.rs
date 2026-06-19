use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::onnx::nemotron::{NemotronModel, NemotronParams};
use transcribe_rs::onnx::Quantization;

/// Usage: `cargo run --example nemotron --features onnx -- [MODEL_DIR] [WAV] [LANG]`
///
/// MODEL_DIR defaults to `models/nemotron-3.5-asr-streaming-0.6b-onnx`,
/// WAV to `samples/jfk.wav`. LANG (e.g. `en-US`, `es-ES`, `auto`) applies only
/// to the multilingual variant.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let model_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "models/nemotron-3.5-asr-streaming-0.6b-onnx".to_string()),
    );
    let wav_path = PathBuf::from(args.next().unwrap_or_else(|| "samples/jfk.wav".to_string()));
    let language = args.next();

    // Use the int8 weights if present, otherwise FP32.
    let quant = if model_dir.join("encoder.int8.onnx").exists() {
        Quantization::Int8
    } else {
        Quantization::FP32
    };

    let load_start = Instant::now();
    let mut model = NemotronModel::load(&model_dir, &quant)?;
    println!(
        "Loaded {:?} Nemotron ({:?}) in {:.2?}",
        model.mode(),
        quant,
        load_start.elapsed()
    );

    let samples = transcribe_rs::audio::read_wav_samples(&wav_path)?;
    let audio_seconds = samples.len() as f64 / 16_000.0;

    let transcribe_start = Instant::now();
    let result = model.transcribe_with(&samples, &NemotronParams { language })?;
    let elapsed = transcribe_start.elapsed();

    println!(
        "Transcribed {:.2}s of audio in {:.2?} ({:.1}x real-time)",
        audio_seconds,
        elapsed,
        audio_seconds / elapsed.as_secs_f64()
    );
    println!("{}", result.text);

    Ok(())
}
