use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::TranscriptionEngine;

fn get_audio_duration(path: &PathBuf) -> Result<f64, Box<dyn std::error::Error>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let duration = reader.duration() as f64 / spec.sample_rate as f64;
    Ok(duration)
}

fn bench_engine(
    name: &str,
    engine: &mut dyn FnMut(&PathBuf) -> Result<String, Box<dyn std::error::Error>>,
    wavs: &[PathBuf],
) {
    println!("  [{name}]");
    for wav in wavs {
        let audio_dur = get_audio_duration(wav).unwrap();
        let start = Instant::now();
        match engine(wav) {
            Ok(text) => {
                let dur = start.elapsed();
                let speed = audio_dur / dur.as_secs_f64();
                println!(
                    "    {} ({:.1}s) -> {:.2?} ({:.2}x realtime)",
                    wav.file_name().unwrap().to_string_lossy(),
                    audio_dur,
                    dur,
                    speed,
                );
                println!("    Text: {}", text);
            }
            Err(e) => {
                println!(
                    "    {} -> ERROR: {}",
                    wav.file_name().unwrap().to_string_lossy(),
                    e,
                );
            }
        }
        println!();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <whisper-model.bin> <qwen-model-dir> <audio1.wav> [audio2.wav ...]",
            args[0]
        );
        std::process::exit(1);
    }

    let whisper_model = PathBuf::from(&args[1]);
    let qwen_model = PathBuf::from(&args[2]);
    let wavs: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();

    println!("==========================================================");
    println!("  Whisper vs Qwen3-ASR Benchmark");
    println!("  Whisper model: {}", whisper_model.display());
    println!("  Qwen model:    {}", qwen_model.display());
    println!("  Audio files:   {}", wavs.len());
    println!("==========================================================");
    println!();

    // Load Whisper
    println!("Loading Whisper...");
    let load_start = Instant::now();
    let mut whisper =
        transcribe_rs::engines::whisper::WhisperEngine::new();
    whisper.load_model(&whisper_model)?;
    println!("  Whisper loaded in {:.2?}", load_start.elapsed());

    // Load Qwen
    println!("Loading Qwen3-ASR...");
    let load_start = Instant::now();
    let mut qwen =
        transcribe_rs::engines::qwen_asr::QwenAsrEngine::new();
    qwen.load_model(&qwen_model)?;
    println!("  Qwen3-ASR loaded in {:.2?}", load_start.elapsed());
    println!();

    bench_engine("Whisper", &mut |wav| {
        let result = whisper.transcribe_file(wav, None)?;
        Ok(result.text)
    }, &wavs);

    bench_engine("Qwen3-ASR", &mut |wav| {
        let result = qwen.transcribe_file(wav, None)?;
        Ok(result.text)
    }, &wavs);

    whisper.unload_model();
    qwen.unload_model();

    Ok(())
}
