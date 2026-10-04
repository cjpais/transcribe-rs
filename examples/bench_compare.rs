//! Side-by-side inference timing benchmark for Qwen3-ASR and Parakeet TDT.
//!
//! Loads one or both models, runs warmup passes then N timed passes on the
//! same audio file, and prints a summary table with load time, mean/min/max
//! inference time, and real-time factor (RTF) for each engine.
//!
//! # Usage
//!
//! ```sh
//! cargo run --example bench_compare --features qwen3 --release -- \
//!     --qwen3 models/qwen3-asr-0.6b --audio samples/jfk.wav
//! ```
//!
//! Both engines:
//!
//! ```sh
//! cargo run --example bench_compare --features "qwen3,onnx" --release -- \
//!     --qwen3 models/qwen3-asr-0.6b \
//!     --parakeet models/parakeet-tdt-0.6b-v3-int8 \
//!     --audio samples/jfk.wav
//! ```
//!
//! # Flags
//!
//! | Flag | Default | Description |
//! |---|---|---|
//! | `--qwen3 <path>` | — | Path to Qwen3-ASR model directory |
//! | `--parakeet <path>` | — | Path to Parakeet-TDT model directory |
//! | `--audio <path>` | `samples/jfk.wav` | Input WAV file (16 kHz mono) |
//! | `--qwen3-quant <q>` | `fp32` | Qwen3 quantization: fp32, fp16, int8, int4 |
//! | `--parakeet-quant <q>` | `int8` | Parakeet quantization: fp32, fp16, int8 |
//! | `--accelerator <a>` | `auto` | ORT accelerator: auto, cpu, cuda, directml, rocm, coreml, webgpu |
//! | `--decoder-gpu` | off | Run decoder on GPU (sets TRANSCRIBE_DECODER_GPU=1) |
//! | `--warmup <n>` | `1` | Number of warmup passes (not timed) |
//! | `--runs <n>` | `3` | Number of timed passes |
//!
//! For more representative RTF numbers use a longer clip (30-60 s). Any 16 kHz
//! mono WAV works; a suitable public-domain source is LibriSpeech test-clean:
//! <https://www.openslr.org/12>

use std::path::PathBuf;
use std::time::{Duration, Instant};

use transcribe_rs::audio::read_wav_samples;
use transcribe_rs::onnx::Quantization;
use transcribe_rs::{
    set_decoder_gpu, set_ort_accelerator, OrtAccelerator, SpeechModel, TranscribeOptions,
};

#[cfg(feature = "qwen3")]
use transcribe_rs::onnx::qwen3::Qwen3Model;

#[cfg(feature = "onnx")]
use transcribe_rs::onnx::parakeet::ParakeetModel;

struct BenchResult {
    label: String,
    load_time: Duration,
    times: Vec<Duration>,
    text: String,
}

impl BenchResult {
    fn mean_secs(&self) -> f64 {
        self.times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / self.times.len() as f64
    }
    fn min_secs(&self) -> f64 {
        self.times
            .iter()
            .map(|d| d.as_secs_f64())
            .fold(f64::INFINITY, f64::min)
    }
    fn max_secs(&self) -> f64 {
        self.times
            .iter()
            .map(|d| d.as_secs_f64())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

fn bench_model(
    label: &str,
    model: &mut dyn SpeechModel,
    samples: &[f32],
    n_warmup: usize,
    n_runs: usize,
    load_time: Duration,
) -> BenchResult {
    let options = TranscribeOptions::default();

    println!("--- {} ---", label);

    for i in 0..n_warmup {
        print!("  Warmup {}/{}... ", i + 1, n_warmup);
        let t = Instant::now();
        let r = model
            .transcribe(samples, &options)
            .expect("transcribe failed");
        println!("{:.3}s → {:?}", t.elapsed().as_secs_f64(), r.text.trim());
    }

    let mut times = Vec::with_capacity(n_runs);
    let mut last_text = String::new();
    for i in 0..n_runs {
        print!("  Run {}/{}... ", i + 1, n_runs);
        let t = Instant::now();
        let r = model
            .transcribe(samples, &options)
            .expect("transcribe failed");
        let elapsed = t.elapsed();
        times.push(elapsed);
        last_text = r.text.trim().to_string();
        println!("{:.3}s", elapsed.as_secs_f64());
    }

    println!();

    BenchResult {
        label: label.to_string(),
        load_time,
        times,
        text: last_text,
    }
}

fn parse_quantization(s: &str) -> Quantization {
    match s.to_ascii_lowercase().as_str() {
        "int8" | "i8" => Quantization::Int8,
        "int4" | "i4" => Quantization::Int4,
        "fp16" | "f16" => Quantization::FP16,
        _ => Quantization::FP32,
    }
}

struct Args {
    qwen3_dir: Option<PathBuf>,
    parakeet_dir: Option<PathBuf>,
    audio_path: PathBuf,
    qwen3_quant: Quantization,
    parakeet_quant: Quantization,
    accelerator: OrtAccelerator,
    decoder_gpu: bool,
    n_warmup: usize,
    n_runs: usize,
}

fn print_help() {
    eprintln!(
        "\
bench_compare — side-by-side inference benchmark for Qwen3-ASR and Parakeet TDT

USAGE:
    cargo run --example bench_compare --features qwen3 --release -- [OPTIONS]

OPTIONS:
    --qwen3 <PATH>           Path to Qwen3-ASR model directory
    --parakeet <PATH>        Path to Parakeet-TDT model directory
    --audio <PATH>           Input WAV file, 16 kHz mono [default: samples/jfk.wav]
    --qwen3-quant <QUANT>    Qwen3 quantization: fp32, fp16, int8, int4 [default: fp32]
    --parakeet-quant <QUANT> Parakeet quantization: fp32, fp16, int8 [default: int8]
    --accelerator <ACCEL>    ORT accelerator: auto, cpu, cuda, directml, rocm,
                             coreml, webgpu [default: auto]
    --decoder-gpu            Run decoder sessions on GPU (default: CPU-only)
    --warmup <N>             Number of warmup passes [default: 1]
    --runs <N>               Number of timed passes [default: 3]
    -h, --help               Print this help message

At least one of --qwen3 or --parakeet must be provided."
    );
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut a = Args {
        qwen3_dir: None,
        parakeet_dir: None,
        audio_path: PathBuf::from("samples/jfk.wav"),
        qwen3_quant: Quantization::FP32,
        parakeet_quant: Quantization::Int8, // Parakeet ships as INT8 only
        accelerator: OrtAccelerator::Auto,
        decoder_gpu: false,
        n_warmup: 1,
        n_runs: 3,
    };

    let mut i = 1;
    while i < args.len() {
        let flag = args[i].as_str();
        match flag {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "--qwen3" | "--parakeet" | "--audio" | "--qwen3-quant" | "--parakeet-quant"
            | "--accelerator" | "--warmup" | "--runs" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for {flag}");
                    std::process::exit(1);
                }
                match flag {
                    "--qwen3" => a.qwen3_dir = Some(PathBuf::from(&args[i])),
                    "--parakeet" => a.parakeet_dir = Some(PathBuf::from(&args[i])),
                    "--audio" => a.audio_path = PathBuf::from(&args[i]),
                    "--qwen3-quant" => a.qwen3_quant = parse_quantization(&args[i]),
                    "--parakeet-quant" => a.parakeet_quant = parse_quantization(&args[i]),
                    "--accelerator" => {
                        a.accelerator = args[i].parse().unwrap_or_else(|e| {
                            eprintln!("Invalid accelerator '{}': {e}", args[i]);
                            std::process::exit(1);
                        });
                    }
                    "--warmup" => a.n_warmup = args[i].parse().unwrap_or(1),
                    "--runs" => a.n_runs = args[i].parse().unwrap_or(3),
                    _ => unreachable!(),
                }
            }
            "--decoder-gpu" => {
                a.decoder_gpu = true;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }
    a
}

fn main() {
    env_logger::init();

    let a = parse_args();

    set_ort_accelerator(a.accelerator);
    set_decoder_gpu(a.decoder_gpu);

    let samples = read_wav_samples(&a.audio_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", a.audio_path.display());
        std::process::exit(1);
    });
    let audio_duration = samples.len() as f64 / 16000.0;

    println!("Audio: {} ({:.2}s)", a.audio_path.display(), audio_duration);
    println!(
        "Accelerator: {:?}  Decoder GPU: {}",
        a.accelerator, a.decoder_gpu
    );
    println!("Warmup: {}  Runs: {}", a.n_warmup, a.n_runs);
    println!();

    let mut results: Vec<BenchResult> = Vec::new();

    // Qwen3-ASR
    #[cfg(feature = "qwen3")]
    if let Some(ref dir) = a.qwen3_dir {
        if !dir.exists() {
            eprintln!("Qwen3 model not found: {}", dir.display());
        } else {
            print!(
                "Loading Qwen3 ({:?}) from {}... ",
                a.qwen3_quant,
                dir.display()
            );
            let t0 = Instant::now();
            match Qwen3Model::load(dir, &a.qwen3_quant) {
                Ok(mut model) => {
                    let load_time = t0.elapsed();
                    println!("{:.3}s", load_time.as_secs_f64());
                    let label = format!(
                        "Qwen3-ASR {:?} ({})",
                        a.qwen3_quant,
                        dir.file_name().unwrap_or_default().to_string_lossy()
                    );
                    results.push(bench_model(
                        &label, &mut model, &samples, a.n_warmup, a.n_runs, load_time,
                    ));
                }
                Err(e) => eprintln!("Failed to load Qwen3: {e}"),
            }
        }
    }

    // Parakeet TDT
    #[cfg(feature = "onnx")]
    if let Some(ref dir) = a.parakeet_dir {
        if !dir.exists() {
            eprintln!("Parakeet model not found: {}", dir.display());
        } else {
            print!(
                "Loading Parakeet ({:?}) from {}... ",
                a.parakeet_quant,
                dir.display()
            );
            let t0 = Instant::now();
            match ParakeetModel::load(dir, &a.parakeet_quant) {
                Ok(mut model) => {
                    let load_time = t0.elapsed();
                    println!("{:.3}s", load_time.as_secs_f64());
                    let label = format!(
                        "Parakeet-TDT {:?} ({})",
                        a.parakeet_quant,
                        dir.file_name().unwrap_or_default().to_string_lossy()
                    );
                    results.push(bench_model(
                        &label, &mut model, &samples, a.n_warmup, a.n_runs, load_time,
                    ));
                }
                Err(e) => eprintln!("Failed to load Parakeet: {e}"),
            }
        }
    }

    if results.is_empty() {
        eprintln!("No models benchmarked. Provide --qwen3 and/or --parakeet paths.");
        std::process::exit(1);
    }

    // Summary table
    println!("=== Summary (audio: {:.2}s) ===", audio_duration);
    println!(
        "{:<36} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "Engine", "Load", "Mean", "Min", "Max", "RTF"
    );
    println!("{}", "-".repeat(82));
    for r in &results {
        let rtf = r.mean_secs() / audio_duration;
        println!(
            "{:<36} {:>8.3}s {:>8.3}s {:>8.3}s {:>8.3}s {:>6.2}x",
            r.label,
            r.load_time.as_secs_f64(),
            r.mean_secs(),
            r.min_secs(),
            r.max_secs(),
            rtf,
        );
    }
    println!();
    for r in &results {
        println!("[{}] {:?}", r.label, r.text);
    }
}
