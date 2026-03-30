//! LLM-based ASR post-processing example using Qwen2.5-0.5B (GGUF).
//!
//! Demonstrates loading a quantized Qwen2.5 model via the library and using it
//! to add punctuation and correct errors in ASR output.
//!
//! Usage:
//!   cargo run --example llm_postprocess --features llm-postprocess --release
//!   cargo run --example llm_postprocess --features llm-postprocess --release -- \
//!     --model models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf \
//!     --tokenizer models/qwen2.5-0.5b/tokenizer.json \
//!     --text "今天天气很好我们去公圆玩吧他说号的"

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use transcribe_rs::llm_postprocess::LlmPostProcessor;

const DEFAULT_MODEL: &str = "models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const DEFAULT_TOKENIZER: &str = "models/qwen2.5-0.5b/tokenizer.json";
const DEFAULT_TEXT: &str = "今天天气很好我们去公圆玩吧他说号的";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (model_path, tokenizer_path, input_text) = parse_args(&args);

    println!("=== LLM 后处理验证 (Qwen2.5-0.5B GGUF) ===\n");

    // 1. Load model
    print!("加载模型... ");
    std::io::stdout().flush()?;
    let load_start = Instant::now();

    let mut processor =
        LlmPostProcessor::from_files(Path::new(&model_path), Path::new(&tokenizer_path))?;

    let load_time = load_start.elapsed();
    println!("完成 ({:.2?})", load_time);

    // 2. Process text
    println!();
    println!("原始文本: {}", input_text);

    let gen_start = Instant::now();
    let result = processor.process(&input_text)?;
    let gen_time = gen_start.elapsed();

    println!("修正文本: {}", result);
    println!();
    println!("--- 统计 ---");
    println!("耗时: {:.2?}", gen_time);
    println!("模型加载: {:.2?}", load_time);

    Ok(())
}

fn parse_args(args: &[String]) -> (String, String, String) {
    let mut model = DEFAULT_MODEL.to_string();
    let mut tokenizer = DEFAULT_TOKENIZER.to_string();
    let mut text = DEFAULT_TEXT.to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model = args[i].clone();
                }
            }
            "--tokenizer" => {
                i += 1;
                if i < args.len() {
                    tokenizer = args[i].clone();
                }
            }
            "--text" => {
                i += 1;
                if i < args.len() {
                    text = args[i].clone();
                }
            }
            _ => {}
        }
        i += 1;
    }

    (model, tokenizer, text)
}
