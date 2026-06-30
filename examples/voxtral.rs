use std::path::PathBuf;
use std::time::Instant;

use transcribe_rs::voxtral::{VoxtralModel, VoxtralParams};
use transcribe_rs::SpeechModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let model_path = std::env::var("VOXTRAL_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/voxtral-mini/Voxtral-Mini-3B-2507-Q4_K_M.gguf"));
    let mmproj_path = std::env::var("VOXTRAL_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("models/voxtral-mini/mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf")
        });
    let wav_path = std::env::var("VOXTRAL_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("samples/jfk.wav"));
    let language = std::env::var("VOXTRAL_LANGUAGE").ok();
    let translate = std::env::var("VOXTRAL_TRANSLATE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes"));
    let repeat = std::env::var("VOXTRAL_REPEAT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1);

    let params = VoxtralParams {
        max_new_tokens: 192,
        ..Default::default()
    };
    let load_start = Instant::now();
    let mut model = VoxtralModel::load_files(&model_path, &mmproj_path, params)?;
    eprintln!("load_time_ms={}", load_start.elapsed().as_millis());

    for index in 0..repeat {
        let start = Instant::now();
        let result = model.transcribe_file(
            &wav_path,
            &transcribe_rs::TranscribeOptions {
                language: language.clone(),
                translate,
                ..Default::default()
            },
        )?;
        eprintln!(
            "run={} transcribe_time_ms={}",
            index + 1,
            start.elapsed().as_millis()
        );
        println!("{}", result.text);
    }
    Ok(())
}
