mod common;

use std::path::PathBuf;

use transcribe_rs::voxtral::{VoxtralModel, VoxtralParams};
use transcribe_rs::SpeechModel;

#[test]
fn test_voxtral_transcribe_jfk() {
    let model_path = match std::env::var("VOXTRAL_MODEL") {
        Ok(path) => PathBuf::from(path),
        Err(_) => return,
    };
    let mmproj_path = match std::env::var("VOXTRAL_MMPROJ") {
        Ok(path) => PathBuf::from(path),
        Err(_) => return,
    };
    let wav_path = PathBuf::from("samples/jfk.wav");

    if !common::require_paths(&[&model_path, &mmproj_path, &wav_path]) {
        return;
    }

    let params = VoxtralParams {
        max_new_tokens: 160,
        ..Default::default()
    };
    let mut model =
        VoxtralModel::load_files(&model_path, &mmproj_path, params).expect("load Voxtral");
    let result = model
        .transcribe_file(
            &wav_path,
            &transcribe_rs::TranscribeOptions {
                language: Some("en".to_string()),
                ..Default::default()
            },
        )
        .expect("transcribe with Voxtral");

    assert!(
        result
            .text
            .to_lowercase()
            .contains("ask not what your country"),
        "unexpected transcript: {}",
        result.text
    );
}
