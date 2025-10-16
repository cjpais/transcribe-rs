use std::path::PathBuf;
use transcribe_rs::engines::whisper::{WhisperEngine, WhisperInferenceParams};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_jfk_transcription() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with default params
    let result = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe");

    let expected = "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}

#[test]
fn test_initial_prompt_functionality() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the Chinese audio file
    let audio_path = PathBuf::from("samples/chinese.wav");

    // First, transcribe without initial prompt
    let result_without_prompt = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe without prompt");

    // Then transcribe with initial prompt that provides formatting guidance
    let params_with_prompt = WhisperInferenceParams {
        initial_prompt: Some("这是一段关于 Coding 的中文语音输入，请分析并整理为文本。代码使用半角标点，非代码使用全角标点：".to_string()),
        language: Some("zh".to_string()),
        ..Default::default()
    };

    let result_with_prompt = engine
        .transcribe_file(&audio_path, Some(params_with_prompt))
        .expect("Failed to transcribe with initial prompt");

    // Both results should not be empty
    assert!(!result_without_prompt.text.trim().is_empty(), "Transcription without prompt should not be empty");
    assert!(!result_with_prompt.text.trim().is_empty(), "Transcription with prompt should not be empty");

    // Count punctuation marks to verify improvement (commas + periods)
    let no_prompt_commas = result_without_prompt.text.matches('，').count();
    let no_prompt_periods = result_without_prompt.text.matches('。').count();
    let no_prompt_total = no_prompt_commas + no_prompt_periods;
    
    let with_prompt_commas = result_with_prompt.text.matches('，').count();
    let with_prompt_periods = result_with_prompt.text.matches('。').count();
    let with_prompt_total = with_prompt_commas + with_prompt_periods;
    
    // Assert that the prompt improves punctuation
    assert!(
        with_prompt_total >= no_prompt_total,
        "Prompt should improve punctuation: without prompt {} punctuation marks, with prompt {} punctuation marks",
        no_prompt_total,
        with_prompt_total
    );
    
    // Verify that both transcriptions are reasonable in length
    assert!(result_without_prompt.text.len() > 10, "Transcription without prompt should have reasonable length");
    assert!(result_with_prompt.text.len() > 10, "Transcription with prompt should have reasonable length");
}

#[test]
fn test_product_names_prompt_effect() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the product names audio file
    let audio_path = PathBuf::from("samples/product_names.wav");

    // First, transcribe without initial prompt
    let result_without_prompt = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe without prompt");

    // Then transcribe with product names in the prompt as a glossary
    let product_names_prompt = "QuirkQuid Quill Inc, P3-Quattro, O3-Omni, B3-BondX, E3-Equity, W3-WrapZ, O2-Outlier, U3-UniFund, M3-Mover";
    let params_with_prompt = WhisperInferenceParams {
        initial_prompt: Some(product_names_prompt.to_string()),
        ..Default::default()
    };

    let result_with_prompt = engine
        .transcribe_file(&audio_path, Some(params_with_prompt))
        .expect("Failed to transcribe with product names prompt");

    // Both results should not be empty
    assert!(!result_without_prompt.text.trim().is_empty(), "Transcription without prompt should not be empty");
    assert!(!result_with_prompt.text.trim().is_empty(), "Transcription with prompt should not be empty");

    // Count occurrences of product names from the prompt
    let prompt_words: Vec<&str> = product_names_prompt.split(", ").collect();
    
    let mut no_prompt_count = 0;
    let mut with_prompt_count = 0;
    
    for word in &prompt_words {
        // Count case-insensitive occurrences
        let word_lower = word.to_lowercase();
        no_prompt_count += result_without_prompt.text.to_lowercase().matches(&word_lower).count();
        with_prompt_count += result_with_prompt.text.to_lowercase().matches(&word_lower).count();
    }
    
    // Assert that the prompt improves the occurrence of product names
    assert!(
        with_prompt_count >= no_prompt_count,
        "Prompt should improve product name recognition: without prompt {} occurrences, with prompt {} occurrences",
        no_prompt_count,
        with_prompt_count
    );

    
    // Verify that both transcriptions are reasonable in length
    assert!(result_without_prompt.text.len() > 20, "Transcription without prompt should have reasonable length");
    assert!(result_with_prompt.text.len() > 20, "Transcription with prompt should have reasonable length");
}

#[test]
fn test_russian_translation() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the Russian audio file
    let audio_path = PathBuf::from("samples/russian.wav");

    // Set up inference params with translate enabled
    let params = WhisperInferenceParams {
        translate: true,
        ..Default::default()
    };

    // Transcribe and translate to English
    let result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe");

    let expected = "Check the connection.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}
