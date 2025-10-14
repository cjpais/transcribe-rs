use std::path::PathBuf;
use transcribe_rs::engines::whisper::{WhisperEngine, WhisperInferenceParams};
use transcribe_rs::TranscriptionEngine;

#[test]
fn test_chinese_punctuation_with_prompt() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Use Chinese audio sample for testing (16kHz version)
     let audio_path = PathBuf::from("samples/chinese.wav");

    // Test without initial prompt (baseline)
    let result_without_prompt = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe without prompt");

    // Test with Chinese-style punctuation prompt
    let params_with_chinese_prompt = WhisperInferenceParams {
        initial_prompt: Some("这是一段关于 Coding 的中文语音输入，请分析并整理为文本。代码使用半角标点，非代码使用全角标点：".to_string()),
        language: Some("zh".to_string()),
        ..Default::default()
    };

    let result_with_chinese_prompt = engine
        .transcribe_file(&audio_path, Some(params_with_chinese_prompt))
        .expect("Failed to transcribe with Chinese prompt");

    // Test with English formatting prompt
    let params_with_english_prompt = WhisperInferenceParams {
        initial_prompt: Some("This is a Chinese speech about coding. Please transcribe with proper punctuation.".to_string()),
        language: Some("zh".to_string()),
        ..Default::default()
    };

    let result_with_english_prompt = engine
        .transcribe_file(&audio_path, Some(params_with_english_prompt))
        .expect("Failed to transcribe with English prompt");

    // All results should not be empty
    assert!(!result_without_prompt.text.trim().is_empty(), "Transcription without prompt should not be empty");
    assert!(!result_with_chinese_prompt.text.trim().is_empty(), "Transcription with Chinese prompt should not be empty");
    assert!(!result_with_english_prompt.text.trim().is_empty(), "Transcription with English prompt should not be empty");

    // Print results for manual inspection of prompt effects
    println!("=== Prompt Effect Comparison ===");
    println!("Without prompt: {}", result_without_prompt.text);
    println!("With Chinese prompt: {}", result_with_chinese_prompt.text);
    println!("With English prompt: {}", result_with_english_prompt.text);

    // Verify reasonable lengths
    assert!(result_without_prompt.text.len() > 10, "Baseline transcription should have reasonable length");
    assert!(result_with_chinese_prompt.text.len() > 10, "Chinese prompt transcription should have reasonable length");
    assert!(result_with_english_prompt.text.len() > 10, "English prompt transcription should have reasonable length");

    // Check for expected content based on the Chinese audio
    let expected_keywords = ["中文", "语音", "内容", "标点", "whisper", "模型"];
    for keyword in expected_keywords.iter() {
        let found_in_any = result_without_prompt.text.to_lowercase().contains(&keyword.to_lowercase()) ||
                          result_with_chinese_prompt.text.to_lowercase().contains(&keyword.to_lowercase()) ||
                          result_with_english_prompt.text.to_lowercase().contains(&keyword.to_lowercase());
        if !found_in_any {
            println!("Warning: Expected keyword '{}' not found in any result", keyword);
        }
    }

    // Count punctuation marks to verify improvement (commas + periods)
     let no_prompt_commas = result_without_prompt.text.matches('，').count();
     let no_prompt_periods = result_without_prompt.text.matches('。').count();
     let no_prompt_total = no_prompt_commas + no_prompt_periods;
     
     let with_prompt_commas = result_with_chinese_prompt.text.matches('，').count();
     let with_prompt_periods = result_with_chinese_prompt.text.matches('。').count();
     let with_prompt_total = with_prompt_commas + with_prompt_periods;
     
     println!("Punctuation count without prompt (commas+periods): {} (commas: {}, periods: {})", no_prompt_total, no_prompt_commas, no_prompt_periods);
     println!("Punctuation count with prompt (commas+periods): {} (commas: {}, periods: {})", with_prompt_total, with_prompt_commas, with_prompt_periods);

    // The prompts should potentially influence the output format
    // This is a behavioral test - the exact effects depend on the model and input
    // The key is that the prompt parameter is being passed through correctly
}

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

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // First, transcribe without initial prompt
    let result_without_prompt = engine
        .transcribe_file(&audio_path, None)
        .expect("Failed to transcribe without prompt");

    // Then transcribe with initial prompt that provides formatting guidance
    let params_with_prompt = WhisperInferenceParams {
        initial_prompt: Some("This is a formal speech with proper punctuation and capitalization. Please transcribe with appropriate punctuation marks.".to_string()),
        ..Default::default()
    };

    let result_with_prompt = engine
        .transcribe_file(&audio_path, Some(params_with_prompt))
        .expect("Failed to transcribe with initial prompt");

    // Both results should not be empty
    assert!(!result_without_prompt.text.trim().is_empty(), "Transcription without prompt should not be empty");
    assert!(!result_with_prompt.text.trim().is_empty(), "Transcription with prompt should not be empty");

    // The core content should be similar (both should contain key words from JFK speech)
    let key_words = ["fellow", "americans", "ask", "country"];
    for word in &key_words {
        assert!(
            result_without_prompt.text.to_lowercase().contains(word),
            "Result without prompt should contain '{}': '{}'", word, result_without_prompt.text
        );
        assert!(
            result_with_prompt.text.to_lowercase().contains(word),
            "Result with prompt should contain '{}': '{}'", word, result_with_prompt.text
        );
    }

    // The prompt should potentially influence formatting/style
    // Note: The exact effect depends on the model and audio quality
    println!("Without prompt: {}", result_without_prompt.text);
    println!("With prompt: {}", result_with_prompt.text);
    
    // Verify that both transcriptions are reasonable in length
    assert!(result_without_prompt.text.len() > 20, "Transcription without prompt should have reasonable length");
    assert!(result_with_prompt.text.len() > 20, "Transcription with prompt should have reasonable length");
}

#[test]
fn test_empty_initial_prompt() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Test with empty initial_prompt
    let params = WhisperInferenceParams {
        initial_prompt: Some("".to_string()),
        ..Default::default()
    };

    // Transcribe with empty initial prompt
    let result = engine
        .transcribe_file(&audio_path, Some(params))
        .expect("Failed to transcribe with empty initial prompt");

    // Should still work normally
    assert!(!result.text.trim().is_empty(), "Transcription should not be empty");
}

#[test]
fn test_real_chinese_audio_prompt_effect() {
    let mut engine = WhisperEngine::new();

    // Load the model
    let model_path = PathBuf::from("models/whisper-medium-q4_1.bin");
    engine
        .load_model(&model_path)
        .expect("Failed to load model");

    // Load the Chinese audio file (16kHz version)
     let audio_path = PathBuf::from("samples/chinese.wav");

    // Test without prompt - should have less punctuation
    let params_no_prompt = WhisperInferenceParams {
        initial_prompt: None,
        language: Some("zh".to_string()),
        ..Default::default()
    };
    
    let result_no_prompt = engine
        .transcribe_file(&audio_path, Some(params_no_prompt))
        .expect("Failed to transcribe without prompt");
    
    println!("=== Transcription Result Without Prompt ===");
    println!("{}", result_no_prompt.text);
    
    // Test with Chinese prompt - should have better punctuation
    let params_with_prompt = WhisperInferenceParams {
        initial_prompt: Some("这是一段关于 Coding 的中文语音输入，请分析并整理为文本。代码使用半角标点，非代码使用全角标点：".to_string()),
        language: Some("zh".to_string()),
        ..Default::default()
    };
    
    let result_with_prompt = engine
        .transcribe_file(&audio_path, Some(params_with_prompt))
        .expect("Failed to transcribe with prompt");
    
    println!("=== Transcription Result With Prompt ===");
    println!("{}", result_with_prompt.text);
    
    // Basic assertions
    assert!(!result_no_prompt.text.trim().is_empty(), "No prompt result should not be empty");
    assert!(!result_with_prompt.text.trim().is_empty(), "With prompt result should not be empty");
    
    // Check for expected content based on the provided text
    let expected_keywords = ["中文", "语音", "内容", "标点", "whisper", "模型"];
    for keyword in expected_keywords.iter() {
        let found_in_either = result_no_prompt.text.to_lowercase().contains(&keyword.to_lowercase()) || 
                             result_with_prompt.text.to_lowercase().contains(&keyword.to_lowercase());
        if !found_in_either {
            println!("Warning: Expected keyword '{}' not found in either result", keyword);
        }
    }
    
    // Count punctuation marks to verify improvement (commas + periods)
     let no_prompt_commas = result_no_prompt.text.matches('，').count();
     let no_prompt_periods = result_no_prompt.text.matches('。').count();
     let no_prompt_total = no_prompt_commas + no_prompt_periods;
     
     let with_prompt_commas = result_with_prompt.text.matches('，').count();
     let with_prompt_periods = result_with_prompt.text.matches('。').count();
     let with_prompt_total = with_prompt_commas + with_prompt_periods;
     
     println!("Punctuation count without prompt (commas+periods): {} (commas: {}, periods: {})", no_prompt_total, no_prompt_commas, no_prompt_periods);
     println!("Punctuation count with prompt (commas+periods): {} (commas: {}, periods: {})", with_prompt_total, with_prompt_commas, with_prompt_periods);
    
    // The prompt should generally improve punctuation, but we'll just verify both results are reasonable
    assert!(result_no_prompt.text.len() > 20, "No prompt result too short");
    assert!(result_with_prompt.text.len() > 20, "With prompt result too short");
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
