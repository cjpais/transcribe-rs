use std::path::PathBuf;

use transcribe_rs::{
    remote::openai::{self, OpenAIRequestParams},
    RemoteTranscriptionEngine,
};

#[tokio::test]
async fn test_jfk_transcription() {
    let engine = openai::default_engine();

    // Load the JFK audio file
    let audio_path = PathBuf::from("samples/jfk.wav");

    // Transcribe with default params
    let result = engine
        .transcribe_file(
            &audio_path,
            OpenAIRequestParams::builder()
                .build()
                .expect("Default parameters shoul be valid"),
        )
        .await
        .expect("Failed to transcribe");

    let expected = "And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country.";
    assert_eq!(
        result.text.trim(),
        expected,
        "\nExpected: '{}'\nActual: '{}'",
        expected,
        result.text.trim()
    );
}
