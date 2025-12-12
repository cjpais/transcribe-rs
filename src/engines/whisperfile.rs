//! Whisperfile speech recognition engine implementation.
//!
//! This module provides a transcription engine that uses Mozilla's whisperfile
//! for speech-to-text conversion. The engine manages the whisperfile server
//! lifecycle automatically - spawning it on model load and stopping it on unload.
//!
//! # Requirements
//!
//! - The whisperfile binary must be available on the system
//! - Whisper model in GGML/GGUF format
//!
//! # Examples
//!
//! ```rust,no_run
//! use transcribe_rs::{TranscriptionEngine, engines::whisperfile::WhisperfileEngine};
//! use std::path::PathBuf;
//!
//! let mut engine = WhisperfileEngine::new(PathBuf::from("/path/to/whisperfile"));
//! engine.load_model(&PathBuf::from("models/ggml-small.bin"))?;
//!
//! let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
//! println!("Transcription: {}", result.text);
//!
//! // Server is automatically stopped when engine is dropped
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{TranscriptionEngine, TranscriptionResult, TranscriptionSegment};
use reqwest::blocking::multipart;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// JSON output structure from whisperfile server (verbose_json format)
#[derive(Deserialize)]
struct WhisperfileOutput {
    text: String,
    #[serde(default)]
    segments: Vec<WhisperfileSegment>,
}

#[derive(Deserialize)]
struct WhisperfileSegment {
    text: String,
    start: f32,
    end: f32,
}

impl From<WhisperfileOutput> for TranscriptionResult {
    fn from(output: WhisperfileOutput) -> Self {
        let segments = if output.segments.is_empty() {
            None
        } else {
            Some(
                output
                    .segments
                    .into_iter()
                    .map(|s| TranscriptionSegment {
                        start: s.start,
                        end: s.end,
                        text: s.text,
                    })
                    .collect(),
            )
        };

        TranscriptionResult {
            text: output.text.trim().to_string(),
            segments,
        }
    }
}

/// Parameters for configuring Whisperfile model loading.
#[derive(Debug, Clone)]
pub struct WhisperfileModelParams {
    /// Port for the whisperfile server (default: 8080)
    pub port: u16,
    /// Host to bind the server to (default: "127.0.0.1")
    pub host: String,
    /// Timeout in seconds to wait for server to start (default: 30)
    pub startup_timeout_secs: u64,
}

impl Default for WhisperfileModelParams {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            startup_timeout_secs: 30,
        }
    }
}

/// Parameters for configuring Whisperfile inference behavior.
#[derive(Debug, Clone)]
pub struct WhisperfileInferenceParams {
    /// Target language for transcription (e.g., "en", "es", "fr").
    /// If None, whisperfile will auto-detect the language.
    pub language: Option<String>,

    /// Whether to translate the transcription to English.
    pub translate: bool,

    /// Temperature for sampling (0.0 = greedy).
    pub temperature: Option<f32>,

    /// Response format hint.
    pub response_format: Option<String>,
}

impl Default for WhisperfileInferenceParams {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            temperature: None,
            response_format: Some("verbose_json".to_string()),
        }
    }
}

/// Whisperfile speech recognition engine.
///
/// This engine manages the whisperfile server lifecycle automatically.
/// When you call `load_model()`, it spawns the whisperfile server process.
/// When the engine is dropped or `unload_model()` is called, the server is stopped.
///
/// # Examples
///
/// ```rust,no_run
/// use transcribe_rs::engines::whisperfile::WhisperfileEngine;
/// use std::path::PathBuf;
///
/// let mut engine = WhisperfileEngine::new(PathBuf::from("/path/to/whisperfile"));
/// ```
pub struct WhisperfileEngine {
    binary_path: PathBuf,
    server_url: String,
    client: reqwest::blocking::Client,
    server_process: Option<Child>,
}

impl WhisperfileEngine {
    /// Create a new Whisperfile engine instance.
    ///
    /// # Arguments
    ///
    /// * `binary_path` - Path to the whisperfile executable
    ///
    /// # Examples
    ///
    /// ```rust
    /// use transcribe_rs::engines::whisperfile::WhisperfileEngine;
    /// use std::path::PathBuf;
    ///
    /// let engine = WhisperfileEngine::new(PathBuf::from("/usr/local/bin/whisperfile"));
    /// ```
    pub fn new(binary_path: impl Into<PathBuf>) -> Self {
        Self {
            binary_path: binary_path.into(),
            server_url: String::new(),
            client: reqwest::blocking::Client::new(),
            server_process: None,
        }
    }

    /// Wait for the server to become ready
    fn wait_for_server(&self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let url = format!("{}/", self.server_url);

        while start.elapsed() < timeout {
            if self
                .client
                .get(&url)
                .timeout(Duration::from_secs(1))
                .send()
                .is_ok()
            {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        Err(format!(
            "Whisperfile server failed to start within {} seconds",
            timeout.as_secs()
        )
        .into())
    }
}

impl Drop for WhisperfileEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl TranscriptionEngine for WhisperfileEngine {
    type InferenceParams = WhisperfileInferenceParams;
    type ModelParams = WhisperfileModelParams;

    fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: Self::ModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Stop any existing server
        self.unload_model();

        // Verify binary exists
        if !self.binary_path.exists() {
            return Err(format!(
                "Whisperfile binary not found: {}",
                self.binary_path.display()
            )
            .into());
        }

        // Verify model exists
        if !model_path.exists() {
            return Err(format!("Model file not found: {}", model_path.display()).into());
        }

        self.server_url = format!("http://{}:{}", params.host, params.port);

        // Spawn the server process
        let child = Command::new(&self.binary_path)
            .arg("--server")
            .arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg(&params.host)
            .arg("--port")
            .arg(params.port.to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to spawn whisperfile server: {}", e))?;

        self.server_process = Some(child);

        // Wait for server to be ready
        self.wait_for_server(Duration::from_secs(params.startup_timeout_secs))?;

        Ok(())
    }

    fn unload_model(&mut self) {
        if let Some(mut child) = self.server_process.take() {
            // Try graceful shutdown first
            let _ = child.kill();
            let _ = child.wait();
        }
        self.server_url.clear();
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        if self.server_process.is_none() {
            return Err("Model not loaded. Call load_model() first.".into());
        }

        // Write samples to a WAV buffer in memory
        let mut wav_buffer = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::new(&mut wav_buffer, spec)?;
        for sample in &samples {
            let sample_i16 = (sample * i16::MAX as f32) as i16;
            writer.write_sample(sample_i16)?;
        }
        writer.finalize()?;

        let wav_data = wav_buffer.into_inner();
        self.transcribe_wav_bytes(wav_data, params)
    }

    fn transcribe_file(
        &mut self,
        wav_path: &Path,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        if self.server_process.is_none() {
            return Err("Model not loaded. Call load_model() first.".into());
        }

        let wav_data = std::fs::read(wav_path)?;
        self.transcribe_wav_bytes(wav_data, params)
    }
}

impl WhisperfileEngine {
    fn transcribe_wav_bytes(
        &self,
        wav_data: Vec<u8>,
        params: Option<WhisperfileInferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let params = params.unwrap_or_default();

        let file_part = multipart::Part::bytes(wav_data)
            .file_name("audio.wav")
            .mime_str("audio/wav")?;

        let mut form = multipart::Form::new().part("file", file_part);

        // Add optional parameters
        if let Some(lang) = &params.language {
            form = form.text("language", lang.clone());
        }

        if params.translate {
            form = form.text("translate", "true");
        }

        if let Some(temp) = params.temperature {
            form = form.text("temperature", temp.to_string());
        }

        if let Some(fmt) = &params.response_format {
            form = form.text("response_format", fmt.clone());
        }

        let url = format!("{}/inference", self.server_url);
        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .map_err(|e| format!("Request to whisperfile server failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(format!("Whisperfile server error {}: {}", status, body).into());
        }

        let json_response = response.text()?;
        let whisperfile_output: WhisperfileOutput = serde_json::from_str(&json_response)?;
        Ok(whisperfile_output.into())
    }
}
