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
use log::{debug, error, info, trace, warn};
use serde::Deserialize;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use ureq::Agent;

/// Custom multipart form-data builder for HTTP requests.
struct MultipartForm {
    boundary: String,
    body: Vec<u8>,
}

impl MultipartForm {
    /// Create a new multipart form with a random boundary.
    fn new() -> Self {
        // Generate a simple boundary using timestamp and a fixed prefix
        let boundary = format!(
            "----transcribe-rs-boundary-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        Self {
            boundary,
            body: Vec::new(),
        }
    }

    /// Add a file part to the form.
    fn file(mut self, name: &str, filename: &str, content_type: &str, data: Vec<u8>) -> Self {
        // Write boundary
        write!(self.body, "--{}\r\n", self.boundary).unwrap();
        // Write content disposition header
        write!(
            self.body,
            "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
            name, filename
        )
        .unwrap();
        // Write content type header
        write!(self.body, "Content-Type: {}\r\n", content_type).unwrap();
        // Blank line before content
        write!(self.body, "\r\n").unwrap();
        // Write file data
        self.body.extend_from_slice(&data);
        // Trailing CRLF
        write!(self.body, "\r\n").unwrap();
        self
    }

    /// Add a text field to the form.
    fn text(mut self, name: &str, value: &str) -> Self {
        // Write boundary
        write!(self.body, "--{}\r\n", self.boundary).unwrap();
        // Write content disposition header
        write!(
            self.body,
            "Content-Disposition: form-data; name=\"{}\"\r\n",
            name
        )
        .unwrap();
        // Blank line before content
        write!(self.body, "\r\n").unwrap();
        // Write text value
        write!(self.body, "{}\r\n", value).unwrap();
        self
    }

    /// Finalize the form and return the content type and body.
    fn build(mut self) -> (String, Vec<u8>) {
        // Write closing boundary
        write!(self.body, "--{}--\r\n", self.boundary).unwrap();
        let content_type = format!("multipart/form-data; boundary={}", self.boundary);
        (content_type, self.body)
    }
}

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

/// GPU acceleration mode for Whisperfile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GPUMode {
    /// Auto-detect the best available GPU (default)
    #[default]
    Auto,
    /// Use Apple Metal (macOS)
    Apple,
    /// Use AMD GPU
    Amd,
    /// Use NVIDIA GPU
    Nvidia,
    /// Disable GPU acceleration
    Disabled,
}

impl GPUMode {
    /// Get the command-line argument value for this GPU mode.
    pub fn as_arg(&self) -> &'static str {
        match self {
            GPUMode::Auto => "auto",
            GPUMode::Apple => "apple",
            GPUMode::Amd => "amd",
            GPUMode::Nvidia => "nvidia",
            GPUMode::Disabled => "disabled",
        }
    }
}

impl std::fmt::Display for GPUMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_arg())
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
    /// GPU acceleration mode (default: Auto)
    pub gpu: GPUMode,
}

impl Default for WhisperfileModelParams {
    fn default() -> Self {
        Self {
            port: 8080,
            host: "127.0.0.1".to_string(),
            startup_timeout_secs: 30,
            gpu: GPUMode::default(),
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
    agent: Agent,
    server_process: Option<Child>,
    /// Flag to signal the log reader thread to stop
    log_shutdown: Arc<AtomicBool>,
    /// Handle to the log reader thread
    log_thread: Option<std::thread::JoinHandle<()>>,
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
            agent: Agent::new_with_defaults(),
            server_process: None,
            log_shutdown: Arc::new(AtomicBool::new(false)),
            log_thread: None,
        }
    }

    /// Wait for the server to become ready
    fn wait_for_server(&self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let url = format!("{}/", self.server_url);

        debug!(
            "Waiting for whisperfile server at {} (timeout: {}s)",
            url,
            timeout.as_secs()
        );

        while start.elapsed() < timeout {
            trace!(
                "Polling whisperfile server... ({:.1}s elapsed)",
                start.elapsed().as_secs_f32()
            );
            if self.agent.get(&url).call().is_ok() {
                info!(
                    "Whisperfile server ready after {:.2}s",
                    start.elapsed().as_secs_f32()
                );
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        error!(
            "Whisperfile server failed to start within {} seconds",
            timeout.as_secs()
        );
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
            warn!(
                "Whisperfile binary not found: {}",
                self.binary_path.display()
            );
            return Err(format!(
                "Whisperfile binary not found: {}",
                self.binary_path.display()
            )
            .into());
        }

        // Verify model exists
        if !model_path.exists() {
            warn!("Model file not found: {}", model_path.display());
            return Err(format!("Model file not found: {}", model_path.display()).into());
        }

        self.server_url = format!("http://{}:{}", params.host, params.port);

        info!(
            "Starting whisperfile server: binary={}, model={}, host={}, port={}, gpu={}",
            self.binary_path.display(),
            model_path.display(),
            params.host,
            params.port,
            params.gpu
        );

        // Spawn the server process with stderr piped for logging
        let mut child = Command::new(&self.binary_path)
            .arg("--server")
            .arg("-m")
            .arg(model_path)
            .arg("--host")
            .arg(&params.host)
            .arg("--port")
            .arg(params.port.to_string())
            .arg("--gpu")
            .arg(params.gpu.as_arg())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                error!("Failed to spawn whisperfile server: {}", e);
                format!("Failed to spawn whisperfile server: {}", e)
            })?;

        debug!("Whisperfile server process spawned (pid: {:?})", child.id());

        // Reset shutdown flag and spawn a thread to read server logs
        self.log_shutdown.store(false, Ordering::SeqCst);

        if let Some(stderr) = child.stderr.take() {
            let shutdown_flag = Arc::clone(&self.log_shutdown);
            let log_thread = std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines() {
                    if shutdown_flag.load(Ordering::SeqCst) {
                        break;
                    }
                    match line {
                        Ok(line) => {
                            debug!("[whisperfile] {}", line);
                        }
                        Err(e) => {
                            trace!("Error reading whisperfile stderr: {}", e);
                            break;
                        }
                    }
                }
                trace!("Whisperfile log reader thread exiting");
            });
            self.log_thread = Some(log_thread);
        }

        self.server_process = Some(child);

        // Wait for server to be ready
        self.wait_for_server(Duration::from_secs(params.startup_timeout_secs))?;

        Ok(())
    }

    fn unload_model(&mut self) {
        // Signal the log reader thread to stop
        self.log_shutdown.store(true, Ordering::SeqCst);

        if let Some(mut child) = self.server_process.take() {
            debug!("Stopping whisperfile server (pid: {:?})", child.id());
            let _ = child.kill();
            let _ = child.wait();
            info!("Whisperfile server stopped");
        }

        // Wait for the log thread to finish
        if let Some(thread) = self.log_thread.take() {
            trace!("Waiting for log reader thread to finish");
            let _ = thread.join();
        }

        self.server_url.clear();
    }

    fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        params: Option<Self::InferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        if self.server_process.is_none() {
            warn!("Attempted to transcribe samples without loading model");
            return Err("Model not loaded. Call load_model() first.".into());
        }

        debug!("Transcribing {} samples", samples.len());

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
            warn!("Attempted to transcribe file without loading model");
            return Err("Model not loaded. Call load_model() first.".into());
        }

        debug!("Transcribing file: {}", wav_path.display());

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

        trace!(
            "Preparing transcription request: {} bytes, language={:?}, translate={}, temp={:?}",
            wav_data.len(),
            params.language,
            params.translate,
            params.temperature
        );

        // Build multipart form using custom builder
        let mut form = MultipartForm::new().file("file", "audio.wav", "audio/wav", wav_data);

        // Add optional parameters
        if let Some(lang) = &params.language {
            form = form.text("language", lang);
        }

        if params.translate {
            form = form.text("translate", "true");
        }

        if let Some(temp) = params.temperature {
            form = form.text("temperature", &temp.to_string());
        }

        if let Some(fmt) = &params.response_format {
            form = form.text("response_format", fmt);
        }

        let (content_type, body) = form.build();

        let url = format!("{}/inference", self.server_url);
        debug!("Sending transcription request to {}", url);

        let start = Instant::now();
        let response = self
            .agent
            .post(&url)
            .content_type(&content_type)
            .send(&body[..])
            .map_err(|e| {
                error!("Request to whisperfile server failed: {}", e);
                format!("Request to whisperfile server failed: {}", e)
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.into_body().read_to_string().unwrap_or_default();
            error!("Whisperfile server error {}: {}", status, body);
            return Err(format!("Whisperfile server error {}: {}", status, body).into());
        }

        let json_response = response.into_body().read_to_string()?;
        let whisperfile_output: WhisperfileOutput = serde_json::from_str(&json_response)?;

        debug!(
            "Transcription completed in {:.2}s ({} chars)",
            start.elapsed().as_secs_f32(),
            whisperfile_output.text.len()
        );
        trace!("Transcription result: {:?}", whisperfile_output.text);

        Ok(whisperfile_output.into())
    }
}
