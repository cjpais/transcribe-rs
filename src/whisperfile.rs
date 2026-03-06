//! Whisperfile speech recognition engine implementation.
//!
//! This module provides a transcription engine that uses Mozilla's whisperfile
//! for speech-to-text conversion. The engine manages the whisperfile server
//! lifecycle automatically.

use crate::{ModelCapabilities, SpeechModel, TranscribeError, TranscriptionResult, TranscriptionSegment};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Whisperfile",
    languages: &["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"],
    supports_timestamps: true,
    supports_translation: true,
    supports_streaming: false,
};
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
    fn new() -> Self {
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

    fn file(mut self, name: &str, filename: &str, content_type: &str, data: Vec<u8>) -> Self {
        write!(self.body, "--{}\r\n", self.boundary).unwrap();
        write!(
            self.body,
            "Content-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n",
            name, filename
        )
        .unwrap();
        write!(self.body, "Content-Type: {}\r\n", content_type).unwrap();
        write!(self.body, "\r\n").unwrap();
        self.body.extend_from_slice(&data);
        write!(self.body, "\r\n").unwrap();
        self
    }

    fn text(mut self, name: &str, value: &str) -> Self {
        write!(self.body, "--{}\r\n", self.boundary).unwrap();
        write!(
            self.body,
            "Content-Disposition: form-data; name=\"{}\"\r\n",
            name
        )
        .unwrap();
        write!(self.body, "\r\n").unwrap();
        write!(self.body, "{}\r\n", value).unwrap();
        self
    }

    fn build(mut self) -> (String, Vec<u8>) {
        write!(self.body, "--{}--\r\n", self.boundary).unwrap();
        let content_type = format!("multipart/form-data; boundary={}", self.boundary);
        (content_type, self.body)
    }
}

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
    #[default]
    Auto,
    Apple,
    Amd,
    Nvidia,
    Disabled,
}

impl GPUMode {
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
    pub port: u16,
    pub host: String,
    pub startup_timeout_secs: u64,
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
    pub language: Option<String>,
    pub translate: bool,
    pub temperature: Option<f32>,
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
/// Manages the whisperfile server lifecycle automatically.
pub struct WhisperfileEngine {
    binary_path: PathBuf,
    server_url: String,
    agent: Agent,
    server_process: Option<Child>,
    log_shutdown: Arc<AtomicBool>,
    log_thread: Option<std::thread::JoinHandle<()>>,
}

impl WhisperfileEngine {
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

    /// Load a model with default parameters.
    pub fn load_model(&mut self, model_path: &Path) -> Result<(), TranscribeError> {
        self.load_model_with_params(model_path, WhisperfileModelParams::default())
    }

    /// Load a model with custom server parameters.
    pub fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: WhisperfileModelParams,
    ) -> Result<(), TranscribeError> {
        self.unload_model();

        if !self.binary_path.exists() {
            warn!(
                "Whisperfile binary not found: {}",
                self.binary_path.display()
            );
            return Err(TranscribeError::ModelNotFound(self.binary_path.clone()));
        }

        if !model_path.exists() {
            warn!("Model file not found: {}", model_path.display());
            return Err(TranscribeError::ModelNotFound(model_path.to_path_buf()));
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
                TranscribeError::Inference(format!("Failed to spawn whisperfile server: {}", e))
            })?;

        debug!("Whisperfile server process spawned (pid: {:?})", child.id());

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

        self.wait_for_server(Duration::from_secs(params.startup_timeout_secs))?;

        Ok(())
    }

    /// Unload the current model and stop the server.
    pub fn unload_model(&mut self) {
        self.log_shutdown.store(true, Ordering::SeqCst);

        if let Some(mut child) = self.server_process.take() {
            debug!("Stopping whisperfile server (pid: {:?})", child.id());
            let _ = child.kill();
            let _ = child.wait();
            info!("Whisperfile server stopped");
        }

        if let Some(thread) = self.log_thread.take() {
            trace!("Waiting for log reader thread to finish");
            let _ = thread.join();
        }

        self.server_url.clear();
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &WhisperfileInferenceParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.transcribe_samples_inner(samples.to_vec(), Some(params.clone()))
    }

    /// Transcribe a WAV file with model-specific parameters.
    pub fn transcribe_file_with(
        &mut self,
        wav_path: &Path,
        params: &WhisperfileInferenceParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        if self.server_process.is_none() {
            warn!("Attempted to transcribe file without loading model");
            return Err(TranscribeError::Inference("Model not loaded. Call load_model() first.".to_string()));
        }

        debug!("Transcribing file: {}", wav_path.display());
        let wav_data = std::fs::read(wav_path)?;
        self.transcribe_wav_bytes(wav_data, Some(params.clone()))
    }

    fn wait_for_server(&self, timeout: Duration) -> Result<(), TranscribeError> {
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
        Err(TranscribeError::Inference(format!(
            "Whisperfile server failed to start within {} seconds",
            timeout.as_secs()
        )))
    }

    fn transcribe_samples_inner(
        &mut self,
        samples: Vec<f32>,
        params: Option<WhisperfileInferenceParams>,
    ) -> Result<TranscriptionResult, TranscribeError> {
        if self.server_process.is_none() {
            warn!("Attempted to transcribe samples without loading model");
            return Err(TranscribeError::Inference("Model not loaded. Call load_model() first.".to_string()));
        }

        debug!("Transcribing {} samples", samples.len());

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

    fn transcribe_wav_bytes(
        &self,
        wav_data: Vec<u8>,
        params: Option<WhisperfileInferenceParams>,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = params.unwrap_or_default();

        trace!(
            "Preparing transcription request: {} bytes, language={:?}, translate={}, temp={:?}",
            wav_data.len(),
            params.language,
            params.translate,
            params.temperature
        );

        let mut form = MultipartForm::new().file("file", "audio.wav", "audio/wav", wav_data);

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
                TranscribeError::Inference(format!("Request to whisperfile server failed: {}", e))
            })?;

        let status = response.status();
        if !status.is_success() {
            let body = response.into_body().read_to_string().unwrap_or_default();
            error!("Whisperfile server error {}: {}", status, body);
            return Err(TranscribeError::Inference(format!("Whisperfile server error {}: {}", status, body)));
        }

        let json_response = response.into_body().read_to_string()
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;
        let whisperfile_output: WhisperfileOutput = serde_json::from_str(&json_response)
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        debug!(
            "Transcription completed in {:.2}s ({} chars)",
            start.elapsed().as_secs_f32(),
            whisperfile_output.text.len()
        );
        trace!("Transcription result: {:?}", whisperfile_output.text);

        Ok(whisperfile_output.into())
    }
}

impl Drop for WhisperfileEngine {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl SpeechModel for WhisperfileEngine {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = WhisperfileInferenceParams {
            language: language.map(|s| s.to_string()),
            ..Default::default()
        };
        self.transcribe_samples_inner(samples.to_vec(), Some(params))
    }
}
