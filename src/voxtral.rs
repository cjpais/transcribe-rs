//! Voxtral transcription engine backed by llama.cpp's libmtmd.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use llama_cpp_4::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel, Special},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
    sampling::LlamaSampler,
};

use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

const SAMPLE_RATE: u32 = 16_000;
const DEFAULT_CONTEXT_SIZE: u32 = 4096;
const DEFAULT_BATCH_SIZE: u32 = 2048;
const DEFAULT_MAX_NEW_TOKENS: usize = 192;

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Voxtral",
    engine_id: "voxtral",
    sample_rate: SAMPLE_RATE,
    languages: &["en", "fr", "de", "es", "it", "pt", "nl", "hi"],
    supports_timestamps: false,
    supports_translation: true,
    supports_streaming: false,
};

/// Runtime knobs for Voxtral generation.
#[derive(Debug, Clone)]
pub struct VoxtralParams {
    /// Language hint. When omitted, Voxtral auto-detects the language.
    pub language: Option<String>,
    /// Ask Voxtral to translate the audio to English instead of transcribing in the source language.
    pub translate: bool,
    /// Maximum autoregressive tokens to emit.
    pub max_new_tokens: usize,
    /// Number of llama.cpp layers to offload. Use a large number to request full offload.
    pub n_gpu_layers: u32,
    /// Context window for each transcription request.
    pub context_size: u32,
    /// Batch size for prompt/audio evaluation.
    pub batch_size: u32,
    /// Threads used by llama.cpp and libmtmd.
    pub n_threads: Option<i32>,
}

impl Default for VoxtralParams {
    fn default() -> Self {
        Self {
            language: None,
            translate: false,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            n_gpu_layers: 99,
            context_size: DEFAULT_CONTEXT_SIZE,
            batch_size: DEFAULT_BATCH_SIZE,
            n_threads: None,
        }
    }
}

/// Loaded Voxtral GGUF model plus multimodal projector.
pub struct VoxtralModel {
    mtmd: MtmdContext,
    model: LlamaModel,
    backend: LlamaBackend,
    params: VoxtralParams,
}

impl VoxtralModel {
    /// Load a Voxtral model directory.
    ///
    /// The directory must contain one text-model `.gguf` and one `mmproj*.gguf` file.
    pub fn load(model_dir: &Path) -> Result<Self, TranscribeError> {
        let model_path = find_model_file(model_dir)?;
        let mmproj_path = find_mmproj_file(model_dir)?;
        Self::load_files(&model_path, &mmproj_path, VoxtralParams::default())
    }

    /// Load explicit text-model and mmproj GGUF files.
    pub fn load_files(
        model_path: &Path,
        mmproj_path: &Path,
        params: VoxtralParams,
    ) -> Result<Self, TranscribeError> {
        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path.to_path_buf()));
        }
        if !mmproj_path.exists() {
            return Err(TranscribeError::ModelNotFound(mmproj_path.to_path_buf()));
        }

        let backend = map_llama(LlamaBackend::init(), "initializing llama.cpp backend")?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(params.n_gpu_layers);
        let model = map_llama(
            LlamaModel::load_from_file(&backend, model_path, &model_params),
            "loading Voxtral GGUF model",
        )?;

        let mut mtmd_params = MtmdContextParams::default().use_gpu(params.n_gpu_layers > 0);
        if let Some(n_threads) = params.n_threads {
            mtmd_params = mtmd_params.n_threads(n_threads);
        }
        let mtmd = map_llama(
            MtmdContext::init_from_file(mmproj_path, &model, mtmd_params),
            "loading Voxtral mmproj",
        )?;
        if !mtmd.supports_audio() {
            return Err(TranscribeError::Config(
                "Voxtral mmproj does not report audio support".to_string(),
            ));
        }

        Ok(Self {
            mtmd,
            model,
            backend,
            params,
        })
    }

    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &VoxtralParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        if samples.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        let text = self.generate(samples, params)?;
        Ok(TranscriptionResult {
            text: clean_generated_text(&text),
            segments: None,
        })
    }

    fn generate(
        &mut self,
        samples: &[f32],
        params: &VoxtralParams,
    ) -> Result<String, TranscribeError> {
        let context_size = NonZeroU32::new(params.context_size).ok_or_else(|| {
            TranscribeError::Config("Voxtral context_size must be greater than zero".to_string())
        })?;

        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(context_size))
            .with_n_batch(params.batch_size)
            .with_n_ubatch(params.batch_size.min(512))
            .with_flash_attention(true);
        if let Some(n_threads) = params.n_threads {
            ctx_params = ctx_params
                .with_n_threads(n_threads)
                .with_n_threads_batch(n_threads);
        }
        let mut ctx = map_llama(
            self.model.new_context(&self.backend, ctx_params),
            "creating Voxtral context",
        )?;

        let bitmap = map_llama(
            MtmdBitmap::from_audio(samples),
            "creating Voxtral audio input",
        )?;
        let prompt = self.build_prompt(params)?;
        let input_text = MtmdInputText::new(&prompt, true, true);
        let mut chunks = MtmdInputChunks::new();
        map_llama(
            self.mtmd.tokenize(&input_text, &[&bitmap], &mut chunks),
            "tokenizing Voxtral prompt",
        )?;

        let mut n_past = 0_i32;
        let n_batch = ctx.n_batch() as i32;
        map_llama(
            self.mtmd
                .eval_chunks(ctx.as_ptr(), &chunks, 0, 0, n_batch, true, &mut n_past),
            "evaluating Voxtral audio prompt",
        )?;

        let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
        let mut batch = LlamaBatch::new(params.batch_size as usize, 1);
        let mut generated = Vec::new();
        let mut pos = n_past;

        for _ in 0..params.max_new_tokens {
            let token = sampler.sample(&ctx, -1);
            if self.model.is_eog_token(token) {
                break;
            }

            generated.extend(map_llama(
                self.model.token_to_bytes(token, Special::Plaintext),
                "decoding Voxtral token",
            )?);
            sampler.accept(token);

            batch.clear();
            map_llama(batch.add(token, pos, &[0], true), "feeding Voxtral token")?;
            map_llama(ctx.decode(&mut batch), "decoding Voxtral token")?;
            pos += 1;
        }

        Ok(String::from_utf8_lossy(&generated).into_owned())
    }

    fn build_prompt(&self, params: &VoxtralParams) -> Result<String, TranscribeError> {
        let marker = MtmdContext::default_marker();
        let task = match (params.translate, params.language.as_deref()) {
            (true, Some(language)) => {
                let language = display_language_name(language);
                format!(
                    "Translate the {language} speech to English. Return only the English translation."
                )
            }
            (true, None) => "Translate the speech to English. Return only the English translation."
                .to_string(),
            (false, Some(language)) => {
                let language = display_language_name(language);
                format!(
                    "Transcribe the {language} speech exactly in {language}. Translation is disabled. Do not translate to any other language. Return only the transcript text, without timestamps."
                )
            }
            (false, None) => {
                "Detect the spoken language. Transcribe the speech verbatim in the detected language. Translation is disabled. Do not translate to any other language. Return only the transcript text, without timestamps."
                    .to_string()
            }
        };
        Ok(format!("[INST] {marker}\n{task}[/INST]"))
    }
}

impl SpeechModel for VoxtralModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = VoxtralParams {
            language: options.language.clone(),
            translate: options.translate,
            ..self.params.clone()
        };
        self.transcribe_with(samples, &params)
    }
}

fn find_model_file(model_dir: &Path) -> Result<PathBuf, TranscribeError> {
    find_gguf(model_dir, |name| !name.starts_with("mmproj"))
}

fn find_mmproj_file(model_dir: &Path) -> Result<PathBuf, TranscribeError> {
    find_gguf(model_dir, |name| name.starts_with("mmproj"))
}

fn find_gguf(
    model_dir: &Path,
    predicate: impl Fn(&str) -> bool,
) -> Result<PathBuf, TranscribeError> {
    let mut candidates = std::fs::read_dir(model_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("gguf")
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(&predicate)
                    .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| TranscribeError::ModelNotFound(model_dir.to_path_buf()))
}

fn clean_generated_text(text: &str) -> String {
    strip_timestamp_markers(text)
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn display_language_name(language: &str) -> &str {
    match language {
        "de" => "German",
        "en" => "English",
        "es" => "Spanish",
        "fr" => "French",
        "it" => "Italian",
        "nl" => "Dutch",
        "pt" => "Portuguese",
        "hi" => "Hindi",
        other => other,
    }
}

fn strip_timestamp_markers(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(start) = rest.find('[') {
        output.push_str(&rest[..start]);
        let Some(relative_end) = rest[start..].find(']') else {
            output.push_str(&rest[start..]);
            return output;
        };

        let end = start + relative_end;
        let marker = &rest[start + 1..end];
        if looks_like_timestamp_marker(marker) {
            rest = &rest[end + 1..];
        } else {
            output.push_str(&rest[start..=end]);
            rest = &rest[end + 1..];
        }
    }

    output.push_str(rest);
    output
}

fn looks_like_timestamp_marker(marker: &str) -> bool {
    let marker = marker.trim();
    marker.contains('-')
        && marker.contains('m')
        && marker.contains('s')
        && marker
            .chars()
            .all(|ch| ch.is_ascii_digit() || matches!(ch, 'm' | 's' | '-' | '.' | ',' | ' ' | '\t'))
}

fn map_llama<T, E: std::fmt::Display>(
    result: Result<T, E>,
    context: &str,
) -> Result<T, TranscribeError> {
    result.map_err(|e| TranscribeError::Inference(format!("{context}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::clean_generated_text;

    #[test]
    fn removes_voxtral_timestamp_markers() {
        let text = "[ 0m0s694ms - 0m1s64ms ] Kannst du das Mistral Modell beschleunigen?";

        assert_eq!(
            clean_generated_text(text),
            "Kannst du das Mistral Modell beschleunigen?"
        );
    }
}
