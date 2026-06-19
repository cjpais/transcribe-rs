//! NVIDIA Nemotron streaming ASR (offline engine).
//!
//! Cache-aware FastConformer + RNN-T, run as an offline [`SpeechModel`] by
//! feeding the whole utterance through the streaming encoder in fixed chunks
//! (threading the encoder cache between chunks) and greedily decoding. Supports
//! both the English-only 0.6B and the multilingual 3.5 0.6B variants —
//! auto-detected from the encoder graph: the multilingual export exposes a
//! `prompt_index` input for language conditioning.
//!
//! Ported from parakeet-rs (MIT), reusing transcribe-rs's `onnx::session`,
//! `rustfft`, and `ndarray` — no new dependencies. See `NEMOTRON_PORT.md` at
//! the repo root for the full implementation spec.
//!
//! Streaming/partial-result output, multi-stream sharing, and per-token
//! confidences are intentionally out of scope here (offline text first).

mod encoder;
mod mel;
mod tokenizer;

use std::path::Path;

use ndarray::{s, Array2, Array3};
use ort::session::Session;

use super::session;
use super::Quantization;
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
};

use encoder::{run_decoder, run_encoder, EncoderCache};
use mel::{log_mel_spectrogram, nemotron_mel_basis, N_MELS};
use tokenizer::SentencePieceVocab;

/// Mel frames fed as the "main" portion of each streaming chunk.
const CHUNK_SIZE: usize = 56;
/// Mel frames of left context prepended to each chunk (the encoder's
/// pre-encode cache region).
const PRE_ENCODE_CACHE: usize = 9;
/// Greedy decode cap on tokens emitted per encoder frame (RNN-T can emit
/// several non-blank tokens before advancing time).
const MAX_SYMBOLS_PER_STEP: usize = 10;
/// Prompt index for language-agnostic ("auto") decoding on the multilingual
/// model — the model picks the language itself.
const AUTO_PROMPT_INDEX: i64 = 101;
/// Fallback decoder LSTM dims if they can't be read from the graph.
const DECODER_LSTM_LAYERS: usize = 2;
const DECODER_LSTM_DIM: usize = 640;

const CAPS_EN: ModelCapabilities = ModelCapabilities {
    name: "Nemotron",
    engine_id: "nemotron",
    sample_rate: 16000,
    languages: &["en"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

const CAPS_MULTI: ModelCapabilities = ModelCapabilities {
    name: "Nemotron",
    engine_id: "nemotron",
    sample_rate: 16000,
    languages: MULTILINGUAL_LANGS,
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Base language codes the multilingual model handles (transcription-ready +
/// broad-coverage + adaptation tiers). [`PROMPT_DICTIONARY`] additionally
/// accepts locale variants (e.g. `en-US`, `pt-BR`) and `auto`.
const MULTILINGUAL_LANGS: &[&str] = &[
    "en", "es", "fr", "it", "pt", "nl", "de", "tr", "ru", "ar", "hi", "ja", "ko", "vi", "uk", "pl",
    "sv", "cs", "da", "bg", "fi", "hr", "sk", "zh", "hu", "ro", "et", "el", "nb", "lt", "lv", "sl",
];

/// Which Nemotron variant a loaded model is. Detected from the encoder graph
/// (the multilingual export exposes a `prompt_index` input).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NemotronMode {
    /// English-only 0.6B (no language conditioning).
    EnglishOnly,
    /// Multilingual 3.5 0.6B (`prompt_index` language conditioning).
    Multilingual,
}

/// Language → prompt embedding index for the multilingual model. Mirrors
/// `cfg.model_defaults.prompt_dictionary` from the `.nemo`, embedded here so a
/// sidecar `config.json` is not required next to the ONNX files.
///
/// See: <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>
const PROMPT_DICTIONARY: &[(&str, i64)] = &[
    ("af-ZA", 54), ("am-ET", 49), ("ar", 7), ("ar-AR", 7), ("auto", 101),
    ("ay-BO", 81), ("az-AZ", 66), ("bg", 30), ("bg-BG", 30), ("bn-IN", 36),
    ("cs", 22), ("cs-CZ", 22), ("da", 25), ("da-DK", 25), ("de", 9),
    ("de-DE", 9), ("el", 21), ("el-GR", 21), ("en", 0), ("en-GB", 1),
    ("en-US", 0), ("enGB", 1), ("es", 3), ("es-ES", 2), ("es-US", 3),
    ("esES", 2), ("et", 60), ("et-EE", 60), ("fa-IR", 38), ("fi", 26),
    ("fi-FI", 26), ("fr", 8), ("fr-CA", 100), ("fr-FR", 8), ("gn-PY", 82),
    ("gu-IN", 42), ("ha-NG", 50), ("haw-US", 97), ("he-IL", 64), ("hi", 6),
    ("hi-HI", 6), ("hi-IN", 6), ("hr", 29), ("hr-HR", 29), ("hu", 23),
    ("hu-HU", 23), ("hy-AM", 68), ("id-ID", 34), ("ig-NG", 53), ("it", 15),
    ("it-IT", 15), ("ja-JA", 10), ("ja-JP", 10), ("ka-GE", 67), ("km-KH", 47),
    ("kn-IN", 43), ("ko", 14), ("ko-KO", 14), ("ko-KR", 14), ("ku-TR", 65),
    ("ky-KG", 71), ("ln-CD", 58), ("lt", 31), ("lt-LT", 31), ("lv", 61),
    ("lv-LV", 61), ("mi-NZ", 96), ("ml-IN", 44), ("mr-IN", 41), ("ms-MY", 35),
    ("mt-MT", 102), ("nah-MX", 83), ("nb", 103), ("nb-NO", 103), ("ne-NP", 46),
    ("nl", 16), ("nl-NL", 16), ("nn", 104), ("nn-NO", 104), ("no", 27),
    ("no-NO", 27), ("ny-MW", 57), ("or-KE", 59), ("pl", 17), ("pl-PL", 17),
    ("pt", 13), ("pt-BR", 12), ("pt-PT", 13), ("qu-PE", 80), ("ro", 20),
    ("ro-RO", 20), ("ru", 11), ("ru-RU", 11), ("rw-RW", 55), ("si-LK", 45),
    ("sk", 28), ("sk-SK", 28), ("sl", 62), ("sl-SI", 62), ("sm-WS", 98),
    ("so-SO", 56), ("sv", 24), ("sv-SE", 24), ("sw-KE", 48), ("ta-IN", 39),
    ("te-IN", 40), ("tg-TJ", 70), ("th-TH", 32), ("to-TO", 99), ("tr", 18),
    ("tr-TR", 18), ("uk", 19), ("uk-UA", 19), ("ur-PK", 37), ("uz-UZ", 69),
    ("vi-VN", 33), ("yo-NG", 52), ("zh-CN", 4), ("zh-TW", 5), ("zh-ZH", 4),
    ("zu-ZA", 51),
];

/// Per-model inference parameters for Nemotron.
#[derive(Debug, Clone, Default)]
pub struct NemotronParams {
    /// Target language for the multilingual model (e.g. `"en-US"`, `"es-ES"`,
    /// `"auto"`). Ignored by the English-only model. When `None`, the
    /// multilingual model decodes in `auto` (language-agnostic) mode.
    pub language: Option<String>,
}

/// Loaded Nemotron streaming model (offline).
pub struct NemotronModel {
    encoder: Session,
    decoder: Session,
    vocab: SentencePieceVocab,
    mel_basis: Array2<f32>,
    mode: NemotronMode,
    num_layers: usize,
    hidden: usize,
    left_context: usize,
    conv_context: usize,
    lstm_layers: usize,
    lstm_dim: usize,
    vocab_size: usize,
    blank_id: usize,
    /// Empty for English-only; the `<xx-XX>` token ids for multilingual.
    lang_tag_ids: Vec<usize>,
}

impl NemotronModel {
    /// Load the model from a directory containing `encoder.onnx`
    /// (+ `encoder.onnx.data`), `decoder_joint.onnx`, and `tokenizer.model`.
    /// Quantized variants (`encoder.int8.onnx`, ...) are selected via
    /// `quantization`, falling back to FP32.
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let encoder_path = session::resolve_model_path(model_dir, "encoder", quantization);
        let decoder_path = session::resolve_model_path(model_dir, "decoder_joint", quantization);
        if !encoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(encoder_path));
        }
        if !decoder_path.exists() {
            return Err(TranscribeError::ModelNotFound(decoder_path));
        }

        let encoder = session::create_session(&encoder_path)?;
        let decoder = session::create_session(&decoder_path)?;

        let vocab = SentencePieceVocab::from_file(model_dir.join("tokenizer.model"))?;
        let vocab_size = vocab.size();
        let blank_id = vocab_size;

        // Read encoder dims (and detect the multilingual prompt input) straight
        // from the graph; fall back to the documented 0.6B defaults.
        let mut num_layers = 24usize;
        let mut hidden = 1024usize;
        let mut left_context = 70usize;
        let mut conv_context = 8usize;
        let mut has_prompt = false;
        for input in encoder.inputs() {
            let name = input.name();
            if name == "prompt_index" {
                has_prompt = true;
                continue;
            }
            if let Some(shape) = input.dtype().tensor_shape() {
                match name {
                    "cache_last_channel" if shape.len() == 4 => {
                        if shape[0] > 0 {
                            num_layers = shape[0] as usize;
                        }
                        if shape[2] > 0 {
                            left_context = shape[2] as usize;
                        }
                        if shape[3] > 0 {
                            hidden = shape[3] as usize;
                        }
                    }
                    "cache_last_time" if shape.len() == 4 && shape[3] > 0 => {
                        conv_context = shape[3] as usize;
                    }
                    _ => {}
                }
            }
        }

        // Read decoder LSTM state dims from `input_states_1` ([layers, 1, dim]).
        let (lstm_layers, lstm_dim) = decoder
            .inputs()
            .iter()
            .find(|i| i.name() == "input_states_1")
            .and_then(|i| i.dtype().tensor_shape().map(|s| s.to_vec()))
            .filter(|s| s.len() == 3 && s[0] > 0 && s[2] > 0)
            .map(|s| (s[0] as usize, s[2] as usize))
            .unwrap_or((DECODER_LSTM_LAYERS, DECODER_LSTM_DIM));

        let mode = if has_prompt {
            NemotronMode::Multilingual
        } else {
            NemotronMode::EnglishOnly
        };
        let lang_tag_ids = if mode == NemotronMode::Multilingual {
            vocab.lang_tag_ids()
        } else {
            Vec::new()
        };

        log::info!(
            "Loaded Nemotron ({:?}): {} layers, hidden {}, left_context {}, vocab {}",
            mode,
            num_layers,
            hidden,
            left_context,
            vocab_size
        );

        Ok(Self {
            encoder,
            decoder,
            vocab,
            mel_basis: nemotron_mel_basis(),
            mode,
            num_layers,
            hidden,
            left_context,
            conv_context,
            lstm_layers,
            lstm_dim,
            vocab_size,
            blank_id,
            lang_tag_ids,
        })
    }

    /// Which variant this model is (auto-detected at load).
    pub fn mode(&self) -> NemotronMode {
        self.mode
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        params: &NemotronParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let prompt_index = self.prompt_index_for(params.language.as_deref());
        let ids = self.process_audio(samples, prompt_index)?;
        let text = self.decode_ids(&ids);
        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    /// Map a language code to the multilingual prompt index. `None` for the
    /// English-only model; unknown codes fall back to `auto`.
    fn prompt_index_for(&self, language: Option<&str>) -> Option<i64> {
        match self.mode {
            NemotronMode::EnglishOnly => None,
            NemotronMode::Multilingual => Some(match language {
                Some(lang) => PROMPT_DICTIONARY
                    .iter()
                    .find_map(|(k, v)| (*k == lang).then_some(*v))
                    .unwrap_or_else(|| {
                        log::warn!("unknown Nemotron language '{lang}', falling back to auto");
                        AUTO_PROMPT_INDEX
                    }),
                None => AUTO_PROMPT_INDEX,
            }),
        }
    }

    /// Drop blank/out-of-vocab and language-tag tokens, then join to text.
    fn decode_ids(&self, ids: &[usize]) -> String {
        let kept: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|id| *id < self.vocab_size && !self.lang_tag_ids.contains(id))
            .collect();
        self.vocab.decode(&kept)
    }

    /// Run the full offline chunk loop, returning every emitted token id.
    fn process_audio(
        &mut self,
        audio: &[f32],
        prompt_index: Option<i64>,
    ) -> Result<Vec<usize>, TranscribeError> {
        let mel = log_mel_spectrogram(audio, &self.mel_basis);
        let total_frames = mel.shape()[1];
        if total_frames == 0 {
            return Ok(Vec::new());
        }

        let mut cache = EncoderCache::zeros(
            self.num_layers,
            self.left_context,
            self.hidden,
            self.conv_context,
        );
        let mut state_1 = Array3::<f32>::zeros((self.lstm_layers, 1, self.lstm_dim));
        let mut state_2 = Array3::<f32>::zeros((self.lstm_layers, 1, self.lstm_dim));
        let mut last_token = self.blank_id as i32;
        let mut ids = Vec::new();

        let expected = PRE_ENCODE_CACHE + CHUNK_SIZE;
        let mut buffer_idx = 0;
        let mut chunk_idx = 0;

        while buffer_idx < total_frames {
            let chunk_end = (buffer_idx + CHUNK_SIZE).min(total_frames);
            let main_len = chunk_end - buffer_idx;

            let mut chunk_data = vec![0.0f32; N_MELS * expected];

            // Pre-encode cache: the PRE_ENCODE_CACHE frames preceding this chunk
            // (zero-padded for the first chunk).
            if chunk_idx > 0 && buffer_idx >= PRE_ENCODE_CACHE {
                let cache_start = buffer_idx - PRE_ENCODE_CACHE;
                for f in 0..PRE_ENCODE_CACHE {
                    for m in 0..N_MELS {
                        chunk_data[m * expected + f] = mel[[m, cache_start + f]];
                    }
                }
            }
            // Main chunk frames.
            for f in 0..main_len {
                for m in 0..N_MELS {
                    chunk_data[m * expected + PRE_ENCODE_CACHE + f] = mel[[m, buffer_idx + f]];
                }
            }

            let mel_chunk = Array3::from_shape_vec((1, N_MELS, expected), chunk_data)?;
            let chunk_length = (PRE_ENCODE_CACHE + main_len) as i64;

            let (encoded, enc_len, next_cache) =
                run_encoder(&mut self.encoder, &mel_chunk, chunk_length, &cache, prompt_index)?;
            cache = next_cache;

            decode_chunk(
                &mut self.decoder,
                &encoded,
                enc_len as usize,
                self.blank_id,
                &mut last_token,
                &mut state_1,
                &mut state_2,
                &mut ids,
            )?;

            buffer_idx += CHUNK_SIZE;
            chunk_idx += 1;
        }

        Ok(ids)
    }
}

/// Greedy RNN-T decode over one chunk's encoded frames, appending emitted
/// (non-blank) token ids and carrying decoder state forward.
#[allow(clippy::too_many_arguments)]
fn decode_chunk(
    decoder: &mut Session,
    encoded: &Array3<f32>,
    enc_frames: usize,
    blank_id: usize,
    last_token: &mut i32,
    state_1: &mut Array3<f32>,
    state_2: &mut Array3<f32>,
    ids: &mut Vec<usize>,
) -> Result<(), TranscribeError> {
    let hidden = encoded.shape()[1];

    for t in 0..enc_frames {
        let col = encoded.slice(s![0, .., t]).to_owned();
        let frame = col.to_shape((1, hidden, 1))?.to_owned();

        for _ in 0..MAX_SYMBOLS_PER_STEP {
            let (logits, next_state_1, next_state_2) =
                run_decoder(decoder, &frame, *last_token, state_1, state_2)?;

            let (max_idx, _) = argmax(&logits);
            if max_idx == blank_id {
                break;
            }

            ids.push(max_idx);
            *last_token = max_idx as i32;
            *state_1 = next_state_1;
            *state_2 = next_state_2;
        }
    }

    Ok(())
}

/// Index of the first maximum in `values`.
fn argmax(values: &[f32]) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    (max_idx, max_val)
}

impl SpeechModel for NemotronModel {
    fn capabilities(&self) -> ModelCapabilities {
        match self.mode {
            NemotronMode::EnglishOnly => CAPS_EN,
            NemotronMode::Multilingual => CAPS_MULTI,
        }
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let params = NemotronParams {
            language: options.language.clone(),
        };
        self.transcribe_with(samples, &params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_dictionary_has_auto_and_aliases() {
        let lookup = |code: &str| PROMPT_DICTIONARY.iter().find_map(|(k, v)| (*k == code).then_some(*v));
        assert_eq!(lookup("auto"), Some(AUTO_PROMPT_INDEX));
        // en and en-US share index 0; en-GB is distinct.
        assert_eq!(lookup("en"), Some(0));
        assert_eq!(lookup("en-US"), Some(0));
        assert_eq!(lookup("en-GB"), Some(1));
        assert_eq!(lookup("xx-ZZ"), None);
    }

    #[test]
    fn argmax_picks_first_max() {
        assert_eq!(argmax(&[0.1, 0.9, 0.3, 0.9]).0, 1);
        assert_eq!(argmax(&[]).0, 0);
    }
}
