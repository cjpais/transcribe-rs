use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use crate::features::{compute_mel, MelConfig, WindowType};
use super::session;
use crate::{ModelCapabilities, SpeechModel, TranscriptionResult};

/// GigaAM v3 e2e_ctc BPE vocabulary (257 tokens: 0-255 subwords + 256 blank).
const VOCAB: &[&str] = &[
    "<unk>", "▁", ".", "е", "а", "с", "и", ",", "о", "т", "н", "м", "у", "й",
    "л", "я", "в", "д", "з", "к", "но", "▁с", "ы", "г", "▁в", "б", "р", "п",
    "то", "ть", "ра", "▁по", "ка", "ш", "ни", "ли", "на", "го", "х", "ро",
    "ва", "▁на", "ю", "ко", "ль", "те", "?", "ч", "ж", "во", "ла", "ре",
    "да", "▁и", "ло", "ст", "-", "ё", "▁не", "ле", "ри", "де", "та", "ны",
    "▁В", "▁С", "ь", "ки", "ер", "▁о", "ви", "ти", "ма", "▁за", "▁А", "▁Т",
    "▁у", "же", "э", "▁М", "ц", "ди", "не", "ру", "че", "ф", "ве", "▁Д",
    "бо", "▁К", "щ", "▁О", "ми", "▁что", "▁«", "»", "ся", "▁По", "▁про",
    "e", "a", "ку", "ну", "▁это", "мо", "жи", "▁ко", "▁П", "▁И", "ча", "му",
    "0", "ты", "ста", "сь", "▁как", "o", "▁мо", "i", "до", "ля", "▁до",
    "▁от", "У", "Б", "ры", "чи", "ци", "▁бы", "▁Включи", "па", "ключ", "по",
    "ду", "▁при", "\u{2014}", "Л", "n", "Р", "сто", "r", "▁так", "сти", "Г",
    "▁На", "Н", "▁об", "▁мне", "l", "Я", "t", "1", "▁За", "s", "Э", "Ч",
    "Е", "▁есть", "ень", "▁Ну", "2", "▁Сбер", "вер", "▁вот", "ение", "смотр",
    "В", "▁раз", "Ф", "▁пере", "ешь", "▁тебя", "u", "3", "5", "d", "y", "Х",
    "4", "З", "S", "С", "h", "c", "m", "9", ":", "8", "6", "7", "M", "B",
    "П", "D", "T", "!", "k", "g", "О", "C", "Ш", "М", "A", "p", "Ю", "P",
    "Т", "К", "А", "L", "b", "Д", "ъ", "H", "%", "F", "v", "V", "R", "O",
    "I", "И", "N", "Ж", "\"", "K", "G", "Ц", "f", "w", "E", "₽", "W", "J",
    "x", "z", "'", "U", "Y", "&", "Z", "X", "+", "/", "Щ", ";", "j", "Й",
    "q", "Q", "°", "Ё", "Ы", "€", "$", "«",
];
const BLANK_ID: usize = 256;

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "GigaAM",
    languages: &["ru"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

pub struct GigaAMModel {
    session: Session,
    mel_config: MelConfig,
}

impl GigaAMModel {
    pub fn load(model_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        if !model_path.exists() {
            return Err(format!("Model file not found: {}", model_path.display()).into());
        }

        log::info!("Loading GigaAM model from {:?}...", model_path);
        let session = session::create_session(model_path)?;

        let mel_config = MelConfig {
            sample_rate: 16000,
            num_mels: 64,
            n_fft: 320,
            hop_length: 160,
            window: WindowType::Hann,
            f_min: 0.0,
            f_max: Some(8000.0),
            pre_emphasis: None,
            snip_edges: false,
            normalize_samples: true,
        };

        Ok(Self {
            session,
            mel_config,
        })
    }

    fn infer(
        &mut self,
        samples: &[f32],
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        if samples.len() < self.mel_config.n_fft {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        // 1. Compute mel spectrogram [num_mels, time]
        let mel = compute_mel(samples, &self.mel_config);
        let time_steps = mel.shape()[1];

        log::debug!(
            "Mel spectrogram shape: [{}, {}]",
            mel.shape()[0],
            mel.shape()[1]
        );

        // 2. Prepare input tensors: features [1, n_mels, time], feature_lengths [1]
        let features = mel.insert_axis(ndarray::Axis(0)); // [1, 64, T]
        let features_dyn = features.into_dyn();
        let feature_lengths = ndarray::arr1(&[time_steps as i64]).into_dyn();

        // 3. Run ONNX forward pass
        let inputs = inputs! {
            "features" => TensorRef::from_array_view(features_dyn.view())?,
            "feature_lengths" => TensorRef::from_array_view(feature_lengths.view())?,
        };
        let outputs = self.session.run(inputs)?;

        // 4. Extract log_probs [1, T', vocab_size]
        let log_probs = outputs[0].try_extract_array::<f32>()?;
        let log_probs = log_probs.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        log::debug!("Log probs shape: {:?}", log_probs.shape());

        // 5. CTC greedy decode
        let text = ctc_greedy_decode_gigaam(&log_probs);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }
}

impl SpeechModel for GigaAMModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe(
        &mut self,
        samples: &[f32],
        _language: Option<&str>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        self.infer(samples)
    }
}

/// CTC greedy decoding specific to GigaAM (uses hardcoded VOCAB array).
fn ctc_greedy_decode_gigaam(log_probs: &ndarray::Array3<f32>) -> String {
    let time_steps = log_probs.shape()[1];
    let vocab_size = log_probs.shape()[2];

    let mut token_ids: Vec<usize> = Vec::with_capacity(time_steps);

    for t in 0..time_steps {
        let mut best_id = 0;
        let mut best_val = f32::NEG_INFINITY;
        for v in 0..vocab_size {
            let val = log_probs[[0, t, v]];
            if val > best_val {
                best_val = val;
                best_id = v;
            }
        }
        token_ids.push(best_id);
    }

    // Collapse consecutive duplicates and remove blanks
    let mut result = String::new();
    let mut prev_id: Option<usize> = None;

    for &id in &token_ids {
        if Some(id) == prev_id {
            continue;
        }
        prev_id = Some(id);

        if id == BLANK_ID || id >= VOCAB.len() {
            continue;
        }

        let token = VOCAB[id];

        if token == "<unk>" {
            continue;
        }

        if let Some(stripped) = token.strip_prefix('▁') {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(stripped);
        } else {
            result.push_str(token);
        }
    }

    result.trim().to_string()
}
