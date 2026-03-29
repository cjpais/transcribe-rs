//! Paraformer ONNX speech recognition engine.
//!
//! Non-autoregressive end-to-end ASR model from FunASR/ModelScope.
//! Uses its own fbank feature extraction (Hamming window, dB scale),
//! LFR frame stacking, and a custom symbol table with `@@` subword joining.

use ndarray::{Array1, Array2};
use ort::inputs;
use ort::session::Session;
use ort::value::TensorRef;
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;

use super::session;
use super::Quantization;
use crate::features::apply_lfr;
use crate::TranscribeError;
use crate::{ModelCapabilities, SpeechModel, TranscribeOptions, TranscriptionResult};

const CAPABILITIES: ModelCapabilities = ModelCapabilities {
    name: "Paraformer",
    engine_id: "paraformer",
    sample_rate: 16000,
    languages: &["zh", "en", "yue"],
    supports_timestamps: false,
    supports_translation: false,
    supports_streaming: false,
};

/// Per-model inference parameters for Paraformer.
#[derive(Debug, Clone, Default)]
pub struct ParaformerParams {
    /// Language hint (currently unused, Paraformer handles zh/en/yue automatically).
    pub language: Option<String>,
}

// ---- Metadata ----

struct ParaformerMetadata {
    lfr_window_size: usize,
    lfr_window_shift: usize,
    blank_id: i32,
    sos_id: i32,
    eos_id: i32,
}

// ---- Symbol Table ----

/// Paraformer-specific symbol table with `@@` subword joining and CJK-aware spacing.
struct ParaformerSymbolTable {
    id_to_sym: HashMap<i32, String>,
}

impl ParaformerSymbolTable {
    /// Load a tokens.txt file where each line is `symbol id`.
    fn load(path: &Path) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Split on last whitespace: "symbol id"
            if let Some(pos) = line.rfind(|c: char| c.is_ascii_whitespace()) {
                let sym = &line[..pos];
                let id_str = line[pos..].trim();
                if let Ok(id) = id_str.parse::<i32>() {
                    id_to_sym.insert(id, sym.to_string());
                }
            }
        }

        log::info!(
            "Loaded Paraformer symbol table with {} tokens from {:?}",
            id_to_sym.len(),
            path
        );
        Ok(Self { id_to_sym })
    }

    fn get(&self, id: i32) -> Option<&str> {
        self.id_to_sym.get(&id).map(|s| s.as_str())
    }

    /// Decode a sequence of token IDs into text.
    fn decode(&self, token_ids: &[i32]) -> String {
        let mut text = String::new();
        let mut prev_join_to_next = false;

        for &id in token_ids {
            let Some(sym) = self.get(id) else {
                continue;
            };
            if is_special_symbol(sym) {
                continue;
            }

            let joins_next = sym.ends_with("@@");
            let clean = sym.trim_end_matches("@@");
            if clean.is_empty() {
                prev_join_to_next = joins_next;
                continue;
            }

            if clean.starts_with('\u{2581}') {
                let piece = clean.trim_start_matches('\u{2581}');
                if !piece.is_empty() {
                    if !text.is_empty() && !text.ends_with(' ') {
                        text.push(' ');
                    }
                    text.push_str(piece);
                }
                prev_join_to_next = joins_next;
                continue;
            }

            if !text.is_empty() && !prev_join_to_next {
                let prev_char = text.chars().last();
                let curr_is_ascii_word = is_ascii_word_piece(clean);
                let prev_is_ascii_word = prev_char.map(is_ascii_word_char).unwrap_or(false);
                let prev_is_cjk = prev_char.map(is_cjk).unwrap_or(false);
                if curr_is_ascii_word && (prev_is_ascii_word || prev_is_cjk) && !text.ends_with(' ')
                {
                    text.push(' ');
                }
            }

            text.push_str(clean);
            prev_join_to_next = joins_next;
        }

        text.trim().to_string()
    }
}

fn is_special_symbol(sym: &str) -> bool {
    sym == "<blank>"
        || sym == "<s>"
        || sym == "</s>"
        || sym == "<unk>"
        || sym == "<OOV>"
        || (sym.starts_with('<') && sym.ends_with('>'))
}

fn is_ascii_word_piece(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_ascii_word_char)
}

fn is_ascii_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn is_cjk(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
}

// ---- Feature Extraction ----

/// Compute Paraformer-style fbank features (Hamming window, dB scale).
///
/// This is NOT the same as Kaldi fbank or the upstream `compute_mel()`.
/// Key differences:
/// - Standard Hamming window (no Povey modification)
/// - No preemphasis, no DC offset removal
/// - dB scale output: `10.0 * log10(energy)` with -80 dB floor
/// - snip_edges=true
///
/// Returns [num_frames, num_bins] (80 bins by default).
fn compute_paraformer_fbank(samples: &[f32]) -> Array2<f32> {
    const NUM_BINS: usize = 80;
    const FFT_SIZE: usize = 512;
    const WINDOW_SIZE: usize = 400;
    const HOP_SIZE: usize = 160;
    const SAMPLE_RATE: f32 = 16000.0;
    const LOW_FREQ: f32 = 0.0;
    const HIGH_FREQ: f32 = 8000.0;

    if samples.len() < WINDOW_SIZE {
        return Array2::zeros((0, NUM_BINS));
    }

    let num_frames = (samples.len() - WINDOW_SIZE) / HOP_SIZE + 1;
    let num_fft_bins = FFT_SIZE / 2 + 1;

    // Standard Hamming window
    let window: Vec<f32> = (0..WINDOW_SIZE)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (WINDOW_SIZE as f32 - 1.0)).cos())
        .collect();

    // Mel filterbank [NUM_BINS, num_fft_bins]
    let mel_banks = mel_filterbank(NUM_BINS, FFT_SIZE, SAMPLE_RATE, LOW_FREQ, HIGH_FREQ);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let mut features = Array2::zeros((num_frames, NUM_BINS));

    for i in 0..num_frames {
        let start = i * HOP_SIZE;

        // Extract and window the frame
        let mut fft_input: Vec<Complex<f32>> = Vec::with_capacity(FFT_SIZE);
        for j in 0..WINDOW_SIZE {
            let sample = if start + j < samples.len() {
                samples[start + j]
            } else {
                0.0
            };
            fft_input.push(Complex::new(sample * window[j], 0.0));
        }
        // Zero-pad to FFT_SIZE
        fft_input.resize(FFT_SIZE, Complex::new(0.0, 0.0));

        fft.process(&mut fft_input);

        // Power spectrum
        let power_spectrum: Vec<f32> = fft_input[..num_fft_bins]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();

        // Apply mel filterbank and convert to dB scale
        for m in 0..NUM_BINS {
            let energy: f32 = mel_banks
                .row(m)
                .iter()
                .zip(power_spectrum.iter())
                .map(|(&w, &p)| w * p)
                .sum();

            // dB scale: 10 * log10(energy), with -80 dB floor
            let db = if energy < 1.0e-10 {
                -80.0
            } else {
                10.0 * energy.log10()
            };
            features[[i, m]] = db.max(-80.0);
        }
    }

    features
}

/// Compute mel filterbank matrix of shape [num_mels, num_fft_bins].
fn mel_filterbank(
    num_mels: usize,
    fft_size: usize,
    sample_rate: f32,
    low_freq: f32,
    high_freq: f32,
) -> Array2<f32> {
    let num_fft_bins = fft_size / 2 + 1;

    let mel_low = hz_to_mel(low_freq);
    let mel_high = hz_to_mel(high_freq);

    let num_points = num_mels + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_points - 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&f| f * fft_size as f32 / sample_rate)
        .collect();

    let mut banks = Array2::zeros((num_mels, num_fft_bins));

    for m in 0..num_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..num_fft_bins {
            let kf = k as f32;
            if kf > left && kf < center {
                banks[[m, k]] = (kf - left) / (center - left);
            } else if kf >= center && kf < right {
                banks[[m, k]] = (right - kf) / (right - center);
            }
        }
    }

    banks
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

// ---- CMVN ----

/// Apply mean-only CMVN normalization (subtract mean, no stddev scaling).
fn apply_mean_cmvn(features: &mut Array2<f32>, mean: &Array1<f32>) {
    let ncols = features.ncols();
    for mut row in features.rows_mut() {
        for j in 0..ncols {
            row[j] -= mean[j];
        }
    }
}

/// Load CMVN mean from an `am.mvn` file (Kaldi-style format).
///
/// Parses the `<LearnRateCoef>` section and extracts the mean vector.
/// Returns `None` if the file doesn't contain the expected format.
fn load_cmvn_mean(path: &Path, target_dim: usize) -> Result<Option<Array1<f32>>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let Some(start_idx) = content.find("<LearnRateCoef>") else {
        return Ok(None);
    };
    let rest = &content[start_idx..];
    let Some(lb_rel) = rest.find('[') else {
        return Ok(None);
    };
    let Some(rb_rel) = rest.find(']') else {
        return Ok(None);
    };
    if rb_rel <= lb_rel {
        return Ok(None);
    }
    let body = &rest[lb_rel + 1..rb_rel];
    let mut values = Vec::new();
    for tok in body.split_whitespace() {
        if let Ok(v) = tok.parse::<f32>() {
            values.push(v);
        }
    }
    if values.len() < target_dim {
        return Ok(None);
    }
    Ok(Some(Array1::from_vec(
        values.into_iter().take(target_dim).collect(),
    )))
}

// ---- Model ----

pub struct ParaformerModel {
    session: Session,
    symbol_table: ParaformerSymbolTable,
    metadata: ParaformerMetadata,
    cmvn_mean: Option<Array1<f32>>,
    speech_input_name: String,
    speech_lengths_input_name: String,
    #[allow(dead_code)]
    logits_output_name: String,
    #[allow(dead_code)]
    token_num_output_name: Option<String>,
}

impl ParaformerModel {
    pub fn load(model_dir: &Path, quantization: &Quantization) -> Result<Self, TranscribeError> {
        let model_path = session::resolve_model_path(model_dir, "model", quantization);
        let tokens_path = model_dir.join("tokens.txt");
        let cmvn_path = model_dir.join("am.mvn");

        if !model_path.exists() {
            return Err(TranscribeError::ModelNotFound(model_path));
        }
        if !tokens_path.exists() {
            return Err(TranscribeError::ModelNotFound(tokens_path));
        }

        log::info!("Loading Paraformer model from {:?}...", model_path);
        let session = session::create_session(&model_path)?;

        // Read metadata from ONNX model
        let lfr_window_size =
            session::read_metadata_i32(&session, "lfr_window_size", Some(7))?.unwrap() as usize;
        let lfr_window_shift =
            session::read_metadata_i32(&session, "lfr_window_shift", Some(6))?.unwrap() as usize;
        let blank_id = session::read_metadata_i32(&session, "blank_id", Some(0))?.unwrap();
        let sos_id = session::read_metadata_i32(&session, "sos_id", Some(1))?.unwrap();
        let eos_id = session::read_metadata_i32(&session, "eos_id", Some(2))?.unwrap();

        let metadata = ParaformerMetadata {
            lfr_window_size,
            lfr_window_shift,
            blank_id,
            sos_id,
            eos_id,
        };

        log::info!(
            "Paraformer metadata: lfr_window={}x{}, blank={}, sos={}, eos={}",
            lfr_window_size,
            lfr_window_shift,
            blank_id,
            sos_id,
            eos_id,
        );

        // Detect I/O names from session
        let inputs: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let outputs: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        let speech_input_name = inputs
            .iter()
            .find(|n| n.contains("speech"))
            .cloned()
            .unwrap_or_else(|| {
                inputs
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "speech".to_string())
            });

        let speech_lengths_input_name = inputs
            .iter()
            .find(|n| n.contains("speech_lengths") || n.contains("length"))
            .cloned()
            .unwrap_or_else(|| {
                inputs
                    .get(1)
                    .cloned()
                    .unwrap_or_else(|| "speech_lengths".to_string())
            });

        let logits_output_name = outputs
            .iter()
            .find(|n| n.contains("logits"))
            .cloned()
            .unwrap_or_else(|| {
                outputs
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "logits".to_string())
            });

        let token_num_output_name = outputs.iter().find(|n| n.contains("token_num")).cloned();

        log::debug!(
            "I/O names: speech={}, lengths={}, logits={}, token_num={:?}",
            speech_input_name,
            speech_lengths_input_name,
            logits_output_name,
            token_num_output_name,
        );

        // Load symbol table
        let symbol_table = ParaformerSymbolTable::load(&tokens_path)?;

        // Load CMVN mean (LFR dim = 80 * lfr_window_size)
        let lfr_dim = 80 * lfr_window_size;
        let cmvn_mean = if cmvn_path.exists() {
            match load_cmvn_mean(&cmvn_path, lfr_dim) {
                Ok(mean) => {
                    if mean.is_some() {
                        log::info!("Loaded CMVN mean from {:?} (dim={})", cmvn_path, lfr_dim);
                    }
                    mean
                }
                Err(e) => {
                    log::warn!("Failed to load CMVN from {:?}: {}", cmvn_path, e);
                    None
                }
            }
        } else {
            log::debug!("No am.mvn file found at {:?}, skipping CMVN", cmvn_path);
            None
        };

        Ok(Self {
            session,
            symbol_table,
            metadata,
            cmvn_mean,
            speech_input_name,
            speech_lengths_input_name,
            logits_output_name,
            token_num_output_name,
        })
    }

    /// Transcribe with model-specific parameters.
    pub fn transcribe_with(
        &mut self,
        samples: &[f32],
        _params: &ParaformerParams,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }

    fn infer(&mut self, samples: &[f32]) -> Result<TranscriptionResult, TranscribeError> {
        // 1. Compute Paraformer fbank features [frames, 80]
        let features = compute_paraformer_fbank(samples);

        if features.nrows() == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        log::debug!(
            "Paraformer fbank: [{}, {}]",
            features.nrows(),
            features.ncols()
        );

        // 2. Apply LFR
        let features = apply_lfr(
            &features,
            self.metadata.lfr_window_size,
            self.metadata.lfr_window_shift,
        );

        log::debug!("After LFR: [{}, {}]", features.nrows(), features.ncols());

        if features.nrows() == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: None,
            });
        }

        // 3. Apply mean-only CMVN
        let mut features = features;
        if let Some(ref mean) = self.cmvn_mean {
            apply_mean_cmvn(&mut features, mean);
        }

        // 4. Forward pass
        let logits = self.forward(&features)?;

        log::debug!("Logits shape: {:?}", logits.shape());

        // 5. Decode (non-autoregressive argmax, NOT CTC)
        let token_ids = self.decode_logits(&logits);

        // 6. Convert to text
        let text = self.symbol_table.decode(&token_ids);

        Ok(TranscriptionResult {
            text,
            segments: None,
        })
    }

    /// Run ONNX forward pass. Returns logits [1, T, vocab_size].
    fn forward(&mut self, features: &Array2<f32>) -> Result<ndarray::Array3<f32>, TranscribeError> {
        let num_frames = features.nrows();

        // Shape: [1, T, D]
        let feat_3d =
            features
                .to_owned()
                .into_shape_with_order((1, num_frames, features.ncols()))?;
        let speech_lengths = ndarray::arr1(&[num_frames as i32]);

        let feat_dyn = feat_3d.into_dyn();
        let lengths_dyn = speech_lengths.into_dyn();

        let t_feat = TensorRef::from_array_view(feat_dyn.view())?;
        let t_lengths = TensorRef::from_array_view(lengths_dyn.view())?;

        let ort_inputs = inputs![
            self.speech_input_name.as_str() => t_feat,
            self.speech_lengths_input_name.as_str() => t_lengths,
        ];

        let outputs = self.session.run(ort_inputs)?;

        let logits = outputs[0].try_extract_array::<f32>()?;
        let logits_owned = logits.to_owned().into_dimensionality::<ndarray::Ix3>()?;

        Ok(logits_owned)
    }

    /// Decode logits using argmax with blank/sos/eos filtering.
    ///
    /// Paraformer is non-autoregressive, so this is a simple argmax over the
    /// vocab dimension, NOT CTC greedy decode.
    fn decode_logits(&self, logits: &ndarray::Array3<f32>) -> Vec<i32> {
        let blank_id = self.metadata.blank_id;
        let sos_id = self.metadata.sos_id;
        let eos_id = self.metadata.eos_id;

        let seq_len = logits.shape()[1];
        let mut token_ids = Vec::new();

        for t in 0..seq_len {
            // Argmax over vocab dimension
            let mut best_id = 0i32;
            let mut best_val = f32::NEG_INFINITY;
            for (v, &val) in logits.slice(ndarray::s![0, t, ..]).iter().enumerate() {
                if val > best_val {
                    best_val = val;
                    best_id = v as i32;
                }
            }

            // Skip blank, sos, eos
            if best_id == blank_id || best_id == sos_id || best_id == eos_id {
                continue;
            }

            token_ids.push(best_id);
        }

        token_ids
    }
}

impl SpeechModel for ParaformerModel {
    fn capabilities(&self) -> ModelCapabilities {
        CAPABILITIES
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.infer(samples)
    }
}
