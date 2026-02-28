//! Shared utilities for Zipformer-based engines (CTC, Transducer).
//!
//! Contains Kaldi-compatible FBank feature extraction, BBPE byte decoding,
//! and the symbol table used by sherpa-onnx Zipformer models.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

/// Kaldi-compatible FBank configuration matching sherpa-onnx / kaldi-native-fbank.
#[derive(Debug, Clone)]
pub struct FbankConfig {
    pub num_bins: usize,
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_size: usize,
    pub sample_rate: i32,
    pub low_freq: f32,
    /// Negative means nyquist + high_freq (Kaldi convention). -400 → 7600 Hz at 16kHz.
    pub high_freq: f32,
    pub preemph_coeff: f32,
    pub snip_edges: bool,
    pub remove_dc_offset: bool,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            num_bins: 80,
            fft_size: 512,
            window_size: 400,
            hop_size: 160,
            sample_rate: 16000,
            low_freq: 20.0,
            high_freq: -400.0,
            preemph_coeff: 0.97,
            snip_edges: false,
            remove_dc_offset: true,
        }
    }
}

// ============== Symbol Table ==============

/// Whether tokens use BBPE byte encoding or standard BPE (literal UTF-8).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenEncoding {
    /// Icefall BBPE: token chars are byte-to-unicode mapped, need decoding.
    Bbpe,
    /// Standard BPE/sentencepiece: token strings are literal UTF-8.
    Bpe,
}

pub struct SymbolTable {
    id_to_sym: HashMap<i32, String>,
    encoding: TokenEncoding,
}

impl SymbolTable {
    /// Load a symbol table with BBPE encoding (default for zh-en models).
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        Self::load_with_encoding(path, TokenEncoding::Bbpe)
    }

    /// Load a symbol table, auto-detecting encoding from sibling files.
    /// If `bbpe.model` exists in the same directory, use BBPE; otherwise standard BPE.
    pub fn load_autodetect(path: &Path) -> Result<Self, std::io::Error> {
        let encoding = if let Some(dir) = path.parent() {
            if dir.join("bbpe.model").exists() {
                log::info!("Detected BBPE encoding (bbpe.model found)");
                TokenEncoding::Bbpe
            } else {
                log::info!("Detected standard BPE encoding (no bbpe.model)");
                TokenEncoding::Bpe
            }
        } else {
            TokenEncoding::Bbpe
        };
        Self::load_with_encoding(path, encoding)
    }

    pub fn load_with_encoding(path: &Path, encoding: TokenEncoding) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();

        for line in contents.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }

            // Format: "token id" (whitespace separated, token can contain spaces)
            let parts: Vec<&str> = line.rsplitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() != 2 {
                continue;
            }

            if let Ok(id) = parts[0].parse::<i32>() {
                id_to_sym.insert(id, parts[1].to_string());
            }
        }

        Ok(Self { id_to_sym, encoding })
    }

    pub fn get(&self, id: i32) -> Option<&str> {
        self.id_to_sym.get(&id).map(|s| s.as_str())
    }

    /// Decode token IDs to text, using the appropriate encoding strategy.
    pub fn decode(&self, token_ids: &[i32]) -> String {
        match self.encoding {
            TokenEncoding::Bbpe => self.decode_bbpe(token_ids),
            TokenEncoding::Bpe => self.decode_bpe(token_ids),
        }
    }

    /// Decode BBPE token IDs to text.
    ///
    /// Icefall BBPE tokens use a byte-to-unicode mapping (PRINTABLE_BASE_CHARS).
    /// We collect all token chars, map each back to a byte, then interpret as UTF-8.
    fn decode_bbpe(&self, token_ids: &[i32]) -> String {
        let mut raw_bytes = Vec::new();

        for &id in token_ids {
            let Some(sym) = self.get(id) else {
                continue;
            };
            if sym.starts_with('<') && sym.ends_with('>') {
                continue;
            }
            for c in sym.chars() {
                if c == '\u{2581}' {
                    // ▁ is the sentencepiece space marker
                    raw_bytes.push(b' ');
                } else if let Some(byte_val) = bbpe_char_to_byte(c) {
                    raw_bytes.push(byte_val);
                }
            }
        }

        let text = String::from_utf8_lossy(&raw_bytes);
        text.trim().to_string()
    }

    /// Decode standard BPE/sentencepiece tokens (literal UTF-8 strings).
    fn decode_bpe(&self, token_ids: &[i32]) -> String {
        let mut text = String::new();

        for &id in token_ids {
            let Some(sym) = self.get(id) else {
                continue;
            };
            if sym.starts_with('<') && sym.ends_with('>') {
                continue;
            }
            // ▁ is the sentencepiece space marker
            text.push_str(&sym.replace('\u{2581}', " "));
        }

        text.trim().to_string()
    }
}

// ============== Icefall BBPE byte mapping ==============

/// Icefall PRINTABLE_BASE_CHARS: maps byte index (0-255) to a Unicode codepoint.
/// Source: icefall/byte_utils.py
const BBPE_CODEPOINTS: [u32; 256] = [
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
    286, 287, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 288, 289, 290, 291, 292,
    293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 308, 309,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 322, 323, 324, 325, 326,
    327, 328, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342,
    343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357,
    358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
    373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388,
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
    404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
    419, 420, 421, 422,
];

/// Convert a BBPE-encoded Unicode char back to its byte value.
pub fn bbpe_char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    // ASCII printable (32-126) maps to itself (bytes 32-126)
    if (32..=126).contains(&cp) {
        return Some(cp as u8);
    }
    // Search the non-identity ranges in the table
    for (byte_val, &mapped_cp) in BBPE_CODEPOINTS.iter().enumerate() {
        if mapped_cp == cp {
            return Some(byte_val as u8);
        }
    }
    None
}

// ============== Kaldi-compatible Fbank Feature Extraction ==============

pub fn compute_mel_filterbank(config: &FbankConfig) -> Vec<Vec<f32>> {
    let num_bins = config.num_bins;
    let fft_size = config.fft_size;
    let sample_rate = config.sample_rate as f32;
    let nyquist = sample_rate / 2.0;

    // Kaldi convention: negative high_freq means nyquist + high_freq
    let low_freq = config.low_freq;
    let high_freq = if config.high_freq <= 0.0 {
        nyquist + config.high_freq
    } else {
        config.high_freq
    };

    let hz_to_mel = |hz: f32| 1127.0 * (1.0 + hz / 700.0).ln();
    let mel_to_hz = |mel: f32| 700.0 * ((mel / 1127.0).exp() - 1.0);

    let low_mel = hz_to_mel(low_freq);
    let high_mel = hz_to_mel(high_freq);

    let num_points = num_bins + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_points - 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz * fft_size as f32) / sample_rate).floor() as usize)
        .collect();

    let half_fft = fft_size / 2 + 1;
    let mut filterbank = vec![vec![0.0f32; half_fft]; num_bins];
    for (i, filter) in filterbank.iter_mut().enumerate() {
        let left = fft_bins[i];
        let center = fft_bins[i + 1];
        let right = fft_bins[i + 2];

        if center > left {
            for j in left..center {
                if j < half_fft {
                    filter[j] = (j - left) as f32 / (center - left) as f32;
                }
            }
        }
        if right > center {
            for j in center..right {
                if j < half_fft {
                    filter[j] = (right - j) as f32 / (right - center) as f32;
                }
            }
        }
    }

    filterbank
}

/// Kaldi-compatible FBank: Povey window, preemphasis, natural log, snip_edges=false.
pub fn compute_fbank_kaldi(samples: &[f32], config: &FbankConfig) -> Array2<f32> {
    let window_size = config.window_size;
    let hop_size = config.hop_size;
    let fft_size = config.fft_size;
    let half_fft = fft_size / 2 + 1;

    if samples.is_empty() {
        return Array2::zeros((0, config.num_bins));
    }

    // Frame count: snip_edges=false pads the signal
    let num_frames = if config.snip_edges {
        if samples.len() < window_size {
            return Array2::zeros((0, config.num_bins));
        }
        (samples.len() - window_size) / hop_size + 1
    } else {
        (samples.len() + hop_size / 2) / hop_size
    };

    if num_frames == 0 {
        return Array2::zeros((0, config.num_bins));
    }

    let filterbank = compute_mel_filterbank(config);

    // Povey window: hamming^0.85
    let window: Vec<f32> = (0..window_size)
        .map(|i| {
            let hamming =
                0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos();
            hamming.powf(0.85)
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut features = Vec::with_capacity(num_frames * config.num_bins);

    for frame_idx in 0..num_frames {
        // Frame start position (snip_edges=false centers the first frame)
        let center = if config.snip_edges {
            frame_idx * hop_size + window_size / 2
        } else {
            frame_idx * hop_size
        };
        let start = center as isize - (window_size as isize / 2);

        // Extract frame with zero-padding at boundaries
        let mut frame = vec![0.0f32; window_size];
        for i in 0..window_size {
            let idx = start + i as isize;
            if idx >= 0 && (idx as usize) < samples.len() {
                frame[i] = samples[idx as usize];
            }
        }

        // Remove DC offset
        if config.remove_dc_offset {
            let mean: f32 = frame.iter().sum::<f32>() / window_size as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }
        }

        // Preemphasis
        if config.preemph_coeff > 0.0 {
            for i in (1..window_size).rev() {
                frame[i] -= config.preemph_coeff * frame[i - 1];
            }
            frame[0] *= 1.0 - config.preemph_coeff;
        }

        // Apply window and zero-pad to FFT size
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        buffer.resize(fft_size, Complex::new(0.0, 0.0));

        fft.process(&mut buffer);

        // Power spectrum
        let mut power = vec![0.0f32; half_fft];
        for (i, c) in buffer.iter().take(half_fft).enumerate() {
            power[i] = c.norm_sqr();
        }

        // Apply mel filterbank and take natural log (Kaldi convention)
        for filter in &filterbank {
            let mut sum = 0.0f32;
            for (i, &w) in filter.iter().enumerate() {
                sum += w * power[i];
            }
            features.push(if sum > f32::EPSILON {
                sum.ln()
            } else {
                (f32::EPSILON).ln()
            });
        }
    }

    Array2::from_shape_vec((num_frames, config.num_bins), features).unwrap()
}
