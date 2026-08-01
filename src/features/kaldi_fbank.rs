//! Kaldi-compatible FBank feature extraction.
//!
//! Implements the same feature pipeline used by sherpa-onnx / kaldi-native-fbank:
//! Povey window (Hamming^0.85), DC offset removal, preemphasis, power spectrum,
//! triangular mel filterbank, and natural log. The `high_freq` field follows the
//! Kaldi sign convention: negative values are treated as `nyquist + high_freq`
//! (e.g. `-400` → 7600 Hz at 16 kHz).

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

/// Kaldi-compatible FBank configuration matching sherpa-onnx / kaldi-native-fbank.
#[derive(Debug, Clone)]
pub struct KaldiFbankConfig {
    pub num_bins: usize,
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_size: usize,
    pub sample_rate: u32,
    pub low_freq: f32,
    /// Negative means nyquist + high_freq (Kaldi convention). -400 → 7600 Hz at 16kHz.
    pub high_freq: f32,
    pub preemph_coeff: f32,
    pub snip_edges: bool,
    pub remove_dc_offset: bool,
}

impl Default for KaldiFbankConfig {
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

/// Compute Kaldi-compatible FBank features from audio samples.
///
/// Returns an array of shape `[num_frames, num_bins]` (time-major).
pub fn compute_kaldi_fbank(samples: &[f32], config: &KaldiFbankConfig) -> Array2<f32> {
    let window_size = config.window_size;
    let hop_size = config.hop_size;
    let fft_size = config.fft_size;
    let half_fft = fft_size / 2 + 1;

    if samples.is_empty() {
        return Array2::zeros((0, config.num_bins));
    }

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

    let filterbank = mel_filterbank(config);

    // Povey window: hamming^0.85
    let window: Vec<f32> = (0..window_size)
        .map(|i| {
            let hamming = 0.54
                - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos();
            hamming.powf(0.85)
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut features = Vec::with_capacity(num_frames * config.num_bins);

    for frame_idx in 0..num_frames {
        let center = if config.snip_edges {
            frame_idx * hop_size + window_size / 2
        } else {
            frame_idx * hop_size
        };
        let start = center as isize - (window_size as isize / 2);

        let mut frame = vec![0.0f32; window_size];
        for (i, sample) in frame.iter_mut().enumerate() {
            let idx = start + i as isize;
            if idx >= 0 && (idx as usize) < samples.len() {
                *sample = samples[idx as usize];
            }
        }

        if config.remove_dc_offset {
            let mean: f32 = frame.iter().sum::<f32>() / window_size as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }
        }

        if config.preemph_coeff > 0.0 {
            for i in (1..window_size).rev() {
                frame[i] -= config.preemph_coeff * frame[i - 1];
            }
            frame[0] *= 1.0 - config.preemph_coeff;
        }

        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        buffer.resize(fft_size, Complex::new(0.0, 0.0));
        fft.process(&mut buffer);

        let power: Vec<f32> = buffer[..half_fft].iter().map(|c| c.norm_sqr()).collect();

        for filter in &filterbank {
            let mut sum = 0.0f32;
            for (i, &w) in filter.iter().enumerate() {
                sum += w * power[i];
            }
            features.push(if sum > f32::EPSILON {
                sum.ln()
            } else {
                f32::EPSILON.ln()
            });
        }
    }

    Array2::from_shape_vec((num_frames, config.num_bins), features).unwrap()
}

/// Build a triangular mel filterbank matrix.
///
/// Returns a `Vec` of `num_bins` filters, each of length `fft_size / 2 + 1`.
fn mel_filterbank(config: &KaldiFbankConfig) -> Vec<Vec<f32>> {
    let num_bins = config.num_bins;
    let fft_size = config.fft_size;
    let sample_rate = config.sample_rate as f32;
    let nyquist = sample_rate / 2.0;
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
            for (idx, val) in filter[left..center.min(half_fft)].iter_mut().enumerate() {
                *val = idx as f32 / (center - left) as f32;
            }
        }
        if right > center {
            for (idx, val) in filter[center..right.min(half_fft)].iter_mut().enumerate() {
                *val = (right - center - idx) as f32 / (right - center) as f32;
            }
        }
    }
    filterbank
}
