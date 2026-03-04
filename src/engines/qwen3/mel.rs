//! Whisper-compatible log-mel spectrogram extraction.
//!
//! Parameters: 128 mel bins, Hann window, n_fft=400, hop=160, Slaney mel scale.
//! This differs from SenseVoice's FBANK (80 bins, Hamming, HTK mel, pre-emphasis)
//! and cannot be shared.

// Multi-array indexed loops are clearer than iterator chains here.
#![allow(clippy::needless_range_loop)]

use ndarray::Array3;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;
use std::sync::LazyLock;

const SAMPLE_RATE: f64 = 16000.0;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const FMIN: f64 = 0.0;
const FMAX: f64 = 8000.0;

/// Cached Hann window (400 f64 values) — deterministic, computed once.
static HANN_WINDOW: LazyLock<Vec<f64>> = LazyLock::new(|| hann_window(N_FFT));

/// Cached Slaney mel filterbank (128×201 f64 matrix) — deterministic, computed once.
static MEL_FILTERS: LazyLock<Vec<Vec<f64>>> =
    LazyLock::new(|| slaney_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX));

/// Compute Whisper-compatible log-mel spectrogram.
///
/// Input: 1-D f32 audio samples at 16kHz.
/// Output: `Array3<f32>` shape `[1, 128, T]`.
pub fn log_mel_spectrogram(audio: &[f32]) -> Array3<f32> {
    if audio.len() <= N_FFT / 2 {
        return Array3::zeros((1, N_MELS, 0));
    }

    let mel_filters = &*MEL_FILTERS;
    let window = &*HANN_WINDOW;
    let num_fft_bins = N_FFT / 2 + 1;

    // Number of STFT frames (torch.stft default: centered, pad_mode reflect not needed
    // since we match the Python output exactly by computing the same frame count)
    let num_frames = audio.len() / HOP_LENGTH + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);

    // Compute magnitude-squared spectrogram
    let mut magnitudes = vec![vec![0.0f64; num_frames]; num_fft_bins];

    for frame_idx in 0..num_frames {
        let center = frame_idx * HOP_LENGTH;
        let mut fft_input: Vec<Complex<f64>> = Vec::with_capacity(N_FFT);

        for i in 0..N_FFT {
            let sample_idx = center as isize + i as isize - (N_FFT / 2) as isize;
            let sample = if sample_idx < 0 {
                // Reflect padding
                audio[(-sample_idx) as usize] as f64
            } else if (sample_idx as usize) >= audio.len() {
                // Reflect padding
                let overshoot = sample_idx as usize - audio.len();
                if audio.len() > 1 && overshoot < audio.len() {
                    audio[audio.len() - 2 - overshoot] as f64
                } else {
                    0.0
                }
            } else {
                audio[sample_idx as usize] as f64
            };
            fft_input.push(Complex::new(sample * window[i], 0.0));
        }

        fft.process(&mut fft_input);

        for k in 0..num_fft_bins {
            magnitudes[k][frame_idx] = fft_input[k].norm_sqr();
        }
    }

    // Drop last STFT frame (WhisperFeatureExtractor quirk)
    let time_steps = num_frames - 1;

    // Apply mel filterbank: mel_spec[m][t] = sum_k(filters[m][k] * magnitudes[k][t])
    let mut mel_spec = vec![vec![0.0f64; time_steps]; N_MELS];
    for m in 0..N_MELS {
        for k in 0..num_fft_bins {
            if mel_filters[m][k] != 0.0 {
                for t in 0..time_steps {
                    mel_spec[m][t] += mel_filters[m][k] * magnitudes[k][t];
                }
            }
        }
    }

    // Log scale with clamping and normalization
    let mut log_spec = vec![vec![0.0f32; time_steps]; N_MELS];
    let mut global_max: f32 = f32::NEG_INFINITY;

    for m in 0..N_MELS {
        for t in 0..time_steps {
            let val = (mel_spec[m][t].max(1e-10).log10()) as f32;
            log_spec[m][t] = val;
            if val > global_max {
                global_max = val;
            }
        }
    }

    // Dynamic range clipping and normalization
    let floor = global_max - 8.0;
    for m in 0..N_MELS {
        for t in 0..time_steps {
            log_spec[m][t] = (log_spec[m][t].max(floor) + 4.0) / 4.0;
        }
    }

    // Pack into [1, N_MELS, time_steps]
    let mut result = Array3::<f32>::zeros((1, N_MELS, time_steps));
    for m in 0..N_MELS {
        for t in 0..time_steps {
            result[[0, m, t]] = log_spec[m][t];
        }
    }
    result
}

/// Hann window of given length.
fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / length as f64).cos()))
        .collect()
}

/// Slaney-normalized mel filterbank matching `librosa.filters.mel(norm='slaney')`.
///
/// Below 1kHz: linear spacing in Hz.
/// Above 1kHz: logarithmic spacing.
/// Each filter is area-normalized (divided by its bandwidth in Hz).
fn slaney_mel_filterbank(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<Vec<f64>> {
    let num_fft_bins = n_fft / 2 + 1;

    // Slaney mel scale: linear below 1kHz, log above
    let f_sp = 200.0 / 3.0; // 66.667 Hz
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - 0.0) / f_sp; // 15.0
    let log_step = 6.4_f64.ln() / 27.0; // step in log region

    let hz_to_mel = |hz: f64| -> f64 {
        if hz < min_log_hz {
            hz / f_sp
        } else {
            min_log_mel + (hz / min_log_hz).ln() / log_step
        }
    };

    let mel_to_hz = |mel: f64| -> f64 {
        if mel < min_log_mel {
            mel * f_sp
        } else {
            min_log_hz * ((mel - min_log_mel) * log_step).exp()
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 uniformly spaced points in mel domain
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // FFT bin frequencies
    let fft_freqs: Vec<f64> = (0..num_fft_bins)
        .map(|k| k as f64 * sr / n_fft as f64)
        .collect();

    let mut filters = vec![vec![0.0f64; num_fft_bins]; n_mels];

    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];

        // Slaney normalization: divide by bandwidth in Hz
        let enorm = 2.0 / (right - left);

        for k in 0..num_fft_bins {
            let freq = fft_freqs[k];
            if freq > left && freq < center {
                filters[m][k] = enorm * (freq - left) / (center - left);
            } else if freq >= center && freq < right {
                filters[m][k] = enorm * (right - freq) / (right - center);
            }
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
        // Hann window: w[0] = 0, w[N/2] = 1
        assert!(w[0].abs() < 1e-10);
        assert!((w[200] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let filters = slaney_mel_filterbank(16000.0, 400, 128, 0.0, 8000.0);
        assert_eq!(filters.len(), 128);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn test_mel_filterbank_nonzero() {
        let filters = slaney_mel_filterbank(16000.0, 400, 128, 0.0, 8000.0);
        // Each filter should have at least some nonzero entries
        for (i, filter) in filters.iter().enumerate() {
            let sum: f64 = filter.iter().sum();
            assert!(sum > 0.0, "Filter {} has zero sum", i);
        }
    }

    #[test]
    fn test_log_mel_spectrogram_shape() {
        // 1 second of silence
        let audio = vec![0.0f32; 16000];
        let mel = log_mel_spectrogram(&audio);
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 128);
        // Expected frames: 16000 / 160 + 1 - 1 = 100
        assert_eq!(mel.shape()[2], 100);
    }

    #[test]
    fn test_log_mel_spectrogram_sine_wave() {
        // 440 Hz sine wave for 0.5s — should produce energy in low mel bins
        let n = 8000;
        let audio: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let mel = log_mel_spectrogram(&audio);
        assert_eq!(mel.shape()[0], 1);
        assert_eq!(mel.shape()[1], 128);
        // Should have non-trivial values (not all identical)
        let min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max > min, "Mel spectrogram should have dynamic range");
    }
}
