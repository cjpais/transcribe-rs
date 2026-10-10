//! Log-mel feature extraction for Nemotron.
//!
//! Ported from parakeet-rs's `audio.rs` (MIT) and adapted to transcribe-rs's
//! `rustfft` dependency (parakeet-rs uses `realfft`; the two produce identical
//! power-spectrum bins for a real input, so this is a mechanical swap — see
//! `stft_power`).
//!
//! Nemotron's ONNX encoder consumes `processed_signal` — a `[1, 128, T]`
//! log-mel spectrogram — directly. Unlike the Parakeet engine there is no
//! preprocessor ONNX in the export, so the front end is computed here in Rust.
//!
//! This matches NeMo's FastConformer streaming featurizer exactly:
//! 512-point FFT over a 400-sample Hann window, 160-sample hop, 128 Slaney
//! mel bins, `log(x + 2^-24)`, and **no** per-feature normalization — the
//! streaming Nemotron models feed raw log-mel "decibels" straight into the
//! encoder. (parakeet-rs's `extract_features_with_cache`, used by the TDT
//! models, *does* normalize; that path is deliberately not reused here.)

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

/// Audio sample rate the model expects (Hz).
pub(super) const SAMPLE_RATE: usize = 16000;
/// FFT size.
pub(super) const N_FFT: usize = 512;
/// Analysis window length (samples). Smaller than [`N_FFT`], so each window
/// is zero-padded up to `N_FFT` before the transform.
pub(super) const WIN_LENGTH: usize = 400;
/// Hop between consecutive frames (samples).
pub(super) const HOP_LENGTH: usize = 160;
/// Number of mel filterbank bins.
pub(super) const N_MELS: usize = 128;
/// Pre-emphasis coefficient.
pub(super) const PREEMPH: f32 = 0.97;
/// Additive guard inside the log. NeMo's featurizer uses
/// `log_zero_guard_type="add"` with value `2^-24`.
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8;

/// First-order pre-emphasis filter: `y[n] = x[n] - coef * x[n-1]`.
pub(super) fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(audio.len());
    out.push(audio[0]);
    for i in 1..audio.len() {
        out.push(audio[i] - coef * audio[i - 1]);
    }
    out
}

/// Symmetric (periodic=false) Hann window, matching parakeet-rs / NeMo.
fn hann_window(window_length: usize) -> Vec<f32> {
    (0..window_length)
        .map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / (window_length as f32 - 1.0)).cos())
        .collect()
}

/// Short-time Fourier transform returning the **power** spectrogram
/// `[n_fft/2 + 1, num_frames]`.
///
/// Center-padded by `n_fft/2` on each side to match librosa/NeMo framing.
/// Implemented with `rustfft` (full complex transform of the zero-imaginary
/// real signal); only the first `n_fft/2 + 1` bins are kept, which equals what
/// parakeet-rs gets from `realfft`.
fn stft_power(audio: &[f32], n_fft: usize, hop_length: usize, win_length: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;

    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad, 0.0);

    if padded.len() < n_fft {
        return Array2::zeros((freq_bins, 0));
    }

    let window = hann_window(win_length);
    let num_frames = (padded.len() - n_fft) / hop_length + 1;

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut spectrogram = Array2::<f32>::zeros((freq_bins, num_frames));
    let mut buf = vec![Complex::new(0.0f32, 0.0); n_fft];

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;

        for c in buf.iter_mut() {
            *c = Complex::new(0.0, 0.0);
        }
        let avail = win_length.min(padded.len() - start);
        for i in 0..avail {
            buf[i] = Complex::new(padded[start + i] * window[i], 0.0);
        }

        fft.process(&mut buf);

        for k in 0..freq_bins {
            spectrogram[[k, frame_idx]] = buf[k].norm_sqr();
        }
    }

    spectrogram
}

// ---- Slaney mel scale (librosa-compatible) ----

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
const LOG_STEP: f64 = 0.068_751_777_420_949_12;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    }
}

fn mel_to_hz_slaney(mel: f64) -> f64 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOG_STEP).exp()
    }
}

/// Build a Slaney-normalized mel filterbank `[n_mels, n_fft/2 + 1]`.
pub(super) fn create_mel_filterbank(n_fft: usize, n_mels: usize, sample_rate: usize) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;
    let mut filterbank = Array2::<f32>::zeros((n_mels, freq_bins));

    let fmax = sample_rate as f64 / 2.0;
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(fmax);

    let mel_points: Vec<f64> = (0..=n_mels + 1)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| i as f64 * sample_rate as f64 / n_fft as f64)
        .collect();

    let fdiff: Vec<f64> = mel_points.windows(2).map(|w| w[1] - w[0]).collect();

    for i in 0..n_mels {
        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - mel_points[i]) / fdiff[i];
            let upper = (mel_points[i + 2] - freq) / fdiff[i + 1];
            filterbank[[i, k]] = 0.0f64.max(lower.min(upper)) as f32;
        }
    }

    // Slaney normalization.
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
        for k in 0..freq_bins {
            filterbank[[i, k]] *= enorm as f32;
        }
    }

    filterbank
}

/// Build the mel filterbank for the Nemotron front end (cached at model load).
pub(super) fn nemotron_mel_basis() -> Array2<f32> {
    create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE)
}

/// Compute the log-mel spectrogram for an utterance: shape `[N_MELS, frames]`.
///
/// No normalization is applied — this is the raw log-mel the streaming
/// Nemotron encoder expects. `mel_basis` should come from
/// [`nemotron_mel_basis`].
pub(super) fn log_mel_spectrogram(audio: &[f32], mel_basis: &Array2<f32>) -> Array2<f32> {
    if audio.is_empty() {
        return Array2::zeros((N_MELS, 0));
    }
    let pre = apply_preemphasis(audio, PREEMPH);
    let power = stft_power(&pre, N_FFT, HOP_LENGTH, WIN_LENGTH);
    let mel = mel_basis.dot(&power);
    mel.mapv(|x| (x + LOG_ZERO_GUARD).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq_hz: f32, sample_rate: usize, num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn stft_concentrates_power_at_expected_bin() {
        // 1 kHz sine at 16 kHz -> bin 1000 * 512 / 16000 = 32.
        let audio = sine_wave(1000.0, SAMPLE_RATE, SAMPLE_RATE);
        let spec = stft_power(&audio, N_FFT, HOP_LENGTH, WIN_LENGTH);

        let expected_bin = 32;
        let freq_bins = N_FFT / 2 + 1;
        let num_frames = spec.shape()[1];

        let mut correct = 0;
        for frame in 2..num_frames.saturating_sub(2) {
            let mut max_bin = 0;
            let mut max_power = 0.0f32;
            for bin in 0..freq_bins {
                if spec[[bin, frame]] > max_power {
                    max_power = spec[[bin, frame]];
                    max_bin = bin;
                }
            }
            if max_bin == expected_bin {
                correct += 1;
            }
        }
        let interior = num_frames.saturating_sub(4);
        assert!(
            correct > interior / 2,
            "expected bin {expected_bin} to dominate, only {correct}/{interior}"
        );
    }

    #[test]
    fn filterbank_has_expected_shape() {
        let fb = create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE);
        assert_eq!(fb.shape(), &[N_MELS, N_FFT / 2 + 1]);
        // Every filter should have some positive weight.
        for i in 0..N_MELS {
            let row_sum: f32 = fb.row(i).iter().sum();
            assert!(row_sum > 0.0, "mel filter {i} is all zero");
        }
    }

    #[test]
    fn log_mel_has_128_bins_and_expected_frame_count() {
        // 1 s of audio -> ~101 frames with center padding (16000/160 + 1).
        let audio = sine_wave(440.0, SAMPLE_RATE, SAMPLE_RATE);
        let basis = nemotron_mel_basis();
        let mel = log_mel_spectrogram(&audio, &basis);
        assert_eq!(mel.shape()[0], N_MELS);
        assert_eq!(mel.shape()[1], SAMPLE_RATE / HOP_LENGTH + 1);
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn silence_yields_log_floor_without_normalization() {
        // Without per-feature normalization, silence must map to a constant
        // ~ln(2^-24) floor everywhere (a normalized path would instead produce
        // NaNs / zeros). This guards against accidentally reusing the
        // normalized featurizer.
        let basis = nemotron_mel_basis();
        let mel = log_mel_spectrogram(&vec![0.0f32; SAMPLE_RATE], &basis);
        let floor = (LOG_ZERO_GUARD).ln();
        assert!(mel.iter().all(|&v| (v - floor).abs() < 1e-3));
    }

    #[test]
    fn empty_audio_is_empty_mel() {
        let basis = nemotron_mel_basis();
        let mel = log_mel_spectrogram(&[], &basis);
        assert_eq!(mel.shape(), &[N_MELS, 0]);
    }
}
