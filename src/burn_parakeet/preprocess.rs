use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::features::mel::{make_window, WindowType};

/// Slaney/Auditory Toolbox mel scale (librosa default, htk=False).
/// Piecewise linear below 1000 Hz, logarithmic above.
fn slaney_hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0
    const LOGSTEP: f32 = 0.068751775; // ln(6.4) / 27.0
    if hz < MIN_LOG_HZ {
        hz / F_SP
    } else {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f32 = 0.06875177;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * LOGSTEP).exp()
    }
}

/// Build a mel filterbank matching librosa.filters.mel(norm="slaney", htk=False).
/// Uses the Slaney mel scale and divides each filter by its bandwidth in Hz.
fn mel_filterbank_slaney(
    num_mels: usize,
    n_fft: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Array2<f32> {
    let num_fft_bins = n_fft / 2 + 1;

    // FFT bin center frequencies in Hz (matching librosa.fft_frequencies)
    let fft_freqs: Vec<f32> = (0..num_fft_bins)
        .map(|k| k as f32 * sample_rate / n_fft as f32)
        .collect();

    // num_mels + 2 equally spaced points in Slaney mel space, converted back to Hz
    let mel_min = slaney_hz_to_mel(f_min);
    let mel_max = slaney_hz_to_mel(f_max);
    let num_points = num_mels + 2;
    let mel_f: Vec<f32> = (0..num_points)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * (i as f32) / (num_points as f32 - 1.0);
            slaney_mel_to_hz(mel)
        })
        .collect();

    // Build triangular filters (librosa style: min of lower/upper ramps)
    let mut fb = Array2::zeros((num_mels, num_fft_bins));

    for m in 0..num_mels {
        let f_lower = mel_f[m];
        let f_center = mel_f[m + 1];
        let f_upper = mel_f[m + 2];
        let fdiff_lower = f_center - f_lower;
        let fdiff_upper = f_upper - f_center;

        for k in 0..num_fft_bins {
            let freq = fft_freqs[k];
            let lower = if fdiff_lower > 0.0 {
                (freq - f_lower) / fdiff_lower
            } else {
                0.0
            };
            let upper = if fdiff_upper > 0.0 {
                (f_upper - freq) / fdiff_upper
            } else {
                0.0
            };
            fb[[m, k]] = lower.min(upper).max(0.0);
        }

        // Slaney normalization: 2 / (f_upper - f_lower)
        let bandwidth = f_upper - f_lower;
        if bandwidth > 0.0 {
            let enorm = 2.0 / bandwidth;
            for k in 0..num_fft_bins {
                fb[[m, k]] *= enorm;
            }
        }
    }

    fb
}

/// NeMo 128-mel preprocessor configuration.
const SAMPLE_RATE: u32 = 16000;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400; // 0.025s * 16000 Hz
const HOP_LENGTH: usize = 160;
const NUM_MELS: usize = 128;
const PRE_EMPHASIS: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 1e-5;

/// Compute NeMo-compatible mel spectrogram features.
///
/// Pipeline: pre-emphasis → reflect pad → STFT → power → mel → log → per-utterance norm.
///
/// Returns features in [num_mels, num_frames] layout (frequency-major),
/// matching the encoder's expected input after transposing to [B, T, num_mels].
pub fn nemo_preprocess(samples: &[f32]) -> Array2<f32> {
    if samples.is_empty() {
        return Array2::zeros((NUM_MELS, 0));
    }

    // 1. Pre-emphasis (signal-level)
    let mut pre = vec![samples[0]; samples.len()];
    for i in 1..samples.len() {
        pre[i] = samples[i] - PRE_EMPHASIS * samples[i - 1];
    }

    // 2. Reflect padding: pad N_FFT/2 on each side (matching MLX's _pad)
    let pad = N_FFT / 2;
    let mut padded = Vec::with_capacity(pre.len() + 2 * pad);
    // Left reflect: x[pad], x[pad-1], ..., x[1]
    for i in (1..=pad).rev() {
        padded.push(pre[i]);
    }
    padded.extend_from_slice(&pre);
    // Right reflect: x[len-2], x[len-3], ..., x[len-1-pad]
    for i in 1..=pad {
        padded.push(pre[pre.len() - 1 - i]);
    }

    // 3. STFT with Hann window (WIN_LENGTH samples, zero-padded to N_FFT)
    let mut window = make_window(WindowType::Hann, WIN_LENGTH);
    window.resize(N_FFT, 0.0);
    let num_frames = (padded.len() - WIN_LENGTH + HOP_LENGTH) / HOP_LENGTH;
    let num_fft_bins = N_FFT / 2 + 1;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let mut power_spectrogram = Array2::zeros((num_fft_bins, num_frames));

    for frame_idx in 0..num_frames {
        let start = frame_idx * HOP_LENGTH;
        let mut fft_input: Vec<Complex<f32>> = (0..N_FFT)
            .map(|j| {
                let sample = if start + j < padded.len() {
                    padded[start + j]
                } else {
                    0.0
                };
                Complex::new(sample * window[j], 0.0)
            })
            .collect();

        fft.process(&mut fft_input);

        // 4. Magnitude spectrum: (|re| + |im|)^2 to match MLX's get_logmel()
        for bin in 0..num_fft_bins {
            let mag = fft_input[bin].re.abs() + fft_input[bin].im.abs();
            power_spectrogram[[bin, frame_idx]] = mag * mag;
        }
    }

    // 5. Mel filterbank (slaney-normalized to match MLX)
    let mel_banks = mel_filterbank_slaney(NUM_MELS, N_FFT, SAMPLE_RATE as f32, 0.0, 8000.0);
    // mel_banks: [NUM_MELS, num_fft_bins], power_spectrogram: [num_fft_bins, num_frames]
    let mel = mel_banks.dot(&power_spectrogram); // [NUM_MELS, num_frames]

    // 6. Log
    let log_mel = mel.mapv(|v| (v + LOG_ZERO_GUARD).ln());

    // 7. Per-utterance normalization: (x - mean) / (std + 1e-5) per frequency bin
    let num_frames_f = num_frames as f32;
    let mut normalized = log_mel.clone();

    for mel_bin in 0..NUM_MELS {
        let row = log_mel.row(mel_bin);
        let mean: f32 = row.sum() / num_frames_f;
        let var: f32 = row.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / num_frames_f;
        let std = var.sqrt();

        for frame in 0..num_frames {
            normalized[[mel_bin, frame]] = (log_mel[[mel_bin, frame]] - mean) / (std + 1e-5);
        }
    }

    normalized
}
