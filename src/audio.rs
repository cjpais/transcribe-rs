//! Audio processing utilities for transcription.
//!
//! This module provides functions for reading and processing audio files
//! to prepare them for transcription engines.

use std::path::Path;

#[cfg(feature = "resampling")]
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

#[cfg(feature = "resampling")]
const RESAMPLE_CHUNK_SIZE: usize = 1024;

/// Read WAV file samples and convert them to the required format.
///
/// This function reads a WAV file and converts it to the format expected by
/// transcription engines: 16kHz sample rate, 16-bit samples, mono channel.
///
/// # Arguments
///
/// * `wav_path` - Path to the WAV file to read
///
/// # Returns
///
/// Returns a vector of f32 samples normalized to the range [-1.0, 1.0].
///
/// # Errors
///
/// This function will return an error if:
/// - The file cannot be opened or read
/// - The WAV format is incorrect (not 16kHz, 16-bit, mono)
/// - The samples cannot be converted to the expected format
///
/// # Examples
///
/// ```rust,no_run
/// use transcribe_rs::audio::read_wav_samples;
/// use std::path::Path;
///
/// let samples = read_wav_samples(Path::new("audio.wav"))?;
/// println!("Loaded {} samples", samples.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Audio Requirements
///
/// The input WAV file must have:
/// - Sample rate: 16,000 Hz
/// - Bit depth: 16 bits per sample
/// - Channels: 1 (mono)
/// - Format: PCM integer samples
pub fn read_wav_samples(wav_path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(wav_path)?;
    let spec = reader.spec();

    let expected_spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    if spec.channels != expected_spec.channels {
        return Err(format!(
            "Expected {} channels, found {}",
            expected_spec.channels, spec.channels
        )
        .into());
    }

    if spec.sample_rate != expected_spec.sample_rate {
        return Err(format!(
            "Expected {} Hz sample rate, found {} Hz",
            expected_spec.sample_rate, spec.sample_rate
        )
        .into());
    }

    if spec.bits_per_sample != expected_spec.bits_per_sample {
        return Err(format!(
            "Expected {} bits per sample, found {}",
            expected_spec.bits_per_sample, spec.bits_per_sample
        )
        .into());
    }

    if spec.sample_format != expected_spec.sample_format {
        return Err(format!("Expected Int sample format, found {:?}", spec.sample_format).into());
    }

    let samples: Result<Vec<f32>, _> = reader
        .samples::<i16>()
        .map(|sample| sample.map(|s| s as f32 / i16::MAX as f32))
        .collect();

    Ok(samples?)
}

/// Mix multi-channel interleaved audio to mono by averaging channels.
///
/// # Arguments
///
/// * `samples` - Interleaved multi-channel audio samples
/// * `channels` - Number of channels in the interleaved data
///
/// # Returns
///
/// Mono audio samples computed by averaging across channels.
/// If `channels` is 1, returns a copy of the input.
/// Trailing samples that don't fill a complete frame are dropped (via `chunks_exact`).
///
/// # Examples
///
/// ```
/// use transcribe_rs::audio::mix_to_mono;
///
/// // Stereo to mono: average each pair
/// let stereo = vec![0.2, 0.8, -0.4, 0.6];
/// let mono = mix_to_mono(&stereo, 2);
/// assert_eq!(mono.len(), 2);
/// ```
pub fn mix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks_exact(ch)
        .map(|chunk| chunk.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Create a persistent SincFixedIn resampler for converting to 16kHz mono.
///
/// Returns `Ok(None)` if audio is already 16kHz mono (no resampling needed).
/// When `Ok(Some(...))`, returns `(resampler, resample_ratio)`.
///
/// # Arguments
///
/// * `sample_rate` - Input audio sample rate in Hz
/// * `channels` - Number of input channels (used only to detect the no-op 16kHz mono case)
///
/// # Returns
///
/// * `Ok(None)` — input is already 16kHz mono, no resampling needed
/// * `Ok(Some((resampler, ratio)))` — a configured `SincFixedIn` resampler and the
///   resample ratio (`16000 / sample_rate`)
///
/// # Errors
///
/// Returns an error if `rubato` fails to construct the resampler (e.g. invalid parameters).
///
/// # Examples
///
/// ```rust,no_run
/// use transcribe_rs::audio::create_resampler;
///
/// // 48kHz mono → needs resampling
/// let result = create_resampler(48000, 1)?;
/// assert!(result.is_some());
///
/// // Already 16kHz mono → no resampler needed
/// let result = create_resampler(16000, 1)?;
/// assert!(result.is_none());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "resampling")]
pub fn create_resampler(
    sample_rate: u32,
    channels: u16,
) -> Result<Option<(SincFixedIn<f32>, f64)>, Box<dyn std::error::Error>> {
    if sample_rate == 16000 && channels == 1 {
        return Ok(None);
    }

    let resample_ratio = 16000.0 / sample_rate as f64;
    let params = SincInterpolationParameters {
        sinc_len: 64,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };

    let resampler = SincFixedIn::<f32>::new(resample_ratio, 8.0, params, RESAMPLE_CHUNK_SIZE, 1)?;
    Ok(Some((resampler, resample_ratio)))
}

/// Resample a chunk of mono audio using a persistent resampler.
///
/// # Arguments
///
/// * `resampler` - A mutable reference to a `SincFixedIn` resampler created by [`create_resampler`]
/// * `mono_samples` - Mono audio samples to resample
///
/// # Returns
///
/// Resampled audio samples. Partial final chunks are zero-padded for the resampler
/// then the output is truncated proportionally to avoid trailing silence.
///
/// # Errors
///
/// Returns an error if the underlying `rubato` resampler fails.
#[cfg(feature = "resampling")]
pub fn resample_chunk(
    resampler: &mut SincFixedIn<f32>,
    mono_samples: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut output = Vec::new();
    let mut pos = 0;

    while pos < mono_samples.len() {
        let end = (pos + RESAMPLE_CHUNK_SIZE).min(mono_samples.len());
        let real_len = end - pos;
        let mut chunk: Vec<f32> = mono_samples[pos..end].to_vec();
        let is_partial = chunk.len() < RESAMPLE_CHUNK_SIZE;
        if is_partial {
            chunk.resize(RESAMPLE_CHUNK_SIZE, 0.0);
        }
        let waves_out = resampler.process(&[chunk], None)?;
        if is_partial {
            let expected_out = (real_len as f64 / RESAMPLE_CHUNK_SIZE as f64
                * waves_out[0].len() as f64)
                .round() as usize;
            output.extend_from_slice(&waves_out[0][..expected_out]);
        } else {
            output.extend_from_slice(&waves_out[0]);
        }
        pos += RESAMPLE_CHUNK_SIZE;
    }

    Ok(output)
}
