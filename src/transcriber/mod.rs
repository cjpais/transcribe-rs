//! Chunked transcription strategies.
//!
//! The [`Transcriber`] trait provides a universal wrapper for non-trivial
//! transcription workflows. Different implementations handle different
//! chunking approaches:
//!
//! - [`VadChunked`] — splits audio on speech/silence boundaries using a [`Vad`](crate::vad::Vad)
//! - [`EnergyAdaptiveChunked`] — fixed-duration chunks with energy-based split point search
//!
//! The model is borrowed per-call (`&mut dyn SpeechModel`), never owned.
//! This works naturally with `Arc<Mutex<Box<dyn SpeechModel>>>` — lock for
//! the call, unlock after.
//!
//! # Examples
//!
//! **File transcription with VAD chunking:**
//! ```ignore
//! use transcribe_rs::transcriber::{Transcriber, VadChunked, VadChunkedConfig};
//! use transcribe_rs::vad::{SmoothedVad, EnergyVad};
//!
//! let vad = SmoothedVad::new(Box::new(EnergyVad::new(480, 0.01)), 15, 15, 2);
//! let mut chunker = VadChunked::new(Box::new(vad), VadChunkedConfig::default(), options);
//! let result = chunker.transcribe_file(&mut model, &path)?;
//! ```
//!
//! **Live audio with per-frame feeding:**
//! ```ignore
//! let mut t = VadChunked::new(vad, config, options);
//! for frame in audio_frames {
//!     for result in t.feed(&mut model, &frame)? {
//!         println!("{}", result.text);
//!     }
//! }
//! let final_result = t.finish(&mut model)?;
//! ```

mod energy_adaptive_chunked;
mod merge;
#[cfg(test)]
pub(crate) mod test_helpers;
mod vad_chunked;

pub use energy_adaptive_chunked::{EnergyAdaptiveChunked, EnergyAdaptiveConfig};
pub use merge::{merge_sequential, merge_sequential_with_separator, DEFAULT_MERGE_SEPARATOR};
pub use vad_chunked::{VadChunked, VadChunkedConfig};

/// Expected sample rate for all transcription audio.
pub(crate) const SAMPLE_RATE: f32 = 16000.0;

/// Compute RMS energy of an audio frame.
pub(crate) fn rms_energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt()
}

use std::path::Path;

use crate::{audio, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult};

/// Transcribe a chunk with optional silence padding and timestamp adjustment.
///
/// Sets leading/trailing silence on [`TranscribeOptions`] and delegates padding
/// and leading-silence timestamp correction to [`SpeechModel::transcribe`].
/// Enforces `min_duration_secs` via zero-padding of the content, then offsets
/// segment timestamps by `chunk_start_secs`.
pub(crate) fn transcribe_padded(
    model: &mut dyn SpeechModel,
    samples: &[f32],
    padding_secs: f32,
    min_duration_secs: f32,
    chunk_start_secs: f32,
    options: &TranscribeOptions,
) -> Result<TranscriptionResult, TranscribeError> {
    let padding_ms = (padding_secs * 1000.0) as u32;

    // The trait will prepend + append padding. Ensure the total
    // (padding + content + padding) meets the minimum duration.
    let pad_total = 2 * padding_ms as usize * audio::SAMPLES_PER_MS;
    let min_total = (min_duration_secs * SAMPLE_RATE) as usize;
    let min_content = min_total.saturating_sub(pad_total);

    let mut content = samples.to_vec();
    if content.len() < min_content {
        content.resize(min_content, 0.0);
    }

    // Override any user-specified padding — chunked transcription manages
    // its own padding to ensure consistent overlap between chunks.
    let mut opts = options.clone();
    opts.leading_silence_ms = Some(padding_ms);
    opts.trailing_silence_ms = Some(padding_ms);

    let mut result = model.transcribe(&content, &opts)?;

    // The trait already subtracted leading silence from timestamps.
    // Offset by this chunk's position in the overall audio.
    if chunk_start_secs > 0.0 {
        result.offset_timestamps(chunk_start_secs);
    }

    Ok(result)
}

/// A chunked transcription strategy.
///
/// Implementations split audio into chunks, transcribe each chunk via
/// a [`SpeechModel`], and merge the results. The model is borrowed
/// per-call, never owned.
///
/// All methods take `&mut self`, making the trait fully object-safe
/// (`Box<dyn Transcriber>` works). After [`finish()`](Transcriber::finish)
/// returns, the transcriber is reset and ready for a new session.
pub trait Transcriber: Send {
    /// Feed audio samples. Internally buffers, detects chunk boundaries,
    /// and transcribes chunks as they become ready.
    ///
    /// Returns all chunk results that completed during this call.
    ///
    /// For live audio: called per-frame (e.g. 480 samples), usually returns
    /// an empty vec, occasionally returns 1 result when a chunk boundary
    /// is detected.
    ///
    /// For files: called with all samples at once, returns N results
    /// (one per chunk that completed during processing).
    fn feed(
        &mut self,
        model: &mut dyn SpeechModel,
        samples: &[f32],
    ) -> Result<Vec<TranscriptionResult>, TranscribeError>;

    /// Transcribe any remaining buffered audio and return the merged
    /// result of the entire session. Resets internal state for reuse.
    fn finish(
        &mut self,
        model: &mut dyn SpeechModel,
    ) -> Result<TranscriptionResult, TranscribeError>;

    /// Convenience: feed all samples and finish in one call.
    ///
    /// The return value of `feed()` is intentionally discarded here —
    /// intermediate results are accumulated internally and merged by
    /// `finish()`.
    fn transcribe(
        &mut self,
        model: &mut dyn SpeechModel,
        samples: &[f32],
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.feed(model, samples)?;
        self.finish(model)
    }

    /// Convenience: load a WAV file, feed, and finish.
    fn transcribe_file(
        &mut self,
        model: &mut dyn SpeechModel,
        path: &Path,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let samples = audio::read_wav_samples(path)?;
        self.transcribe(model, &samples)
    }
}
