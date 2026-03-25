//! Voice Activity Detection (VAD).
//!
//! This module provides the [`Vad`] trait for frame-level voice activity detection,
//! along with built-in implementations:
//!
//! - [`EnergyVad`] — simple RMS energy threshold (pure Rust, no dependencies)
//! - [`SmoothedVad`] — wraps any `Vad` with onset detection and hangover
//! - [`SileroVad`] — Silero ONNX model (requires `vad-silero` feature)
//!
//! # Recommended Parameters
//!
//! | Parameter | Value | Meaning |
//! |-----------|-------|---------|
//! | Silero threshold | 0.3 | 30% speech probability to register |
//! | Prefill frames | 15 | 450ms audio history before speech |
//! | Hangover frames | 15 | 450ms continuation after speech ends |
//! | Onset frames | 2 | 60ms consecutive speech to trigger |
//! | Frame size | 480 | 30ms at 16kHz |

#[cfg(feature = "vad-silero")]
mod silero;
#[cfg(feature = "vad-silero")]
pub use silero::SileroVad;

use std::collections::VecDeque;

use crate::TranscribeError;

/// Voice activity detection.
///
/// Implementations classify fixed-size audio frames as speech or non-speech.
/// The frame size is determined by the implementation (e.g. 480 samples for
/// Silero at 16kHz).
pub trait Vad: Send {
    /// Number of samples required per frame.
    ///
    /// Silero: 480 (30ms at 16kHz). EnergyVad: configurable.
    fn frame_size(&self) -> usize;

    /// Returns `true` if the frame contains speech.
    ///
    /// The frame must be exactly [`frame_size()`](Vad::frame_size) samples.
    fn is_speech(&mut self, frame: &[f32]) -> Result<bool, TranscribeError>;

    /// Drain any buffered prefill audio and return it as a flat sample vector.
    ///
    /// Call this immediately after [`is_speech()`](Vad::is_speech) returns `true`
    /// for the first time (speech onset) to recover pre-onset audio that the
    /// smoothing/onset logic consumed. The returned samples do **not** include
    /// the current frame — the caller should append that separately.
    ///
    /// Returns an empty vec by default (no prefill support).
    fn drain_prefill(&mut self) -> Vec<f32> {
        Vec::new()
    }

    /// Reset internal state. Call between unrelated audio segments.
    fn reset(&mut self);
}

/// Simple RMS energy-based VAD. Zero dependencies.
///
/// Classifies a frame as speech if its RMS energy exceeds the configured
/// threshold. Suitable for clean audio with minimal background noise.
/// Use [`SileroVad`] for noisy environments.
pub struct EnergyVad {
    frame_size: usize,
    threshold_rms: f32,
}

impl EnergyVad {
    /// Create a new `EnergyVad`.
    ///
    /// - `frame_size`: number of samples per frame
    /// - `threshold_rms`: RMS threshold above which a frame is classified as speech
    pub fn new(frame_size: usize, threshold_rms: f32) -> Self {
        Self {
            frame_size,
            threshold_rms,
        }
    }
}

impl Vad for EnergyVad {
    fn frame_size(&self) -> usize {
        self.frame_size
    }

    fn is_speech(&mut self, frame: &[f32]) -> Result<bool, TranscribeError> {
        if frame.len() != self.frame_size {
            return Err(TranscribeError::Audio(format!(
                "expected {} samples, got {}",
                self.frame_size,
                frame.len()
            )));
        }
        let rms = (frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32).sqrt();
        Ok(rms > self.threshold_rms)
    }

    fn reset(&mut self) {}
}

/// Wraps any [`Vad`] with onset detection and hangover smoothing.
///
/// - **Onset detection**: requires `onset_frames` consecutive speech frames
///   before transitioning to the speech state.
/// - **Hangover**: continues reporting speech for `hangover_frames` frames
///   after the inner VAD stops detecting voice.
/// - **Prefill buffer**: maintains a ring buffer of recent frames (up to
///   `prefill_frames + 1`) for use by higher-level chunking logic.
///
/// The `is_speech()` method returns the smoothed speech/non-speech decision.
/// Use [`in_speech()`](SmoothedVad::in_speech) to query the current state
/// without feeding a new frame.
pub struct SmoothedVad {
    inner: Box<dyn Vad>,
    onset_frames: usize,
    hangover_frames: usize,
    prefill_frames: usize,
    // internal state
    frame_buffer: VecDeque<Vec<f32>>,
    hangover_counter: usize,
    onset_counter: usize,
    in_speech: bool,
    /// Set on speech onset, cleared by `drain_prefill()`. Guards against
    /// draining at the wrong time or double-draining.
    at_onset: bool,
}

impl SmoothedVad {
    /// Create a new `SmoothedVad` wrapping the given inner VAD.
    ///
    /// - `inner`: the underlying VAD implementation
    /// - `prefill_frames`: number of historical frames to retain in the ring buffer
    /// - `hangover_frames`: number of non-speech frames to wait before leaving speech state
    /// - `onset_frames`: number of consecutive speech frames required to enter speech state
    pub fn new(
        inner: Box<dyn Vad>,
        prefill_frames: usize,
        hangover_frames: usize,
        onset_frames: usize,
    ) -> Self {
        Self {
            inner,
            onset_frames,
            hangover_frames,
            prefill_frames,
            frame_buffer: VecDeque::new(),
            hangover_counter: 0,
            onset_counter: 0,
            in_speech: false,
            at_onset: false,
        }
    }

    /// Whether currently in a speech region (after onset, before hangover expires).
    pub fn in_speech(&self) -> bool {
        self.in_speech
    }

    /// Access the prefill frame buffer.
    ///
    /// Contains up to `prefill_frames + 1` most recent frames. Useful for
    /// higher-level chunking logic that needs audio history when speech onset
    /// is detected.
    pub fn frame_buffer(&self) -> &VecDeque<Vec<f32>> {
        &self.frame_buffer
    }
}

impl Vad for SmoothedVad {
    fn frame_size(&self) -> usize {
        self.inner.frame_size()
    }

    fn is_speech(&mut self, frame: &[f32]) -> Result<bool, TranscribeError> {
        // Buffer frame for prefill (skip copy if prefill disabled).
        // The current frame is included in the buffer so that
        // `drain_prefill()` can pop it off — callers add the current
        // frame themselves.
        if self.prefill_frames > 0 {
            self.frame_buffer.push_back(frame.to_vec());
            while self.frame_buffer.len() > self.prefill_frames + 1 {
                self.frame_buffer.pop_front();
            }
        }

        let voice = self.inner.is_speech(frame)?;

        match (self.in_speech, voice) {
            (false, true) => {
                self.onset_counter += 1;
                if self.onset_counter >= self.onset_frames {
                    self.in_speech = true;
                    self.at_onset = true;
                    self.hangover_counter = self.hangover_frames;
                    self.onset_counter = 0;
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            (true, true) => {
                self.hangover_counter = self.hangover_frames;
                Ok(true)
            }
            (true, false) => {
                if self.hangover_counter > 0 {
                    self.hangover_counter -= 1;
                    Ok(true)
                } else {
                    self.in_speech = false;
                    Ok(false)
                }
            }
            (false, false) => {
                self.onset_counter = 0;
                Ok(false)
            }
        }
    }

    fn drain_prefill(&mut self) -> Vec<f32> {
        if !self.at_onset {
            return Vec::new();
        }
        self.at_onset = false;
        // Remove the current frame (pushed during the is_speech call that
        // triggered this drain) — the caller appends it separately.
        self.frame_buffer.pop_back();
        let frame_size = self.inner.frame_size();
        let mut out = Vec::with_capacity(self.frame_buffer.len() * frame_size);
        for buf in self.frame_buffer.drain(..) {
            out.extend(buf);
        }
        out
    }

    fn reset(&mut self) {
        self.frame_buffer.clear();
        self.hangover_counter = 0;
        self.onset_counter = 0;
        self.in_speech = false;
        self.at_onset = false;
        self.inner.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── EnergyVad tests ──────────────────────────────────────────────

    #[test]
    fn energy_vad_silence_below_threshold() {
        let mut vad = EnergyVad::new(480, 0.01);
        let silence = vec![0.0f32; 480];
        assert!(!vad.is_speech(&silence).unwrap());
    }

    #[test]
    fn energy_vad_loud_signal_above_threshold() {
        let mut vad = EnergyVad::new(480, 0.01);
        let loud = vec![1.0f32; 480];
        assert!(vad.is_speech(&loud).unwrap());
    }

    #[test]
    fn energy_vad_wrong_frame_size() {
        let mut vad = EnergyVad::new(480, 0.01);
        let short = vec![0.0f32; 100];
        assert!(vad.is_speech(&short).is_err());
    }

    #[test]
    fn energy_vad_threshold_boundary() {
        let mut vad = EnergyVad::new(480, 0.5);

        // Well below threshold
        let below = vec![0.1f32; 480];
        assert!(!vad.is_speech(&below).unwrap());

        // Well above threshold
        let above = vec![0.9f32; 480];
        assert!(vad.is_speech(&above).unwrap());
    }

    #[test]
    fn energy_vad_frame_size_getter() {
        let vad = EnergyVad::new(320, 0.01);
        assert_eq!(vad.frame_size(), 320);
    }

    // ── SmoothedVad tests ────────────────────────────────────────────

    /// Helper: creates a SmoothedVad wrapping an EnergyVad
    fn make_smoothed(onset: usize, hangover: usize, prefill: usize) -> SmoothedVad {
        SmoothedVad::new(
            Box::new(EnergyVad::new(480, 0.01)),
            prefill,
            hangover,
            onset,
        )
    }

    #[test]
    fn smoothed_onset_requires_n_frames() {
        let mut vad = make_smoothed(3, 5, 0);
        let speech = vec![1.0f32; 480];

        // First two speech frames should NOT trigger onset
        assert!(!vad.is_speech(&speech).unwrap());
        assert!(!vad.in_speech());
        assert!(!vad.is_speech(&speech).unwrap());
        assert!(!vad.in_speech());

        // Third speech frame triggers onset
        assert!(vad.is_speech(&speech).unwrap());
        assert!(vad.in_speech());
    }

    #[test]
    fn smoothed_hangover_extends_speech() {
        let mut vad = make_smoothed(1, 3, 0);
        let speech = vec![1.0f32; 480];
        let silence = vec![0.0f32; 480];

        // Enter speech
        assert!(vad.is_speech(&speech).unwrap());
        assert!(vad.in_speech());

        // Hangover: 3 silent frames should still report speech
        assert!(vad.is_speech(&silence).unwrap()); // hangover 2
        assert!(vad.in_speech());
        assert!(vad.is_speech(&silence).unwrap()); // hangover 1
        assert!(vad.in_speech());
        assert!(vad.is_speech(&silence).unwrap()); // hangover 0
        assert!(vad.in_speech());

        // After hangover expires, should report non-speech
        assert!(!vad.is_speech(&silence).unwrap());
        assert!(!vad.in_speech());
    }

    #[test]
    fn smoothed_speech_resets_hangover() {
        let mut vad = make_smoothed(1, 2, 0);
        let speech = vec![1.0f32; 480];
        let silence = vec![0.0f32; 480];

        // Enter speech
        assert!(vad.is_speech(&speech).unwrap());

        // One silent frame (hangover 1 remaining)
        assert!(vad.is_speech(&silence).unwrap());

        // Speech again resets hangover
        assert!(vad.is_speech(&speech).unwrap());

        // Now need full 2 silent frames again
        assert!(vad.is_speech(&silence).unwrap()); // hangover 1
        assert!(vad.is_speech(&silence).unwrap()); // hangover 0
        assert!(!vad.is_speech(&silence).unwrap()); // expired
    }

    #[test]
    fn smoothed_onset_counter_resets_on_silence() {
        let mut vad = make_smoothed(3, 0, 0);
        let speech = vec![1.0f32; 480];
        let silence = vec![0.0f32; 480];

        // Two speech frames (need 3 for onset)
        assert!(!vad.is_speech(&speech).unwrap());
        assert!(!vad.is_speech(&speech).unwrap());

        // Silence resets the onset counter
        assert!(!vad.is_speech(&silence).unwrap());

        // Need to start over — two more speech frames not enough
        assert!(!vad.is_speech(&speech).unwrap());
        assert!(!vad.is_speech(&speech).unwrap());
        assert!(!vad.in_speech());

        // Third consecutive speech frame triggers
        assert!(vad.is_speech(&speech).unwrap());
        assert!(vad.in_speech());
    }

    #[test]
    fn smoothed_reset_clears_state() {
        let mut vad = make_smoothed(1, 5, 10);
        let speech = vec![1.0f32; 480];

        // Enter speech and feed a few frames
        assert!(vad.is_speech(&speech).unwrap());
        assert!(vad.is_speech(&speech).unwrap());
        assert!(vad.in_speech());
        assert!(!vad.frame_buffer().is_empty());

        // Reset
        vad.reset();
        assert!(!vad.in_speech());
        assert!(vad.frame_buffer().is_empty());
    }

    #[test]
    fn smoothed_prefill_buffer_size() {
        let mut vad = make_smoothed(1, 0, 5);
        let speech = vec![1.0f32; 480];

        // Feed 10 frames — buffer should cap at prefill_frames + 1 = 6
        for _ in 0..10 {
            let _ = vad.is_speech(&speech).unwrap();
        }
        assert_eq!(vad.frame_buffer().len(), 6);
    }

    #[test]
    fn smoothed_frame_size_delegates() {
        let vad = make_smoothed(1, 0, 0);
        assert_eq!(vad.frame_size(), 480);
    }

    #[test]
    fn smoothed_drain_prefill_returns_pre_onset_frames() {
        let mut vad = make_smoothed(2, 0, 5);
        let speech = vec![1.0f32; 480];
        let silence = vec![0.0f32; 480];

        // Feed 3 silence frames (buffered as prefill)
        assert!(!vad.is_speech(&silence).unwrap());
        assert!(!vad.is_speech(&silence).unwrap());
        assert!(!vad.is_speech(&silence).unwrap());

        // Feed 2 speech frames to trigger onset (onset_frames=2)
        assert!(!vad.is_speech(&speech).unwrap()); // onset_counter=1
        assert!(vad.is_speech(&speech).unwrap()); // onset_counter=2, triggers

        // drain_prefill should return pre-onset frames (excluding current)
        let prefill = vad.drain_prefill();
        // Buffer had: [S1, S2, S3, F1, F2], pop F2 → return [S1, S2, S3, F1]
        assert_eq!(prefill.len(), 4 * 480);
    }

    #[test]
    fn smoothed_drain_prefill_empty_without_prefill() {
        let mut vad = make_smoothed(1, 0, 0); // prefill disabled
        let speech = vec![1.0f32; 480];

        assert!(vad.is_speech(&speech).unwrap());
        let prefill = vad.drain_prefill();
        assert!(prefill.is_empty());
    }

    #[test]
    fn smoothed_no_buffering_when_prefill_zero() {
        let mut vad = make_smoothed(1, 0, 0);
        let speech = vec![1.0f32; 480];

        for _ in 0..10 {
            let _ = vad.is_speech(&speech).unwrap();
        }
        assert!(vad.frame_buffer().is_empty());
    }
}
