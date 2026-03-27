use crate::vad::Vad;
use crate::{SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult};

use super::merge::merge_sequential_with_separator;
use super::{rms_energy, transcribe_padded, Transcriber, SAMPLE_RATE};

/// Configuration for [`VadChunked`].
pub struct VadChunkedConfig {
    /// Minimum chunk duration in seconds. Chunks shorter than this
    /// are carried forward and merged with the next speech region
    /// (unlike [`EnergyAdaptiveConfig::min_chunk_secs`] which skips
    /// short remainders). Final chunks in `finish()` are zero-padded
    /// to this duration before transcription.
    pub min_chunk_secs: f32,
    /// Maximum chunk duration in seconds. Force-split if VAD finds
    /// no silence within this duration.
    pub max_chunk_secs: f32,
    /// Seconds of silence to prepend and append to each chunk before
    /// transcription. Helps models that need surrounding context.
    pub padding_secs: f32,
    /// If set, when force-splitting at `max_chunk_secs`, scan backward
    /// over this many seconds of audio to find the lowest-energy frame
    /// and split there instead of hard-cutting. This avoids splitting
    /// mid-word during long monologues. `None` disables (hard cut).
    pub smart_split_search_secs: Option<f32>,
    /// Separator inserted between chunk texts when merging.
    /// Use `" "` for most languages, `""` for CJK.
    pub merge_separator: String,
}

impl Default for VadChunkedConfig {
    fn default() -> Self {
        Self {
            min_chunk_secs: 1.0,
            max_chunk_secs: 30.0,
            padding_secs: 0.0,
            smart_split_search_secs: None,
            merge_separator: " ".into(),
        }
    }
}

/// VAD-based chunked transcription.
///
/// Splits audio on speech/silence boundaries detected by a [`Vad`].
/// Each speech region is transcribed independently, then merged in
/// [`finish()`](Transcriber::finish).
///
/// Works for both live audio (small per-frame feeds) and file
/// transcription (one large feed).
pub struct VadChunked {
    vad: Box<dyn Vad>,
    config: VadChunkedConfig,
    options: TranscribeOptions,
    // internal state
    speech_buffer: Vec<f32>,
    /// Accumulates sub-frame remainders across `feed()` calls so they
    /// can be combined with the next call's samples to form a complete
    /// frame for VAD processing.
    pending: Vec<f32>,
    in_speech: bool,
    elapsed_samples: usize,
    /// Sample offset where the first sample entered the current speech
    /// buffer. Used for accurate timestamp calculation, especially when
    /// short speech carries forward across silence gaps.
    speech_start_sample: Option<usize>,
    chunk_index: usize,
    results: Vec<TranscriptionResult>,
}

impl VadChunked {
    pub fn new(vad: Box<dyn Vad>, config: VadChunkedConfig, options: TranscribeOptions) -> Self {
        Self {
            vad,
            config,
            options,
            speech_buffer: Vec::new(),
            pending: Vec::new(),
            in_speech: false,
            elapsed_samples: 0,
            speech_start_sample: None,
            chunk_index: 0,
            results: Vec::new(),
        }
    }

    /// When force-splitting, scan backward over `search_secs` of the
    /// speech buffer to find the frame with the lowest RMS energy. Splits
    /// the buffer there, transcribes the first part, and keeps the
    /// remainder for the next chunk.
    fn smart_split_buffer(
        &mut self,
        model: &mut dyn SpeechModel,
        search_secs: f32,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let frame_size = self.vad.frame_size();
        let search_samples = (search_secs * SAMPLE_RATE) as usize;
        let buf_len = self.speech_buffer.len();

        // Find the lowest-energy frame in the search window
        let search_start = buf_len.saturating_sub(search_samples);
        // Align to frame boundaries
        let search_start = (search_start / frame_size) * frame_size;

        let mut min_rms = f32::MAX;
        let mut best_offset = buf_len; // default: split at end (same as hard cut)

        let mut offset = search_start;
        while offset + frame_size <= buf_len {
            let frame = &self.speech_buffer[offset..offset + frame_size];
            let rms = rms_energy(frame);
            if rms < min_rms {
                min_rms = rms;
                best_offset = offset + frame_size; // split after this frame
            }
            offset += frame_size;
        }

        log::info!(
            "smart split: search window {:.2}s, best split at {:.2}s (rms={:.4}), buffer={:.2}s",
            search_secs,
            best_offset as f32 / SAMPLE_RATE,
            min_rms,
            buf_len as f32 / SAMPLE_RATE,
        );

        // Drain the chunk; remainder stays in speech_buffer.
        let chunk: Vec<f32> = self.speech_buffer.drain(..best_offset).collect();
        let chunk_start_secs = self.speech_start_sample.unwrap_or_else(|| {
            self.elapsed_samples
                .saturating_sub(self.speech_buffer.len() + chunk.len())
        }) as f32
            / SAMPLE_RATE;

        // Update speech_start_sample for the remainder
        if self.speech_buffer.is_empty() {
            self.speech_start_sample = None;
        } else {
            self.speech_start_sample = self.speech_start_sample.map(|s| s + best_offset);
        }

        self.transcribe_chunk(model, chunk, chunk_start_secs)
    }

    /// Take the entire speech buffer and transcribe it as one chunk.
    fn flush_speech_buffer(
        &mut self,
        model: &mut dyn SpeechModel,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let samples = std::mem::take(&mut self.speech_buffer);
        let chunk_start_secs = self
            .speech_start_sample
            .unwrap_or_else(|| self.elapsed_samples.saturating_sub(samples.len()))
            as f32
            / SAMPLE_RATE;
        self.speech_start_sample = None;
        self.transcribe_chunk(model, samples, chunk_start_secs)
    }

    /// Transcribe a chunk of audio samples at a given position. Does not
    /// touch `self.speech_buffer` or `self.speech_start_sample` — callers
    /// manage buffer state before calling this.
    fn transcribe_chunk(
        &mut self,
        model: &mut dyn SpeechModel,
        samples: Vec<f32>,
        chunk_start_secs: f32,
    ) -> Result<TranscriptionResult, TranscribeError> {
        log::info!(
            "chunk {}: start={:.2}s duration={:.2}s samples={} padding={:.0}ms",
            self.chunk_index,
            chunk_start_secs,
            samples.len() as f32 / SAMPLE_RATE,
            samples.len(),
            self.config.padding_secs * 1000.0,
        );

        self.chunk_index += 1;

        let result = transcribe_padded(
            model,
            &samples,
            self.config.padding_secs,
            self.config.min_chunk_secs,
            chunk_start_secs,
            &self.options,
        )?;

        log::info!("  -> \"{}\"", result.text.trim());

        self.results.push(result.clone());
        Ok(result)
    }

    fn finish_inner(
        &mut self,
        model: &mut dyn SpeechModel,
    ) -> Result<TranscriptionResult, TranscribeError> {
        // Flush any pending sub-frame samples into the speech buffer.
        // Pending holds at most frame_size-1 samples (~29ms at 480/16kHz) that
        // were never VAD-classified. This is practically harmless: if we're in a
        // speech region the tail belongs there, and if not, the model will produce
        // empty/whitespace text that merge filters out.
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            if self.speech_buffer.is_empty() && self.speech_start_sample.is_none() {
                self.speech_start_sample = Some(self.elapsed_samples);
            }
            self.speech_buffer.extend_from_slice(&pending);
            self.elapsed_samples += pending.len();
        }

        if !self.speech_buffer.is_empty() {
            log::info!(
                "finish: transcribing remaining buffer ({:.2}s)",
                self.speech_buffer.len() as f32 / SAMPLE_RATE
            );
            self.flush_speech_buffer(model)?;
        }
        log::info!("session complete: {} chunks transcribed", self.chunk_index);
        Ok(merge_sequential_with_separator(
            &self.results,
            &self.config.merge_separator,
        ))
    }

    fn reset_state(&mut self) {
        self.results.clear();
        self.speech_buffer.clear();
        self.pending.clear();
        self.in_speech = false;
        self.elapsed_samples = 0;
        self.speech_start_sample = None;
        self.chunk_index = 0;
        self.vad.reset();
    }
}

impl Transcriber for VadChunked {
    fn feed(
        &mut self,
        model: &mut dyn SpeechModel,
        samples: &[f32],
    ) -> Result<Vec<TranscriptionResult>, TranscribeError> {
        let frame_size = self.vad.frame_size();
        let mut new_results = Vec::new();

        // Combine pending sub-frame samples from previous call
        let combined;
        let to_process = if self.pending.is_empty() {
            samples
        } else {
            self.pending.extend_from_slice(samples);
            combined = std::mem::take(&mut self.pending);
            &combined
        };

        for frame in to_process.chunks(frame_size) {
            if frame.len() < frame_size {
                // Partial frame at end — hold for next feed() call
                self.pending.extend_from_slice(frame);
                continue;
            }

            let is_speech = self.vad.is_speech(frame)?;
            self.elapsed_samples += frame_size;

            if is_speech {
                if !self.in_speech {
                    // Speech onset — recover pre-onset audio from VAD prefill
                    let prefill = self.vad.drain_prefill();
                    if self.speech_start_sample.is_none() {
                        self.speech_start_sample =
                            Some((self.elapsed_samples - frame_size).saturating_sub(prefill.len()));
                    }
                    self.speech_buffer.extend_from_slice(&prefill);
                }
                self.speech_buffer.extend_from_slice(frame);
                self.in_speech = true;

                // Force-split if exceeding max duration
                let chunk_secs = self.speech_buffer.len() as f32 / SAMPLE_RATE;
                if chunk_secs >= self.config.max_chunk_secs {
                    log::info!(
                        "force-splitting at {:.2}s (max_chunk_secs={:.2})",
                        self.elapsed_samples as f32 / SAMPLE_RATE,
                        self.config.max_chunk_secs
                    );
                    let result = if let Some(search_secs) = self.config.smart_split_search_secs {
                        self.smart_split_buffer(model, search_secs)?
                    } else {
                        self.flush_speech_buffer(model)?
                    };
                    new_results.push(result);
                }
            } else if self.in_speech {
                // Speech -> silence transition: transcribe the chunk
                self.in_speech = false;
                if !self.speech_buffer.is_empty() {
                    let chunk_secs = self.speech_buffer.len() as f32 / SAMPLE_RATE;
                    if chunk_secs >= self.config.min_chunk_secs {
                        log::info!(
                            "speech->silence at {:.2}s, chunk buffered={:.2}s",
                            self.elapsed_samples as f32 / SAMPLE_RATE,
                            chunk_secs
                        );
                        new_results.push(self.flush_speech_buffer(model)?);
                    } else {
                        // Keep short speech in buffer so it merges with the
                        // next speech region (don't lose brief utterances)
                        log::debug!(
                            "carrying forward short chunk ({:.2}s < min {:.2}s)",
                            chunk_secs,
                            self.config.min_chunk_secs
                        );
                    }
                }
            }
        }

        Ok(new_results)
    }

    fn finish(
        &mut self,
        model: &mut dyn SpeechModel,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let result = self.finish_inner(model);
        self.reset_state();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcriber::test_helpers::{make_silence, make_speech, FailOnNthModel, MockModel};
    use crate::vad::{EnergyVad, SmoothedVad};

    #[test]
    fn vad_chunked_basic_speech_then_silence() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0, // no minimum for test
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed speech frames
        let speech = make_speech(480, 10); // 10 frames = 300ms
        let results = t.feed(&mut model, &speech).unwrap();
        assert!(results.is_empty()); // no silence boundary yet

        // Feed silence to trigger transcription
        let silence = make_silence(480, 5);
        let results = t.feed(&mut model, &silence).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text, "chunk_4800"); // 10 * 480 samples
    }

    #[test]
    fn vad_chunked_finish_transcribes_remainder() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed only speech, no silence
        let speech = make_speech(480, 10);
        let results = t.feed(&mut model, &speech).unwrap();
        assert!(results.is_empty());

        // finish should transcribe the remainder
        let final_result = t.finish(&mut model).unwrap();
        assert_eq!(final_result.text, "chunk_4800");
    }

    #[test]
    fn vad_chunked_max_duration_force_splits() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            max_chunk_secs: 0.06, // 60ms = 960 samples = 2 frames
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed 10 frames of speech — should force-split multiple times
        let speech = make_speech(480, 10);
        let results = t.feed(&mut model, &speech).unwrap();
        assert!(results.len() >= 4); // 10 frames / 2 frames per chunk
    }

    #[test]
    fn vad_chunked_short_speech_carries_forward() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 1.0, // 1 second minimum
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed very short speech (1 frame = 30ms) then silence
        let speech = make_speech(480, 1);
        t.feed(&mut model, &speech).unwrap();

        let silence = make_silence(480, 5);
        let results = t.feed(&mut model, &silence).unwrap();
        assert!(results.is_empty()); // too short, not transcribed yet

        // Feed more speech — the short chunk should merge with this one
        let speech2 = make_speech(480, 40); // 1.2s
        t.feed(&mut model, &speech2).unwrap();

        let silence2 = make_silence(480, 5);
        let results = t.feed(&mut model, &silence2).unwrap();
        assert_eq!(results.len(), 1);
        // Should contain the original 1 frame + 40 new frames = 41 * 480 = 19680 samples
        assert_eq!(results[0].text, "chunk_19680");
    }

    #[test]
    fn vad_chunked_carry_forward_timestamp_correct() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 1.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Speech at t=0 (1 frame), silence gap, then more speech
        let speech = make_speech(480, 1);
        t.feed(&mut model, &speech).unwrap();

        let silence = make_silence(480, 5);
        t.feed(&mut model, &silence).unwrap();

        let speech2 = make_speech(480, 40);
        t.feed(&mut model, &speech2).unwrap();

        let silence2 = make_silence(480, 5);
        let results = t.feed(&mut model, &silence2).unwrap();
        assert_eq!(results.len(), 1);

        // Timestamp should reflect the start of the first speech (t=0),
        // not the back-calculated position
        let segs = results[0].segments.as_ref().unwrap();
        assert!(
            segs[0].start < 0.01,
            "expected start near 0.0, got {}",
            segs[0].start
        );
    }

    #[test]
    fn vad_chunked_timestamps_adjusted() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed silence first (1 second = ~33 frames at 480)
        let silence = make_silence(480, 34);
        t.feed(&mut model, &silence).unwrap();

        // Then speech
        let speech = make_speech(480, 10);
        t.feed(&mut model, &speech).unwrap();

        // Then silence to trigger
        let silence2 = make_silence(480, 5);
        let results = t.feed(&mut model, &silence2).unwrap();
        assert_eq!(results.len(), 1);

        // Segment timestamps should be offset by the initial silence
        let segs = results[0].segments.as_ref().unwrap();
        assert!(segs[0].start > 0.9); // ~1.02 seconds of silence preceded it
    }

    #[test]
    fn vad_chunked_timestamps_clamped_to_zero() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            padding_secs: 0.5, // 500ms padding
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed speech immediately (chunk_start_secs ≈ 0)
        let speech = make_speech(480, 10);
        t.feed(&mut model, &speech).unwrap();

        let silence = make_silence(480, 5);
        let results = t.feed(&mut model, &silence).unwrap();
        assert_eq!(results.len(), 1);

        // With 500ms padding and chunk starting near 0, timestamps must be >= 0
        let segs = results[0].segments.as_ref().unwrap();
        assert!(
            segs[0].start >= 0.0,
            "timestamp should not be negative, got {}",
            segs[0].start
        );
        assert!(
            segs[0].end >= 0.0,
            "timestamp should not be negative, got {}",
            segs[0].end
        );
    }

    #[test]
    fn vad_chunked_propagates_transcription_error() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            max_chunk_secs: 0.06, // force split every 2 frames
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = FailOnNthModel::new(2); // fail on 2nd chunk

        // Feed enough speech for 4+ chunks — should error on 2nd chunk
        let speech = make_speech(480, 10);
        let result = t.feed(&mut model, &speech);
        assert!(result.is_err());
    }

    #[test]
    fn vad_chunked_transcribe_convenience() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        let speech = make_speech(480, 10);
        let result = t.transcribe(&mut model, &speech).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn vad_chunked_smart_split_finds_low_energy() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            max_chunk_secs: 0.3, // 300ms = 10 frames to force split
            smart_split_search_secs: Some(0.15), // search last 150ms (5 frames)
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Build 12 frames of speech where frame 7 (0-indexed) is quieter.
        // Frames 0-6: loud (1.0), frame 7: quiet (0.1), frames 8-11: loud (1.0)
        // Search window covers frames 5-9 (last 5 frames of the 10-frame max).
        // Smart split should pick frame 7 as the split point.
        let mut audio = Vec::new();
        for i in 0..12 {
            let val = if i == 7 { 0.1 } else { 1.0 };
            audio.extend(vec![val; 480]);
        }

        let results = t.feed(&mut model, &audio).unwrap();
        // Should have at least 1 result from force-split
        assert!(!results.is_empty());

        // The first chunk should be split at frame 8 (after the quiet frame 7),
        // so it should contain 8 * 480 = 3840 samples
        assert_eq!(results[0].text, "chunk_3840");
    }

    #[test]
    fn vad_chunked_smart_split_disabled_hard_cuts() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            max_chunk_secs: 0.06,          // 60ms = 2 frames
            smart_split_search_secs: None, // disabled
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // All uniform energy — with smart split disabled, hard-cuts at exactly 2 frames
        let speech = make_speech(480, 6);
        let results = t.feed(&mut model, &speech).unwrap();
        assert_eq!(results.len(), 3); // 6 frames / 2 = 3 exact chunks
        assert_eq!(results[0].text, "chunk_960"); // 2 * 480
    }

    #[test]
    fn vad_chunked_multiple_speech_regions() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // speech -> silence -> speech -> silence
        let mut audio = Vec::new();
        audio.extend(make_speech(480, 5));
        audio.extend(make_silence(480, 5));
        audio.extend(make_speech(480, 8));
        audio.extend(make_silence(480, 5));

        let results = t.feed(&mut model, &audio).unwrap();
        assert_eq!(results.len(), 2); // two speech regions
    }

    #[test]
    fn vad_chunked_object_safe() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut transcriber: Box<dyn Transcriber> = Box::new(VadChunked::new(
            Box::new(vad),
            config,
            TranscribeOptions::default(),
        ));
        let mut model = MockModel;

        let speech = make_speech(480, 10);
        let result = transcriber.transcribe(&mut model, &speech).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn vad_chunked_prefill_captures_onset_audio() {
        // SmoothedVad with onset_frames=2 and prefill_frames=5.
        // Without prefill, the first onset frame (which SmoothedVad returns
        // false for) would be lost. With prefill, it's recovered.
        let inner = EnergyVad::new(480, 0.01);
        let vad = SmoothedVad::new(Box::new(inner), 5, 0, 2);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed 3 frames of speech then silence to trigger transcription.
        // With onset_frames=2: frame 1 returns false, frame 2 triggers onset.
        // Prefill recovers frame 1, so all 3 speech frames are captured.
        let speech = make_speech(480, 3);
        t.feed(&mut model, &speech).unwrap();

        let silence = make_silence(480, 5);
        let results = t.feed(&mut model, &silence).unwrap();
        assert_eq!(results.len(), 1);
        // All 3 frames captured (not just 2): 3 * 480 = 1440
        assert_eq!(results[0].text, "chunk_1440");
    }

    #[test]
    fn vad_chunked_prefill_timestamp_accounts_for_prefill() {
        let inner = EnergyVad::new(480, 0.01);
        let vad = SmoothedVad::new(Box::new(inner), 5, 0, 2);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // 3 silence frames, then 3 speech frames, then silence
        let silence = make_silence(480, 3);
        t.feed(&mut model, &silence).unwrap();

        let speech = make_speech(480, 3);
        t.feed(&mut model, &speech).unwrap();

        let silence2 = make_silence(480, 5);
        let results = t.feed(&mut model, &silence2).unwrap();
        assert_eq!(results.len(), 1);

        // Prefill includes the 3 silence frames + onset frame 1.
        // Speech start should account for prefill reaching back into the
        // silence region. The first speech frame starts at sample 3*480=1440,
        // and prefill includes 3 silence frames before it, so start ≈ 0.
        let segs = results[0].segments.as_ref().unwrap();
        assert!(
            segs[0].start < 0.01,
            "expected start near 0.0, got {}",
            segs[0].start
        );
    }

    #[test]
    fn vad_chunked_pending_frame_alignment() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());
        let mut model = MockModel;

        // Feed 1000 samples (2 full frames of 480 + 40 remainder)
        // The 40 remainder should go to pending, not speech_buffer
        let speech = make_speech(1, 1000); // 1000 samples of speech
        t.feed(&mut model, &speech).unwrap();

        // Feed another 1000: pending(40) + 1000 = 1040 → 2 frames + 80 pending
        t.feed(&mut model, &speech).unwrap();

        // Feed silence to trigger. The 80 pending speech samples combine with
        // the start of silence to form a frame that VAD still classifies as
        // speech (RMS > threshold), so 5 full frames total.
        let silence = make_silence(480, 5);
        let results = t.feed(&mut model, &silence).unwrap();
        assert_eq!(results.len(), 1);

        // 4 full speech frames (1920) + 1 mixed frame (80 speech + 400 silence = 480)
        assert_eq!(results[0].text, "chunk_2400");
    }

    #[test]
    fn vad_chunked_reusable_after_error() {
        let vad = EnergyVad::new(480, 0.01);
        let config = VadChunkedConfig {
            min_chunk_secs: 0.0,
            ..Default::default()
        };
        let mut t = VadChunked::new(Box::new(vad), config, TranscribeOptions::default());

        // First session: force an error during finish
        let speech = make_speech(480, 10);
        t.feed(&mut FailOnNthModel::new(1), &speech).unwrap();
        assert!(t.finish(&mut FailOnNthModel::new(1)).is_err());

        // Second session: should work cleanly after error reset
        let mut model = MockModel;
        let result = t.transcribe(&mut model, &speech).unwrap();
        assert_eq!(result.text, "chunk_4800");
    }
}
