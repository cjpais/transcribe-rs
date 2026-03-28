use crate::{SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult};

use super::merge::merge_sequential_with_separator;
use super::{rms_energy, transcribe_padded, Transcriber, SAMPLE_RATE};

/// Configuration for [`EnergyAdaptiveChunked`].
pub struct EnergyAdaptiveConfig {
    /// Target chunk duration in seconds (e.g. 30.0). The actual chunk
    /// will be within `target ± search_window` of this value.
    pub target_chunk_secs: f32,
    /// Scan +/- this far from the target for a low-energy split point
    /// (e.g. 3.0). Set to 0.0 for fixed-duration chunks.
    pub search_window_secs: f32,
    /// Seconds of silence to prepend and append to each chunk before
    /// transcription.
    pub padding_secs: f32,
    /// Minimum chunk duration in seconds. Chunks shorter than this are
    /// zero-padded before transcription. Remainders shorter than this
    /// in `finish()` are skipped entirely (unlike [`VadChunkedConfig::min_chunk_secs`]
    /// which carries short speech forward to merge with the next region).
    pub min_chunk_secs: f32,
    /// Frame size in samples for energy analysis when searching for
    /// split points. Smaller values give finer temporal resolution;
    /// larger values are faster to scan. Default: 480 (30ms at 16kHz).
    pub frame_size: usize,
    /// Separator inserted between chunk texts when merging.
    /// Use `" "` for most languages, `""` for CJK.
    pub merge_separator: String,
}

impl Default for EnergyAdaptiveConfig {
    fn default() -> Self {
        Self {
            target_chunk_secs: 30.0,
            search_window_secs: 3.0,
            padding_secs: 0.0,
            min_chunk_secs: 0.0,
            frame_size: 480,
            merge_separator: " ".into(),
        }
    }
}

/// Adaptive chunked transcription using energy-based split point search.
///
/// Targets a fixed chunk duration but adjusts the actual split point to
/// land on a low-energy frame (natural pause) within a configurable
/// search window around the target. No VAD model needed — just RMS
/// energy analysis on buffered audio.
///
/// Good default for file transcription when you don't have or want a
/// neural VAD model. Avoids splitting mid-word by finding natural
/// pause points via energy analysis.
pub struct EnergyAdaptiveChunked {
    config: EnergyAdaptiveConfig,
    options: TranscribeOptions,
    // internal state
    buffer: Vec<f32>,
    elapsed_samples: usize,
    chunk_index: usize,
    results: Vec<TranscriptionResult>,
}

impl EnergyAdaptiveChunked {
    pub fn new(config: EnergyAdaptiveConfig, options: TranscribeOptions) -> Self {
        Self {
            config,
            options,
            buffer: Vec::new(),
            elapsed_samples: 0,
            chunk_index: 0,
            results: Vec::new(),
        }
    }

    /// Find the best split point in the buffer, searching around
    /// `target_samples` for the minimum-energy frame.
    fn find_split_point(&self, target_samples: usize) -> usize {
        let search_samples = (self.config.search_window_secs * SAMPLE_RATE) as usize;
        let search_start = target_samples.saturating_sub(search_samples);
        let search_end = (target_samples + search_samples).min(self.buffer.len());

        // Align to frame boundaries
        let search_start = (search_start / self.config.frame_size) * self.config.frame_size;

        let mut min_rms = f32::MAX;
        let mut best_offset = target_samples.min(self.buffer.len());

        let mut offset = search_start;
        while offset + self.config.frame_size <= search_end {
            let frame = &self.buffer[offset..offset + self.config.frame_size];
            let rms = rms_energy(frame);
            if rms < min_rms {
                min_rms = rms;
                best_offset = offset + self.config.frame_size;
            }
            offset += self.config.frame_size;
        }

        log::debug!(
            "energy adaptive: target={:.2}s, split at {:.2}s (rms={:.4})",
            target_samples as f32 / SAMPLE_RATE,
            best_offset as f32 / SAMPLE_RATE,
            min_rms,
        );

        best_offset
    }

    fn transcribe_chunk(
        &mut self,
        model: &mut dyn SpeechModel,
        chunk: &[f32],
        chunk_start_samples: usize,
    ) -> Result<TranscriptionResult, TranscribeError> {
        let chunk_start_secs = chunk_start_samples as f32 / SAMPLE_RATE;

        log::info!(
            "chunk {}: start={:.2}s duration={:.2}s samples={} padding={:.0}ms",
            self.chunk_index,
            chunk_start_secs,
            chunk.len() as f32 / SAMPLE_RATE,
            chunk.len(),
            self.config.padding_secs * 1000.0,
        );

        self.chunk_index += 1;

        let result = transcribe_padded(
            model,
            chunk,
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
        if !self.buffer.is_empty() {
            let chunk = std::mem::take(&mut self.buffer);
            let chunk_secs = chunk.len() as f32 / SAMPLE_RATE;
            if chunk_secs >= self.config.min_chunk_secs {
                let chunk_start = self.elapsed_samples - chunk.len();
                self.transcribe_chunk(model, &chunk, chunk_start)?;
            } else {
                log::debug!(
                    "skipping short remainder ({:.2}s < min {:.2}s)",
                    chunk_secs,
                    self.config.min_chunk_secs
                );
            }
        }
        Ok(merge_sequential_with_separator(
            &self.results,
            &self.config.merge_separator,
        ))
    }

    fn reset_state(&mut self) {
        self.buffer.clear();
        self.results.clear();
        self.elapsed_samples = 0;
        self.chunk_index = 0;
    }
}

impl Transcriber for EnergyAdaptiveChunked {
    fn feed(
        &mut self,
        model: &mut dyn SpeechModel,
        samples: &[f32],
    ) -> Result<Vec<TranscriptionResult>, TranscribeError> {
        self.buffer.extend_from_slice(samples);
        self.elapsed_samples += samples.len();

        let target_samples = (self.config.target_chunk_secs * SAMPLE_RATE) as usize;
        let search_samples = (self.config.search_window_secs * SAMPLE_RATE) as usize;
        // We need at least target + search_window of audio to find the best
        // split point (the optimal point could be past the target).
        let min_buffer_for_split = target_samples + search_samples;

        let mut new_results = Vec::new();

        while self.buffer.len() >= min_buffer_for_split {
            let split_at = self.find_split_point(target_samples);
            let chunk: Vec<f32> = self.buffer.drain(..split_at).collect();
            let chunk_start = self.elapsed_samples - self.buffer.len() - chunk.len();
            new_results.push(self.transcribe_chunk(model, &chunk, chunk_start)?);
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
    use crate::transcriber::test_helpers::MockModel;

    #[test]
    fn energy_adaptive_splits_at_low_energy() {
        // 3s target, 0.5s search window
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 3.0,
            search_window_secs: 0.5,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // Build 5s of audio: loud except for a quiet region at ~2.8s
        // (within search window of target 3.0s)
        let mut audio = vec![1.0f32; 16000 * 5];
        // Insert quiet region at 2.7-2.8s
        let quiet_start = (2.7 * SAMPLE_RATE) as usize;
        let quiet_end = (2.8 * SAMPLE_RATE) as usize;
        for s in &mut audio[quiet_start..quiet_end] {
            *s = 0.01;
        }

        let results = t.feed(&mut model, &audio).unwrap();
        assert_eq!(results.len(), 1);

        // The split should have landed in the quiet region (around 2.7-2.8s)
        // rather than exactly at 3.0s. The chunk size in samples tells us where.
        let chunk_samples: usize = results[0]
            .text
            .strip_prefix("chunk_")
            .unwrap()
            .parse()
            .unwrap();
        let chunk_secs = chunk_samples as f32 / SAMPLE_RATE;
        assert!(
            chunk_secs >= 2.5 && chunk_secs <= 3.5,
            "split at {chunk_secs:.2}s, expected within search window of 3.0s"
        );
        // Should be closer to 2.8 than to 3.0
        assert!(
            chunk_secs < 3.0,
            "split at {chunk_secs:.2}s, should prefer quiet region before target"
        );
    }

    #[test]
    fn energy_adaptive_uniform_audio_splits_near_target() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 1.0,
            search_window_secs: 0.3,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // Uniform audio — all frames have equal energy
        let audio = vec![0.5f32; 16000 * 3];
        let results = t.feed(&mut model, &audio).unwrap();

        // Should still produce chunks (first minimum-energy frame wins)
        assert!(!results.is_empty());
    }

    #[test]
    fn energy_adaptive_remainder_in_finish() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 3.0,
            search_window_secs: 0.5,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 2s of audio — not enough for a chunk in feed()
        let audio = vec![0.5f32; 16000 * 2];
        let results = t.feed(&mut model, &audio).unwrap();
        assert!(results.is_empty());

        // finish() should transcribe the remainder
        let final_result = t.finish(&mut model).unwrap();
        assert_eq!(final_result.text, "chunk_32000");
    }

    #[test]
    fn energy_adaptive_min_chunk_skips_short_remainder() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 3.0,
            search_window_secs: 0.5,
            min_chunk_secs: 3.0, // remainder must be >= 3s
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 2s of audio — shorter than min_chunk_secs
        let audio = vec![0.5f32; 16000 * 2];
        t.feed(&mut model, &audio).unwrap();

        let final_result = t.finish(&mut model).unwrap();
        assert_eq!(final_result.text, ""); // skipped
    }

    #[test]
    fn energy_adaptive_timestamps_correct() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 1.0,
            search_window_secs: 0.0,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // search_window=0 means split exactly at target (fixed-duration chunks)
        let audio = vec![0.5f32; 16000 * 3];
        let results = t.feed(&mut model, &audio).unwrap();

        // With 0 search window, min_buffer_for_split = target = 1s.
        // 3s of audio yields 3 chunks (48000 → 32000 → 16000 → 0).
        assert_eq!(results.len(), 3);

        if results.len() >= 2 {
            let seg0 = &results[0].segments.as_ref().unwrap()[0];
            let seg1 = &results[1].segments.as_ref().unwrap()[0];
            assert!((seg0.start - 0.0).abs() < 0.01);
            assert!((seg1.start - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn energy_adaptive_timestamps_clamped_to_zero() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 1.0,
            search_window_secs: 0.0,
            padding_secs: 0.5,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 2];
        let results = t.feed(&mut model, &audio).unwrap();
        assert!(!results.is_empty());

        // With 500ms padding on the first chunk (start=0), timestamps must be >= 0
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
    fn energy_adaptive_transcribe_convenience() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 1.0,
            search_window_secs: 0.3,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 5];
        let result = t.transcribe(&mut model, &audio).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn energy_adaptive_empty_input() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 1.0,
            search_window_secs: 0.3,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let result = t.transcribe(&mut model, &[]).unwrap();
        assert_eq!(result.text, "");
    }

    #[test]
    fn energy_adaptive_object_safe() {
        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 1.0,
            search_window_secs: 0.3,
            ..Default::default()
        };
        let mut transcriber: Box<dyn Transcriber> = Box::new(EnergyAdaptiveChunked::new(
            config,
            TranscribeOptions::default(),
        ));
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 5];
        let result = transcriber.transcribe(&mut model, &audio).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn energy_adaptive_reusable_after_error() {
        use crate::transcriber::test_helpers::FailOnNthModel;

        let config = EnergyAdaptiveConfig {
            target_chunk_secs: 3.0,
            search_window_secs: 0.5,
            ..Default::default()
        };
        let mut t = EnergyAdaptiveChunked::new(config, TranscribeOptions::default());

        // First session: feed 2s (below 3.5s split threshold), error during finish
        let audio = vec![0.5f32; 16000 * 2];
        t.feed(&mut FailOnNthModel::new(99), &audio).unwrap();
        assert!(t.finish(&mut FailOnNthModel::new(1)).is_err());

        // Second session: should work cleanly after error reset
        let mut model = MockModel;
        let result = t.transcribe(&mut model, &audio).unwrap();
        assert!(!result.text.is_empty());
    }
}
