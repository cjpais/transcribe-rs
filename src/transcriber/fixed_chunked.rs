use crate::{SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult};

use super::merge::merge_sequential_with_separator;
use super::{transcribe_padded, Transcriber, SAMPLE_RATE};

/// Configuration for [`FixedChunked`].
pub struct FixedChunkedConfig {
    /// Chunk duration in seconds. Audio is split at this interval.
    /// Should be short enough to stay within the model's encoder
    /// limits (e.g. 30s for Conformer-based models like Parakeet).
    pub chunk_duration_secs: f32,
    /// Overlap in seconds between adjacent chunks. The tail of each
    /// chunk is kept in the buffer so the next chunk starts with
    /// shared context. Helps avoid garbled words at chunk boundaries.
    /// Set to 0.0 for hard cuts.
    pub overlap_secs: f32,
    /// Seconds of silence to prepend and append to each chunk before
    /// transcription.
    pub padding_secs: f32,
    /// Minimum chunk duration in seconds. Remainders shorter than this
    /// in `finish()` are skipped entirely.
    pub min_chunk_secs: f32,
    /// Separator inserted between chunk texts when merging.
    /// Use `" "` for most languages, `""` for CJK.
    pub merge_separator: String,
}

impl Default for FixedChunkedConfig {
    fn default() -> Self {
        Self {
            chunk_duration_secs: 30.0,
            overlap_secs: 1.0,
            padding_secs: 0.0,
            min_chunk_secs: 0.0,
            merge_separator: " ".into(),
        }
    }
}

/// Fixed-duration chunked transcription with configurable overlap.
///
/// Splits audio into equal-sized chunks and transcribes each one
/// independently. No VAD model or energy analysis needed — just a
/// straightforward time-based split.
///
/// The optional overlap keeps a small tail from each chunk in the
/// buffer so the next chunk starts with shared audio context. This
/// prevents the model from seeing a hard cut mid-word at chunk
/// boundaries.
///
/// Good default when the model has a hard sequence-length limit
/// (e.g. Conformer encoders with fixed positional embeddings) and
/// you want the simplest possible chunking with no extra dependencies.
pub struct FixedChunked {
    config: FixedChunkedConfig,
    options: TranscribeOptions,
    // internal state
    buffer: Vec<f32>,
    elapsed_samples: usize,
    chunk_index: usize,
    results: Vec<TranscriptionResult>,
}

impl FixedChunked {
    pub fn new(config: FixedChunkedConfig, options: TranscribeOptions) -> Self {
        Self {
            config,
            options,
            buffer: Vec::new(),
            elapsed_samples: 0,
            chunk_index: 0,
            results: Vec::new(),
        }
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

impl Transcriber for FixedChunked {
    fn feed(
        &mut self,
        model: &mut dyn SpeechModel,
        samples: &[f32],
    ) -> Result<Vec<TranscriptionResult>, TranscribeError> {
        self.buffer.extend_from_slice(samples);
        self.elapsed_samples += samples.len();

        let chunk_samples = (self.config.chunk_duration_secs * SAMPLE_RATE) as usize;
        let overlap_samples = (self.config.overlap_secs * SAMPLE_RATE) as usize;
        let step_samples = chunk_samples.saturating_sub(overlap_samples).max(1);

        let mut new_results = Vec::new();

        while self.buffer.len() >= chunk_samples {
            let chunk: Vec<f32> = self.buffer[..chunk_samples].to_vec();
            let chunk_start = self.elapsed_samples - self.buffer.len();

            // Drain only the non-overlapping portion so the next chunk
            // starts with `overlap_samples` of shared context.
            self.buffer.drain(..step_samples);

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
    use crate::transcriber::test_helpers::{FailOnNthModel, MockModel};

    #[test]
    fn fixed_splits_at_chunk_duration() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            overlap_secs: 0.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 3s of audio with 1s chunks, no overlap -> 3 chunks in feed
        let audio = vec![0.5f32; 16000 * 3];
        let results = t.feed(&mut model, &audio).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "chunk_16000");
        assert_eq!(results[1].text, "chunk_16000");
        assert_eq!(results[2].text, "chunk_16000");
    }

    #[test]
    fn fixed_overlap_retains_tail() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            overlap_secs: 0.5, // 8000 samples overlap
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 2s of audio. step = 16000 - 8000 = 8000 samples.
        // Chunk 1 at buffer=32000: take 16000, drain 8000. buffer=24000.
        // Chunk 2 at buffer=24000: take 16000, drain 8000. buffer=16000.
        // Chunk 3 at buffer=16000: take 16000, drain 8000. buffer=8000.
        // buffer < chunk_samples, stop.
        let audio = vec![0.5f32; 16000 * 2];
        let results = t.feed(&mut model, &audio).unwrap();
        assert_eq!(results.len(), 3);

        // Each chunk is 16000 samples (1 second)
        for r in &results {
            assert_eq!(r.text, "chunk_16000");
        }

        // finish() transcribes the 8000-sample remainder
        let final_result = t.finish(&mut model).unwrap();
        assert!(final_result.text.contains("chunk_8000"));
    }

    #[test]
    fn fixed_remainder_in_finish() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 3.0,
            overlap_secs: 0.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 2s of audio, 3s chunk -> nothing in feed, all in finish
        let audio = vec![0.5f32; 16000 * 2];
        let results = t.feed(&mut model, &audio).unwrap();
        assert!(results.is_empty());

        let final_result = t.finish(&mut model).unwrap();
        assert_eq!(final_result.text, "chunk_32000");
    }

    #[test]
    fn fixed_min_chunk_skips_short_remainder() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 3.0,
            overlap_secs: 0.0,
            min_chunk_secs: 3.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 2s < 3s min -> skipped in finish
        let audio = vec![0.5f32; 16000 * 2];
        t.feed(&mut model, &audio).unwrap();

        let final_result = t.finish(&mut model).unwrap();
        assert_eq!(final_result.text, "");
    }

    #[test]
    fn fixed_timestamps_correct_no_overlap() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            overlap_secs: 0.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 3];
        let results = t.feed(&mut model, &audio).unwrap();
        assert_eq!(results.len(), 3);

        let seg0 = &results[0].segments.as_ref().unwrap()[0];
        let seg1 = &results[1].segments.as_ref().unwrap()[0];
        let seg2 = &results[2].segments.as_ref().unwrap()[0];
        assert!((seg0.start - 0.0).abs() < 0.01);
        assert!((seg1.start - 1.0).abs() < 0.01);
        assert!((seg2.start - 2.0).abs() < 0.01);
    }

    #[test]
    fn fixed_timestamps_correct_with_overlap() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            overlap_secs: 0.5,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // step = 0.5s. Chunks start at 0.0, 0.5, 1.0, ...
        let audio = vec![0.5f32; 16000 * 2];
        let results = t.feed(&mut model, &audio).unwrap();

        let seg0 = &results[0].segments.as_ref().unwrap()[0];
        let seg1 = &results[1].segments.as_ref().unwrap()[0];
        assert!((seg0.start - 0.0).abs() < 0.01);
        assert!((seg1.start - 0.5).abs() < 0.01);
    }

    #[test]
    fn fixed_timestamps_clamped_to_zero() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            overlap_secs: 0.0,
            padding_secs: 0.5,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 2];
        let results = t.feed(&mut model, &audio).unwrap();
        assert!(!results.is_empty());

        let segs = results[0].segments.as_ref().unwrap();
        assert!(
            segs[0].start >= 0.0,
            "timestamp should not be negative, got {}",
            segs[0].start
        );
    }

    #[test]
    fn fixed_transcribe_convenience() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 5];
        let result = t.transcribe(&mut model, &audio).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn fixed_empty_input() {
        let config = FixedChunkedConfig::default();
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        let result = t.transcribe(&mut model, &[]).unwrap();
        assert_eq!(result.text, "");
    }

    #[test]
    fn fixed_object_safe() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 1.0,
            ..Default::default()
        };
        let mut transcriber: Box<dyn Transcriber> =
            Box::new(FixedChunked::new(config, TranscribeOptions::default()));
        let mut model = MockModel;

        let audio = vec![0.5f32; 16000 * 3];
        let result = transcriber.transcribe(&mut model, &audio).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn fixed_reusable_after_error() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 3.0,
            overlap_secs: 0.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());

        // First session: error during finish
        let audio = vec![0.5f32; 16000 * 2];
        t.feed(&mut FailOnNthModel::new(99), &audio).unwrap();
        assert!(t.finish(&mut FailOnNthModel::new(1)).is_err());

        // Second session: should work after error reset
        let mut model = MockModel;
        let result = t.transcribe(&mut model, &audio).unwrap();
        assert!(!result.text.is_empty());
    }

    #[test]
    fn fixed_short_audio_single_pass() {
        let config = FixedChunkedConfig {
            chunk_duration_secs: 30.0,
            ..Default::default()
        };
        let mut t = FixedChunked::new(config, TranscribeOptions::default());
        let mut model = MockModel;

        // 5s audio with 30s chunks -> single chunk in finish
        let audio = vec![0.5f32; 16000 * 5];
        let results = t.feed(&mut model, &audio).unwrap();
        assert!(results.is_empty());

        let final_result = t.finish(&mut model).unwrap();
        assert_eq!(final_result.text, "chunk_80000");
    }
}
