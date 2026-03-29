use super::SAMPLE_RATE;
use crate::{
    ModelCapabilities, SpeechModel, TranscribeError, TranscribeOptions, TranscriptionResult,
    TranscriptionSegment,
};

/// Mock model that returns the sample count as text.
pub(crate) struct MockModel;

impl SpeechModel for MockModel {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            name: "mock",
            engine_id: "mock",
            sample_rate: 16000,
            languages: &[],
            supports_timestamps: false,
            supports_translation: false,
            supports_streaming: false,
        }
    }

    fn transcribe_raw(
        &mut self,
        samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        Ok(TranscriptionResult {
            text: format!("chunk_{}", samples.len()),
            segments: Some(vec![TranscriptionSegment {
                start: 0.0,
                end: samples.len() as f32 / SAMPLE_RATE,
                text: format!("chunk_{}", samples.len()),
            }]),
        })
    }
}

/// Mock model that fails on the Nth call.
pub(crate) struct FailOnNthModel {
    call_count: usize,
    fail_on: usize,
}

impl FailOnNthModel {
    pub(crate) fn new(fail_on: usize) -> Self {
        Self {
            call_count: 0,
            fail_on,
        }
    }
}

impl SpeechModel for FailOnNthModel {
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            name: "mock",
            engine_id: "mock",
            sample_rate: 16000,
            languages: &[],
            supports_timestamps: false,
            supports_translation: false,
            supports_streaming: false,
        }
    }

    fn transcribe_raw(
        &mut self,
        _samples: &[f32],
        _options: &TranscribeOptions,
    ) -> Result<TranscriptionResult, TranscribeError> {
        self.call_count += 1;
        if self.call_count == self.fail_on {
            return Err(TranscribeError::Inference("mock failure".into()));
        }
        Ok(TranscriptionResult {
            text: format!("chunk_{}", self.call_count),
            segments: None,
        })
    }
}

pub(crate) fn make_speech(frame_size: usize, num_frames: usize) -> Vec<f32> {
    vec![1.0f32; frame_size * num_frames]
}

pub(crate) fn make_silence(frame_size: usize, num_frames: usize) -> Vec<f32> {
    vec![0.0f32; frame_size * num_frames]
}
