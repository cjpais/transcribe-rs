//! Silero VAD — neural voice activity detection using the Silero ONNX model.
//!
//! Requires the `vad-silero` feature flag and a `silero_vad_v4.onnx` model file.
//!
//! The model file is NOT bundled — the consumer provides the path:
//!
//! ```ignore
//! use transcribe_rs::vad::SileroVad;
//! let vad = SileroVad::new("/path/to/silero_vad_v4.onnx", 0.3)?;
//! ```

use std::path::Path;

use ndarray::{Array1, Array3, ArrayView2};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use super::Vad;
use crate::TranscribeError;

/// Number of samples per frame: 30ms at 16kHz.
const FRAME_SAMPLES: usize = 480;

/// Silero VAD using an ONNX model with LSTM state.
///
/// Classifies 30ms audio frames (480 samples at 16kHz) as speech or
/// non-speech using the Silero VAD v4 model. Maintains internal LSTM
/// hidden and cell states across frames for temporal context.
///
/// # Example
///
/// ```ignore
/// use transcribe_rs::vad::{SileroVad, Vad};
///
/// let mut vad = SileroVad::new("silero_vad_v4.onnx", 0.3)?;
/// let frame = vec![0.0f32; 480]; // 30ms of silence
/// let is_speech = vad.is_speech(&frame)?;
/// ```
pub struct SileroVad {
    session: Session,
    h: Array3<f32>,  // LSTM hidden state (2, 1, 64)
    c: Array3<f32>,  // LSTM cell state (2, 1, 64)
    sr: Array1<i64>, // sample rate tensor
    threshold: f32,
}

impl SileroVad {
    /// Create a new `SileroVad` from a Silero ONNX model file.
    ///
    /// - `model_path`: path to `silero_vad_v4.onnx`
    /// - `threshold`: speech probability threshold (recommended: 0.3)
    pub fn new(model_path: impl AsRef<Path>, threshold: f32) -> Result<Self, TranscribeError> {
        let path = model_path.as_ref();
        let session = Session::builder()
            .map_err(|e| TranscribeError::Config(format!("ort session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TranscribeError::Config(format!("ort optimization level: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| TranscribeError::Config(format!("ort intra threads: {e}")))?
            .with_inter_threads(1)
            .map_err(|e| TranscribeError::Config(format!("ort inter threads: {e}")))?
            .commit_from_file(path)
            .map_err(|e| {
                if !path.exists() {
                    TranscribeError::ModelNotFound(path.to_path_buf())
                } else {
                    TranscribeError::Inference(format!("failed to load VAD model: {e}"))
                }
            })?;

        Ok(Self {
            session,
            h: Array3::zeros((2, 1, 64)),
            c: Array3::zeros((2, 1, 64)),
            sr: Array1::from_vec(vec![16000i64]),
            threshold,
        })
    }

    /// Current speech probability threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Update the speech probability threshold (0.0–1.0).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Run inference and return the raw speech probability (0.0–1.0).
    ///
    /// Unlike [`is_speech()`](Vad::is_speech), this returns the model's
    /// confidence rather than a thresholded boolean. Useful for custom
    /// smoothing, visualization, or sensitivity tuning.
    pub fn speech_probability(&mut self, frame: &[f32]) -> Result<f32, TranscribeError> {
        if frame.len() != FRAME_SAMPLES {
            return Err(TranscribeError::Audio(format!(
                "expected {FRAME_SAMPLES} samples, got {}",
                frame.len()
            )));
        }

        let input = ArrayView2::from_shape((1, FRAME_SAMPLES), frame)
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        let t_input = TensorRef::from_array_view(input.into_dyn())
            .map_err(|e| TranscribeError::Inference(format!("tensor input: {e}")))?;
        let t_sr = TensorRef::from_array_view(self.sr.view().into_dyn())
            .map_err(|e| TranscribeError::Inference(format!("tensor sr: {e}")))?;
        let t_h = TensorRef::from_array_view(self.h.view().into_dyn())
            .map_err(|e| TranscribeError::Inference(format!("tensor h: {e}")))?;
        let t_c = TensorRef::from_array_view(self.c.view().into_dyn())
            .map_err(|e| TranscribeError::Inference(format!("tensor c: {e}")))?;

        let outputs = self
            .session
            .run(inputs![
                "input" => t_input,
                "sr" => t_sr,
                "h" => t_h,
                "c" => t_c,
            ])
            .map_err(|e| TranscribeError::Inference(e.to_string()))?;

        // Extract updated LSTM states
        let hn = outputs
            .get("hn")
            .ok_or_else(|| TranscribeError::Inference("missing output: hn".to_string()))?
            .try_extract_array::<f32>()
            .map_err(|e| TranscribeError::Inference(format!("extract hn: {e}")))?;
        self.h = hn
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .map_err(|e| TranscribeError::Inference(format!("reshape hn: {e}")))?;

        let cn = outputs
            .get("cn")
            .ok_or_else(|| TranscribeError::Inference("missing output: cn".to_string()))?
            .try_extract_array::<f32>()
            .map_err(|e| TranscribeError::Inference(format!("extract cn: {e}")))?;
        self.c = cn
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .map_err(|e| TranscribeError::Inference(format!("reshape cn: {e}")))?;

        let output = outputs
            .get("output")
            .ok_or_else(|| TranscribeError::Inference("missing output: output".to_string()))?
            .try_extract_array::<f32>()
            .map_err(|e| TranscribeError::Inference(format!("extract output: {e}")))?;

        Ok(output[[0, 0]])
    }
}

impl Vad for SileroVad {
    fn frame_size(&self) -> usize {
        FRAME_SAMPLES
    }

    fn is_speech(&mut self, frame: &[f32]) -> Result<bool, TranscribeError> {
        Ok(self.speech_probability(frame)? > self.threshold)
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}
