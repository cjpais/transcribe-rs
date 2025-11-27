use anyhow::Result;
use ndarray::{Array, Array3, ArrayBase, Dim, OwnedRepr};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::path::Path;

pub struct SileroVad {
    session: Session,
    h: Array3<f32>,
    c: Array3<f32>,
    sr: ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>,
}

impl SileroVad {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session,
            h: Array::zeros((2, 1, 64)),
            c: Array::zeros((2, 1, 64)),
            sr: Array::from_elem((1,), 16000.0),
        })
    }

    pub fn push_frame(&mut self, frame: &[f32]) -> Result<VadResult> {
        // frame size should be 480 for 30ms at 16kHz
        // input shape: [1, 480]
        let input_array = Array::from_shape_vec((1, frame.len()), frame.to_vec())?;

        let input = Value::from_array(input_array)?;
        let sr = Value::from_array(self.sr.clone())?;
        let h = Value::from_array(self.h.clone())?;
        let c = Value::from_array(self.c.clone())?;

        // Inputs
        let inputs = ort::inputs![
            "input" => input,
            "sr" => sr,
            "h" => h,
            "c" => c,
        ];

        let outputs = self.session.run(inputs)?;

        // Outputs: output, hn, cn
        let (_, output_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let probability = output_data[0];

        // Update states
        let (hn_shape, hn_data) = outputs["hn"].try_extract_tensor::<f32>()?;
        let (cn_shape, cn_data) = outputs["cn"].try_extract_tensor::<f32>()?;

        let hn_shape_usize: Vec<usize> = hn_shape.iter().map(|&x| x as usize).collect();
        let cn_shape_usize: Vec<usize> = cn_shape.iter().map(|&x| x as usize).collect();

        self.h = Array::from_shape_vec(hn_shape_usize, hn_data.to_vec())?.into_dimensionality()?;
        self.c = Array::from_shape_vec(cn_shape_usize, cn_data.to_vec())?.into_dimensionality()?;

        Ok(VadResult { probability })
    }

    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}

pub struct VadResult {
    pub probability: f32,
}

impl VadResult {
    pub fn is_speech(&self) -> bool {
        self.probability > 0.5
    }
}
