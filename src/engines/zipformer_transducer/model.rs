use std::path::Path;

use ndarray::Array2;
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use super::super::zipformer_common::*;

#[derive(thiserror::Error, Debug)]
pub enum ZipformerTransducerError {
    #[error("ORT error: {0}")]
    Ort(#[from] ort::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Tokens file not found: {0}")]
    TokensNotFound(String),
    #[error("Output not found: {0}")]
    OutputNotFound(String),
}

pub struct ZipformerTransducerModel {
    encoder_session: Session,
    decoder_session: Session,
    joiner_session: Session,
    symbol_table: SymbolTable,
    blank_id: i32,
    context_size: usize,
    // Encoder I/O names
    enc_x_name: String,
    enc_x_lens_name: String,
    enc_out_name: String,
    enc_out_lens_name: String,
    // Decoder I/O names
    dec_y_name: String,
    dec_out_name: String,
    // Joiner I/O names
    join_enc_name: String,
    join_dec_name: String,
    join_logit_name: String,
}

impl Drop for ZipformerTransducerModel {
    fn drop(&mut self) {
        log::debug!("Dropping ZipformerTransducerModel");
    }
}

impl ZipformerTransducerModel {
    pub fn new(model_dir: &Path, quantized: bool) -> Result<Self, ZipformerTransducerError> {
        let suffix = if quantized { "int8" } else { "fp32" };

        let encoder_path = Self::find_model_file(model_dir, "encoder", suffix)?;
        let decoder_path = Self::find_model_file(model_dir, "decoder", suffix)?;
        let joiner_path = Self::find_model_file(model_dir, "joiner", suffix)?;

        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(ZipformerTransducerError::TokensNotFound(
                tokens_path.display().to_string(),
            ));
        }

        log::info!("Loading Zipformer Transducer encoder from {:?}", encoder_path);
        let encoder_session = Self::init_session(&encoder_path)?;
        log::info!("Loading Zipformer Transducer decoder from {:?}", decoder_path);
        let decoder_session = Self::init_session(&decoder_path)?;
        log::info!("Loading Zipformer Transducer joiner from {:?}", joiner_path);
        let joiner_session = Self::init_session(&joiner_path)?;

        let symbol_table = SymbolTable::load_autodetect(&tokens_path)?;

        // Detect streaming models: they have cached_* state inputs in the encoder
        let enc_inputs: Vec<String> = encoder_session.inputs.iter().map(|i| i.name.clone()).collect();
        let has_cached_inputs = enc_inputs.iter().any(|n| n.starts_with("cached_"));
        if has_cached_inputs {
            return Err(ZipformerTransducerError::ModelNotFound(
                "This is a streaming model (encoder has cached_* state inputs). \
                 Streaming models require fixed-size chunk input and state management. \
                 Please use an offline (non-streaming) model instead."
                    .to_string(),
            ));
        }

        // Detect encoder I/O names
        let enc_outputs: Vec<String> = encoder_session.outputs.iter().map(|o| o.name.clone()).collect();

        let enc_x_name = Self::find_name(&enc_inputs, &["x"])
            .unwrap_or_else(|| enc_inputs[0].clone());
        let enc_x_lens_name = Self::find_name(&enc_inputs, &["x_lens", "x_length"])
            .unwrap_or_else(|| enc_inputs.get(1).cloned().unwrap_or_else(|| enc_inputs[0].clone()));
        let enc_out_name = Self::find_name(&enc_outputs, &["encoder_out"])
            .unwrap_or_else(|| enc_outputs[0].clone());
        let enc_out_lens_name = Self::find_name(&enc_outputs, &["encoder_out_lens", "encoder_out_length"])
            .unwrap_or_else(|| enc_outputs.get(1).cloned().unwrap_or_else(|| enc_outputs[0].clone()));

        // Detect decoder I/O names
        let dec_inputs: Vec<String> = decoder_session.inputs.iter().map(|i| i.name.clone()).collect();
        let dec_outputs: Vec<String> = decoder_session.outputs.iter().map(|o| o.name.clone()).collect();

        let dec_y_name = Self::find_name(&dec_inputs, &["y"])
            .unwrap_or_else(|| dec_inputs[0].clone());
        let dec_out_name = Self::find_name(&dec_outputs, &["decoder_out"])
            .unwrap_or_else(|| dec_outputs[0].clone());

        // Context size is 2 for all sherpa-onnx zipformer transducer models
        let context_size = 2;
        log::info!("Using context_size={}", context_size);

        // Detect joiner I/O names
        let join_inputs: Vec<String> = joiner_session.inputs.iter().map(|i| i.name.clone()).collect();
        let join_outputs: Vec<String> = joiner_session.outputs.iter().map(|o| o.name.clone()).collect();

        let join_enc_name = Self::find_name(&join_inputs, &["encoder_out"])
            .unwrap_or_else(|| join_inputs[0].clone());
        let join_dec_name = Self::find_name(&join_inputs, &["decoder_out"])
            .unwrap_or_else(|| join_inputs.get(1).cloned().unwrap_or_else(|| join_inputs[0].clone()));
        let join_logit_name = Self::find_name(&join_outputs, &["logit"])
            .unwrap_or_else(|| join_outputs[0].clone());

        log::info!(
            "Encoder I/O: x='{}', x_lens='{}' -> out='{}', out_lens='{}'",
            enc_x_name, enc_x_lens_name, enc_out_name, enc_out_lens_name
        );
        log::info!(
            "Decoder I/O: y='{}' -> out='{}'",
            dec_y_name, dec_out_name
        );
        log::info!(
            "Joiner I/O: enc='{}', dec='{}' -> logit='{}'",
            join_enc_name, join_dec_name, join_logit_name
        );

        let blank_id = 0;

        Ok(Self {
            encoder_session,
            decoder_session,
            joiner_session,
            symbol_table,
            blank_id,
            context_size,
            enc_x_name,
            enc_x_lens_name,
            enc_out_name,
            enc_out_lens_name,
            dec_y_name,
            dec_out_name,
            join_enc_name,
            join_dec_name,
            join_logit_name,
        })
    }

    /// Find an ONNX model file by component name, trying various naming conventions.
    /// e.g. encoder-epoch-34-avg-19.int8.onnx, encoder.int8.onnx, encoder.onnx
    fn find_model_file(
        model_dir: &Path,
        component: &str,
        suffix: &str,
    ) -> Result<std::path::PathBuf, ZipformerTransducerError> {
        // Try exact names first
        let exact_suffixed = model_dir.join(format!("{component}.{suffix}.onnx"));
        if exact_suffixed.exists() {
            return Ok(exact_suffixed);
        }
        let exact = model_dir.join(format!("{component}.onnx"));
        if exact.exists() {
            return Ok(exact);
        }

        // Glob for pattern: component*.suffix.onnx then component*.onnx
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            let files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();

            // Prefer suffixed (e.g. int8) variant
            if let Some(f) = files.iter().find(|f| {
                f.starts_with(component)
                    && f.ends_with(&format!(".{suffix}.onnx"))
            }) {
                return Ok(model_dir.join(f));
            }

            // Fall back to any matching component ONNX
            if let Some(f) = files.iter().find(|f| {
                f.starts_with(component) && f.ends_with(".onnx")
            }) {
                return Ok(model_dir.join(f));
            }
        }

        Err(ZipformerTransducerError::ModelNotFound(format!(
            "No {component}*.onnx found in {}",
            model_dir.display()
        )))
    }

    fn find_name(names: &[String], candidates: &[&str]) -> Option<String> {
        for &candidate in candidates {
            if let Some(n) = names.iter().find(|n| n.as_str() == candidate) {
                return Some(n.clone());
            }
        }
        None
    }

    fn init_session(path: &Path) -> Result<Session, ZipformerTransducerError> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(path)?;

        for input in &session.inputs {
            log::info!(
                "  input: name={}, type={:?}",
                input.name,
                input.input_type
            );
        }
        for output in &session.outputs {
            log::info!(
                "  output: name={}, type={:?}",
                output.name,
                output.output_type
            );
        }
        Ok(session)
    }

    fn compute_fbank(&self, samples: &[f32]) -> Array2<f32> {
        compute_fbank_kaldi(samples, &FbankConfig::default())
    }

    /// Run the encoder: features [N,T,80] + lens [N] -> encoder_out [N,T',D] + encoder_out_lens [N]
    fn run_encoder(
        &mut self,
        features: &Array2<f32>,
    ) -> Result<(ndarray::Array3<f32>, i64), ZipformerTransducerError> {
        let num_frames = features.nrows();
        let num_features = features.ncols();

        let feat_3d = features
            .to_owned()
            .into_shape_with_order((1, num_frames, num_features))?;
        let feat_dyn = feat_3d.into_dyn();
        let lens = ndarray::arr1(&[num_frames as i64]).into_dyn();

        let inputs = inputs![
            self.enc_x_name.as_str() => TensorRef::from_array_view(feat_dyn.view())?,
            self.enc_x_lens_name.as_str() => TensorRef::from_array_view(lens.view())?,
        ];

        let outputs = self.encoder_session.run(inputs)?;

        let encoder_out = outputs
            .get(self.enc_out_name.as_str())
            .ok_or_else(|| ZipformerTransducerError::OutputNotFound(self.enc_out_name.clone()))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()?;

        let encoder_out_lens = outputs
            .get(self.enc_out_lens_name.as_str())
            .and_then(|v| v.try_extract_array::<i64>().ok())
            .and_then(|arr| arr.as_slice().and_then(|s| s.first().copied()))
            .unwrap_or(encoder_out.shape()[1] as i64);

        log::debug!(
            "Encoder output: shape={:?}, lens={}",
            encoder_out.shape(),
            encoder_out_lens
        );

        Ok((encoder_out, encoder_out_lens))
    }

    /// Run the decoder: y [N, context_size] (i64) -> decoder_out [N, D]
    fn run_decoder(
        &mut self,
        context: &[i64],
    ) -> Result<ndarray::Array2<f32>, ZipformerTransducerError> {
        let y = ndarray::Array2::from_shape_vec(
            (1, self.context_size),
            context.to_vec(),
        )?;
        let y_dyn = y.into_dyn();

        let inputs = inputs![
            self.dec_y_name.as_str() => TensorRef::from_array_view(y_dyn.view())?,
        ];

        let outputs = self.decoder_session.run(inputs)?;

        let decoder_out = outputs
            .get(self.dec_out_name.as_str())
            .ok_or_else(|| ZipformerTransducerError::OutputNotFound(self.dec_out_name.clone()))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok(decoder_out)
    }

    /// Run the joiner: encoder_out [N, D] + decoder_out [N, D] -> logit [N, vocab_size]
    fn run_joiner(
        &mut self,
        encoder_out_frame: &ndarray::ArrayView1<f32>,
        decoder_out: &ndarray::Array2<f32>,
    ) -> Result<ndarray::Array2<f32>, ZipformerTransducerError> {
        // encoder_out_frame is [D], reshape to [1, D]
        let enc = encoder_out_frame
            .to_owned()
            .into_shape_with_order((1, encoder_out_frame.len()))?
            .into_dyn();

        let dec_dyn = decoder_out.clone().into_dyn();

        let inputs = inputs![
            self.join_enc_name.as_str() => TensorRef::from_array_view(enc.view())?,
            self.join_dec_name.as_str() => TensorRef::from_array_view(dec_dyn.view())?,
        ];

        let outputs = self.joiner_session.run(inputs)?;

        let logit = outputs
            .get(self.join_logit_name.as_str())
            .ok_or_else(|| ZipformerTransducerError::OutputNotFound(self.join_logit_name.clone()))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix2>()?;

        Ok(logit)
    }

    /// Greedy search decoding for transducer models.
    fn greedy_search(
        &mut self,
        features: &Array2<f32>,
    ) -> Result<Vec<i32>, ZipformerTransducerError> {
        let (encoder_out, encoder_out_lens) = self.run_encoder(features)?;
        let t_max = (encoder_out_lens as usize).min(encoder_out.shape()[1]);

        // Initialize decoder context with blank_id
        let mut context = vec![self.blank_id as i64; self.context_size];
        let mut decoder_out = self.run_decoder(&context)?;

        let mut tokens = Vec::new();

        for t in 0..t_max {
            let enc_frame = encoder_out.slice(ndarray::s![0, t, ..]);
            let logit = self.run_joiner(&enc_frame, &decoder_out)?;

            // argmax over vocab dimension
            let logit_row = logit.row(0);
            let mut max_id = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in logit_row.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_id = i;
                }
            }

            if max_id as i32 != self.blank_id {
                tokens.push(max_id as i32);
                // Slide context window and re-run decoder
                context.rotate_left(1);
                *context.last_mut().unwrap() = max_id as i64;
                decoder_out = self.run_decoder(&context)?;
            }
        }

        Ok(tokens)
    }

    pub fn transcribe(
        &mut self,
        samples: &[f32],
    ) -> Result<ZipformerTransducerResult, ZipformerTransducerError> {
        let features = self.compute_fbank(samples);
        if features.nrows() == 0 {
            return Ok(ZipformerTransducerResult {
                text: String::new(),
                token_ids: Vec::new(),
            });
        }

        log::debug!(
            "Computed {} frames, {} dims",
            features.nrows(),
            features.ncols()
        );

        let token_ids = self.greedy_search(&features)?;
        let text = self.symbol_table.decode(&token_ids);
        log::debug!("Decoded {} tokens: {:?} -> {}", token_ids.len(), token_ids, text);

        Ok(ZipformerTransducerResult { text, token_ids })
    }
}

pub struct ZipformerTransducerResult {
    pub text: String,
    pub token_ids: Vec<i32>,
}
