use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::path::Path;

/// Create an ONNX session with standard settings.
pub fn create_session(path: &Path) -> Result<Session, ort::Error> {
    let providers = vec![CPUExecutionProvider::default().build()];

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers(providers)?
        .with_parallel_execution(true)?
        .commit_from_file(path)?;

    for input in &session.inputs {
        log::info!(
            "Model input: name={}, type={:?}",
            input.name,
            input.input_type
        );
    }
    for output in &session.outputs {
        log::info!(
            "Model output: name={}, type={:?}",
            output.name,
            output.output_type
        );
    }

    Ok(session)
}

/// Create an ONNX session with configurable thread count.
///
/// `num_threads` = 0 means use the ORT default (all available cores).
/// For sessions running many sequential steps (e.g. autoregressive decoder),
/// a lower thread count reduces synchronization overhead.
pub fn create_session_with_threads(
    path: &Path,
    num_threads: usize,
) -> Result<Session, ort::Error> {
    // Sequential execution: for the autoregressive decoder, nodes are largely a linear chain
    // (attention → FFN × 28 layers). Parallel mode adds scheduling overhead without benefit.
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_parallel_execution(false)?;

    if num_threads > 0 {
        builder = builder.with_intra_threads(num_threads)?;
    }

    let session = builder
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(path)?;

    Ok(session)
}

/// Resolve a model file path for the requested quantization level.
///
/// Looks for `{name}.{suffix}.onnx` based on the quantization variant,
/// falling back to `{name}.onnx` (FP32) if the requested file doesn't exist.
pub fn resolve_model_path(dir: &Path, name: &str, quantization: &super::Quantization) -> std::path::PathBuf {
    let suffix = match quantization {
        super::Quantization::FP32 => None,
        super::Quantization::FP16 => Some("fp16"),
        super::Quantization::Int8 => Some("int8"),
    };

    if let Some(suffix) = suffix {
        let path = dir.join(format!("{}.{}.onnx", name, suffix));
        if path.exists() {
            log::info!("Loading {} model: {}", suffix, path.display());
            return path;
        }
        log::warn!("{} model not found at {}, falling back to {}.onnx", suffix, path.display(), name);
    }

    dir.join(format!("{}.onnx", name))
}

/// Read a custom metadata string from an ONNX session.
pub fn read_metadata_str(
    session: &Session,
    key: &str,
) -> Result<Option<String>, ort::Error> {
    let meta = session.metadata()?;
    meta.custom(key)
}

/// Read a custom metadata i32 value, with optional default.
pub fn read_metadata_i32(
    session: &Session,
    key: &str,
    default: Option<i32>,
) -> Result<Option<i32>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key)
        .map_err(|e| crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e)))?;
    match str_val {
        Some(v) => Ok(Some(v.parse::<i32>().map_err(|e| {
            crate::TranscribeError::Config(format!("failed to parse '{}': {}", key, e))
        })?)),
        None => Ok(default),
    }
}

/// Read a comma-separated float vector from metadata.
pub fn read_metadata_float_vec(
    session: &Session,
    key: &str,
) -> Result<Option<Vec<f32>>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key)
        .map_err(|e| crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e)))?;
    match str_val {
        Some(v) => {
            let floats: Result<Vec<f32>, _> =
                v.split(',').map(|s| s.trim().parse::<f32>()).collect();
            Ok(Some(floats.map_err(|e| {
                crate::TranscribeError::Config(format!("failed to parse floats in '{}': {}", key, e))
            })?))
        }
        None => Ok(None),
    }
}
