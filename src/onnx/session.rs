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
pub fn create_session_with_threads(
    path: &Path,
    num_threads: usize,
) -> Result<Session, ort::Error> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;

    if num_threads > 0 {
        builder = builder.with_intra_threads(num_threads)?;
    }

    let session = builder
        .with_execution_providers([CPUExecutionProvider::default().build()])?
        .commit_from_file(path)?;

    Ok(session)
}

/// Resolve a model file path, optionally trying quantized variant first.
///
/// If `quantized` is true, looks for `{name}.int8.onnx` first, falling back to `{name}.onnx`.
pub fn resolve_model_path(dir: &Path, name: &str, quantized: bool) -> std::path::PathBuf {
    if quantized {
        let int8_path = dir.join(format!("{}.int8.onnx", name));
        if int8_path.exists() {
            log::info!("Loading quantized model: {}", int8_path.display());
            return int8_path;
        }
        log::warn!("Quantized model not found, falling back to {}.onnx", name);
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
) -> Result<Option<i32>, Box<dyn std::error::Error>> {
    match read_metadata_str(session, key)? {
        Some(v) => Ok(Some(v.parse::<i32>()?)),
        None => Ok(default),
    }
}

/// Read a comma-separated float vector from metadata.
pub fn read_metadata_float_vec(
    session: &Session,
    key: &str,
) -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
    match read_metadata_str(session, key)? {
        Some(v) => {
            let floats: Result<Vec<f32>, _> =
                v.split(',').map(|s| s.trim().parse::<f32>()).collect();
            Ok(Some(floats?))
        }
        None => Ok(None),
    }
}
