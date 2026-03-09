use ort::execution_providers::CPUExecutionProvider;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use std::path::Path;

#[cfg(target_os = "windows")]
use ort::execution_providers::directml::DirectMLExecutionProvider;

fn base_session_builder() -> Result<SessionBuilder, ort::Error> {
    Session::builder()?.with_optimization_level(GraphOptimizationLevel::Level3)
}

fn configured_session_builder(
    parallel_execution: Option<bool>,
    intra_threads: Option<usize>,
) -> Result<SessionBuilder, ort::Error> {
    let mut builder = base_session_builder()?;

    if let Some(num_threads) = intra_threads.filter(|num_threads| *num_threads > 0) {
        builder = builder.with_intra_threads(num_threads)?;
    }

    #[cfg(target_os = "windows")]
    {
        let _ = parallel_execution;

        let builder = builder
            .with_parallel_execution(false)?
            .with_memory_pattern(false)?;

        return match builder.with_execution_providers([
            DirectMLExecutionProvider::default()
                .with_device_id(0)
                .build()
                .error_on_failure(),
            CPUExecutionProvider::default().build(),
        ]) {
            Ok(builder) => {
                log::info!(
                    "ONNX Runtime session registered DirectMLExecutionProvider on Windows (device 0) with CPU fallback enabled"
                );
                Ok(builder)
            }
            Err(e) => {
                log::warn!(
                    "DirectMLExecutionProvider registration failed on Windows: {}. Falling back to CPUExecutionProvider only",
                    e
                );

                let mut cpu_builder = base_session_builder()?;
                if let Some(num_threads) = intra_threads.filter(|num_threads| *num_threads > 0) {
                    cpu_builder = cpu_builder.with_intra_threads(num_threads)?;
                }

                cpu_builder
                    .with_parallel_execution(false)?
                    .with_memory_pattern(false)?
                    .with_execution_providers([CPUExecutionProvider::default().build()])
            }
        };
    }

    #[cfg(not(target_os = "windows"))]
    {
        if let Some(parallel_execution) = parallel_execution {
            builder = builder.with_parallel_execution(parallel_execution)?;
        }

        builder.with_execution_providers([CPUExecutionProvider::default().build()])
    }
}

/// Create an ONNX session with standard settings.
pub fn create_session(path: &Path) -> Result<Session, ort::Error> {
    let session = configured_session_builder(Some(true), None)?.commit_from_file(path)?;

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
pub fn create_session_with_threads(path: &Path, num_threads: usize) -> Result<Session, ort::Error> {
    configured_session_builder(None, Some(num_threads))?.commit_from_file(path)
}

/// Resolve a model file path for the requested quantization level.
///
/// Looks for `{name}.{suffix}.onnx` based on the quantization variant,
/// falling back to `{name}.onnx` (FP32) if the requested file doesn't exist.
pub fn resolve_model_path(
    dir: &Path,
    name: &str,
    quantization: &super::Quantization,
) -> std::path::PathBuf {
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
        log::warn!(
            "{} model not found at {}, falling back to {}.onnx",
            suffix,
            path.display(),
            name
        );
    }

    dir.join(format!("{}.onnx", name))
}

/// Read a custom metadata string from an ONNX session.
pub fn read_metadata_str(session: &Session, key: &str) -> Result<Option<String>, ort::Error> {
    let meta = session.metadata()?;
    meta.custom(key)
}

/// Read a custom metadata i32 value, with optional default.
pub fn read_metadata_i32(
    session: &Session,
    key: &str,
    default: Option<i32>,
) -> Result<Option<i32>, crate::TranscribeError> {
    let str_val = read_metadata_str(session, key).map_err(|e| {
        crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e))
    })?;
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
    let str_val = read_metadata_str(session, key).map_err(|e| {
        crate::TranscribeError::Config(format!("failed to read metadata '{}': {}", key, e))
    })?;
    match str_val {
        Some(v) => {
            let floats: Result<Vec<f32>, _> =
                v.split(',').map(|s| s.trim().parse::<f32>()).collect();
            Ok(Some(floats.map_err(|e| {
                crate::TranscribeError::Config(format!(
                    "failed to parse floats in '{}': {}",
                    key, e
                ))
            })?))
        }
        None => Ok(None),
    }
}
