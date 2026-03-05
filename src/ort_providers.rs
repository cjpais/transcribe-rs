use ort::execution_providers::{CPUExecutionProvider, ExecutionProviderDispatch};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "directml")]
use ort::execution_providers::DirectMLExecutionProvider;
#[cfg(feature = "coreml")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(feature = "webgpu")]
use ort::execution_providers::WebGPUExecutionProvider;

/// Runtime selection of which GPU execution provider to use.
///
/// Set once at startup (or when the user changes the setting) via
/// [`set_gpu_provider`].  Read by [`execution_providers`] each time an
/// ORT session is created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum GpuProvider {
    /// Try all compiled-in GPU EPs, then CPU (default).
    Auto = 0,
    /// CPU only — no GPU acceleration.
    #[serde(rename = "cpu")]
    CpuOnly = 1,
    /// DirectML (Windows).
    #[serde(rename = "directml")]
    DirectMl = 2,
    /// CUDA (Linux).
    Cuda = 3,
    /// CoreML (macOS).
    #[serde(rename = "coreml")]
    CoreMl = 4,
    /// WebGPU (cross-platform).
    #[serde(rename = "webgpu")]
    WebGpu = 5,
}

impl GpuProvider {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => GpuProvider::CpuOnly,
            2 => GpuProvider::DirectMl,
            3 => GpuProvider::Cuda,
            4 => GpuProvider::CoreMl,
            5 => GpuProvider::WebGpu,
            _ => GpuProvider::Auto,
        }
    }
}

impl fmt::Display for GpuProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuProvider::Auto => write!(f, "auto"),
            GpuProvider::CpuOnly => write!(f, "cpu"),
            GpuProvider::DirectMl => write!(f, "directml"),
            GpuProvider::Cuda => write!(f, "cuda"),
            GpuProvider::CoreMl => write!(f, "coreml"),
            GpuProvider::WebGpu => write!(f, "webgpu"),
        }
    }
}

impl FromStr for GpuProvider {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(GpuProvider::Auto),
            "cpu" => Ok(GpuProvider::CpuOnly),
            "directml" => Ok(GpuProvider::DirectMl),
            "cuda" => Ok(GpuProvider::Cuda),
            "coreml" => Ok(GpuProvider::CoreMl),
            "webgpu" => Ok(GpuProvider::WebGpu),
            _ => Err(format!("unknown GPU provider: {}", s)),
        }
    }
}

static GPU_PROVIDER: AtomicU8 = AtomicU8::new(GpuProvider::Auto as u8);

/// Set the runtime GPU provider selection.
///
/// Must be called before any ORT sessions are created.  Changing the
/// provider while models are loaded has no effect on existing sessions.
pub fn set_gpu_provider(provider: GpuProvider) {
    GPU_PROVIDER.store(provider as u8, Ordering::Release);
}

/// Read the current GPU provider selection.
pub fn get_gpu_provider() -> GpuProvider {
    GpuProvider::from_u8(GPU_PROVIDER.load(Ordering::Acquire))
}

/// Return which GPU providers are available in this build.
///
/// Always includes `Auto` and `CpuOnly`; GPU-specific variants are
/// present only when the corresponding Cargo feature is compiled in.
pub fn available_providers() -> Vec<GpuProvider> {
    #[allow(unused_mut)]
    let mut v = vec![GpuProvider::Auto, GpuProvider::CpuOnly];
    #[cfg(feature = "directml")]
    v.push(GpuProvider::DirectMl);
    #[cfg(feature = "cuda")]
    v.push(GpuProvider::Cuda);
    #[cfg(feature = "coreml")]
    v.push(GpuProvider::CoreMl);
    #[cfg(feature = "webgpu")]
    v.push(GpuProvider::WebGpu);
    v
}

/// Return an ordered list of execution providers to try.
///
/// Respects the runtime [`GpuProvider`] selection.  When a specific
/// provider is selected, only that provider + CPU fallback are returned.
/// In `Auto` mode, all compiled-in GPU EPs are tried before CPU.
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let selection = get_gpu_provider();

    match selection {
        GpuProvider::CpuOnly => {
            return vec![CPUExecutionProvider::default().build()];
        }
        #[cfg(feature = "directml")]
        GpuProvider::DirectMl => {
            return vec![
                DirectMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        #[cfg(feature = "cuda")]
        GpuProvider::Cuda => {
            return vec![
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        #[cfg(feature = "coreml")]
        GpuProvider::CoreMl => {
            return vec![
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        #[cfg(feature = "webgpu")]
        GpuProvider::WebGpu => {
            return vec![
                WebGPUExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ];
        }
        GpuProvider::Auto => {}
        #[allow(unreachable_patterns)]
        _ => {
            // The selected provider wasn't compiled into this build.
            // Fall back to CPU-only rather than silently trying other GPU EPs.
            log::warn!(
                "Selected GPU provider {:?} is not available in this build, falling back to CPU",
                selection
            );
            return vec![CPUExecutionProvider::default().build()];
        }
    }

    // Auto: all compiled-in GPU EPs + CPU
    let mut providers = Vec::new();

    #[cfg(feature = "cuda")]
    providers.push(CUDAExecutionProvider::default().build());

    #[cfg(feature = "directml")]
    providers.push(DirectMLExecutionProvider::default().build());

    #[cfg(feature = "coreml")]
    providers.push(CoreMLExecutionProvider::default().build());

    #[cfg(feature = "webgpu")]
    providers.push(WebGPUExecutionProvider::default().build());

    providers.push(CPUExecutionProvider::default().build());
    providers
}

/// Return CPU-only execution providers.
///
/// Use this for models that contain operators with known GPU EP bugs
/// (e.g. Conv2d on DirectML producing NaN).  This bypasses the global
/// [`GpuProvider`] setting entirely.
pub fn cpu_execution_providers() -> Vec<ExecutionProviderDispatch> {
    vec![CPUExecutionProvider::default().build()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Restore Auto after each test since the AtomicU8 is process-global.
    struct RestoreAuto;
    impl Drop for RestoreAuto {
        fn drop(&mut self) {
            set_gpu_provider(GpuProvider::Auto);
        }
    }

    #[test]
    fn from_u8_round_trip() {
        let variants = [
            GpuProvider::Auto,
            GpuProvider::CpuOnly,
            GpuProvider::DirectMl,
            GpuProvider::Cuda,
            GpuProvider::CoreMl,
            GpuProvider::WebGpu,
        ];
        for v in variants {
            assert_eq!(GpuProvider::from_u8(v as u8), v);
        }
    }

    #[test]
    fn from_u8_unknown_defaults_to_auto() {
        assert_eq!(GpuProvider::from_u8(6), GpuProvider::Auto);
        assert_eq!(GpuProvider::from_u8(255), GpuProvider::Auto);
    }

    #[test]
    fn set_get_round_trip() {
        let _guard = RestoreAuto;
        let variants = [
            GpuProvider::Auto,
            GpuProvider::CpuOnly,
            GpuProvider::DirectMl,
            GpuProvider::Cuda,
            GpuProvider::CoreMl,
            GpuProvider::WebGpu,
        ];
        for v in variants {
            set_gpu_provider(v);
            assert_eq!(get_gpu_provider(), v);
        }
    }

    #[test]
    fn available_providers_baseline() {
        let providers = available_providers();
        assert!(providers.contains(&GpuProvider::Auto));
        assert!(providers.contains(&GpuProvider::CpuOnly));
        assert!(providers.len() >= 2);
    }

    #[test]
    fn available_providers_no_duplicates() {
        let providers = available_providers();
        let mut seen = Vec::new();
        for p in &providers {
            assert!(!seen.contains(p), "duplicate provider: {:?}", p);
            seen.push(*p);
        }
    }

    #[test]
    fn default_provider_is_auto() {
        let _guard = RestoreAuto;
        // Reset to a known state first
        set_gpu_provider(GpuProvider::Auto);
        assert_eq!(get_gpu_provider(), GpuProvider::Auto);
    }

    #[test]
    fn display_round_trip() {
        let cases = [
            (GpuProvider::Auto, "auto"),
            (GpuProvider::CpuOnly, "cpu"),
            (GpuProvider::DirectMl, "directml"),
            (GpuProvider::Cuda, "cuda"),
            (GpuProvider::CoreMl, "coreml"),
            (GpuProvider::WebGpu, "webgpu"),
        ];
        for (variant, expected) in cases {
            assert_eq!(variant.to_string(), expected);
            assert_eq!(expected.parse::<GpuProvider>().unwrap(), variant);
        }
    }

    #[test]
    fn from_str_unknown_errors() {
        assert!("vulkan".parse::<GpuProvider>().is_err());
        assert!("AUTO".parse::<GpuProvider>().is_err());
        assert!("".parse::<GpuProvider>().is_err());
    }

    #[test]
    fn serde_round_trip() {
        let cases = [
            (GpuProvider::Auto, "\"auto\""),
            (GpuProvider::CpuOnly, "\"cpu\""),
            (GpuProvider::DirectMl, "\"directml\""),
            (GpuProvider::Cuda, "\"cuda\""),
            (GpuProvider::CoreMl, "\"coreml\""),
            (GpuProvider::WebGpu, "\"webgpu\""),
        ];
        for (variant, json) in cases {
            assert_eq!(serde_json::to_string(&variant).unwrap(), json);
            assert_eq!(serde_json::from_str::<GpuProvider>(json).unwrap(), variant);
        }
    }
}
