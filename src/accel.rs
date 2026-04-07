//! Per-engine accelerator preferences.
//!
//! Each engine family has its own accelerator enum containing only the options
//! meaningful for that engine. Call the appropriate setter early in your program
//! before loading models.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicI32, AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ORT accelerator
// ---------------------------------------------------------------------------

/// Preferred hardware accelerator for ORT-based engines (SenseVoice, GigaAM, Parakeet, Moonshine).
///
/// Each variant requires its corresponding `ort-*` feature flag to be enabled at compile time.
/// If the selected accelerator's feature is not enabled, session creation falls back to CPU
/// with a log warning.
///
/// **Binary size note:** Enabling `ort-cuda` pulls in the CUDA execution provider libraries
/// (~800 MB+), significantly increasing the final binary and its runtime dependencies
/// (CUDA toolkit / cuDNN). Prefer `CpuOnly` or lighter providers unless GPU acceleration
/// is specifically required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
#[repr(u8)]
pub enum OrtAccelerator {
    /// Automatically select the best available execution provider (default).
    /// DirectML and WebGPU are excluded from auto-selection because they
    /// require sequential execution mode; set them explicitly to use.
    Auto = 0,
    /// Force CPU-only execution — no GPU providers.
    #[serde(rename = "cpu", alias = "cpu_only")]
    CpuOnly = 1,
    /// NVIDIA CUDA (requires `ort-cuda` feature; adds ~800 MB to binary size).
    Cuda = 2,
    /// NVIDIA TensorRT (requires `ort-tensorrt` feature; builds on CUDA with optimised graph compilation).
    #[serde(rename = "tensorrt", alias = "tensor_rt")]
    TensorRt = 7,
    /// Microsoft DirectML (Windows).
    #[serde(rename = "directml", alias = "direct_ml")]
    DirectMl = 3,
    /// AMD ROCm.
    Rocm = 4,
    /// Apple CoreML (macOS/iOS — Neural Engine, GPU, or CPU).
    #[serde(rename = "coreml")]
    CoreMl = 5,
    /// WebGPU via Dawn (Windows, Linux, WebAssembly).
    #[serde(rename = "webgpu")]
    WebGpu = 6,
    /// XNNPACK CPU acceleration (ARM, x86_64). Optimised for Conv/Gemm/MatMul
    /// kernels; uses its own threadpool independent of the session intra-op pool.
    #[serde(rename = "xnnpack")]
    Xnnpack = 8,
}

static ORT_ACCELERATOR: AtomicU8 = AtomicU8::new(OrtAccelerator::Auto as u8);

/// Set the global ORT accelerator preference.
///
/// Call once, early in the program, before any ORT models are loaded.
pub fn set_ort_accelerator(pref: OrtAccelerator) {
    ORT_ACCELERATOR.store(pref as u8, Ordering::Relaxed);
}

/// Get the current ORT accelerator preference.
pub fn get_ort_accelerator() -> OrtAccelerator {
    OrtAccelerator::from_u8(ORT_ACCELERATOR.load(Ordering::Relaxed))
}

impl OrtAccelerator {
    /// Return the list of ORT accelerators that are compiled-in for the current build.
    ///
    /// Always includes `CpuOnly`. Only includes GPU accelerators whose corresponding
    /// feature flag is enabled.
    pub fn available() -> Vec<OrtAccelerator> {
        #[allow(unused_mut)]
        let mut v = vec![OrtAccelerator::CpuOnly];

        #[cfg(feature = "ort-cuda")]
        v.push(OrtAccelerator::Cuda);

        #[cfg(feature = "ort-tensorrt")]
        v.push(OrtAccelerator::TensorRt);

        #[cfg(feature = "ort-directml")]
        v.push(OrtAccelerator::DirectMl);

        #[cfg(feature = "ort-rocm")]
        v.push(OrtAccelerator::Rocm);

        #[cfg(feature = "ort-coreml")]
        v.push(OrtAccelerator::CoreMl);

        #[cfg(feature = "ort-webgpu")]
        v.push(OrtAccelerator::WebGpu);

        #[cfg(feature = "ort-xnnpack")]
        v.push(OrtAccelerator::Xnnpack);

        v
    }

    fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Auto,
            1 => Self::CpuOnly,
            2 => Self::Cuda,
            3 => Self::DirectMl,
            4 => Self::Rocm,
            5 => Self::CoreMl,
            6 => Self::WebGpu,
            7 => Self::TensorRt,
            8 => Self::Xnnpack,
            _ => Self::Auto,
        }
    }
}

impl Default for OrtAccelerator {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for OrtAccelerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Auto => "auto",
            Self::CpuOnly => "cpu",
            Self::Cuda => "cuda",
            Self::TensorRt => "tensorrt",
            Self::DirectMl => "directml",
            Self::Rocm => "rocm",
            Self::CoreMl => "coreml",
            Self::WebGpu => "webgpu",
            Self::Xnnpack => "xnnpack",
        };
        f.write_str(s)
    }
}

impl FromStr for OrtAccelerator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" | "cpu_only" | "cpuonly" => Ok(Self::CpuOnly),
            "cuda" => Ok(Self::Cuda),
            "tensorrt" | "trt" | "tensor_rt" => Ok(Self::TensorRt),
            "directml" | "dml" => Ok(Self::DirectMl),
            "rocm" => Ok(Self::Rocm),
            "coreml" | "core_ml" => Ok(Self::CoreMl),
            "webgpu" | "web_gpu" => Ok(Self::WebGpu),
            "xnnpack" => Ok(Self::Xnnpack),
            other => Err(format!("unknown ORT accelerator: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Whisper accelerator
// ---------------------------------------------------------------------------

/// Preferred hardware accelerator for the whisper.cpp engine.
///
/// The actual GPU backend (Metal, Vulkan, etc.) is selected at compile time
/// via whisper-rs feature flags. This enum only controls whether GPU is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum WhisperAccelerator {
    /// Automatically select the best available backend (default — uses GPU if available).
    Auto = 0,
    /// Force CPU-only execution.
    CpuOnly = 1,
    /// Explicitly request GPU execution.
    Gpu = 2,
}

static WHISPER_ACCELERATOR: AtomicU8 = AtomicU8::new(WhisperAccelerator::Auto as u8);

/// Set the global whisper.cpp accelerator preference.
///
/// Call once, early in the program, before any Whisper models are loaded.
pub fn set_whisper_accelerator(pref: WhisperAccelerator) {
    WHISPER_ACCELERATOR.store(pref as u8, Ordering::Relaxed);
}

/// Get the current whisper.cpp accelerator preference.
pub fn get_whisper_accelerator() -> WhisperAccelerator {
    WhisperAccelerator::from_u8(WHISPER_ACCELERATOR.load(Ordering::Relaxed))
}

impl WhisperAccelerator {
    /// Return the list of Whisper accelerators available for the current build.
    ///
    /// Always includes `CpuOnly`. Includes `Gpu` when whisper-rs was compiled
    /// with a GPU backend (Metal on macOS, Vulkan on Windows/Linux, CUDA on Windows).
    pub fn available() -> Vec<WhisperAccelerator> {
        #[allow(unused_mut)]
        let mut v = vec![WhisperAccelerator::CpuOnly];

        #[cfg(any(
            feature = "whisper-metal",
            feature = "whisper-vulkan",
            feature = "whisper-cuda"
        ))]
        v.push(WhisperAccelerator::Gpu);

        v
    }

    /// Returns `true` if GPU should be used.
    pub fn use_gpu(&self) -> bool {
        *self != Self::CpuOnly
    }

    fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Auto,
            1 => Self::CpuOnly,
            2 => Self::Gpu,
            _ => Self::Auto,
        }
    }
}

impl Default for WhisperAccelerator {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for WhisperAccelerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Auto => "auto",
            Self::CpuOnly => "cpu",
            Self::Gpu => "gpu",
        };
        f.write_str(s)
    }
}

impl FromStr for WhisperAccelerator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" | "cpu_only" | "cpuonly" => Ok(Self::CpuOnly),
            "gpu" => Ok(Self::Gpu),
            other => Err(format!("unknown Whisper accelerator: {other}")),
        }
    }
}

// ---------------------------------------------------------------------------
// Whisper GPU device selection
// ---------------------------------------------------------------------------

/// Auto-select: let the library pick the best GPU (default).
pub const GPU_DEVICE_AUTO: i32 = -1;

static WHISPER_GPU_DEVICE: AtomicI32 = AtomicI32::new(GPU_DEVICE_AUTO);

/// Set the preferred GPU device index for whisper.cpp.
///
/// - `-1` (default, [`GPU_DEVICE_AUTO`]): automatically select the best device
///   (prefers dedicated GPUs over integrated ones).
/// - `0, 1, 2, …`: use a specific device by backend index.
///
/// Call before loading a Whisper model; takes effect on next model load.
pub fn set_whisper_gpu_device(device: i32) {
    WHISPER_GPU_DEVICE.store(device, Ordering::Relaxed);
}

/// Get the current whisper GPU device preference.
///
/// Returns [`GPU_DEVICE_AUTO`] (`-1`) for automatic selection, or a
/// non-negative backend-specific device index.
pub fn get_whisper_gpu_device() -> i32 {
    WHISPER_GPU_DEVICE.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Tests that mutate global accelerator state must hold this lock.
    static ACCEL_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that serialises access to global state and restores defaults when dropped.
    struct AccelGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
    impl AccelGuard {
        fn new() -> Self {
            let g = ACCEL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            Self(g)
        }
    }
    impl Drop for AccelGuard {
        fn drop(&mut self) {
            set_ort_accelerator(OrtAccelerator::Auto);
            set_whisper_accelerator(WhisperAccelerator::Auto);
            set_whisper_gpu_device(GPU_DEVICE_AUTO);
        }
    }

    // -- ORT tests --

    #[test]
    fn ort_default_is_auto() {
        let _g = AccelGuard::new();
        set_ort_accelerator(OrtAccelerator::Auto);
        assert_eq!(get_ort_accelerator(), OrtAccelerator::Auto);
    }

    #[test]
    fn ort_set_and_get() {
        let _g = AccelGuard::new();
        set_ort_accelerator(OrtAccelerator::Cuda);
        assert_eq!(get_ort_accelerator(), OrtAccelerator::Cuda);
        set_ort_accelerator(OrtAccelerator::CpuOnly);
        assert_eq!(get_ort_accelerator(), OrtAccelerator::CpuOnly);
    }

    #[test]
    fn ort_display_roundtrip() {
        for pref in [
            OrtAccelerator::Auto,
            OrtAccelerator::CpuOnly,
            OrtAccelerator::Cuda,
            OrtAccelerator::TensorRt,
            OrtAccelerator::DirectMl,
            OrtAccelerator::Rocm,
            OrtAccelerator::CoreMl,
            OrtAccelerator::WebGpu,
            OrtAccelerator::Xnnpack,
        ] {
            let s = pref.to_string();
            let parsed: OrtAccelerator = s.parse().unwrap();
            assert_eq!(parsed, pref);
        }
    }

    #[test]
    fn ort_parse_aliases() {
        assert_eq!(
            "dml".parse::<OrtAccelerator>().unwrap(),
            OrtAccelerator::DirectMl
        );
        assert_eq!(
            "CPU".parse::<OrtAccelerator>().unwrap(),
            OrtAccelerator::CpuOnly
        );
        assert_eq!(
            "cpu_only".parse::<OrtAccelerator>().unwrap(),
            OrtAccelerator::CpuOnly
        );
        assert_eq!(
            "trt".parse::<OrtAccelerator>().unwrap(),
            OrtAccelerator::TensorRt
        );
    }

    #[test]
    fn ort_parse_unknown_errors() {
        assert!("foobar".parse::<OrtAccelerator>().is_err());
    }

    #[test]
    fn ort_serde_roundtrip() {
        for (pref, expected) in [
            (OrtAccelerator::Auto, "\"auto\""),
            (OrtAccelerator::CpuOnly, "\"cpu\""),
            (OrtAccelerator::Cuda, "\"cuda\""),
            (OrtAccelerator::TensorRt, "\"tensorrt\""),
            (OrtAccelerator::DirectMl, "\"directml\""),
            (OrtAccelerator::Rocm, "\"rocm\""),
            (OrtAccelerator::CoreMl, "\"coreml\""),
            (OrtAccelerator::WebGpu, "\"webgpu\""),
            (OrtAccelerator::Xnnpack, "\"xnnpack\""),
        ] {
            let json = serde_json::to_string(&pref).unwrap();
            assert_eq!(json, expected, "serialize {:?}", pref);
            let back: OrtAccelerator = serde_json::from_str(&json).unwrap();
            assert_eq!(back, pref, "deserialize {}", json);
        }
    }

    #[test]
    fn ort_serde_backward_compat() {
        // Old snake_case forms from before the serde(rename) overrides
        // must still deserialize for backward compatibility.
        let old_cpu: OrtAccelerator = serde_json::from_str("\"cpu_only\"").unwrap();
        assert_eq!(old_cpu, OrtAccelerator::CpuOnly);
        let old_dml: OrtAccelerator = serde_json::from_str("\"direct_ml\"").unwrap();
        assert_eq!(old_dml, OrtAccelerator::DirectMl);
        let old_trt: OrtAccelerator = serde_json::from_str("\"tensor_rt\"").unwrap();
        assert_eq!(old_trt, OrtAccelerator::TensorRt);
    }

    #[test]
    fn ort_available_always_includes_cpu() {
        let avail = OrtAccelerator::available();
        assert!(avail.contains(&OrtAccelerator::CpuOnly));
    }

    #[test]
    fn ort_from_u8_invalid_returns_auto() {
        assert_eq!(OrtAccelerator::from_u8(255), OrtAccelerator::Auto);
    }

    // -- Whisper tests --

    #[test]
    fn whisper_default_is_auto() {
        let _g = AccelGuard::new();
        set_whisper_accelerator(WhisperAccelerator::Auto);
        assert_eq!(get_whisper_accelerator(), WhisperAccelerator::Auto);
    }

    #[test]
    fn whisper_set_and_get() {
        let _g = AccelGuard::new();
        set_whisper_accelerator(WhisperAccelerator::CpuOnly);
        assert_eq!(get_whisper_accelerator(), WhisperAccelerator::CpuOnly);
        set_whisper_accelerator(WhisperAccelerator::Gpu);
        assert_eq!(get_whisper_accelerator(), WhisperAccelerator::Gpu);
    }

    #[test]
    fn whisper_display_roundtrip() {
        for pref in [
            WhisperAccelerator::Auto,
            WhisperAccelerator::CpuOnly,
            WhisperAccelerator::Gpu,
        ] {
            let s = pref.to_string();
            let parsed: WhisperAccelerator = s.parse().unwrap();
            assert_eq!(parsed, pref);
        }
    }

    #[test]
    fn whisper_use_gpu_flag() {
        assert!(WhisperAccelerator::Auto.use_gpu());
        assert!(!WhisperAccelerator::CpuOnly.use_gpu());
        assert!(WhisperAccelerator::Gpu.use_gpu());
    }

    #[test]
    fn whisper_parse_unknown_errors() {
        assert!("foobar".parse::<WhisperAccelerator>().is_err());
    }

    #[test]
    fn whisper_serde_roundtrip() {
        let pref = WhisperAccelerator::Gpu;
        let json = serde_json::to_string(&pref).unwrap();
        assert_eq!(json, "\"gpu\"");
        let back: WhisperAccelerator = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pref);
    }

    #[test]
    fn whisper_available_always_includes_cpu() {
        let avail = WhisperAccelerator::available();
        assert!(avail.contains(&WhisperAccelerator::CpuOnly));
    }

    #[test]
    fn whisper_from_u8_invalid_returns_auto() {
        assert_eq!(WhisperAccelerator::from_u8(255), WhisperAccelerator::Auto);
    }

    // -- GPU device tests --

    #[test]
    fn gpu_device_default_is_auto() {
        let _g = AccelGuard::new();
        set_whisper_gpu_device(GPU_DEVICE_AUTO);
        assert_eq!(get_whisper_gpu_device(), GPU_DEVICE_AUTO);
    }

    #[test]
    fn gpu_device_set_and_get() {
        let _g = AccelGuard::new();
        set_whisper_gpu_device(1);
        assert_eq!(get_whisper_gpu_device(), 1);
        set_whisper_gpu_device(GPU_DEVICE_AUTO);
        assert_eq!(get_whisper_gpu_device(), GPU_DEVICE_AUTO);
    }
}
