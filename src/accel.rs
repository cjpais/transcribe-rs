//! Framework-agnostic accelerator preference.
//!
//! Call [`set_accelerator`] early in your program to select a preferred execution
//! provider. Engine implementations read the preference via [`get_accelerator`]
//! when building inference sessions.

use std::fmt;
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

/// Preferred hardware accelerator for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum AcceleratorPreference {
    /// Automatically select the best available accelerator (default).
    Auto = 0,
    /// Force CPU-only execution — no GPU providers.
    CpuOnly = 1,
    /// NVIDIA CUDA.
    Cuda = 2,
    /// Microsoft DirectML (Windows).
    DirectMl = 3,
    /// AMD ROCm.
    Rocm = 4,
}

static ACCELERATOR: AtomicU8 = AtomicU8::new(AcceleratorPreference::Auto as u8);

/// Set the global accelerator preference.
///
/// This should be called once, early in the program, before any models are loaded.
/// It is safe to call from any thread.
pub fn set_accelerator(pref: AcceleratorPreference) {
    ACCELERATOR.store(pref as u8, Ordering::Relaxed);
}

/// Get the current accelerator preference.
pub fn get_accelerator() -> AcceleratorPreference {
    AcceleratorPreference::from_u8(ACCELERATOR.load(Ordering::Relaxed))
}

/// Return the list of accelerators that are compiled-in for the current build.
///
/// Always includes `CpuOnly`. Only includes GPU accelerators whose corresponding
/// feature flag is enabled.
pub fn available_accelerators() -> Vec<AcceleratorPreference> {
    #[allow(unused_mut)]
    let mut v = vec![AcceleratorPreference::CpuOnly];

    #[cfg(feature = "ort-cuda")]
    v.push(AcceleratorPreference::Cuda);

    #[cfg(feature = "ort-directml")]
    v.push(AcceleratorPreference::DirectMl);

    #[cfg(feature = "ort-rocm")]
    v.push(AcceleratorPreference::Rocm);

    v
}

impl AcceleratorPreference {
    /// Returns true if GPU should be used (i.e. not explicitly CPU-only).
    pub fn use_gpu(&self) -> bool {
        *self != Self::CpuOnly
    }

    fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Auto,
            1 => Self::CpuOnly,
            2 => Self::Cuda,
            3 => Self::DirectMl,
            4 => Self::Rocm,
            _ => Self::Auto,
        }
    }
}

impl fmt::Display for AcceleratorPreference {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Auto => "auto",
            Self::CpuOnly => "cpu",
            Self::Cuda => "cuda",
            Self::DirectMl => "directml",
            Self::Rocm => "rocm",
        };
        f.write_str(s)
    }
}

impl FromStr for AcceleratorPreference {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" | "cpu_only" | "cpuonly" => Ok(Self::CpuOnly),
            "cuda" => Ok(Self::Cuda),
            "directml" | "dml" => Ok(Self::DirectMl),
            "rocm" => Ok(Self::Rocm),
            other => Err(format!("unknown accelerator: {other}")),
        }
    }
}

impl Default for AcceleratorPreference {
    fn default() -> Self {
        Self::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RAII guard that restores Auto preference when dropped.
    struct AccelGuard;
    impl Drop for AccelGuard {
        fn drop(&mut self) {
            set_accelerator(AcceleratorPreference::Auto);
        }
    }

    #[test]
    fn default_is_auto() {
        let _g = AccelGuard;
        assert_eq!(get_accelerator(), AcceleratorPreference::Auto);
    }

    #[test]
    fn set_and_get() {
        let _g = AccelGuard;
        set_accelerator(AcceleratorPreference::Cuda);
        assert_eq!(get_accelerator(), AcceleratorPreference::Cuda);
        set_accelerator(AcceleratorPreference::CpuOnly);
        assert_eq!(get_accelerator(), AcceleratorPreference::CpuOnly);
    }

    #[test]
    fn display_roundtrip() {
        for pref in [
            AcceleratorPreference::Auto,
            AcceleratorPreference::CpuOnly,
            AcceleratorPreference::Cuda,
            AcceleratorPreference::DirectMl,
            AcceleratorPreference::Rocm,
        ] {
            let s = pref.to_string();
            let parsed: AcceleratorPreference = s.parse().unwrap();
            assert_eq!(parsed, pref);
        }
    }

    #[test]
    fn parse_aliases() {
        assert_eq!("dml".parse::<AcceleratorPreference>().unwrap(), AcceleratorPreference::DirectMl);
        assert_eq!("CPU".parse::<AcceleratorPreference>().unwrap(), AcceleratorPreference::CpuOnly);
        assert_eq!("cpu_only".parse::<AcceleratorPreference>().unwrap(), AcceleratorPreference::CpuOnly);
    }

    #[test]
    fn use_gpu_flag() {
        assert!(AcceleratorPreference::Auto.use_gpu());
        assert!(!AcceleratorPreference::CpuOnly.use_gpu());
        assert!(AcceleratorPreference::Cuda.use_gpu());
    }

    #[test]
    fn parse_unknown_errors() {
        assert!("foobar".parse::<AcceleratorPreference>().is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let pref = AcceleratorPreference::Cuda;
        let json = serde_json::to_string(&pref).unwrap();
        assert_eq!(json, "\"cuda\"");
        let back: AcceleratorPreference = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pref);
    }

    #[test]
    fn available_always_includes_cpu() {
        let avail = available_accelerators();
        assert!(avail.contains(&AcceleratorPreference::CpuOnly));
    }

    #[test]
    fn from_u8_invalid_returns_auto() {
        assert_eq!(AcceleratorPreference::from_u8(255), AcceleratorPreference::Auto);
    }
}
