//! GPU device enumeration and auto-selection for whisper.cpp inference.
//!
//! Provides [`list_gpu_devices`] to enumerate available GPUs and
//! [`auto_select_gpu_device`] to pick the best one automatically
//! (preferring dedicated GPUs over integrated by total VRAM).

use log::info;
use serde::Serialize;
use std::ffi::CStr;
use whisper_rs::whisper_rs_sys;

/// Whether a GPU is dedicated or integrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GpuKind {
    /// Dedicated GPU (discrete card with its own VRAM).
    Dedicated,
    /// Integrated GPU (shares system memory).
    Integrated,
}

/// Information about a GPU device available for inference.
#[derive(Debug, Clone, Serialize)]
pub struct GpuDeviceInfo {
    /// Device ID to pass to [`crate::whisper_cpp::WhisperLoadParams::gpu_device`].
    ///
    /// This is the Nth GPU/IGPU in ggml's global device list, matching the
    /// index that whisper.cpp uses to select a device.
    pub id: i32,
    /// Human-readable device name (e.g. "NVIDIA GeForce RTX 4060").
    pub name: String,
    /// Whether this is a dedicated or integrated GPU.
    pub kind: GpuKind,
    /// Total VRAM in bytes.
    pub total_vram: usize,
    /// Free VRAM in bytes.
    pub free_vram: usize,
}

/// List GPU devices available for whisper inference.
///
/// Uses the GGML backend API to enumerate GPU devices across all backends
/// (Vulkan, CUDA, Metal, etc.). Device IDs match the index that whisper.cpp
/// uses internally to select a GPU.
///
/// On Metal, `gpu_device` is ignored by whisper.cpp (there is only one Metal
/// device), but the device is still listed for informational purposes.
pub fn list_gpu_devices() -> Vec<GpuDeviceInfo> {
    // On some platforms (e.g. Windows) whisper_rs_sys generates these as i32,
    // on others as u32. Cast to i32 for cross-platform compatibility.
    let type_gpu = whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU as i32;
    let type_igpu = whisper_rs_sys::ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_IGPU as i32;

    unsafe {
        let count = whisper_rs_sys::ggml_backend_dev_count();
        let mut gpu_devices = Vec::new();
        let mut gpu_index: i32 = 0;

        for i in 0..count {
            let dev = whisper_rs_sys::ggml_backend_dev_get(i);
            if dev.is_null() {
                continue;
            }

            // Only include actual GPU devices (dedicated or integrated),
            // skip CPU and ACCEL (e.g. Apple Accelerate framework).
            let dev_type = whisper_rs_sys::ggml_backend_dev_type(dev) as i32;
            if dev_type != type_gpu && dev_type != type_igpu {
                continue;
            }

            let name = {
                let ptr = whisper_rs_sys::ggml_backend_dev_description(dev);
                if ptr.is_null() {
                    format!("GPU {gpu_index}")
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };

            let mut free: usize = 0;
            let mut total: usize = 0;
            whisper_rs_sys::ggml_backend_dev_memory(dev, &mut free, &mut total);

            let kind = if dev_type == type_gpu {
                GpuKind::Dedicated
            } else {
                GpuKind::Integrated
            };

            gpu_devices.push(GpuDeviceInfo {
                id: gpu_index,
                name,
                kind,
                total_vram: total,
                free_vram: free,
            });

            gpu_index += 1;
        }

        gpu_devices
    }
}

/// Auto-select the best GPU device for whisper inference.
///
/// Prefers dedicated GPUs over integrated, then picks the device with the most
/// total VRAM as a tiebreaker.
///
/// Returns `0` if no devices are found or enumeration fails.
pub fn auto_select_gpu_device() -> i32 {
    let devices = list_gpu_devices();
    if devices.is_empty() {
        return 0;
    }

    let best = devices
        .iter()
        .max_by_key(|d| {
            let kind_priority = match d.kind {
                GpuKind::Dedicated => 1u8,
                GpuKind::Integrated => 0,
            };
            (kind_priority, d.total_vram)
        })
        .unwrap(); // safe: devices is non-empty

    info!(
        "Auto-selected GPU device {} '{}' ({:?}, {} MB VRAM)",
        best.id,
        best.name,
        best.kind,
        best.total_vram / (1024 * 1024),
    );

    best.id
}
