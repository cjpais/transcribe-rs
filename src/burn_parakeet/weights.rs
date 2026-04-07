use memmap2::Mmap;
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::path::Path;

use crate::TranscribeError;

/// Memory-map a safetensors file for zero-copy weight access.
pub fn mmap_file(path: &Path) -> Result<Mmap, TranscribeError> {
    let file = std::fs::File::open(path).map_err(TranscribeError::Io)?;
    unsafe { Mmap::map(&file) }.map_err(TranscribeError::Io)
}

/// Get f32 data for a named tensor. Zero-copy when data is 4-byte aligned
/// (common with safetensors on little-endian), copies otherwise.
pub fn get_f32<'a>(st: &'a SafeTensors<'a>, name: &str) -> Result<Cow<'a, [f32]>, TranscribeError> {
    let view = st.tensor(name).map_err(|e| {
        TranscribeError::Config(format!("Missing weight {name}: {e}"))
    })?;
    let bytes = view.data();
    assert!(
        bytes.len() % 4 == 0,
        "Weight {name}: byte count {} not divisible by 4",
        bytes.len()
    );
    let ptr = bytes.as_ptr();
    if ptr as usize % std::mem::align_of::<f32>() == 0 {
        // Zero-copy reinterpret — safe on little-endian (ARM Mac, x86)
        Ok(Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(ptr as *const f32, bytes.len() / 4)
        }))
    } else {
        // Unaligned fallback (rare)
        Ok(Cow::Owned(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ))
    }
}

/// Transpose a row-major [rows, cols] matrix to [cols, rows] on CPU.
pub fn cpu_transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * cols);
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// MLX conv1d weight [out_c, kernel, in_c] → Burn [out_c, in_c/groups, kernel] on CPU.
pub fn cpu_conv1d_reorder(data: &[f32], out_c: usize, kernel: usize, in_c: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), out_c * kernel * in_c);
    let mut out = vec![0.0f32; out_c * in_c * kernel];
    for o in 0..out_c {
        for k in 0..kernel {
            for i in 0..in_c {
                out[o * (in_c * kernel) + i * kernel + k] =
                    data[o * (kernel * in_c) + k * in_c + i];
            }
        }
    }
    out
}

/// MLX conv2d weight [out_c, kH, kW, in_c] → Burn [out_c, in_c/groups, kH, kW] on CPU.
pub fn cpu_conv2d_reorder(
    data: &[f32],
    out_c: usize,
    kh: usize,
    kw: usize,
    in_c: usize,
) -> Vec<f32> {
    debug_assert_eq!(data.len(), out_c * kh * kw * in_c);
    let mut out = vec![0.0f32; out_c * in_c * kh * kw];
    for o in 0..out_c {
        for h in 0..kh {
            for w in 0..kw {
                for i in 0..in_c {
                    out[o * (in_c * kh * kw) + i * (kh * kw) + h * kw + w] =
                        data[o * (kh * kw * in_c) + h * (kw * in_c) + w * in_c + i];
                }
            }
        }
    }
    out
}
