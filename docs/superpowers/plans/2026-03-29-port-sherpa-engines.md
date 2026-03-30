# Port Paraformer/Zipformer Engines to Upstream Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Paraformer, Zipformer CTC, Zipformer Transducer engines and punctuation model from our fork's old `TranscriptionEngine` trait to upstream's `SpeechModel` trait architecture, enabling PR merge into cjpais/transcribe-rs.

**Architecture:** Create a feature branch from `upstream/main` (v0.3.5). Add shared Kaldi fbank and BBPE decode modules. Port each engine as a new module under `src/onnx/`. Port punct.rs as a standalone feature. All engines use upstream's `session.rs`, `TranscribeError`, and `Quantization`.

**Tech Stack:** Rust, ort 2.0.0-rc.12, ndarray 0.17, rustfft 6, serde/serde_json

---

## File Structure

### New files to create:
- `src/features/kaldi_fbank.rs` — Kaldi-compatible fbank (Povey window, DC removal, preemphasis, neg high_freq)
- `src/decode/bbpe.rs` — BBPE byte-to-unicode symbol table + text normalization
- `src/onnx/paraformer/mod.rs` — ParaformerModel + SpeechModel impl
- `src/onnx/zipformer_ctc/mod.rs` — ZipformerCtcModel + SpeechModel impl
- `src/onnx/zipformer_transducer/mod.rs` — ZipformerTransducerModel + SpeechModel impl
- `src/punct.rs` — PunctModel (standalone punctuation post-processor)
- `examples/paraformer.rs` — Paraformer example
- `examples/zipformer_ctc.rs` — Zipformer CTC example
- `examples/zipformer_transducer.rs` — Zipformer Transducer example
- `tests/paraformer.rs` — Paraformer test
- `tests/zipformer_ctc.rs` — Zipformer CTC test
- `tests/zipformer_transducer.rs` — Zipformer Transducer test

### Files to modify:
- `src/features/mod.rs` — add `pub mod kaldi_fbank;`
- `src/decode/mod.rs` — add `pub mod bbpe;`
- `src/onnx/mod.rs` — add `pub mod paraformer; pub mod zipformer_ctc; pub mod zipformer_transducer;`
- `src/lib.rs` — add `pub mod punct;` under punct feature
- `src/error.rs` — no changes needed (already has ort::Error and serde_json::Error From impls)
- `Cargo.toml` — add `punct` feature, examples, tests

### Reference files (read from backup branch):
- `backup/pre-upstream-port:src/engines/paraformer/model.rs` — Paraformer inference logic
- `backup/pre-upstream-port:src/engines/paraformer/features.rs` — Paraformer fbank (simpler, non-Kaldi)
- `backup/pre-upstream-port:src/engines/paraformer/tokens.rs` — Paraformer symbol table
- `backup/pre-upstream-port:src/engines/zipformer_common.rs` — Kaldi fbank + BBPE + SymbolTable
- `backup/pre-upstream-port:src/engines/zipformer_ctc/model.rs` — CTC inference
- `backup/pre-upstream-port:src/engines/zipformer_transducer/model.rs` — Transducer inference
- `backup/pre-upstream-port:src/punct.rs` — Punctuation model

---

## Key API Mapping (old → new)

### ort rc.10 → rc.12
- `session.inputs` → `session.inputs()`
- `session.outputs` → `session.outputs()`
- `input.name` → `input.name()`
- `input.input_type` → `input.dtype()`
- `output.name` → `output.name()`
- `output.output_type` → `output.dtype()`
- `CPUExecutionProvider::default().build()` → use `session::create_session()` (handles all EPs)
- `metadata.custom(key)?` returns `Option<String>` (no Result wrapping in rc.12; use `session::read_metadata_str`)

### Trait mapping
- `TranscriptionEngine::transcribe_samples(samples, params)` → `SpeechModel::transcribe_raw(samples, &TranscribeOptions)`
- `Box<dyn Error>` → `TranscribeError`
- `ParaformerModel::new(dir, quantized: bool)` → `ParaformerModel::load(dir, &Quantization)`
- Custom error enums → `TranscribeError::{ModelNotFound, Inference, Config, ...}`

### Feature extraction mapping
- Paraformer uses standard fbank (Hamming window, dB scale) — use upstream `compute_mel()` with appropriate `MelConfig`
- Zipformer uses Kaldi fbank (Povey window, natural log, DC removal) — use new `kaldi_fbank.rs`
- LFR/CMVN — use upstream `features::apply_lfr` and `features::apply_cmvn`

---

### Task 1: Create feature branch from upstream/main

**Files:** None (git operations only)

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b feat/sherpa-engines upstream/main
```

- [ ] **Step 2: Verify clean state**

```bash
cargo check --features onnx
```

Expected: compiles clean on upstream/main

- [ ] **Step 3: Commit (empty, branch marker)**

No commit needed — clean branch from upstream.

---

### Task 2: Add Kaldi fbank feature extraction

**Files:**
- Create: `src/features/kaldi_fbank.rs`
- Modify: `src/features/mod.rs`

- [ ] **Step 1: Create `src/features/kaldi_fbank.rs`**

Port from `backup/pre-upstream-port:src/engines/zipformer_common.rs` (the `compute_fbank_kaldi` function and `FbankConfig`), adapting to upstream style:

```rust
//! Kaldi-compatible FBank feature extraction.
//!
//! Matches the behavior of kaldi-native-fbank / sherpa-onnx for Zipformer
//! and Paraformer models that expect Kaldi-style features.

use ndarray::Array2;
use rustfft::{num_complex::Complex, FftPlanner};

/// Kaldi-compatible FBank configuration.
#[derive(Debug, Clone)]
pub struct KaldiFbankConfig {
    pub num_bins: usize,
    pub fft_size: usize,
    pub window_size: usize,
    pub hop_size: usize,
    pub sample_rate: u32,
    pub low_freq: f32,
    /// Negative means nyquist + high_freq (Kaldi convention). -400 → 7600 Hz at 16 kHz.
    pub high_freq: f32,
    pub preemph_coeff: f32,
    pub snip_edges: bool,
    pub remove_dc_offset: bool,
}

impl Default for KaldiFbankConfig {
    fn default() -> Self {
        Self {
            num_bins: 80,
            fft_size: 512,
            window_size: 400,
            hop_size: 160,
            sample_rate: 16000,
            low_freq: 20.0,
            high_freq: -400.0,
            preemph_coeff: 0.97,
            snip_edges: false,
            remove_dc_offset: true,
        }
    }
}

/// Compute Kaldi-compatible FBank features.
///
/// Key differences from standard mel spectrogram:
/// - Povey window (Hamming^0.85) instead of plain Hamming/Hann
/// - DC offset removal per frame
/// - Preemphasis applied per frame (reverse order)
/// - snip_edges=false centers first frame and zero-pads boundaries
/// - Natural log energy (not dB)
/// - Negative high_freq interpreted as nyquist + value
///
/// Returns `[num_frames, num_bins]`.
pub fn compute_kaldi_fbank(samples: &[f32], config: &KaldiFbankConfig) -> Array2<f32> {
    let window_size = config.window_size;
    let hop_size = config.hop_size;
    let fft_size = config.fft_size;
    let half_fft = fft_size / 2 + 1;

    if samples.is_empty() {
        return Array2::zeros((0, config.num_bins));
    }

    let num_frames = if config.snip_edges {
        if samples.len() < window_size {
            return Array2::zeros((0, config.num_bins));
        }
        (samples.len() - window_size) / hop_size + 1
    } else {
        (samples.len() + hop_size / 2) / hop_size
    };

    if num_frames == 0 {
        return Array2::zeros((0, config.num_bins));
    }

    let nyquist = config.sample_rate as f32 / 2.0;
    let high_freq = if config.high_freq <= 0.0 {
        nyquist + config.high_freq
    } else {
        config.high_freq
    };

    let filterbank = mel_filterbank(config.num_bins, fft_size, config.sample_rate as f32, config.low_freq, high_freq);

    // Povey window: hamming^0.85
    let window: Vec<f32> = (0..window_size)
        .map(|i| {
            let hamming = 0.54
                - 0.46
                    * (2.0 * std::f32::consts::PI * i as f32 / (window_size as f32 - 1.0)).cos();
            hamming.powf(0.85)
        })
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut features = Vec::with_capacity(num_frames * config.num_bins);

    for frame_idx in 0..num_frames {
        let center = if config.snip_edges {
            frame_idx * hop_size + window_size / 2
        } else {
            frame_idx * hop_size
        };
        let start = center as isize - (window_size as isize / 2);

        // Extract frame with zero-padding at boundaries
        let mut frame = vec![0.0f32; window_size];
        for i in 0..window_size {
            let idx = start + i as isize;
            if idx >= 0 && (idx as usize) < samples.len() {
                frame[i] = samples[idx as usize];
            }
        }

        // Remove DC offset
        if config.remove_dc_offset {
            let mean: f32 = frame.iter().sum::<f32>() / window_size as f32;
            for s in frame.iter_mut() {
                *s -= mean;
            }
        }

        // Preemphasis (reverse order to avoid overwriting)
        if config.preemph_coeff > 0.0 {
            for i in (1..window_size).rev() {
                frame[i] -= config.preemph_coeff * frame[i - 1];
            }
            frame[0] *= 1.0 - config.preemph_coeff;
        }

        // Apply window and FFT
        let mut buffer: Vec<Complex<f32>> = frame
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();
        buffer.resize(fft_size, Complex::new(0.0, 0.0));
        fft.process(&mut buffer);

        // Power spectrum
        let power: Vec<f32> = buffer[..half_fft].iter().map(|c| c.norm_sqr()).collect();

        // Apply mel filterbank and take natural log
        for filter in &filterbank {
            let energy: f32 = filter.iter().zip(power.iter()).map(|(&w, &p)| w * p).sum();
            features.push(if energy > f32::EPSILON {
                energy.ln()
            } else {
                f32::EPSILON.ln()
            });
        }
    }

    Array2::from_shape_vec((num_frames, config.num_bins), features).unwrap()
}

fn mel_filterbank(
    num_bins: usize,
    fft_size: usize,
    sample_rate: f32,
    low_freq: f32,
    high_freq: f32,
) -> Vec<Vec<f32>> {
    let half_fft = fft_size / 2 + 1;

    let hz_to_mel = |hz: f32| 1127.0 * (1.0 + hz / 700.0).ln();
    let mel_to_hz = |mel: f32| 700.0 * ((mel / 1127.0).exp() - 1.0);

    let low_mel = hz_to_mel(low_freq);
    let high_mel = hz_to_mel(high_freq);

    let num_points = num_bins + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_points - 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let fft_bins: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz * fft_size as f32) / sample_rate).floor() as usize)
        .collect();

    let mut filterbank = vec![vec![0.0f32; half_fft]; num_bins];
    for (i, filter) in filterbank.iter_mut().enumerate() {
        let left = fft_bins[i];
        let center = fft_bins[i + 1];
        let right = fft_bins[i + 2];

        if center > left {
            for j in left..center {
                if j < half_fft {
                    filter[j] = (j - left) as f32 / (center - left) as f32;
                }
            }
        }
        if right > center {
            for j in center..right {
                if j < half_fft {
                    filter[j] = (right - j) as f32 / (right - center) as f32;
                }
            }
        }
    }

    filterbank
}
```

- [ ] **Step 2: Register in `src/features/mod.rs`**

Add after existing exports:

```rust
pub mod kaldi_fbank;
pub use kaldi_fbank::{compute_kaldi_fbank, KaldiFbankConfig};
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check --features audio-features
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/features/kaldi_fbank.rs src/features/mod.rs
git commit -m "feat: add Kaldi-compatible fbank feature extraction"
```

---

### Task 3: Add BBPE decode module

**Files:**
- Create: `src/decode/bbpe.rs`
- Modify: `src/decode/mod.rs`

- [ ] **Step 1: Create `src/decode/bbpe.rs`**

Port from `backup/pre-upstream-port:src/engines/zipformer_common.rs` (SymbolTable, BBPE mapping, normalize_text):

```rust
//! BBPE (Byte-level BPE) symbol table for Icefall/sherpa-onnx models.
//!
//! Supports two encoding modes:
//! - BBPE: byte-to-unicode mapped tokens (Icefall zh-en models)
//! - BPE: standard sentencepiece tokens (literal UTF-8)
//!
//! Auto-detects encoding by checking for `bbpe.model` sibling file.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Token encoding mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenEncoding {
    /// Icefall BBPE: token chars are byte-to-unicode mapped, need decoding.
    Bbpe,
    /// Standard BPE/sentencepiece: token strings are literal UTF-8.
    Bpe,
}

/// Symbol table with BBPE/BPE decoding support.
pub struct BbpeSymbolTable {
    id_to_sym: HashMap<i32, String>,
    encoding: TokenEncoding,
}

impl BbpeSymbolTable {
    /// Load with auto-detected encoding.
    /// If `bbpe.model` exists in the same directory as `path`, use BBPE; otherwise BPE.
    pub fn load(path: &Path) -> Result<Self, std::io::Error> {
        let encoding = if let Some(dir) = path.parent() {
            if dir.join("bbpe.model").exists() {
                log::info!("Detected BBPE encoding (bbpe.model found)");
                TokenEncoding::Bbpe
            } else {
                log::info!("Detected standard BPE encoding (no bbpe.model)");
                TokenEncoding::Bpe
            }
        } else {
            TokenEncoding::Bbpe
        };
        Self::load_with_encoding(path, encoding)
    }

    /// Load with explicit encoding.
    pub fn load_with_encoding(
        path: &Path,
        encoding: TokenEncoding,
    ) -> Result<Self, std::io::Error> {
        let contents = fs::read_to_string(path)?;
        let mut id_to_sym = HashMap::new();

        for line in contents.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            // Format: "token id" (split on last whitespace; token can contain spaces)
            let parts: Vec<&str> = line.rsplitn(2, |c: char| c.is_whitespace()).collect();
            if parts.len() == 2 {
                if let Ok(id) = parts[0].parse::<i32>() {
                    id_to_sym.insert(id, parts[1].to_string());
                }
            }
        }

        log::info!(
            "Loaded {} tokens from {:?} (encoding={:?})",
            id_to_sym.len(),
            path,
            encoding
        );
        Ok(Self { id_to_sym, encoding })
    }

    /// Decode token IDs to text.
    pub fn decode(&self, token_ids: &[i32]) -> String {
        match self.encoding {
            TokenEncoding::Bbpe => self.decode_bbpe(token_ids),
            TokenEncoding::Bpe => self.decode_bpe(token_ids),
        }
    }

    fn decode_bbpe(&self, token_ids: &[i32]) -> String {
        let mut raw_bytes = Vec::new();

        for &id in token_ids {
            let Some(sym) = self.id_to_sym.get(&id) else {
                continue;
            };
            if sym.starts_with('<') && sym.ends_with('>') {
                continue;
            }
            for c in sym.chars() {
                if c == '\u{2581}' {
                    raw_bytes.push(b' ');
                } else if let Some(byte_val) = bbpe_char_to_byte(c) {
                    raw_bytes.push(byte_val);
                }
            }
        }

        let text = String::from_utf8_lossy(&raw_bytes);
        normalize_text(text.trim())
    }

    fn decode_bpe(&self, token_ids: &[i32]) -> String {
        let mut text = String::new();

        for &id in token_ids {
            let Some(sym) = self.id_to_sym.get(&id) else {
                continue;
            };
            if sym.starts_with('<') && sym.ends_with('>') {
                continue;
            }
            text.push_str(&sym.replace('\u{2581}', " "));
        }

        normalize_text(text.trim())
    }
}

// ---- Text normalization ----

fn is_cjk(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' |
        '\u{3400}'..='\u{4DBF}' |
        '\u{F900}'..='\u{FAFF}' |
        '\u{2E80}'..='\u{2EFF}' |
        '\u{3000}'..='\u{303F}' |
        '\u{FF00}'..='\u{FFEF}'
    )
}

/// Remove spaces between CJK characters and lowercase English text.
fn normalize_text(text: &str) -> String {
    let text = text.to_lowercase();
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());

    for i in 0..chars.len() {
        let c = chars[i];
        if c == ' ' {
            let prev_cjk = i > 0 && is_cjk(chars[i - 1]);
            let next_cjk = i + 1 < chars.len() && is_cjk(chars[i + 1]);
            if prev_cjk && next_cjk {
                continue;
            }
        }
        result.push(c);
    }

    result
}

// ---- Icefall BBPE byte mapping ----

/// Icefall PRINTABLE_BASE_CHARS: maps byte index (0-255) to a Unicode codepoint.
const BBPE_CODEPOINTS: [u32; 256] = [
    256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
    271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
    286, 287, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 288, 289, 290, 291, 292,
    293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 308, 309,
    310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 322, 323, 324, 325, 326,
    327, 328, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342,
    343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357,
    358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372,
    373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388,
    389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403,
    404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418,
    419, 420, 421, 422,
];

fn bbpe_char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32;
    if (32..=126).contains(&cp) {
        return Some(cp as u8);
    }
    for (byte_val, &mapped_cp) in BBPE_CODEPOINTS.iter().enumerate() {
        if mapped_cp == cp {
            return Some(byte_val as u8);
        }
    }
    None
}
```

- [ ] **Step 2: Register in `src/decode/mod.rs`**

Add after existing exports:

```rust
pub mod bbpe;
pub use bbpe::BbpeSymbolTable;
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check --features audio-features
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/decode/bbpe.rs src/decode/mod.rs
git commit -m "feat: add BBPE symbol table for Icefall/sherpa-onnx models"
```

---

### Task 4: Add ParaformerModel

**Files:**
- Create: `src/onnx/paraformer/mod.rs`
- Modify: `src/onnx/mod.rs`

- [ ] **Step 1: Create `src/onnx/paraformer/mod.rs`**

Port from backup branch, adapting to upstream patterns. Key changes:
- Use `session::create_session()` instead of manual session builder
- Use `session::resolve_model_path()` for quantization
- Use `session::read_metadata_i32/float_vec()` for metadata
- Use upstream `features::compute_mel()` with `MelConfig` (Paraformer uses standard Hamming window fbank, NOT Kaldi fbank)
- Use upstream `features::apply_lfr()` and `features::apply_cmvn()`
- Return `TranscribeError` instead of custom error enum
- Implement `SpeechModel` trait
- Use ort rc.12 API (`session.inputs()`, `input.name()`, etc.)

The complete file should include:
1. `CAPABILITIES` const
2. `ParaformerParams` struct (empty for now — Paraformer is language-auto)
3. `ParaformerModel` struct with session, symbol_table, metadata, cmvn, I/O names
4. `ParaformerModel::load(dir, &Quantization)` constructor
5. Paraformer-specific `SymbolTable` (inline, handles `@@` joining and `▁` markers — different from BBPE)
6. Metadata parsing via `session::read_metadata_i32`
7. CMVN loading from ONNX metadata or `am.mvn` file
8. `compute_features()` → `compute_mel()` + `apply_lfr()` + `apply_cmvn()`
9. `forward()` → run ONNX session
10. `decode_logits()` → argmax with eos/blank/sos filtering
11. `SpeechModel` impl with `transcribe_raw()`

**Important Paraformer-specific details:**
- Paraformer uses dB scale fbank (10*log10), NOT natural log — use `MelConfig` with `pre_emphasis: None` and standard Hamming window, then manually apply 10*log10 scaling. Actually, looking at the old code more carefully: it uses `10.0 * sum.log10()` with `-80.0` floor. The upstream `compute_mel` with `pre_emphasis: None` uses `ln()`. We need to match the original behavior.
- Solution: Use upstream `compute_mel` with custom `MelConfig{pre_emphasis: None, ...}` — BUT upstream's `compute_mel_spectrogram` uses `ln()`, not `10*log10`. We need to either (a) modify the output, or (b) implement inline. Option (b) is safer to avoid breaking existing models. Implement a private `compute_paraformer_fbank()` inside the module that matches the original exactly.
- LFR default: window_size=7, window_shift=6 (from ONNX metadata)
- CMVN: mean subtraction only (old code uses `apply_mean_cmvn` which subtracts mean; upstream `apply_cmvn` multiplies by inv_stddev too). For Paraformer we only have neg_mean, no inv_stddev. So we do mean-only CMVN inline.
- Symbol table: Paraformer tokens use `@@` for subword joining and `▁` for spaces, plus special tokens `<blank>`, `<s>`, `</s>`, `<unk>`. This is different from both upstream's SymbolTable and the BBPE SymbolTable. Keep it inline in the module.

- [ ] **Step 2: Register in `src/onnx/mod.rs`**

Add:
```rust
pub mod paraformer;
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check --features onnx
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/onnx/paraformer/ src/onnx/mod.rs
git commit -m "feat: add Paraformer ONNX engine"
```

---

### Task 5: Add ZipformerCtcModel

**Files:**
- Create: `src/onnx/zipformer_ctc/mod.rs`
- Modify: `src/onnx/mod.rs`

- [ ] **Step 1: Create `src/onnx/zipformer_ctc/mod.rs`**

Port from backup branch. Key adaptations:
- Use `session::create_session()` for session creation
- Use `compute_kaldi_fbank()` from `features::kaldi_fbank`
- Use upstream `ctc_greedy_decode()` from `decode::ctc` — BUT note: upstream CTC takes `ArrayView3<f32>` with shape [batch, time, vocab] and `&[i64]` lengths. Our old code had custom CTC with `Array2`. Need to reshape to 3D for upstream API.
- Use `BbpeSymbolTable` from `decode::bbpe` for token decoding
- Model file discovery: keep our smart fallback logic (scan directory for *.onnx) but also try `session::resolve_model_path()` first
- Streaming model rejection: keep the `cached_*` input detection
- Return `TranscribeError`
- Implement `SpeechModel` trait

- [ ] **Step 2: Register in `src/onnx/mod.rs`**

Add:
```rust
pub mod zipformer_ctc;
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check --features onnx
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/onnx/zipformer_ctc/ src/onnx/mod.rs
git commit -m "feat: add Zipformer CTC ONNX engine"
```

---

### Task 6: Add ZipformerTransducerModel

**Files:**
- Create: `src/onnx/zipformer_transducer/mod.rs`
- Modify: `src/onnx/mod.rs`

- [ ] **Step 1: Create `src/onnx/zipformer_transducer/mod.rs`**

Port from backup branch. This is the most complex engine (3 sessions):
- Use `session::create_session()` for all 3 sessions
- Use `compute_kaldi_fbank()` for features
- Use `BbpeSymbolTable` for token decoding
- Keep the multi-file model discovery logic (`find_model_file` for encoder/decoder/joiner with various naming patterns)
- Keep streaming model rejection
- Keep the RNN-T greedy search decoding loop (no upstream equivalent)
- context_size=2 hardcoded
- Return `TranscribeError`
- Implement `SpeechModel` trait

**Important:** The transducer's `find_model_file` looks for `{component}-*.{suffix}.onnx` patterns (e.g., `encoder-epoch-34-avg-19.int8.onnx`). This is unique to sherpa-onnx transducer models and must be preserved.

- [ ] **Step 2: Register in `src/onnx/mod.rs`**

Add:
```rust
pub mod zipformer_transducer;
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check --features onnx
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/onnx/zipformer_transducer/ src/onnx/mod.rs
git commit -m "feat: add Zipformer Transducer ONNX engine"
```

---

### Task 7: Add PunctModel

**Files:**
- Create: `src/punct.rs`
- Modify: `src/lib.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add `punct` feature to `Cargo.toml`**

In `[features]` section, add:
```toml
# Neural punctuation restoration (CT-Transformer)
punct = ["dep:ort", "dep:ndarray"]
```

Update `all` feature to include `punct`:
```toml
all = ["onnx", "whisper-cpp", "whisperfile", "openai", "punct"]
```

- [ ] **Step 2: Create `src/punct.rs`**

Port from backup branch with these adaptations:
- Use `session::create_session()` instead of manual session builder (but note: punct uses `#[cfg(feature = "punct")]` not `#[cfg(feature = "onnx")]`, and `session` module is under `onnx` feature. So we need to build the session manually for punct, OR gate punct under onnx.)
- **Decision:** Gate punct session creation manually (like `vad-silero` does — it also uses ort directly without the onnx feature). Use `ort` directly:

```rust
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
```

- Keep the custom `PunctError` enum (it's not a SpeechModel, so TranscribeError doesn't fit perfectly. But for consistency, convert to `TranscribeError`.)
- **Decision:** Use `TranscribeError` for consistency with the rest of the crate. The From impls already exist for ort::Error, serde_json::Error, and io::Error.
- Use ort rc.12 API for session inputs/outputs
- Keep all inference logic unchanged

- [ ] **Step 3: Register in `src/lib.rs`**

Add after existing module declarations:
```rust
#[cfg(feature = "punct")]
pub mod punct;
```

- [ ] **Step 4: Verify compilation**

```bash
cargo check --features punct
cargo check --features "onnx,punct"
```

Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add src/punct.rs src/lib.rs Cargo.toml
git commit -m "feat: add neural punctuation restoration model"
```

---

### Task 8: Add examples

**Files:**
- Create: `examples/paraformer.rs`
- Create: `examples/zipformer_ctc.rs`
- Create: `examples/zipformer_transducer.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Create examples**

Follow the upstream pattern from `examples/gigaam.rs`. Each example:
- Accepts model_dir and wav_path as positional args with defaults
- Supports `--int8` flag
- Shows load time, transcribe time, real-time speedup
- Displays text and segments

Default model paths:
- Paraformer: `models/sherpa-onnx-paraformer-zh-2025-10-07`
- Zipformer CTC: `models/sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16`
- Zipformer Transducer: `models/sherpa-onnx-zipformer-zh-en-2023-11-22`

Default wav: `samples/zh.wav`

- [ ] **Step 2: Add example declarations to `Cargo.toml`**

```toml
[[example]]
name = "paraformer"
required-features = ["onnx"]

[[example]]
name = "zipformer_ctc"
required-features = ["onnx"]

[[example]]
name = "zipformer_transducer"
required-features = ["onnx"]
```

- [ ] **Step 3: Verify examples compile**

```bash
cargo check --example paraformer --features onnx
cargo check --example zipformer_ctc --features onnx
cargo check --example zipformer_transducer --features onnx
```

Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add examples/paraformer.rs examples/zipformer_ctc.rs examples/zipformer_transducer.rs Cargo.toml
git commit -m "feat: add examples for Paraformer and Zipformer engines"
```

---

### Task 9: Add tests

**Files:**
- Create: `tests/paraformer.rs`
- Create: `tests/zipformer_ctc.rs`
- Create: `tests/zipformer_transducer.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Create test files**

Follow upstream pattern from `tests/gigaam.rs`. Each test:
- Uses `mod common;` for `require_paths`
- Skips if model/wav not found (graceful skip, not failure)
- Loads model with `Quantization::Int8`
- Transcribes a test WAV
- Asserts expected output text

- [ ] **Step 2: Add test declarations to `Cargo.toml`**

```toml
[[test]]
name = "paraformer"
required-features = ["onnx"]

[[test]]
name = "zipformer_ctc"
required-features = ["onnx"]

[[test]]
name = "zipformer_transducer"
required-features = ["onnx"]
```

- [ ] **Step 3: Verify tests compile**

```bash
cargo test --no-run --features onnx
```

Expected: PASS (tests compile; may skip at runtime if models not present)

- [ ] **Step 4: Commit**

```bash
git add tests/paraformer.rs tests/zipformer_ctc.rs tests/zipformer_transducer.rs Cargo.toml
git commit -m "test: add tests for Paraformer and Zipformer engines"
```

---

### Task 10: Full verification

- [ ] **Step 1: Verify all features compile**

```bash
cargo check --features onnx
cargo check --features punct
cargo check --features "onnx,punct"
cargo check --features all
```

Expected: all PASS

- [ ] **Step 2: Run cargo clippy**

```bash
cargo clippy --features "onnx,punct" -- -D warnings
```

Expected: PASS (no warnings)

- [ ] **Step 3: Run cargo fmt**

```bash
cargo fmt --check
```

Expected: PASS

- [ ] **Step 4: Run tests with models (if available)**

```bash
cargo test --features onnx -- --nocapture
```

- [ ] **Step 5: Run examples with models (if available)**

```bash
cargo run --example paraformer --features onnx -- models/sherpa-onnx-paraformer-zh-2025-10-07 samples/zh.wav --int8
cargo run --example zipformer_ctc --features onnx -- models/sherpa-onnx-zipformer-ctc-small-zh-int8-2025-07-16 samples/zh.wav --int8
cargo run --example zipformer_transducer --features onnx -- models/sherpa-onnx-zipformer-zh-en-2023-11-22 samples/zh.wav --int8
```

- [ ] **Step 6: Final commit (if any fmt/clippy fixes)**

```bash
git add -A
git commit -m "chore: fix clippy warnings and formatting"
```
