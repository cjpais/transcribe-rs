# Burn Parakeet Performance Analysis

## Current Performance

Reference speeds: ~40x realtime (CPU), ~110x realtime (MLX/GPU).
Burn/wgpu implementation is significantly slower than both.

---

## Model Load (~20 seconds)

The 2.5GB safetensors file (697 tensors) goes through a pipeline that creates **~1,630 GPU tensor allocations**:

### Double initialization (~8-10s)
- `FastConformerEncoder::new()`, `TdtPredictor::new()`, `JointNetwork::new()` create **~817 GPU tensors with random weights** first
- Then `load_encoder_weights()` etc. create **~812 more GPU tensors** from the real weights, immediately discarding the random ones
- That's ~5GB of wasted CPU->GPU traffic

### ~~HashMap intermediate (~2-3s)~~ FIXED
- ~~`load_safetensors()` reads the entire 2.5GB file, parses every byte to f32, stores in `HashMap<String, Vec<f32>>`~~
- Now uses mmap + zero-copy `&[f32]` reinterpretation from safetensors. No 2.5GB CPU allocation.

### Individual GPU uploads (~3-4s)
- 812 separate `Tensor::from_floats()` calls, each creating a wgpu buffer + staging upload
- Each upload has fixed overhead (buffer creation, command submission)

### ~~Post-upload format conversions (~1-2s)~~ FIXED
- ~~~245 transpose operations (MLX [out,in] -> Burn [in,out] for every linear)~~
- ~~~77 permute operations (MLX NHWC -> Burn NCHW for convs)~~
- Now done on CPU before GPU upload — `cpu_transpose`, `cpu_conv1d_reorder`, `cpu_conv2d_reorder`.

### Potential fixes
- Skip random init: use `Param::from_tensor()` directly instead of init+overwrite
- Eliminate HashMap: use safetensors mmap, reinterpret bytes as `&[f32]` directly (little-endian on ARM Mac)
- Batch uploads: concatenate weights into fewer, larger GPU transfers
- Transpose on CPU before upload (avoid GPU kernel launches for format conversion)

---

## Inference — Encoder

### 1. Redundant BatchNorm in ConvModule (HIGH IMPACT)
**Location:** `encoder.rs:96-102`

BN is correctly folded into depthwise conv weights at load time (`mod.rs:370-393`), setting mean=0, var=1, weight=1, bias=0. But the forward pass still computes:

```
(x - 0) / sqrt(1 + 1e-5) * 1 + 0
```

This is ~10 GPU kernel launches (4 `.val()` + 4 reshapes + subtract + divide + sqrt + multiply + add) per ConvModule × 24 layers = **~240 wasted GPU kernel launches per forward pass**.

**Fix:** Skip BN computation entirely when weights are folded. Add a `bn_folded` flag.

### 2. Positional encoding recomputed on CPU (MODERATE)
**Location:** `encoder.rs:311-339`

`make_pos_encoding()` builds a (2*seq_len-1) × 1024 sinusoidal encoding using scalar CPU math, then transfers to GPU. For seq_len=125, that's 255K floats computed in a nested loop.

**Fix:** Cache after first computation (seq_len depends on input length but is deterministic), or compute using GPU tensor ops.

### 3. rel_shift allocations in attention (MODERATE)
**Location:** `attention.rs:127-139`

Creates a zeros tensor, concatenates, reshapes 2x, slices 2x — 6 GPU ops × 24 layers = **144 GPU kernel launches**. The `Cat` allocates a new buffer every time.

**Fix:** Could pre-allocate the padded buffer or use an in-place shift. Lower priority than #1.

---

## Inference — Decoder

### 4. Full logits GPU->CPU transfer per step (HIGH IMPACT)
**Location:** `decoder.rs:276`

```rust
let logits_data: Vec<f32> = logits.to_data().to_vec().unwrap();
```

Transfers 8198 f32s (32KB) back to CPU **and forces a full GPU pipeline sync** every decoder step. For 10s audio (~125+ encoder frames), that's 125+ round-trip GPU syncs. Each sync flushes the entire wgpu command queue.

**Fix:** Do argmax on GPU with `Tensor::argmax()`, transfer only 2 scalar indices (token + duration). Still requires a sync per step (autoregressive), but eliminates the 32KB data transfer and moves the argmax computation to GPU.

### 5. LSTM batch-2 padding workaround (MODERATE)
**Location:** `decoder.rs:48-51`

Pads inputs to batch=2 to work around cubecl autotune divide-by-zero with M=1 matmul, then slices back. This doubles matmul compute, adds 2 zeros + 2 cat + 1 slice per LSTM call × 2 layers per step.

**Fix:** Transpose the matmul (put the "1" on N instead of M), or check if newer burn/cubecl versions fix the bug.

### 6. Debug logging in hot loop (LOW)
**Location:** `decoder.rs:282-288`

The `t < 3` conditional + top-5 sort runs every step even when not printing. Minor impact but easy to remove.

---

## Priority Order

| # | Fix | Location | Impact | Risk |
|---|-----|----------|--------|------|
| 1 | Skip folded BN | encoder.rs | HIGH | None (no numerical change) |
| 2 | GPU argmax in decoder | decoder.rs | HIGH | None (no numerical change) |
| 3 | Skip random weight init | mod.rs | HIGH (load time) | Low |
| 4 | Eliminate HashMap / mmap safetensors | weights.rs | HIGH (load time) | Low |
| 5 | Cache positional encoding | encoder.rs | MODERATE | None |
| 6 | Fix LSTM padding | decoder.rs | MODERATE | Low (need to verify cubecl bug) |
| 7 | Optimize rel_shift | attention.rs | MODERATE | Low |
