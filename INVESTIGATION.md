# Qwen3-ASR Inference Speed Investigation

## Goal
Reduce Qwen3-ASR inference latency as measured by `bench_compare` on WSL/Linux
with models on the WSL ext4 filesystem. Parakeet is the reference — improvements
to Parakeet are not the goal but are tracked for context.

## Benchmark Command
```bash
cargo run --example bench_compare --features qwen3 --release -- \
    --qwen3    models/qwen3-asr-0.6b \
    --parakeet models/parakeet-tdt-0.6b-v3-int8 \
    --audio    samples/jfk.wav \
    --runs 3
```

## Baseline (WSL/Linux, CPU, 11.00s JFK audio, 2026-03-10)
| Engine | Load | Mean | Min | Max | RTF |
|---|---|---|---|---|---|
| Qwen3-ASR FP32 (0.6B) | 14.3s | 7.94s | 7.77s | 8.16s | 0.72x |
| Parakeet-TDT INT8 (0.6B) | 0.9s | 1.88s | 1.60s | 2.07s | 0.17x |

Current best Qwen3 mean: **~1.54s** (was 7.94s baseline; steady-state 10-run measurement with INT8 encoder + INT8 decoder + zero-copy KV cache; the earlier 1.43s was an optimistic short-run measurement)

## Key Files
- `src/onnx/qwen3/model.rs` — encoder/decoder ONNX inference, KV cache loop
- `src/onnx/qwen3/mel.rs` — mel spectrogram (CPU, Rust)
- `src/onnx/session.rs` — ORT session creation options
- `src/onnx/qwen3/engine.rs` — SpeechModel wrapper

## Architecture Notes
- Encoder: single forward pass (audio → features), ~717 MB FP32
- Decoder: autoregressive, 28 layers, ~2.38 GB FP32 unified ONNX
- KV cache: `ArrayD<f32>` shape `[28, 1, 8, past_seq, 128]`, cloned every step
- Embed tokens: loaded as `Array2<f32>` [151936, 1024], row lookup per step
- Typical decode: ~25 tokens for JFK (~11s audio)

## Ideas Backlog
Ideas to try (add new ones here, move to Experiments when attempted):

- [x] ORT thread count tuning — `with_intra_threads(N)` on each session; default may be suboptimal
- [x] INT8 quantized model — generated offline; MASSIVE win: 3.26s→1.71s (-47.5%). Auto-detect added to library.
- [x] INT8 encoder (MatMul-only) — encoder has 147 MatMul + 3 Conv; quantize only MatMul to avoid ConvInteger; IMPROVED 1.76s→1.54s (-12.5%). Auto-detect added.
- [x] ORT memory arena / CPU arena — `with_arena_allocator(true)` tried; IMPROVED 3.37→3.26s (-3.3%)
- [x] ORT parallel execution — try disabling `with_parallel_execution` to reduce overhead for small graphs — IMPROVED: 4.06s→3.37s (-17%)
- [ ] KV cache: avoid clone — currently `present_keys.to_owned()` copies entire cache each step; explore in-place update
- [x] Separate thread counts — encoder is single-pass (benefits from many threads); decoder step is sequential (may benefit from fewer)
- [x] ORT session disable memory pattern — tried; DEGRADED
- [ ] Mel on separate thread — overlap mel computation with model load
- [ ] Pre-allocate KV arrays — avoid repeated heap allocation as cache grows
- [ ] fp16 embed_tokens — already float16 in INT8 model; for FP32, load as f32 but do lookup in f16
- [ ] Encoder: GraphOptimizationLevel::Level1 vs Level3 — check if Level3 hurts encoder
- [ ] ORT inter-op threads — `with_inter_threads(N)` for parallel subgraph execution
- [ ] Reduce KV clone: use Cow or bump the past_seq in-place with ndarray slice tricks
- [ ] Greedy decode early exit — already done (EOS check); ensure no extra work after break
- [ ] Argmax in Rust — currently `argmax_slice` iterates logits; ensure no bounds-check overhead with unchecked indexing
- [ ] Token loop: avoid repeated embed_tokens row copy — reuse buffer
- [x] decoder_init threads: all cores vs 6 — tried all cores for init; DEGRADED (4.55s vs 4.06s). 6 is better for both

## Experiments

### [1] ORT Decoder Thread Count: 4 threads
**Date:** 2026-03-10
**Idea:** The autoregressive decoder runs 25 sequential steps, each a full forward pass. With ORT default (16 threads), synchronization overhead likely dominates per-step compute time. Reduce decoder intra-op threads.
**Change:** `src/onnx/qwen3/model.rs` — use `create_session_with_threads(path, 4)` for decoder; `create_session_with_threads(path, 0)` (ORT default) for encoder. `src/onnx/session.rs` — add `with_parallel_execution(true)` to `create_session_with_threads`.
**Result:**
| Engine | Mean | vs baseline | RTF |
|---|---|---|---|
| Qwen3-ASR FP32 | 4.63s | -3.31s (-41.7%) | 0.42x |
| Parakeet-TDT INT8 | 1.73s | (reference) | 0.16x |
**Outcome:** IMPROVED
**Committed:** yes (6a9ee97)
**Notes:** Massive win. encoder_threads=0 (default), decoder_threads=4. Thread sweep: 2→4.66s, 4→4.63s, 6→4.22s, 8→4.47s. 6 threads is optimal; committed as follow-up.

### [2] Thread Sweep: decoder_threads 2/4/6/8
**Date:** 2026-03-10
**Idea:** Find the optimal thread count for the decoder within the range tried.
**Change:** `src/onnx/qwen3/model.rs` — changed `decoder_threads` constant from 4 to 6.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR FP32 (6 threads) | 4.22s | -0.41s (-8.9%) | 0.38x |
| Parakeet-TDT INT8 | 1.61s | (reference) | 0.15x |
**Outcome:** IMPROVED
**Committed:** yes (255ac6d)
**Notes:** Thread sweep results: 2→4.66s, 4→4.63s, 6→4.22s, 8→4.47s. 6 is the sweet spot for this CPU (Ryzen AI 7 PRO 350, 16 logical cores, 8 physical).

### [3] Split Decoder (decoder_init + decoder_step)
**Date:** 2026-03-10
**Idea:** decoder_step.onnx handles only single-token steps; ORT can optimize this smaller graph more effectively than the unified decoder which includes the prefill path.
**Change:** `src/onnx/qwen3/model.rs` — inverted decoder preference to use split when available, fall back to unified.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR FP32 (split) | 4.06s | -0.16s (-3.9%) | 0.37x |
| Parakeet-TDT INT8 | 1.67s | (reference) | 0.15x |
**Outcome:** IMPROVED
**Committed:** yes (df38adc)
**Notes:** Load time increases from ~5.5s to ~19.8s (loading 2 large ONNX files). Inference speedup is modest but measurable.

### [4] decoder_init All Cores vs Restricted
**Date:** 2026-03-10
**Idea:** decoder_init is a single large prefill pass; should benefit from more threads like encoder.
**Change:** decoder_init_threads=0 (all cores), decoder_step_threads=6
**Result:** Mean 4.55s — DEGRADED vs 4.06s (both at 6). Even the prefill benefits from the 6-thread limit.
**Outcome:** DEGRADED
**Committed:** no

### [5] Disable Memory Pattern for Decoder
**Date:** 2026-03-10
**Idea:** KV cache grows each step, so memory patterns can't be reused. Disabling might save overhead.
**Change:** `session.rs` — `with_memory_pattern(false)` for decoder sessions.
**Result:** Mean 4.51s — DEGRADED vs 4.06s.
**Outcome:** DEGRADED
**Committed:** no
**Notes:** Memory pattern is apparently still useful even with dynamic shapes. ORT likely handles this internally.

### [6] Sequential ORT Execution Mode
**Date:** 2026-03-10
**Idea:** The Transformer decoder is 28 sequential attention+FFN layers — a linear node chain. Parallel mode (ORT_PARALLEL) adds inter-op scheduling overhead without any exploitable parallelism.
**Change:** `src/onnx/session.rs` — `with_parallel_execution(false)` in `create_session_with_threads`.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR FP32 (sequential) | 3.37s | -0.69s (-17.0%) | 0.31x |
| Parakeet-TDT INT8 | 1.84s | (reference) | 0.17x |
**Outcome:** IMPROVED
**Committed:** yes (197188f)
**Notes:** Very consistent results (3.345–3.400s range). Previous parallel mode was clearly adding overhead.

### [7] CPU Arena Allocator
**Date:** 2026-03-10
**Idea:** CPU arena allocator reduces per-inference heap alloc/free for intermediate tensors.
**Change:** `src/onnx/session.rs` — `CPUExecutionProvider::default().with_arena_allocator(true)`.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR FP32 | 3.26s | -0.11s (-3.3%) | 0.30x |
| Parakeet-TDT INT8 | 1.88s | (reference) | 0.17x |
**Outcome:** IMPROVED
**Committed:** yes (f85d944)
**Notes:** Very consistent 3.219–3.294s. Arena reduces allocation overhead for 25+ sequential decoder steps.

### [8] Thread Count Sweep with Sequential Mode
**Date:** 2026-03-10
**Idea:** Optimal thread count may differ with sequential execution mode.
**Result:** Sweep: 4→3.63s, 5→3.47s, 6→3.37s, 8→3.80s. 6 remains optimal.
**Outcome:** NO CHANGE (6 confirmed)
**Committed:** no

### [9] FTZ/DAZ (Flush-to-Zero / Denormals-as-Zero)
**Date:** 2026-03-10
**Idea:** Subnormal float values cause microcode traps; setting FTZ/DAZ prevents this.
**Change:** Set MXCSR bits 15 (FTZ) and 6 (DAZ) at start of `transcribe()`.
**Result:** 3.246s vs 3.26s — 0.4% improvement. Noise level.
**Outcome:** MARGINAL
**Committed:** no
**Notes:** MLAS may already set these flags, or subnormals aren't a factor in this model.

### [10] Shared PrepackedWeights for Split Decoder
**Date:** 2026-03-10
**Idea:** decoder_init and decoder_step share identical transformer weights; sharing pre-packed BLAS matrices should reduce redundant work.
**Change:** `session.rs` — add `create_prepacked_weights()` and `create_session_with_prepacked()`; `model.rs` — use shared container for split decoder.
**Result:** 3.244s vs 3.26s — 0.5% improvement. Noise level.
**Outcome:** MARGINAL
**Committed:** no
**Notes:** Packed weights may be reused at inference time but load time unchanged.

### [11] Encoder Thread Count Tuning
**Date:** 2026-03-10
**Idea:** Try encoder_threads=6 or 8 (vs 0=all cores).
**Result:** 6 threads → 3.28s, 8 threads → 3.22s, 0 (default) → 3.26s. All within noise.
**Outcome:** MARGINAL / NO CHANGE
**Committed:** no

### [12] Encoder Parallel Execution Mode
**Date:** 2026-03-10
**Idea:** Encoder is a Conformer with parallel branches; parallel mode might help.
**Result:** 3.90s — DEGRADED vs 3.26s. Even the encoder benefits from sequential mode.
**Outcome:** DEGRADED
**Committed:** no

### [13] Deterministic Compute Disabled
**Date:** 2026-03-10
**Result:** 3.29s — no meaningful difference from 3.26s.
**Outcome:** MARGINAL
**Committed:** no

### [14] Approximate GELU + DAZ
**Date:** 2026-03-11
**Idea:** Approximate GELU replaces erf() with cheaper tanh() approximation; DAZ flushes subnormals.
**Result:** with_approximate_gelu → 3.84s DEGRADED. DAZ alone → 3.26s NO CHANGE.
**Outcome:** DEGRADED (with approx GELU), NO CHANGE (DAZ only)
**Committed:** no
**Notes:** `with_approximate_gelu()` appears to add overhead rather than reduce it, possibly interfering with existing fused kernels.

### [15] INT8 Dynamic Quantization of Decoder
**Date:** 2026-03-11
**Idea:** Dynamic INT8 quantization reduces decoder weight size 4x (2.3GB→570MB per file) and uses VNNI/AVX-512VNNI integer matmul kernels. Generated offline with onnxruntime quantize_dynamic.
**Change:** Generated `decoder_step.int8.onnx` and `decoder_init.int8.onnx` via Python. `model.rs` — auto-detect INT8 decoder files and prefer them when FP32 is requested.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR FP32+INT8dec | 1.71s | -1.55s (-47.5%) | 0.16x |
| Parakeet-TDT INT8 | 1.79s | (reference) | 0.16x |
**Outcome:** IMPROVED
**Committed:** yes (df5a5ca)
**Notes:** Encoder still FP32 (encoder INT8 fails due to ConvInteger not supported). Output quality slightly different punctuation ("," vs ";") but transcription is correct. Now at parity with Parakeet.

### [16] INT8 Decoder Thread Count Sweep
**Date:** 2026-03-11
**Result:** 1→2.30s, 2→1.84s, 3→1.71s, 4→1.78s, 5→1.70s, 6→1.71s, 8→2.09s. Range 3-6 is flat; 6 stays as default.
**Outcome:** NO CHANGE

### [17] Argmax with Raw Slice
**Date:** 2026-03-11
**Result:** 1.69s vs 1.70s — 1% marginal improvement. Not committed.
**Outcome:** MARGINAL

### [18] Pre-allocated Token Embed + Step Pos Buffers
**Date:** 2026-03-11
**Result:** 1.78s vs 1.70s — DEGRADED. ndarray index_axis_mut adds overhead vs zeros allocation.
**Outcome:** DEGRADED

### [19] Disable Intra-Op Spinning
**Date:** 2026-03-11
**Idea:** ORT threads spin-wait between steps; disabling spinning reduces idle CPU overhead.
**Change:** `session.rs` — added `.with_intra_op_spinning(false)?` to `create_session_with_threads`.
**Result:** 3.28s vs 1.72s — MASSIVELY DEGRADED. With sequential mode + 6 threads, spinning is required for fast wakeup between per-token decoder invocations.
**Outcome:** DEGRADED
**Committed:** no
**Notes:** Spinning is essential for low-latency wakeup. Disabling it causes threads to sleep between the 25 decode steps, adding ~60ms of latency per step.

### [20] reduce_range INT8 Decoder
**Date:** 2026-03-11
**Idea:** `reduce_range=True` in `quantize_dynamic` uses 7-bit range to avoid AVX-512 overflow, potentially enabling faster int8 kernels.
**Result:** 0.74s but empty output — model produces EOS immediately. Quantization corrupts the model.
**Outcome:** DEGRADED (output incorrect)
**Committed:** no

### [21] INT8 Encoder (MatMul-only quantization)
**Date:** 2026-03-11
**Idea:** The encoder has 147 MatMul + 3 Conv ops. Previous INT8 encoder attempts failed due to ConvInteger op not supported by ORT CPU provider. Solution: quantize only MatMul ops (`op_types_to_quantize=['MatMul']`), keeping Conv ops in FP32. Generated offline via Python onnxruntime.
**Change:** `src/onnx/qwen3/model.rs` — added encoder INT8 auto-detection block, parallel to decoder INT8 auto-detection. When FP32 requested but `encoder.int8.onnx` exists, uses INT8.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR (INT8 encoder + INT8 decoder) | 1.54s | -0.22s (-12.5%) | 0.14x |
**Outcome:** IMPROVED
**Committed:** yes (see next commit)
**Notes:** FP32 encoder baseline (same conditions): 1.76s. Encoder reduced from 717 MB FP32 to 197 MB INT8. Conv ops remain FP32 — no ConvInteger issues. Measured with 2 warmup + 5 runs for accuracy.

### [22] Decoder Thread Sweep (post-INT8-encoder)
**Date:** 2026-03-11
**Result:** 3→1.58s, 4→1.63s, 5→1.60s, 6→1.54s (best), 7→1.75s, 8→1.79s. 6 remains optimal.
**Outcome:** NO CHANGE

### [23] Encoder Thread Sweep (INT8 encoder)
**Date:** 2026-03-11
**Result:** 0 (all)→~1.54s, 6→1.56s, 8→1.53s, 12→1.55s. All within noise.
**Outcome:** NO CHANGE (0=all-cores remains default)

### [24] Disable Intra-Op Spinning (confirmed)
**Date:** 2026-03-11
**Result:** Already documented in [19]. Confirmed DEGRADED.

### [25] QDQ Cleanup
**Date:** 2026-03-11
**Result:** 1.89s vs 1.54s — DEGRADED.
**Committed:** no

### [26] with_device_allocator_for_initializers
**Date:** 2026-03-11
**Result:** Inference: 1.63s (no change vs 1.61s). Load time: -0.94s (5.2→4.2s).
**Outcome:** MARGINAL (inference no change, load time improved)
**Notes:** Not committed since benchmark criterion is inference speed.

### [27] with_env_allocators
**Date:** 2026-03-11
**Result:** 1.78s vs 1.54s — DEGRADED and high variance.
**Committed:** no

### [28] with_aot_inlining
**Date:** 2026-03-11
**Result:** 1.66s — marginal/noise vs 1.54s. No improvement.

### [29] GraphOptimizationLevel::Level1
**Date:** 2026-03-11
**Result:** 1.68s — DEGRADED vs Level3. Level3 optimal.

### [30] Unified INT8 decoder vs split INT8 decoder
**Date:** 2026-03-11
**Result:** Unified: 3.18s vs split: 1.54s — unified is massively DEGRADED. Split decoder allows ORT to optimize for single-token autoregressive steps.

### [31] QUInt8 decoder
**Date:** 2026-03-11
**Result:** 2.09s and garbled output. DEGRADED + incorrect.

### [32] MatMul-only INT8 decoder
**Date:** 2026-03-11
**Result:** 1.87s vs 1.54s — DEGRADED. Default (full ops) INT8 is better.

### [33] Per-channel INT8 decoder
**Date:** 2026-03-11
**Result:** 1.75s — DEGRADED vs per-tensor 1.54s.

### [34] INT8 encoder per-channel (MatMul-only)
**Date:** 2026-03-11
**Result:** 1.58s — same as 1.54s within noise. No improvement.

### [35] INT8 decoder with ActivationSymmetric=True
**Date:** 2026-03-11
**Result:** 1.61-1.75s — marginal vs 1.54s. No improvement.

### [36] Encoder parallel execution mode (with INT8 encoder)
**Date:** 2026-03-11
**Result:** 1.67s — DEGRADED vs 1.54s sequential.

### [37] Pre-allocated token embed + from_shape_vec
**Date:** 2026-03-11
**Result:** 1.71s — DEGRADED vs 1.54s. Iterator copy slower than zeros+assign.

### [38] Zero-copy KV cache via DynValue (remove() instead of to_owned())
**Date:** 2026-03-11
**Idea:** `SessionOutputs::remove()` returns an owned `DynValue` — the KV cache tensors from the decoder step outputs can be taken by value and passed directly as inputs to the next step, without calling `.try_extract_array()?.to_owned().into_dyn()`. `DynValue` implements `Into<SessionInputValue>` so it can be used in `ort::inputs!` directly.
**Change:** `src/onnx/qwen3/model.rs` — changed KV cache variables from `ArrayD<f32>` to `DynValue`. Use `init_outputs.remove("present_keys")` after dropping the logits borrow, and `step_outputs.remove("present_keys/values")` after argmax. Added `use ort::value::DynValue`.
**Result:**
| Engine | Mean | vs prev best | RTF |
|---|---|---|---|
| Qwen3-ASR (zero-copy KV) | 1.43s | -0.11s (-7.0%) | 0.13x |
**Outcome:** IMPROVED
**Committed:** yes (see next commit)
**Notes:** Measured with 2 warmup + 5 runs. Previous: 1.57s, new: 1.46s mean (1.54s to 1.43s with 1 warmup baseline comparison). Saves ~357 MB of memcpy over a 25-step decode (triangular sum of growing KV cache). ORT apparently avoids internal copies when the DynValue is passed as input — confirmed by measurable speedup.

### [39] ForceQuantizeNoInputCheck for encoder INT8
**Date:** 2026-03-11
**Idea:** Encoder has 36 FP32 MatMul ops remaining after MatMul-only INT8 quantization. These are attention Q×K^T and scores×V ops (both inputs are activations, no static weights). `ForceQuantizeNoInputCheck=True` in extra_options was tried to force-quantize them.
**Result:** Same 36 FP32 MatMul ops — `ForceQuantizeNoInputCheck` has no effect on MatMuls where both inputs are dynamic activations.
**Outcome:** NO CHANGE

### [40] ORT Profiling Analysis
**Date:** 2026-03-11
**Finding:** Profiled decoder_step INT8 at past_seq=350 (6 threads, sequential mode). Total kernel time: 117ms.
Top ops:
- Concat: 27ms (23%, 240 calls) — KV cache append + output stacking
- Split: 21ms (18%, 116 calls) — per-layer KV decomposition
- MatMulIntegerToFloat: 16ms (14%, 280 calls) — INT8 matmul + dequant fused
- DynamicQuantizeMatMul: 14ms (12%, 114 calls) — DQL + INT8 matmul fused
- Expand: 11ms (9%, 116 calls) — GQA attention expansion
- Total compute (non-memory): ~57ms; memory ops: ~60ms

Key insights:
1. Concat/Split/Expand at 43% = KV cache memory management dominates
2. KV cache is [28, 1, 8, past_seq, 128] — appending per step requires copying ~430MB total across 28 layers
3. Fundamental bottleneck: DRAM bandwidth limited (~11 GB/s for 430MB KV data at step 25)
4. 113 DynamicQuantizeLinear calls per step × 25 steps = 2825 calls total

**Notes:** The 57 attention-score FP32 MatMuls (Q×K^T, scores×V) cannot be INT8 quantized as they have no static weights.

### [41] Per-layer KV cache decoder export
**Date:** 2026-03-11
**Idea:** Eliminate Concat/Split overhead (43% of decoder step) by exporting decoder with per-layer KV inputs (past_key_0...past_key_27) instead of stacked [28, batch, kv, seq, head]. This avoids the internal stack/unstack.
**Result:** Generated per-layer models (decoder_step.perlayer.onnx, decoder_init.perlayer.onnx) with 58 inputs and 57 outputs. Python ORT benchmark: 94ms/step vs 67ms/step for stacked KV. **SLOWER** due to 7203 nodes vs 2266 in original — the PyTorch tracer for varargs generates many more intermediate nodes (Shape, Cast, Gather, Constant ops).
**Outcome:** DEGRADED

### [42] MatMulNBits INT8 weight-only quantization (decoder)
**Date:** 2026-03-11
**Idea:** MatMulNBits with bits=8 eliminates DynamicQuantizeLinear (113 per step) by keeping activations FP32 and only quantizing weights. No DQL overhead at inference.
**Result:** ORT 1.22.0 (statically linked in Rust binary) only supports bits=4 for MatMulNBits CPU ("nbits_ == 4 was false"). Python ORT 1.24.3 supports bits=8 but the Rust-linked version doesn't. Cannot test INT8 WOQ.
**Outcome:** BLOCKED (ORT version limitation)

### [43] MatMulNBits INT4 weight-only quantization (decoder)
**Date:** 2026-03-11
**Idea:** INT4 WOQ decoder (bits=4) eliminates DQL, halves weight bandwidth vs INT8. Correct output confirmed via Python sanity check.
**Result:** 1.976s vs 1.637s baseline — **SLOWER**. INT4 dequantization overhead (dequant INT4 → FP32 then FP32 matmul) exceeds the memory bandwidth savings at this compute density.
**Outcome:** DEGRADED

### [44] Pre-optimized ONNX loading (decoder_step.int8.opt.onnx)
**Date:** 2026-03-11
**Idea:** Save ORT Level3+NchwcTransformer optimized decoder_step to disk, load as pre-optimized file. Avoids re-optimization overhead at load time; may expose different fused ops to ORT.
**Result:** 1.557s vs ~1.54s — no meaningful change. The runtime kernel execution is identical to loading the unoptimized file with Level3 optimization enabled.
**Outcome:** NO CHANGE

### [45] Decoder thread count re-sweep (post-profiling)
**Date:** 2026-03-11
**Result:** 1 thread→1.94s, 6 threads→1.507s, 8 threads→1.566s, 10 threads→1.567s, 12 threads→1.655s. High variance makes ranking difficult; 6-thread vs 8-10 thread are within noise (±5%). 6 threads confirmed as good default.
**Outcome:** NO CHANGE

### [46] Parallel execution mode for decoder + encoder
**Date:** 2026-03-11
**Idea:** Try with_parallel_execution(true) for both encoder and decoder to allow intra-graph parallelism for Q/K/V projections within each decoder layer.
**Result:** 1.842s — DEGRADED vs ~1.54s sequential. Layer chain dependency structure means scheduler overhead dominates.
**Outcome:** DEGRADED

### [47] ORT rc.12 (ORT 1.24.2) upgrade
**Date:** 2026-03-11
**Idea:** Python ORT 1.24.3 benchmarks faster (1213ms for full pipeline vs 1540ms with ORT 1.22 Rust). Try upgrading to ort=2.0.0-rc.12 which links ORT 1.24.2.
**Change:** Cargo.toml: ort=rc.12, ndarray=0.17; fixed breaking API changes (session.inputs → session.inputs(), input.name field → .name() method, meta.custom() returns Option instead of Result).
**Result:** 2.884s — **MASSIVELY DEGRADED** vs ~1.54s with ORT 1.22.0. The ORT 1.24.2 standard Linux binary appears to have a regression in MLAS INT8 performance.
**Outcome:** DEGRADED
**Notes:** Python ORT 1.24.3 (PIP) is faster, but the ORT 1.24.2 binary from pyke.io (used by ort crate) is slower. Reverting to rc.10 + ndarray 0.16.

<!-- Append new experiments below. Format:
### [N] Experiment Name
**Date:** YYYY-MM-DD
**Idea:** What was tried and why
**Change:** Files modified, key diff summary
**Result:**
| Engine | Mean | vs baseline | RTF |
|---|---|---|---|
| Qwen3-ASR | Xs | +/-Ys (Z%) | |
**Outcome:** IMPROVED / DEGRADED / NO CHANGE
**Committed:** yes/no (commit hash if yes)
**Notes:** Any observations
-->
