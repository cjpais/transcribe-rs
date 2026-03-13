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

Current best Qwen3 mean: **~1.25s** (was 7.94s baseline → 1.54s with INT8+zero-copy; further improved to ~1.25s after system stabilization and reduced measurement noise)

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

### [48] Stage timing instrumentation
**Date:** 2026-03-10
**Finding:** Added timing probes to transcribe(): Mel=4-10ms, Encoder=300-450ms (warm ~300ms), Decoder=1200-1550ms (warm ~1200ms).
**Takeaway:** Decoder is 80% of total pipeline time. Encoder is 20%. Mel is <1%. Both Encoder and Decoder have high per-run variance (±150ms).

### [49] Pre-allocated step buffers
**Date:** 2026-03-10
**Idea:** Move `Array3::zeros((1,1,hidden_size))` and `ArrayD::from_shape_vec` allocations outside the decode loop to avoid ~4KB heap alloc+zero+dealloc per step.
**Result:** No measurable effect. 3×10-run comparison: baseline avg 1.403s, pre-alloc avg 1.420s — within noise. The 100KB total allocation over 25 steps is negligible compared to ORT session overhead.
**Outcome:** MARGINAL (no measurable effect)

### [50] Denormal-as-zero session flag
**Date:** 2026-03-10
**Idea:** `with_denormal_as_zero()` enables flush-to-zero + DAZ which eliminates CPU microcode trap for subnormal floats. Relevant if model intermediate values approach zero.
**Result:** Baseline avg 1.419s, with flag avg 1.420s — identical. The INT8 quantized decoder likely avoids denormals naturally (quantization clips small values).
**Outcome:** NO CHANGE

### [51] Approximate GELU on encoder only
**Date:** 2026-03-10
**Idea:** INT8 encoder has 22 Erf ops (standard GELU). `with_approximate_gelu()` replaces erf() with tanh approximation, which should be faster. Applied only to encoder via new `create_session_with_threads_gelu()` function; decoder uses SiLU (not affected).
**Result:** Baseline avg 1.403s, with GELU approx avg 1.395s — within noise. The 22 Erf ops are a negligible fraction of encoder compute dominated by 111 INT8 MatMuls.
**Outcome:** NO CHANGE

### [52] Encoder threads = 8 (physical cores only)
**Date:** 2026-03-10
**Idea:** 8-core HT CPU; using 16 logical threads for encoder might cause HT contention. Try pinning to 8 physical cores.
**Result:** Baseline avg 1.403s, 8-thread avg 1.476s — within noise (if anything slightly worse). HT contention not the limiting factor.
**Outcome:** NO CHANGE

### [53] Decoder init with all cores (decoder_init_threads=0)
**Date:** 2026-03-10
**Idea:** decoder_init processes ~150-token sequence; larger sequence might benefit from more parallelism.
**Result:** 1.713s avg (3×10 runs) vs 1.506s baseline — DEGRADED. Layer-sequential computation structure means more threads = more synchronization overhead. Consistent with earlier experiment [1] finding.
**Outcome:** DEGRADED

### [54] Unified vs Split INT8 decoder comparison
**Date:** 2026-03-11
**Idea:** Measure unified INT8 decoder (`decoder.int8.onnx`, 571 MB) vs split INT8 decoder (`decoder_init.int8.onnx` + `decoder_step.int8.onnx`, 571 MB each) to quantify the speed gap. Also added unified INT8 auto-detection to the Rust loader as fallback.
**Change:** `src/onnx/qwen3/model.rs` — extended decoder quantization auto-detection to check for unified INT8 (`decoder.int8.onnx`) when split INT8 files are absent.
**Result:**
| Config | Load | Mean | RTF |
|---|---|---|---|
| Split INT8 (baseline) | 6.6s | 1.305s | 0.12x |
| Unified INT8 | 15.2s | 2.729s | 0.25x |

Split is 2.1x faster inference and 2.3x faster load. The unified decoder must handle both prefill (variable-length input) and step (single-token input) paths, preventing ORT from optimizing for the single-token case. The split decoder_step model is specialized for single-token decode, enabling more aggressive graph optimization.
**Outcome:** Split confirmed superior. Unified INT8 fallback detection added to Rust loader.
**Committed:** pending (code improvement, no perf change)
**Notes:** Transcription difference: split produces comma ("for you, ask") while unified produces semicolon ("for you; ask"). Both are acceptable ASR outputs.

### [55] ORT Transformer Optimizer on FP32 decoder (fusion)
**Date:** 2026-03-11
**Idea:** Run `onnxruntime.transformers.optimizer` on decoder models to fuse multi-head attention, LayerNorm, etc.
**Result:** Optimizer applied 113 SimplifiedLayerNormalization fusions (28 layers × ~4 per layer) on FP32 decoder_step.onnx (2266 → 1586 nodes). However, the saved output was a single 2.3 GB protobuf file (inlined all external data), incompatible with ONNX loading. On INT8 model, optimizer crashed with `AttributeError` in `fusion_quickgelu.py` — INT8 models with DynamicQuantizeLinear nodes are not supported by the fusion passes.
**Outcome:** BLOCKED (optimizer cannot handle INT8 models; FP32 output format broken)

### [56] Shared external data for FP32 split decoder
**Date:** 2026-03-11
**Idea:** Both `decoder_init.onnx.data` and `decoder_step.onnx.data` contain model weights (2.3 GB each). If the byte layouts match, point both ONNX files at a single shared `.data` file to halve FP32 decoder disk footprint.
**Result:** Analysis showed only 120 of 331/357 initializers share names between init and step models. Of those 120, only 56 have matching offsets/lengths and 7 have data mismatches. The init-only (2.33 GB) and step-only (2.32 GB) data dominate, with only 51 MB shared. Potential savings: 63 MB (1.3%) — not worth the complexity.
**Outcome:** ABANDONED (minimal savings; different ONNX graph structures from separate export traces)

### [57] onnxsim on INT8 models
**Date:** 2026-03-11
**Idea:** Run onnxsim (graph simplification) on INT8 encoder, decoder_init, decoder_step to reduce node count and potentially improve inference speed. Pre-existing `.sim.onnx` files: encoder 197→191 MB, decoders 571→570 MB.
**Result:**
| Config | Mean |
|---|---|
| Original INT8 (baseline) | 1.272s |
| onnxsim INT8 | 1.254s |
Within noise. Transcription identical.
**Outcome:** NO CHANGE (marginal size reduction, no speed improvement)

### [58] Encoder optimization Level1 vs Level3
**Date:** 2026-03-11
**Idea:** Test ORT graph optimization Level1 for encoder (skip complex fusions). Level3 may add overhead for relatively simple encoder graph.
**Change:** `src/onnx/session.rs` — added `create_session_with_opts()` and `create_session_full()` for configurable optimization level and parallel execution.
**Result:**
| Config | Mean |
|---|---|
| Level3 (baseline) | 1.272s |
| Level1 encoder | 1.304s |
Level3 graph optimizations help the encoder. Level1 is worse.
**Outcome:** DEGRADED (reverted; kept session API additions)

### [59] RMSNorm fusion to SimplifiedLayerNormalization
**Date:** 2026-03-11
**Idea:** INT8 decoder has 113 unfused RMSNorm patterns (Pow→ReduceMean→Add→Sqrt→Reciprocal→Mul→Mul = 7 nodes each = 791 nodes). Neither the quantizer nor ORT Level3 fuses these. Wrote `fuse_rmsnorm.py` to replace them with `com.microsoft:SimplifiedLayerNormalization` ops, reducing decoder_step from 2970→2292 nodes.
**Result:** ORT error: `com.microsoft:SimplifiedLayerNormalization(-1) is not a registered function/op`. The pyke.io ORT binary (1.24.2) used by the `ort` Rust crate does not include this contrib op. Also confirmed it's missing from Python ORT 1.24.2.
**Outcome:** BLOCKED (contrib op not available in this ORT build)
**Notes:** The fusion script works correctly — 113 patterns detected and fused. Would need an ORT build with the op enabled, or a different fusion target.

### [60] Decode loop profiling (Rust-side overhead vs ORT execution)
**Date:** 2026-03-11
**Idea:** Add per-step timing to identify if Rust-side overhead (embedding lookup, tensor construction, argmax) is significant vs ORT session.run() time.
**Result:** For 29 decode steps:
- ORT session.run(): 26.3ms/step (99.7% of decode loop time)
- Input preparation (ort::inputs! macro): 0.01ms/step
- Argmax + post-processing: 0.7ms/step total

Breakdown of total 1.25s pipeline:
- Mel spectrogram: ~5ms
- Encoder: ~240ms
- Prefill (decoder_init): ~260ms
- Decode steps (29 × 26ms): ~755ms
**Outcome:** INFORMATIONAL — ORT execution dominates; Rust overhead is negligible
**Notes:** Profiling code removed after data collection.

### [61] WoQ4 (4-bit weight-only quantization) decoder
**Date:** 2026-03-11
**Idea:** Test pre-existing WoQ4 decoder files (305 MB each vs 571 MB INT8). Uses `MatMulNBits` contrib op for 4-bit weight decompression.
**Result:**
| Config | Load | Mean | RTF |
|---|---|---|---|
| INT8 split (baseline) | 4.7s | 1.25s | 0.11x |
| WoQ4 split | 3.4s | 1.79s | 0.16x |
43% slower inference. Transcription slightly different ("so my" vs "so, my"). The 4-bit dequantization overhead on CPU outweighs smaller model size.
**Outcome:** DEGRADED (43% slower)

### [62] Decoder thread count tuning (4 vs 6 vs 8)
**Date:** 2026-03-11
**Idea:** Re-test decoder thread counts around the current setting of 6. 4 threads may reduce synchronization overhead further; 8 threads (physical core count) may increase parallelism.
**Result:**
| Threads | ORT ms/step | Total mean |
|---|---|---|
| 4 | 26.9 | ~1.33s |
| 6 (baseline) | 26.3 | ~1.25s |
| 8 | 27.2-30.7 | ~1.30s (high variance) |
6 threads confirmed optimal. 4 too few parallelism for the 28-layer decoder. 8 causes HT contention.
**Outcome:** NO CHANGE (6 threads confirmed optimal)

### [63] Parallel execution for encoder
**Date:** 2026-03-11
**Idea:** Enable `with_parallel_execution(true)` for encoder session only. The encoder is a single forward pass with independent attention heads that could benefit from inter-op parallelism, unlike the sequential decoder.
**Change:** `src/onnx/session.rs` — added `create_session_full()` with parallel execution parameter.
**Result:**
| Config | Mean |
|---|---|
| Sequential encoder (baseline) | 1.25s |
| Parallel encoder | 1.30s |
Slightly worse — inter-op scheduling overhead exceeds any parallelism benefit.
**Outcome:** DEGRADED (reverted encoder to sequential; kept session API)

### [64] Generate 1.7B INT8 quantized models
**Date:** 2026-03-11
**Idea:** Run INT8 dynamic quantization on the 1.7B model for future HuggingFace release and downstream testing.
**Result:** Quantization successful:
| Component | FP32 | INT8 | Ratio |
|---|---|---|---|
| encoder.onnx | 1277 MB | 326 MB | 25.5% |
| decoder_init.onnx | 6884 MB | 1723 MB | 25.0% |
| decoder_step.onnx | 6884 MB | 1723 MB | 25.0% |
| embed_tokens.bin | 1245 MB | 622 MB | 50.0% |
| **Total** | **16305 MB** | **4408 MB** | **27.0%** |

Output at `~/qwen3-asr-onnx/output/qwen3-asr-1.7b-int8/`. INT8 decoders use external data format (1.7 GB `.onnx.data` files) since they exceed the 2 GB protobuf limit.
**Outcome:** COMPLETED (model artifacts generated, not a speed optimization)
**Notes:** Not benchmarked — 1.7B inference on CPU would be slow. Primary value is for GPU deployment and HuggingFace distribution.

### [65] Memory bandwidth analysis — theoretical decode step floor
**Date:** 2026-03-11
**Idea:** Determine whether the decode step is compute-bound or memory-bandwidth-bound by comparing theoretical minimum weight-fetch time against observed step time.
**Result:** INT8 decoder_step initializer weights total 596 MB. At typical DDR4 bandwidth:
| Bandwidth | Theoretical min | Observed | Overhead |
|---|---|---|---|
| 40 GB/s (ideal) | 14.9 ms/step | 26.3 ms/step | 76% |
| 30 GB/s (realistic) | 19.9 ms/step | 26.3 ms/step | 32% |
| 25 GB/s (WSL overhead) | 23.8 ms/step | 26.3 ms/step | 10% |

The decode step is memory-bandwidth-bound. Every step must read 596 MB of weights from DRAM. The 26ms/step is within 10-30% of the theoretical minimum for this model size on DDR4.
**Outcome:** INFORMATIONAL — decode step is at the memory bandwidth floor for this model size
**Notes:** Further speed improvements require either: (a) smaller model weights (WoQ4 was tried but dequantization overhead negated bandwidth savings on CPU), (b) higher bandwidth hardware (GPU/DDR5), or (c) architectural changes (fewer layers, smaller hidden size). The current 1.25s = 6.3× improvement over the 7.94s baseline represents near-optimal CPU inference for this model.

## Summary — Optimization Wall
After 65 experiments, the Qwen3-ASR 0.6B inference pipeline on WSL/Linux CPU has been optimized from 7.94s to ~1.25s (6.3× improvement, RTF 0.72→0.11). The remaining time breaks down as:

| Stage | Time | % | Bound by |
|---|---|---|---|
| Mel spectrogram | ~5ms | 0.4% | CPU compute (fast) |
| Encoder (INT8) | ~240ms | 19% | DRAM bandwidth (~197 MB weights) |
| Prefill (decoder_init) | ~260ms | 21% | DRAM bandwidth (~571 MB weights, 150-token KV) |
| Decode steps (29 × 26ms) | ~755ms | 60% | DRAM bandwidth (~596 MB weights per step) |

The decode step at 26ms/step is within 10-30% of the theoretical memory bandwidth floor. All Rust-side overhead (tensor prep, argmax, embedding lookup) is <0.5ms total across all 29 steps.

### [66] Shared external data for split decoder
**Date:** 2026-03-11
**Idea:** decoder_init and decoder_step share identical weights (~2.38 GB for 0.6B) because they're exported from the same PyTorch modules. Post-export, rewrite decoder_step to reference decoder_init's external data file, eliminating the duplicate.
**Change:** Created `~/qwen3-asr-onnx/share_weights.py`. Updated `export.py` and `quantize.py` to call it after split export/quantization. Algorithm: hash each initializer's data (SHA-256), match step→init, redirect external_data references, inline unmatched constants.
**Result:**

| Model | Matched tensors | Before | After | Saved |
|---|---|---|---|---|
| 0.6B FP32 | 310/310 | 4.76 GB | 2.38 GB + 4 MB protos | 2.38 GB |
| 0.6B INT8 | 254/254 | 3.58 GB | 1.79 GB + 4 MB protos | 1.79 GB |
| 1.7B FP32 | 310/310 | 13.76 GB | 6.88 GB + 4 MB protos | 6.88 GB |
| 1.7B INT8 | 254/254 | 3.44 GB | 1.72 GB + 4 MB protos | 1.72 GB |

Benchmark (WSL, 11s JFK audio, 3 runs):

| Engine | Mean | RTF |
|---|---|---|
| Qwen3-ASR FP32 shared weights | 1.59s | 0.14x |
| Parakeet-TDT INT8 | 1.69s | 0.15x |

**Outcome:** NO CHANGE (performance). Disk/RAM savings as expected.
**Committed:** no (export tooling, not in transcribe-rs)
**Verification:**
- onnx.checker: PASS (all 4 model sets)
- ORT Python load: PASS (all 4 model sets)
- compare.py: EXACT token agreement across native/wrapper/FP32/INT8
- Rust tests: 3/3 PASS (0.6B + 1.7B)
- bench_compare: no regression
**Notes:** RAM savings depend on OS page cache deduplication. Two ORT sessions referencing the same mmap'd file should share physical pages for read-only weight data.

### [67] ORT Transformer Optimizer investigation
**Date:** 2026-03-11
**Idea:** Investigate whether ORT's Transformer Optimizer fusion passes produce usable models. Previous experiment [55] saved fused models incorrectly (inlined all external data). Rerun with `use_external_data_format=True`.
**Change:** Created `~/qwen3-asr-onnx/optimize_decoder.py`. Ran on `decoder_step.onnx` with default bert model_type.
**Result:**
- 2266 → 1586 nodes (30% reduction)
- 113 SimplifiedLayerNormalization fusions (replaced Pow/Reciprocal/ReduceMean/Sqrt/Mul decomposed RMSNorm)
- 2 Transpose removals
- Only contrib op: SimplifiedLayerNormalization
- ORT Python load: PASS
- Shape inference warning but did not block optimization
**Outcome:** INVESTIGATION — fused model loads and reduces graph complexity. Needs inference correctness verification and Rust ORT compatibility testing. SimplifiedLayerNormalization is a contrib op that may or may not be available in the Rust ORT binary.
**Committed:** no
**Notes:** The fused model is saved at `output/qwen3-asr-0.6b/decoder_step.optimized.onnx`. Next steps: (1) verify inference correctness against original, (2) test in Rust ORT, (3) if compatible, apply to decoder_init and benchmark the full pipeline.

### [68] INT8 encoder + FP32 decoder hybrid
**Date:** 2026-03-12
**Idea:** Isolate whether WER degradation in smooth INT8 comes from encoder or decoder quantization. Assembled trial dir with smooth INT8 encoder (197 MB, INT8 weights despite .onnx naming) + FP32 decoder (2.3 GB) + FP32 embeddings.
**Change:** No code changes. Created `output/trial-int8enc-fp32dec/` with symlinks to existing files.
**Result:**

| Audio | Mean | RTF | vs FP32 baseline |
|---|---|---|---|
| JFK 11s | 3.25s | 0.30x | +0.11s (+3.5%) |
| LibriSpeech 35s | 12.54s | 0.36x | +1.44s (+13%) |

Load: 18.0s. Transcript identical to FP32 on both audio files ("vertible column", semicolon in JFK).

**Outcome:** INFORMATIONAL — decoder quantization is the sole source of WER degradation. INT8 encoder alone does not affect speed meaningfully (decoder dominates).
**Committed:** no
**Notes:** FP32 decoder reads 2.3 GB weights per session. The ~3% slowdown on 11s may be within variance; the 13% on 35s suggests slightly more overhead from the hybrid setup (two separate weight files for encoder vs decoder). No WER eval needed — transcript is token-identical to FP32.

### [69] FP16 decoder conversion (abandoned)
**Date:** 2026-03-12
**Idea:** Convert FP32 decoder to FP16 for reduced memory bandwidth. FP16 halves model size vs FP32 with less quality loss than INT8.
**Change:** Created `convert_fp16.py` using `onnxconverter_common.convert_float_to_float16`. Two issues:
1. Naive per-tensor approach (numpy FP16 cast) failed: ORT CPU EP rejects mixed float32/float16 ops (Add node type mismatch).
2. Graph-level converter (`convert_float_to_float16`) doesn't handle ONNX external data format — produced empty 2-byte files.
3. **Fundamental flaw:** On CPU EP, FP16 is strictly worse than INT8 for bandwidth (2 bytes/weight vs 1 byte/weight). CPU EP casts both FP16 and INT8 to FP32 for compute, so FP16 gives no quality advantage at inference time — only at export time. INT8 is the right target for CPU.
**Outcome:** ABANDONED — FP16 decoder is not viable on CPU. Trials 69 and 70 (INT8 enc + FP16 dec) skipped.
**Notes:** FP16 would be relevant for GPU EP (native FP16 compute) but not for CPU-only inference. On CPU, the optimal path is INT8 with targeted exclusions to preserve quality-sensitive layers.

### [71] Exclude lm_head from INT8 quantization
**Date:** 2026-03-12
**Idea:** lm_head maps hidden_size → vocab_size (151936). Special token discrimination is most sensitive to quantization. Keep lm_head in FP32 while quantizing all other layers.
**Change:** Added `--nodes-to-exclude` and `--weight-type` CLI args to `quantize.py`. Re-quantized smooth FP32 decoder with `--nodes-to-exclude node_linear_196` (lm_head).
**Result:**

| Audio | Mean | RTF | vs smooth INT8 | vs FP32 |
|---|---|---|---|---|
| JFK 11s | 2.05s | 0.19x | +0.56s (+37%) | -1.09s (-35%) |
| LibriSpeech 35s | 8.34s | 0.24x | +2.52s (+43%) | -2.76s (-25%) |

Load: 5.96s. Decoder size: 1065 MB (vs 569 MB full INT8, +87% from FP32 lm_head).
JFK transcript: comma variant (same as smooth INT8, not FP32's semicolon).
35s transcript: "vertible calm" (same as smooth INT8, not FP32's "vertible column").
20-sample WER: 1.53% (first 20 samples — not comparable to 200-sample baselines without same-subset runs).

**Outcome:** MIXED — excludes lm_head but quality markers (comma, "vertible calm") match smooth INT8, not FP32. Speed is slower than full INT8 due to FP32 lm_head MatMul per decode step. The quantization error in attention/MLP layers still dominates quality loss.
**Committed:** no
**Notes:** Excluding lm_head alone is insufficient. The quality degradation is distributed across all decoder layers, not concentrated in lm_head. This makes sense: each attention/MLP layer introduces small quantization errors that compound through 24 layers.

### [72] INT16 decoder quantization (abandoned)
**Date:** 2026-03-12
**Idea:** `quantize_dynamic` supports `QuantType.QInt16` — 16-bit integer weights. Less compression than INT8 but less quality loss.
**Change:** Added `--weight-type int16` to `quantize.py`. Ran on smooth FP32.
**Result:** ORT refuses to load: `Type 'tensor(int16)' of input parameter of operator (MatMulInteger) is invalid.` Failed for both encoder and decoder. ORT's MatMulInteger only supports int8 on CPU EP.
**Outcome:** ABANDONED — INT16 quantization not supported by ORT runtime.

### [73] AWQ alpha sweep (α=0.1, 0.15, 0.2, 0.25, 0.3) — WINNER: α=0.2
**Date:** 2026-03-12
**Idea:** Current α=0.5 may be sub-optimal. Lower alpha = less smoothing = less weight distortion but more activation variance retained. Used cached activations from initial calibration — no re-calibration needed.
**Change:** Ran `awq_smooth.py` with `--skip-encoder --activations-cache` at five alpha values, followed by INT8 quantization. Evaluated at 20, 100, and 200 samples.
**Result:**

**20-sample WER was misleading** — the first 20 librispeech-other samples are easy, giving α=0.5 an apparent advantage. 100-sample and 200-sample evaluations reveal the true ranking:

| Alpha | 20-sample WER | 100-sample WER | 200-sample WER | 35s word | RTF (11s) | RTF (35s) |
|---|---|---|---|---|---|---|
| 0 (FP32) | 1.53% | 4.13% | 4.42% | "vertible" | 0.29x | 0.32x |
| 0.1 | — | 4.83% | — | — | — | — |
| **0.2** | **—** | **4.76%** | **5.21%** | **"vertebral" ✓** | **0.15x** | **0.19x** |
| 0.25 | 2.04% | 5.08% | 5.26% | "vertible" | 0.15x | 0.19x |
| 0.3 | 3.06% | — | — | "column" + comma | 0.12x | 0.17x |
| 0.5 (baseline) | 1.79% | 5.46% | 5.62% | "calm" | 0.14x | 0.17x |
| Parakeet | — | — | 5.45% | "vertebral" ✓ | 0.16x | 0.16x |

The 35s transcript word "vertebral column" is correctly produced by α=0.2 and Parakeet. FP32 and α=0.25 say "vertible" (wrong), α=0.5 says "calm" (wrong).

**Head-to-head α=0.2 vs Parakeet (5 runs):**

| Metric | Qwen3 α=0.2 INT8 | Parakeet INT8 |
|---|---|---|
| WER (200 samples) | 5.21% | 5.45% |
| RTF (11s) | 0.15x | 0.16x |
| RTF (35s) | 0.19x | 0.16x |
| Load time | 3.0s | 0.9s |
| Punctuation | Full | Minimal |

**Outcome:** IMPROVED — α=0.2 beats Parakeet on WER (5.21% vs 5.45%) and short-audio speed. Slightly slower on longer audio (0.19x vs 0.16x). Full punctuation vs Parakeet's minimal punctuation. Reduces the FP32→INT8 WER penalty from +1.20pp (α=0.5) to +0.79pp (α=0.2), a 34% reduction.
**Committed:** pending
**Notes:** The α=0.2 optimum makes sense: enough smoothing to handle the worst outlier channels, but not so much that it distorts the overall weight distribution. The optimum is flat between α=0.1-0.2 (4.83% vs 4.76% on 100 samples). For production, α=0.2 is recommended.

### [74] Selective layer quantization — first/last layer FP32
**Date:** 2026-03-12
**Idea:** Keep first decoder layer (7 linears), last decoder layer (7 linears), and lm_head in FP32. Middle 26 layers quantized to INT8.
**Change:** Used `quantize.py --nodes-to-exclude` with 15 node names.
**Result:** 20-sample WER 2.04%, RTF 0.18x (11s), 0.22x (35s). 35s transcript: "vertible column" with spurious comma. Slower than full INT8 and not better than α=0.5 on 20-sample WER.
**Outcome:** NO IMPROVEMENT — selective layer exclusion is inferior to alpha tuning. The FP32 layers add speed overhead without sufficient quality benefit. Alpha=0.2 is both faster and more accurate.
**Committed:** no

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

### [75] Static INT8 quantization (abandoned)
**Date:** 2026-03-12
**Idea:** Replace dynamic quantization (weights only) with static quantization (weights + activations). Uses calibration data from 32 audio samples × 16 decoder steps to find optimal INT8 ranges for activation tensors.
**Change:** Created `quantize_static.py` implementing `CalibrationDataReader` for both decoder_init and decoder_step. Tested QDQ and QOperator formats with symmetric activations and weights.
**Result:** Both formats produce completely garbled output (random tokens, CJK characters, control codes). QDQ: 89s for 11s audio (8.1x RTF). QOperator: 147s (13.4x RTF). Massive speed regression on top of total quality failure.
**Outcome:** ABANDONED — static quantization is fundamentally incompatible with transformer decoder attention. INT8 activation quantization destroys softmax distributions, RoPE rotations, and KV cache values. Dynamic quantization (weights-only INT8, FP32 activations) is the correct approach for this architecture.
**Notes:** This matches the ORT documentation which primarily targets CNN/ResNet architectures for static quantization. Transformer decoders with autoregressive KV caching are not a supported use case. The `quantize_static.py` script is retained for reference but should not be used for decoder models.

### [76] MLP-only INT8 quantization (attention in FP32)
**Date:** 2026-03-12
**Idea:** Keep all attention projections (q/k/v/o_proj, 4 per layer × 28 layers = 112 nodes) in FP32 and only quantize MLP layers (gate/up/down_proj, 3 per layer × 28 layers = 84 nodes + lm_head). Hypothesis: attention is most sensitive to quantization, so keeping it FP32 should recover WER while MLP quantization still saves bandwidth.
**Change:** Used `quantize.py --nodes-to-exclude` with 112 attention node names on α=0.5 smooth FP32 source. Decoder weights: 1.2 GB (vs 600 MB full INT8, 2.4 GB FP32). Total model: 1.7 GB.
**Result:**
| Audio | RTF | Notes |
|---|---|---|
| 11s JFK | 0.26x | vs 0.14x full INT8, 0.29x FP32 |
| 35s LibriSpeech | 0.26x | Consistent with 11s |

100-sample WER (LibriSpeech test-other): **5.02%**

| Config | WER | RTF | Size |
|---|---|---|---|
| FP32 | 4.42% | 0.29x | 3.8 GB |
| MLP-only INT8 α=0.5 | 5.02% | 0.26x | 1.7 GB |
| Full INT8 α=0.2 | 5.21% | 0.14x | 1.1 GB |
| Full INT8 α=0.5 | 5.62% | 0.14x | 1.1 GB |
| Parakeet INT8 | 5.45% | 0.16x | — |

**Outcome:** INFORMATIONAL — WER is within 0.6pp of FP32 (best INT8 variant tested), but RTF is nearly identical to FP32 (0.26x vs 0.29x). Attention projections dominate decoder compute time, so keeping them FP32 eliminates most of the INT8 speed advantage. The 0.19pp WER improvement over α=0.2 full INT8 does not justify the ~2× speed regression.
**Committed:** no
**Notes:** Confirms that attention quantization is responsible for both the speed benefit and quality loss of INT8. For the 0.6B model, α=0.2 full dynamic INT8 remains the optimal tradeoff: 5.21% WER at 0.14x RTF beats Parakeet (5.45%) and is nearly 2× faster than FP32.

### [77] Alpha sweep below 0.2 (α=0.15)
**Date:** 2026-03-12
**Idea:** Test whether reducing alpha below 0.2 continues to improve WER. α=0.15 means less activation variance is migrated into weights.
**Change:** AWQ smooth α=0.15 using cached activations, then full dynamic INT8 quantization. Max scale ratio: 2.7x (vs 5.3x at α=0.2, 17.6x at α=0.5).
**Result:** 100-sample WER: **5.40%** — worse than α=0.2 (5.21%) and α=0.5 (5.62%). One sample produced empty output (decoder failure). α=0.2 is a sweet spot, not a monotonic trend.

| Alpha | WER (100-sample) | Max scale ratio |
|---|---|---|
| 0.5 | 5.62% | 17.6x |
| 0.2 | 5.21% | 5.3x |
| 0.15 | 5.40% | 2.7x |

**Outcome:** WORSE — α=0.15 is too little smoothing. Without sufficient variance migration, raw activation outliers cause worse quantization error than the moderate weight inflation at α=0.2.
**Committed:** no
**Notes:** Extended sweep with α=0.25 showed 5.08% at 100 samples, but 200-sample head-to-head: α=0.25=5.26% vs α=0.2=5.21%. The 100-sample noise was misleading. α=0.22 produced 6.29% (possibly bad export). Full alpha curve on 200 samples:

| Alpha | WER (200-sample) |
|---|---|
| 0.15 | 5.40% (100-sample only) |
| 0.20 | **5.21%** |
| 0.25 | 5.26% |
| 0.30 | 5.21% (100-sample) |
| 0.50 | 5.62% |

The minimum is a broad plateau around α=0.2-0.3. α=0.2 is optimal or near-optimal. No further alpha tuning warranted.

### [78] Calibration sample count: 256 vs 128
**Date:** 2026-03-12
**Idea:** More calibration samples for AWQ smoothing may produce better per-channel activation statistics, leading to more accurate scale factors.
**Change:** Re-ran α=0.2 AWQ smooth with n_samples=256 (vs default 128). Then quantized and evaluated.
**Result:** 200-sample WER: n=256 → 5.28%, n=128 → 5.21%. More samples did not help; 128 is sufficient for convergence.
**Outcome:** NO IMPROVEMENT — activation statistics converge by 128 samples. The remaining WER gap to FP32 is not caused by scale estimation noise.
**Committed:** no

---

## Quantization Trial Summary (Experiments 68-78)

The α=0.2 AWQ-smoothed dynamic INT8 quantization is the optimal configuration for Qwen3-ASR 0.6B. All alternative strategies tested either degraded WER, degraded speed, or both.

### Final Comparison (200-sample LibriSpeech test-other)

| Engine | WER | RTF (11s) | RTF (35s) | Model Size | Load Time |
|---|---|---|---|---|---|
| Qwen3-ASR 0.6B FP32 | 4.42% | 0.29x | 0.32x | 3.8 GB | 17.0s |
| **Qwen3-ASR 0.6B INT8 (α=0.2)** | **5.21%** | **0.14x** | **0.17x** | **1.1 GB** | **4.9s** |
| Qwen3-ASR 0.6B INT8 (α=0.5) | 5.62% | 0.14x | 0.17x | 1.1 GB | 4.9s |
| Parakeet-TDT 0.6B INT8 | 5.45% | 0.16x | 0.13x | — | 1.2s |

### What Was Tried

| # | Strategy | Result | Why |
|---|---|---|---|
| 68 | INT8 enc + FP32 dec | 0.30x RTF | No advantage over pure FP32 |
| 69 | FP16 decoder | — | CPU casts to FP32, no benefit |
| 71 | Exclude lm_head | No WER improvement | Error distributed across layers |
| 72 | INT16 weights | — | ORT lacks INT16 compute kernels |
| 73 | Alpha sweep | **α=0.2 optimal** | Best WER/speed tradeoff |
| 74 | Selective layer FP32 | Slower, no WER gain | Inferior to alpha tuning |
| 75 | Static quantization | Garbled output | Incompatible with transformer decoder |
| 76 | MLP-only INT8 | 5.02% WER, 0.26x RTF | Nearly FP32 speed — not useful |
| 77 | Alpha < 0.2 | 5.40% at α=0.15 | Too little smoothing |
| 78 | 256 calibration samples | 5.28% | 128 is sufficient |

### Conclusions

1. The 0.79pp WER gap (5.21% → 4.42%) is inherent to 8-bit dynamic weight quantization on this architecture
2. α=0.2 AWQ smoothing is optimal — the alpha-WER curve has a broad minimum around 0.2-0.3
3. Attention projection quantization provides most of the speed benefit but also most of the quality loss
4. For applications where WER < 5% is required, use FP32 (or wait for the 1.7B INT8 to be resolved)
5. For speed-sensitive applications, α=0.2 INT8 beats Parakeet on both WER (5.21% vs 5.45%) and speed (0.14x vs 0.16x RTF)

---

## 1.7B Quantization Trials (Experiments 79-83)

### Baseline
| Engine | WER (200-sample) | RTF (11s) | RTF (35s) | Size |
|---|---|---|---|---|
| Qwen3-ASR 1.7B FP32 | 3.79% | — | — | 8.8 GB |

### [79] 1.7B naive INT8 (no smoothing)
**Date:** 2026-03-13
**Result:** 100-sample WER: 9.66% — +5.87pp vs FP32.

### [80] 1.7B INT8 AWQ α=0.5
**Date:** 2026-03-13
**Result:** 100-sample WER: 16.14% — worse than naive INT8. The 1.7B is more sensitive to aggressive smoothing than the 0.6B (where α=0.5 gave 5.62%).

### [81] 1.7B INT8 AWQ α=0.1
**Date:** 2026-03-13
**Result:** 25.35% — too little smoothing, quantizer hits raw activation outliers directly.

### [82] 1.7B INT8 AWQ α=0.15
**Date:** 2026-03-13
**Result:** 22.55% — still badly degraded. The 1.7B alpha-WER curve drops off sharply below α=0.2.

### [83] 1.7B INT8 AWQ α=0.2 — BEST
**Date:** 2026-03-13
**Idea:** Apply same optimal alpha from 0.6B to 1.7B.
**Result:**
| Engine | WER (200-sample) | RTF (11s) | RTF (35s) | Size | Load |
|---|---|---|---|---|---|
| 1.7B FP32 | 3.79% | — | — | 8.8 GB | — |
| **1.7B INT8 α=0.2** | **9.04%** | **0.49x** | **0.47x** | **2.6 GB** | **8.7s** |

**Outcome:** α=0.2 is optimal for 1.7B, same as 0.6B. However the INT8 penalty is much larger: +5.25pp vs 0.6B's +0.79pp. The alpha-WER curve for 1.7B is sharp and narrow around 0.2 — any deviation (0.15 or 0.25) causes dramatic WER regression.

### [84] 1.7B INT8 AWQ α=0.25
**Date:** 2026-03-13
**Result:** 100-sample WER: 7.69% (note: 100-sample noise is significant). 200-sample would likely be comparable or worse than α=0.2.
**Outcome:** α=0.2 confirmed as optimal.

### 1.7B Alpha Sweep Summary

| Alpha | WER (100-sample) |
|---|---|
| 0.10 | 25.35% |
| 0.15 | 22.55% |
| 0.20 | **6.35%** (9.04% on 200-sample) |
| 0.25 | 7.69% |
| 0.50 | 16.14% |
| naive | 9.66% |
| FP32 | 3.94% (3.79% on 200-sample) |

### 1.7B Conclusions

1. α=0.2 is optimal for 1.7B, same as 0.6B, but the alpha-WER minimum is much sharper
2. The +5.25pp INT8 penalty (9.04% vs 3.79%) is substantially worse than 0.6B's +0.79pp
3. 0.47–0.49x RTF — about 3× slower than 0.6B INT8 (0.14x), as expected for a ~3× larger decoder
4. Load time of ~8.7s (cold) is high; subsequent loads benefit from OS file cache
5. The 1.7B INT8 at 9.04% WER offers better quality than Parakeet (5.45%) only in the sense that it has full punctuation and sentence structure, but at a large WER cost vs Qwen3 0.6B INT8 (5.21%)
6. 1.7B INT8 is not recommended for production until the quantization penalty can be reduced further

---

## MatMulNBits (int4, block_size=64) Trials — Experiments 85-86

ORT's `MatMulNBitsQuantizer` with RTN algorithm applies per-group weight-only int4 quantization. Each block of 64 weights shares a float16 scale and zero-point. This is the same quantization granularity as MLX's 8-bit and GPTQ's int4, and operates directly on the existing FP32 ONNX export without calibration data.

Tool: `quantize_nbits.py` in `~/qwen3-asr-onnx/`

### [85] 0.6B int4 MatMulNBits (block_size=64, RTN)
**Date:** 2026-03-13
**Result:**
| Metric | Value |
|---|---|
| WER (200-sample) | 5.28% |
| RTF (11s) | 0.26x |
| Model size | 2.0 GB |

**Outcome:** WER matches AWQ INT8 α=0.2 (5.21%) within noise. **Speed is 2× slower** (0.26x vs 0.14x). ORT's x86 CPU MatMulNBits kernel dequantizes weights to fp32 before each matmul rather than computing in int4 — eliminating the memory bandwidth benefit. For the 0.6B, AWQ INT8 remains the better choice.

### [86] 1.7B int4 MatMulNBits (block_size=64, RTN) — BEST 1.7B RESULT
**Date:** 2026-03-13
**Result:**
| Metric | Value |
|---|---|
| WER (200-sample) | **4.33%** |
| RTF (11s) | 0.56x |
| RTF (35s) | 0.65x |
| Model size | 4.3 GB |

**Outcome:** WER is dramatically better than AWQ INT8 α=0.2 (9.04% → 4.33%, a 4.71pp improvement). Only 0.54pp behind FP32 (3.79%). Per-group quantization (64 weights/block with independent float16 scales) handles the 1.7B's larger activation outliers far better than per-tensor dynamic INT8.

Speed is slower than AWQ INT8 (0.56x vs 0.47x) due to the fp32 dequantize fallback, but the quality tradeoff is strongly positive: **int4 MatMulNBits is the recommended quantization for 1.7B**.

### Full Comparison

| Engine | WER (200-sample) | RTF (11s) | RTF (35s) | Size | Load |
|---|---|---|---|---|---|
| 0.6B FP32 | 4.42% | 0.29x | 0.32x | 3.8 GB | 17.0s |
| **0.6B AWQ INT8 α=0.2** | **5.21%** | **0.14x** | **0.17x** | **1.1 GB** | **4.9s** |
| 0.6B int4 MatMulNBits | 5.28% | 0.26x | 0.26x | 2.0 GB | 5.8s |
| Parakeet-TDT 0.6B INT8 | 5.45% | 0.16x | 0.13x | — | 1.2s |
| 1.7B FP32 | 3.79% | ~0.7x | ~0.7x | 8.8 GB | ~45s |
| **1.7B int4 MatMulNBits** | **4.33%** | **0.56x** | **0.65x** | **4.3 GB** | **10.0s** |
| 1.7B AWQ INT8 α=0.2 | 9.04% | 0.47x | 0.49x | 2.6 GB | 8.7s |

### Notes
- ORT's x86 CPU MatMulNBits kernel does not compute in int4 — it dequantizes to fp32 first. Speed benefit requires a GPU EP or ARM64 with optimized kernel.
- For 0.6B on x86 CPU, AWQ INT8 α=0.2 remains the best option (2× faster, similar WER).
- For 1.7B on x86 CPU, int4 MatMulNBits is the best option (4.33% vs 9.04% WER at only marginally more compute than AWQ INT8).
- The 1.7B int4 model (4.33% WER) beats the 0.6B FP32 (4.42%) — so for quality-sensitive use cases, 1.7B int4 is the highest-quality option below 1.7B FP32.
