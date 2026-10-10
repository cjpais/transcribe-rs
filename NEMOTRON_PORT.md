# Nemotron streaming ASR — offline engine port

Implementation spec for adding NVIDIA **Nemotron streaming** ASR to
transcribe-rs as a native ONNX engine (`src/onnx/nemotron/`). Targets both the
English-only 0.6B and the **multilingual 3.5 0.6B** variants
([nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)).

Relates to issue #31 ("Nemotron streaming") and PR #36. Unlike PR #36 this is a
**native port** of the validated logic from
[`altunenes/parakeet-rs`](https://github.com/altunenes/parakeet-rs) (MIT) — **no
new dependencies, no `parakeet-rs` crate dependency** — and is **offline-only**
(implements `SpeechModel`, not a streaming trait), matching the maintainer's
stated scope on #31/#36.

## Architecture

Cache-aware FastConformer encoder + RNN-T decoder/joint, with (multilingual
only) a prompt-MLP that conditions decoding on a language id. The model is a
*streaming* model; we run it **offline** by feeding the whole utterance through
the streaming encoder in fixed chunks, threading the encoder cache between
chunks, then greedily decoding. From the caller's perspective it's a normal
record-then-transcribe engine.

## ONNX contract (source of truth)

Files in the model dir: `encoder.onnx` (+ `encoder.onnx.data`),
`decoder_joint.onnx`, `tokenizer.model`. Quantized variants follow the existing
`{name}.int8.onnx` convention via `session::resolve_model_path`.

### `encoder.onnx`

| Input | Shape | Type | Notes |
|---|---|---|---|
| `processed_signal` | `[1, 128, T]` | f32 | log-mel, computed in Rust |
| `processed_signal_length` | `[1]` | i64 | = T frames in this chunk |
| `cache_last_channel` | `[24, 1, L, 1024]` | f32 | L = 56 (multilingual) / 70 (en) |
| `cache_last_time` | `[24, 1, 1024, 8]` | f32 | conv_context = 8 |
| `cache_last_channel_len` | `[1]` | i64 | starts at `0` |
| `prompt_index` | `[1]` | i64 | **multilingual only**; presence = variant detection |

Outputs: `encoded` `[1, 1024, T_out]`, `encoded_len` (i64 scalar),
`cache_last_channel_next`, `cache_last_time_next`, `cache_last_channel_len_next`
(threaded into the next chunk).

Dims (num_layers=24, hidden=1024, L, conv_context) are read from the graph's
input shapes at load, not hard-coded.

### `decoder_joint.onnx`

Identical contract to the existing Parakeet engine:

| Input | Shape | Type |
|---|---|---|
| `encoder_outputs` | `[1, 1024, 1]` | f32 |
| `targets` | `[1, 1]` | i32 |
| `target_length` | `[1]` | i32 |
| `input_states_1` / `input_states_2` | `[2, 1, 640]` | f32 |

Outputs: `outputs` (logits, length `vocab_size + 1`), `output_states_1`,
`output_states_2`. **`blank_id = vocab_size`** (appended; not a `<blk>` vocab row).

### Constants

`CHUNK_SIZE = 56` mel frames per chunk, `PRE_ENCODE_CACHE = 9` carried mel
frames prepended to each chunk, `SUBSAMPLING_FACTOR = 8` ⇒ one encoder output
frame = 80 ms (used for token timestamps). Greedy decode `MAX_SYMBOLS_PER_STEP = 10`.

### Mel front end (the one fiddly part)

`n_fft = 512`, `win_length = 400` (zero-padded to 512), `hop = 160`,
`n_mels = 128`, Slaney filterbank, pre-emphasis `0.97`, `log(x + 2^-24)`, and
**no per-feature normalization** — the streaming Nemotron models feed raw
log-mel "decibels" into the encoder. (parakeet-rs's `extract_features_with_cache`
*does* normalize; that path is the wrong one to reuse.) `MelConfig` in
`features::` can't express `win_length != n_fft`, so the front end is ported
into `nemotron/mel.rs` against `rustfft` rather than bent through `compute_mel`.

## Reuse map

| Need | Already in transcribe-rs | Action |
|---|---|---|
| Session create + quant file resolve | `onnx::session` | reuse |
| RNN-T greedy decode loop | `parakeet::{decode_step,create_decoder_state}` | copy + retarget shapes |
| Timestamp grouping token→word→segment | `parakeet::convert_timestamps` | reuse / factor out |
| `TranscriptionResult` / `SpeechModel` | `lib.rs` | reuse |
| FFT / ndarray / once_cell | `rustfft`, `ndarray`, `once_cell` | reuse (no new deps) |
| Cache-aware encoder + 4D cache threading | — | **port** (`encoder.rs`) |
| Non-normalized NeMo log-mel | — | **port** (`mel.rs`) |
| SentencePiece `.model` protobuf parse | — | **port** (`tokenizer.rs`) |
| Prompt dictionary + `<xx-XX>` stripping | — | **port** (engine) |

## File plan (`src/onnx/nemotron/`)

| File | Responsibility | Status |
|---|---|---|
| `mel.rs` | pre-emphasis + STFT (win 400 / fft 512) + Slaney mel + log; no norm | **done** (+5 tests) |
| `tokenizer.rs` | SentencePiece `.model` protobuf loader, lang-tag detection | **done** (+3 tests) |
| `encoder.rs` | `EncoderCache` (4D tensors) + `run_encoder`/`run_decoder` + dim auto-detect | **done** |
| `mod.rs` | `NemotronModel`/`NemotronParams`, chunk loop, decode loop, prompt dict, `impl SpeechModel` | **done** (+2 tests) |

Plus: register `pub mod nemotron;` in `src/onnx/mod.rs` (**done**);
`tests/nemotron.rs` + `examples/nemotron.rs` + `Cargo.toml`
`required-features = ["onnx"]` entries (**done**).

## Validation

`cargo check --features onnx --lib --tests --examples` is clean (zero
warnings). 11 unit tests + 1 integration test pass.

End-to-end against the real multilingual FP32 export
(`altunenes/parakeet-rs/nemotron-3.5-asr-streaming-0.6b-onnx`), CPU, ~2.5x
real-time:

| Audio | Lang | Output |
|---|---|---|
| jfk.wav | `en-US` | "And so my fellow Americans ask not what your country can do for you. Ask what you can do for your country." |
| german.wav | `de-DE` | "Am Strand der Badeanzug die Badehose, die Sandalen, ..." |
| russian.wav | `ru-RU` | "Проверка связи." |
| german.wav | `auto` | (same German — auto language detection) |

Run it (this machine — `ort` static link fails, so use `load-dynamic` + a
supplied onnxruntime ≥ API 17):

```bash
ORT_DYLIB_PATH='D:\dev\ort-runtime\onnxruntime.dll' \
  cargo run --example nemotron --features onnx,ort/load-dynamic -- \
  models/nemotron-3.5-asr-streaming-0.6b-onnx samples/jfk.wav en-US
```

## `SpeechModel` mapping

```text
NemotronModel::load(dir, &Quantization)   // encoder + decoder_joint + tokenizer.model;
                                          // detect prompt_index input -> mode
NemotronParams { language, timestamp_granularity }
transcribe_with(samples, &params)         // language -> prompt_index (default "auto" = 101);
                                          // reset state; chunk loop; greedy decode; strip <xx-XX>
impl SpeechModel { capabilities(); transcribe_raw() }   // streaming = false, timestamps = true
```

`CAPABILITIES`: `name = "Nemotron"`, `engine_id = "nemotron"`,
`sample_rate = 16000`, `languages =` prompt-dictionary keys (multilingual) /
`["en"]` (en-only). Decoder state (`state_1/2`, `last_token`, encoder cache)
lives on `&mut self`, reset per call — no `Arc<Mutex>`/multi-stream plumbing.

## Deliberately out of scope (for now)

Streaming trait / partial results; multi-stream shared handles; per-token
`logprob` richness. These can follow once the streaming-API design on #31 lands.

## Open decisions

- **Tokenizer input**: consume the shipped `tokenizer.model` (chosen — protobuf
  parser is ~160 self-contained lines and consumes the HF bundle unchanged) vs.
  also emitting a `vocab.txt`.
- **FP32 first**: validate correctness against the existing 2.45 GB FP32 export,
  then add the int8 packaging step (orthogonal to the engine).
- **Timestamps**: include from the start (frame → 80 ms) by reusing Parakeet's
  grouping.

## Downstream (Handy)

Bump the `transcribe-rs` pin across the per-target `Cargo.toml`s; add one
`ModelInfo` row (`engine_type: EngineType::Nemotron`, `is_directory: true`, int8
tarball URL, multilingual `supported_languages`). Re-check the ort/Intel-Mac
question at that point.
