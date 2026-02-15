# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build/check/test with all features
cargo build-all
cargo check-all
cargo test-all

# Build/test a single engine
cargo build --features parakeet
cargo test --features parakeet

# Run a single test by name
cargo test --features parakeet test_jfk_transcription_fp32

# Run an example
cargo run --example parakeet --features parakeet
cargo run --example sense_voice --features sense_voice -- --int8 models/sense-voice-int8 samples/jfk.wav
```

No CI is configured. No linter or formatter configuration exists beyond standard `cargo fmt` and `cargo clippy`.

## Architecture

Single-crate library with feature-gated transcription engine backends. No default features are enabled.

### Core Traits (`src/lib.rs`)

- **`TranscriptionEngine`** — Batch transcription. Associated types `InferenceParams` and `ModelParams` allow per-engine configuration. Default `transcribe_file` reads WAV then delegates to `transcribe_samples`.
- **`StreamingTranscriptionEngine`** — Incremental chunk-based transcription via `push_samples`/`get_transcript`/`reset`. Associated type `ModelParams` mirrors `TranscriptionEngine`.
- **`RemoteTranscriptionEngine`** (async, `openai` feature) — Async API-backed transcription.

All batch engines return `TranscriptionResult { text, segments: Option<Vec<TranscriptionSegment>> }`.

### Engine Modules (`src/engines/`)

Each engine is gated behind a Cargo feature flag. Engines that use ONNX Runtime (`parakeet`, `moonshine`, `sense_voice`) share `ort` and `ndarray` as optional deps.

| Feature | Engine | Model format | Notes |
|---------|--------|-------------|-------|
| `whisper` | `WhisperEngine` | Single GGML file | Metal (macOS) / Vulkan (Linux/Windows) |
| `parakeet` | `ParakeetEngine` | ONNX directory (6-7 files) | FP32/Int8 quantization, token/word/segment timestamps |
| `moonshine` | `MoonshineEngine` | ONNX directory (3 files) | Custom BPE tokenizer, KV cache for autoregressive decoding |
| `sense_voice` | `SenseVoiceEngine` | ONNX + token vocab | CTC-based, FBANK feature extraction via rustfft |
| `whisperfile` | `WhisperfileEngine` | GGML file | Spawns local server process, custom multipart HTTP client |
| `nemotron-streaming` | `NemotronStreamingEngine` | ONNX directory | Wraps `parakeet-rs`, implements `StreamingTranscriptionEngine` |
| `openai` | `OpenAIEngine` | Remote API | Async, requires `OPENAI_API_KEY` env var |

### Audio (`src/audio.rs`)

`read_wav_samples` validates and reads 16kHz/16-bit/mono WAV files. `mix_to_mono` is always available. Resampling utilities (`create_resampler`, `resample_chunk`) are gated behind the `resampling` feature (pulled in transitively by `nemotron-streaming`).

### Testing

Integration tests in `tests/` are feature-gated (`required-features` in Cargo.toml). Tests skip gracefully when model files are absent (checked via `Path::exists`). Models are stored in `models/` (gitignored) and sample audio in `samples/`.
