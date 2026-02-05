# Streaming Adapter Research

Research notes moved from the main spec. These document implementation details and comparisons that informed the design but aren't essential for understanding the API.

## How sherpa-onnx implements `is_ready()`

Sherpa-onnx's WebSocket server bridges its pull-based C++ core with push-based WebSocket clients. This is essentially the same adapter pattern we need.

**Core implementation** (`online-recognizer-transducer-impl.h`):

```cpp
bool IsReady(OnlineStream *s) const override {
  return s->GetNumProcessedFrames() + model_->ChunkSize() < s->NumFramesReady();
}
```

In plain English: `ready = (unprocessed_frames >= chunk_size)`

**The three numbers:**

| Value | Meaning |
|-------|---------|
| `NumFramesReady()` | Total feature frames computed from audio so far |
| `GetNumProcessedFrames()` | Frames already decoded |
| `ChunkSize()` | Model's minimum frames needed per decode step (e.g., 32-64) |

**Visual example:**

```
Audio samples: ████████████████████████████████████
                          | feature extraction
Feature frames: [0][1][2][3][4][5][6][7][8][9][10][11]...
                 └─── processed ───┘ └── available ──┘
                     (6 frames)         (6 frames)
                     
ChunkSize = 4

IsReady? = (6 + 4) < 12 -> 10 < 12 -> YES
```

**Sherpa's adapter architecture:**

```
WebSocket -> Connection::samples (queue) -> OnlineStream -> Timer polls IsReady() -> DecodeStreams()
```

Key mechanisms:
- **Per-connection buffer** - `std::deque<std::vector<float>>` for incoming audio chunks
- **Timer-driven polling** - 10ms loop checks `IsReady()` on all connections
- **Batch processing** - Groups up to 5 ready streams for efficient GPU/CPU use
- **Double buffering** - Raw audio queue + feature extractor internal buffer

Each model implementation defines its own `IsReady()` via polymorphism:

```cpp
// Transducer impl
class OnlineRecognizerTransducerImpl : public OnlineRecognizer {
  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + model_->ChunkSize() < s->NumFramesReady();
  }
};

// Paraformer impl  
class OnlineRecognizerParaformerImpl : public OnlineRecognizer {
  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() + 61 < s->NumFramesReady();  // hardcoded
  }
};
```

**Comparison with our spec:**

| Aspect | Sherpa WebSocket | Our Spec |
|--------|------------------|----------|
| Buffer | `deque<vector<float>>` | `VecDeque<f32>` |
| Polling | Timer-driven (10ms) | On-demand via `is_ready()` |
| Batching | Yes (multi-stream) | No (single stream) |
| Readiness | `IsReady()` per-model | `is_ready()` with default |

Sherpa's adapter is **server-centric** (manages many connections, batch decoding), while our adapter is **client-centric** (single stream, simpler). The `is_ready()` pattern lets smart backends optimize polling without requiring it from all implementations.

## Comparison of streaming adapter patterns (Sherpa, Vosk, NeMo)

Research into three major ASR frameworks reveals different approaches to bridging push-based audio input with pull-based decoding. None is universally "better"—each has trade-offs suited to different contexts.

**Pattern comparison:**

| Aspect | Sherpa | Vosk | NeMo |
|--------|--------|------|------|
| **Pattern name** | Buffer-and-Poll | Process-As-You-Receive | Batch-and-Decode |
| **Decode trigger** | Timer (10ms poll) | On every audio push | When buffer fills |
| **Readiness check** | Explicit `IsReady()` | Implicit (always ready) | Implicit (buffer fullness) |
| **Buffering** | Per-connection queue | Internal to recognizer | `FeatureFrameBufferer` |
| **State management** | Opaque `OnlineStream` | Internal FST state | Explicit `partial_hypotheses` |
| **Multi-stream batching** | Yes | No | Yes (native) |

**Trade-offs:**

| Pattern | Strengths | Weaknesses | Best for |
|---------|-----------|------------|----------|
| **Sherpa (poll)** | Explicit control, batching, predictable CPU usage | Timer overhead, slight latency | Servers with many concurrent streams |
| **Vosk (immediate)** | Simplest, lowest per-chunk latency | No batching, CPU spikes on audio arrival | Single-stream, real-time apps |
| **NeMo (batch)** | GPU-optimized, CUDA graphs, explicit state | More complex, higher latency | GPU inference, research/flexibility |

**Key insight:** The "best" approach depends on:
- **Hardware** — GPU batching (NeMo) vs CPU single-stream (Vosk)
- **Deployment** — Server with many clients (Sherpa) vs embedded/desktop (Vosk)
- **Model architecture** — Some models have natural chunk boundaries, others don't

**Our spec's flexibility:**

| Backend style | `is_ready()` impl | Example |
|---------------|-------------------|---------|
| Explicit readiness (Sherpa-like) | Override with model-specific check | `SherpaDecoder` |
| Internal buffering (Vosk-like) | Use default `true` | Simple backends |
| Batch-oriented (NeMo-like) | Override, return `true` when batch ready | `NemoDecoder` |

The default `is_ready() -> true` accommodates all patterns—backends with explicit readiness override it, others just work.

## Callback trait bounds: `Fn` vs `FnMut`

The `StreamingEngine` callback uses `Fn` not `FnMut`:

```rust
pub fn start_listening<F>(&self, callback: F) -> Result<(), Error>
where
    F: Fn(Result<Transcript, Error>) -> Result<(), Error> + Send + 'static;
```

`Fn` means the callback cannot mutate captured state. This works for forwarding (e.g., `app.emit()`). To accumulate results, use `Mutex<Vec<_>>` or a channel.

## Legacy wrapper implementation

`transcribe-rs` internally depends on `transcription-rs` and transforms types:

```rust
// transcribe-rs/Cargo.toml
[dependencies]
transcription-rs = "0.1"

// transcribe-rs/src/lib.rs
use transcription_rs::Transcript;

pub struct TranscriptionResult {
    pub text: String,
    pub segments: Option<Vec<TranscriptionSegment>>,
}

pub struct TranscriptionSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

impl TranscriptionEngine for WhisperEngine {
    fn transcribe_samples(&mut self, samples: Vec<f32>, params: Option<Self::InferenceParams>)
        -> Result<TranscriptionResult, Box<dyn std::error::Error>>
    {
        // Delegate to new crate
        let transcripts = transcription_rs::BatchDecoder::transcribe_samples(
            self, &samples, params
        )?;

        // Transform to old type
        Ok(TranscriptionResult {
            text: transcripts.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" "),
            segments: Some(transcripts.into_iter().map(|t| TranscriptionSegment {
                start: t.start.unwrap_or(0.0),
                end: t.end.unwrap_or(0.0),
                text: t.text,
            }).collect()),
        })
    }
}
