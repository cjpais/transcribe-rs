# transcription-rs Appendix

Reference material for the [transcription-rs API spec](transcription-rs.md) and sub-specs.

## Data Types

> *Pseudocode shows data shapes. See implementation for language-specific details (ownership, error handling, etc.).*

Both API layers return the same `Transcript` type with optional word-level detail. Supporting types include:
- `Word` for per-word timing
- `Speaker` for diarization
- `Alternative` for n-best hypotheses.

### Type Definitions

```
STRUCT Transcript
    text: String

    -- Result-level finality
    is_final: Bool               -- this result won't be revised
    is_endpoint: Bool            -- natural speech boundary detected (silence/pause)
    segment_id: Int              -- same ID for all revisions of one utterance

    -- Timing (seconds from stream/file start)
    start: Float?
    end: Float?

    -- Confidence
    confidence: Float?           -- 0.0-1.0, overall confidence score

    -- Language
    language: String?            -- detected/specified language code (e.g., "en", "en-US")
    language_confidence: Float?  -- 0.0-1.0, confidence in language detection

    -- Speaker diarization
    speaker: Speaker?            -- speaker identifier for this segment

    -- Multi-channel audio
    channel: Int?                -- audio channel index (0-based)

    -- Word-level detail
    words: List<Word>?

    -- N-best alternatives
    alternatives: List<Alternative>?  -- alternative transcriptions for same audio

    -- Raw backend response for debug/niche fields (requires include_raw: true)
    raw: JSON?

    -- Check if this result supersedes a previous partial (same segment, more final)
    METHOD supersedes(other: Transcript) -> Bool


-- Speaker identifier - backends use different schemes
ENUM Speaker
    Id(Int)              -- numeric ID (0, 1, 2...) - Deepgram, Azure, Rev.ai
    Label(String)        -- string label ("A", "B", "speaker_1") - AssemblyAI, Google


-- Alternative transcription hypothesis (n-best)
STRUCT Alternative
    text: String
    confidence: Float?
    words: List<Word>?


STRUCT Word
    text: String                 -- the word text
    punctuated: String?          -- with punctuation/caps (e.g., "yeah" -> "Yeah.")
    start: Float                 -- start time (seconds)
    end: Float                   -- end time (seconds)
    confidence: Float?           -- 0.0-1.0
    speaker: Speaker?            -- speaker for this word (may differ from segment)
```

### Design Notes

- `is_final` - "this result text won't be revised" (matches Deepgram/AWS `IsPartial: false`). Use for UI display decisions.
- `is_endpoint` - "speaker paused/stopped, segment complete" (matches Deepgram `speech_final`, sherpa `is_endpoint()`). Use to trigger `reset()`.
- After `input_finished()`, drain until `is_ready()` returns false. `get_result()` returns `None` when no result is available.
- `segment_id` groups all revisions of one utterance (partials share the same ID as their final). Use `supersedes()` to check if a new result replaces an old partial.
- `words` replaces both `tokens` and `timestamps` arrays - richer, aligned with cloud APIs. If a backend lacks word-level timing, `words` is `None` (not a vec of words with missing timestamps).
- `confidence` at word level: serves as both model certainty and stability indicator. Backends with boolean `Stable` map to `1.0`/`0.5`. For streaming, interim results may populate `confidence` with stability scores (how likely this partial will change).
- `speaker` - backends vary: Deepgram/Azure use numeric IDs, AssemblyAI/Google use string labels. The `Speaker` enum preserves this distinction.
- `channel` - for stereo/multi-channel audio (e.g., call center: agent=0, customer=1).
- `alternatives` - n-best hypotheses for the same audio segment. Primary hypothesis is in `text`/`words`; alternatives provide ranked fallbacks with their own confidence and optional word detail.
- `raw` - full backend response for debugging or accessing niche fields. **Disabled by default**; enable via `Config { include_raw: true, .. }`.
- `punctuated` - Deepgram's `punctuated_word` pattern. Useful when you need both raw ("yeah") and display ("Yeah.") forms.
- Batch API returns this type with `is_final: true`, `is_endpoint: true` always.

---

## Internal Implementation

### High-Level Adapter Internals

```
METHOD start_listening(callback)
    audio_channel = new Channel
    result_channel = new Channel
    engine = self.create_engine()

    -- Decode thread - uses low-level API
    SPAWN THREAD
        LOOP
            samples = audio_channel.receive()
            IF channel_closed THEN
                -- Flush remaining audio
                engine.input_finished()
                FOR result IN drain_results(engine)
                    result_channel.send(Ok(result))
                BREAK

            engine.accept_waveform(samples)

            -- Drain all results, send each to callback thread
            WHILE engine.is_ready()
                IF engine.decode() FAILS THEN
                    result_channel.send(Err(error))
                    BREAK
                result = engine.get_result()
                IF result EXISTS THEN
                    result_channel.send(Ok(result))

            -- Reset after segment boundary (not inside drain loop)
            IF engine.is_endpoint() THEN
                engine.reset()

    -- Callback thread
    SPAWN THREAD
        FOR result IN result_channel
            IF callback(result) FAILS THEN
                BREAK

    self.audio_channel = audio_channel
```

### Thread Model

```
CONSUMER THREAD                 transcription-rs INTERNAL THREADS

┌──────────────────┐           ┌──────────────────┐    ┌──────────────────┐
│ audio callback   │           │ Decode Thread    │    │ Callback Thread  │
│                  │           │                  │    │                  │
│ source           │   chan    │ accept_waveform  │chan│ loop {           │
│  .push_audio() ─────────────▶│ drain_results() ─────▶│   callback(r)   │
│                  │           │ for each result  │    │ }                │
└──────────────────┘           └──────────────────┘    └──────────────────┘
                                                              │
                                                              ▼
                                                        app.emit()
```

### Helper: drain_results()

Example pattern (not included in crate—customize error handling as needed):

```
-- Drain all pending results from engine (runs decode loop internally).
-- Stops on error; use decode() directly for error handling.
FUNCTION drain_results(engine: StreamingTranscriptionEngine) -> List<Transcript>
    results = []
    WHILE engine.is_ready()
        IF engine.decode() FAILS THEN
            BREAK
        result = engine.get_result()
        IF result EXISTS THEN
            results.append(result)
    RETURN results
```

### Why is_ready() has a default implementation

Different models have different chunk size requirements baked into their architecture:

| Model | ChunkSize | Approx. Audio Duration |
|-------|-----------|------------------------|
| Zipformer2 | 32-64 frames | 320-640ms |
| Paraformer | 61 frames | 610ms |
| Whisper streaming | varies | model-dependent |
| Deepgram (push) | N/A | server decides |

The `is_ready()` check is inherently model-specific. Backends like `SherpaEngine` can implement smart readiness checks that delegate to the underlying model, while simpler backends or push-based adapters can use the default `true` (always attempt decode, let `decode()` return early if needed).

---

## FAQ

### How does a consumer switch between online, sentence, or simulated streaming?

By choosing a backend. The streaming strategy is an implementation detail—all backends expose the same `StreamingTranscriptionEngine` interface, so usage code is identical:

```
engine = SherpaEngine.new(config)  -- backend choice determines strategy
source = StreamingTranscriptionSource.new(engine)
source.start_listening(callback)
source.push_audio(samples)
```

| Strategy                                          | Implementation                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Online streaming** (Sherpa/NeMo)                | Implements `StreamingTranscriptionEngine` directly                              |
| **Sentence-based** (VAD)                          | Wrapper buffers until VAD signals end of speech, then calls batch transcription |
| **Simulated streaming** (streaming-whisper style) | Wrapper with sliding window, returns partials via `StreamingTranscriptionEngine`|

**Design note:** Whether VAD runs inside or outside the engine is an implementation choice per-strategy:

| Strategy | VAD Role |
|----------|----------|
| **Online** | Typically none or energy-gating only — continuous partials needed |
| **Sentence-based** | Core component — VAD defines segment boundaries |
| **Simulated** | Optional — may use VAD to trigger final commit |

The key insight: `accept_waveform(samples)` as the universal input allows all strategies to be implemented as layers. Native streaming models implement `StreamingTranscriptionEngine` directly; wrappers add VAD/windowing logic on top of batch models while exposing the same interface. This keeps the abstraction boundary clean — consumers don't care which strategy a backend uses.

### What's the difference between is_final and is_endpoint?

- `is_final` — "this text won't be revised". Use for UI display decisions (show as committed text).
- `is_endpoint` — "speaker paused/stopped, segment complete". Use to trigger `reset()`.

A result can be `is_final` without being `is_endpoint` (text is stable but speaker hasn't paused yet).

### Why route ElevenLabs/OpenAI through PushAdapter? They're already push-based.

`PushAdapter` converts push-based backends into the pull-based `StreamingTranscriptionEngine` interface. This lets users:

- Call `decode()` / `get_result()` synchronously instead of only using callbacks
- Use the same consumption pattern for local and cloud backends
- Swap backends without changing how results are consumed

Without this conversion, cloud backends would only be usable via callbacks, breaking API uniformity.

### How do errors propagate?

| Error Source      | High Level                         | Low Level                            |
| ----------------- | ---------------------------------- | ------------------------------------ |
| Decode fails      | Sent as `Err(e)` to callback       | Returns error from `decode()`        |
| Consumer error    | Callback returns `Err`, loop stops | Consumer handles directly            |
| End of stream     | `stop_listening()` signals end     | Consumer calls `input_finished()`    |
| Endpoint detected | Auto-reset after drain             | Consumer calls `reset()` after drain |

### How does VAD (Voice Activity Detection) fit in?

VAD is **orthogonal to transcription**—it's an audio preprocessing concern. This spec keeps VAD out of the core API because use cases vary (some backends have it built-in, others don't need it).

| Concept | What it detects | Where it lives |
|---------|-----------------|----------------|
| **VAD** | "Is someone speaking right now?" | Pre-processor (before ASR) |
| **`is_endpoint()`** | "Did the speaker pause/finish?" | Decoder (informed by language model) |

VAD filters silence *before* audio reaches the engine. `is_endpoint()` detects pauses *during* decoding, often using linguistic cues that VAD can't see.

---

## Reference

### API Survey

| API | Finality | Endpoint | Confidence | Speaker | Channel | Alternatives |
|-----|----------|----------|------------|---------|---------|--------------|
| sherpa-onnx | N/A | `is_endpoint()` | N/A | N/A | N/A | N/A |
| Deepgram | `is_final` | `speech_final` | 0.0-1.0 | `speaker: int` | `channel` | `alternatives[]` |
| AssemblyAI | `message_type` | `end_of_turn` | 0.0-1.0 | `speaker: "A","B"` | `channel` | N/A |
| AWS Transcribe | `IsPartial` | natural segments | 0.0-1.0, `Stable` | `Speaker: string` | `ChannelId` | `Alternatives[]` |
| Google Cloud | `isFinal` | `speechEventType` | 0.0-1.0, `stability` | `speakerLabel` | `channelTag` | `alternatives[]` |
| Azure | `RecognitionStatus` | implicit | 0.0-1.0 | `speaker: int` | `channel` | `NBest[]` |
| OpenAI Realtime | event types | VAD events | via `logprobs` | `speaker` | N/A | N/A |
| ElevenLabs | `partial`/`committed` | `commit_strategy` | `logprob` | N/A | N/A | N/A |
| Rev.ai | `type: final/partial` | implicit | 0.0-1.0 | `speaker: int` | N/A | N/A |
| Vosk | `partial` vs `result` | implicit | `conf` | `spk` (embedding) | N/A | `alternatives[]` |

### Field Mappings to Transcript

| API Field | Transcript Field | Notes |
|-----------|------------------|-------|
| `is_final`, `isFinal`, `!IsPartial` | `is_final` | |
| `speech_final`, `end_of_turn`, `speechEventType` | `is_endpoint` | |
| `confidence`, `conf` | `confidence` | 0.0-1.0 |
| `stability` (Google interim) | `confidence` | Maps stability → confidence for partials |
| `Stable: bool` (AWS) | `confidence` | `true` → 1.0, `false` → 0.5 |
| `speaker`, `speakerLabel`, `Speaker` | `speaker` | `Speaker::Id` or `Speaker::Label` |
| `channel`, `channelTag`, `ChannelId` | `channel` | 0-based index |
| `alternatives`, `NBest` | `alternatives` | |
| `punctuated_word` (Deepgram) | `Word.punctuated` | |
| `language`, `languageCode` | `language` | |
| `LanguageIdentification[].Score` | `language_confidence` | |

### Compatible 3rd-Party APIs

| API | Streaming | Transport | Deployment |
|-----|-----------|-----------|------------|
| **sherpa-onnx** | pull | local | local |
| **NeMo** | pull | local | local |
| **Vosk** | pull | local | local |
| **Deepgram** | push | WebSocket | cloud |
| **OpenAI Realtime** | push | WebSocket | cloud |
| **ElevenLabs** | push | WebSocket | cloud |
| **AssemblyAI** | push | WebSocket | cloud |
| **AWS Transcribe** | push | WebSocket | cloud |
| **Azure Speech** | push | SDK/WebSocket | cloud |
| **Google Cloud Speech** | push | gRPC | cloud |
| **Rev.ai** | push | WebSocket | cloud |
| **Voxtral Realtime** | push | WebSocket | cloud/local |
| **whisper.cpp** | batch only | local | local |

### Batch API Comparison

**`transcription-rs`** (new crate):

```
INTERFACE BatchTranscriptionEngine
    transcribe_samples(samples: List<Float>, ...) -> List<Transcript>
    transcribe_file(path: String, ...) -> List<Transcript>
```

**`transcribe-rs`** (deprecated wrapper):

```
-- Old signature preserved - wraps transcription-rs internally
INTERFACE TranscriptionEngine
    transcribe_samples(samples: List<Float>, ...) -> TranscriptionResult
    transcribe_file(path: String, ...) -> TranscriptionResult
```

| Aspect        | `transcription-rs`  | `transcribe-rs` (wrapper)                    |
| ------------- | ------------------- | -------------------------------------------- |
| Samples param | borrowed reference  | owned copy                                   |
| Return type   | `List<Transcript>`  | `TranscriptionResult` with nested `segments` |
| Streaming     | Full support        | Not exposed                                  |

---

## Migration

### Naming Convention

| Pattern | Role | Style | You... | Examples |
|---------|------|-------|--------|----------|
| `*Engine` | both | pull | call it | `StreamingTranscriptionEngine`, `SherpaEngine` |
| `*Source` (backend) | contributor | push | implement for push APIs | `ElevenLabsSource`, `OpenAISource` |
| `StreamingTranscriptionSource` | app dev | push | receive callbacks | high-level API |

`PushAdapter` bridges the two: wraps a backend `*Source` to implement `*Engine`.

### Why no high-level batch API?

Batch transcription is synchronous and simple—users call `BatchTranscriptionEngine` directly. Streaming needs `StreamingTranscriptionSource` to handle threading, callbacks, and push→pull conversion.

### Why a new crate

| Crate                  | Status     | Purpose                                                    |
| ---------------------- | ---------- | ---------------------------------------------------------- |
| **`transcription-rs`** | New        | Rich API with streaming support, `Transcript` type         |
| **`transcribe-rs`**    | Deprecated | Thin wrapper, transforms to old `TranscriptionResult` type |

### Changes from legacy

| Legacy (transcribe-rs) | New (transcription-rs) | Notes |
|------------------------|------------------------|-------|
| `TranscriptionEngine` | `BatchTranscriptionEngine` | Added "Batch" prefix for clarity |
| `TranscriptionResult` | `Transcript` | Richer fields |
| — | `StreamingTranscriptionSource` | New: high-level streaming API (callback-based) |
| — | `StreamingTranscriptionEngine` | New: low-level streaming interface (pull-based) |
| — | `PushSource` | New: for push-based backends |
| — | `PushAdapter` | New: wraps `PushSource` as `StreamingTranscriptionEngine` |

### Migration Steps

```toml
# Old
[dependencies]
transcribe-rs = "0.2"  # now wraps transcription-rs

# New (recommended)
[dependencies]
transcription-rs = "0.1"
```

For code that used `TranscriptionResult.text` (joined text), use the included helper:

```
-- Before (legacy)
result = engine.transcribe(audio)
print(result.text)  -- TranscriptionResult.text was pre-joined

-- After (new)
transcripts = engine.transcribe(audio)
print(joined_text(transcripts))  -- explicitly join segments
```
