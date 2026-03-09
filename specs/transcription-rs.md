# transcription-rs API Design

## Overview

**transcription-rs** unifies transcription APIs behind a common interface.

**Why:** transcribe-rs handles batch transcription well, but:
- No streaming support (partial results as you speak)
- Streaming is more complex than batch — async operations, interim results that get revised, endpoint detection
- Current `TranscriptionResult` type has gaps
- Streaming APIs vary widely: some push results via callbacks, others require polling; fields and semantics differ

**What's new:**
- Streaming transcription support via `StreamingTranscriptionEngine`
- One `Transcript` type for all results (batch, streaming partial, streaming final)
- Swappable backends — switch providers without changing app code

**Optional adapters** handle the push/pull impedance mismatch: most cloud APIs are push-based (callbacks), most local models are pull-based (you call, get result). Adapters let implementors write minimal wrappers matching their API's native style; apps can consume either push or pull.

## Sub-Specs

### (A) Transcript Type — *highly recommended*

**Problem:** The current `TranscriptionResult` type isn't suitable for streaming APIs and has gaps even for batch. Different return types force code duplication and make switching between batch and streaming harder than it should be.

**Solution:** One comprehensive `Transcript` type for all results (batch, streaming partial, streaming final). The `raw` field leaves the door open for any gaps in this spec and future 3rd-party API innovations.

→ Details: *transcription-rs-a-transcript-type.md (coming soon)*

### (B) StreamingTranscriptionEngine (Low-Level) — *required*

**Problem:** No streaming transcription support currently. Need a common interface so backends are swappable. Backend contributors need a clear, simple target to implement.

**Solution:** `StreamingTranscriptionEngine` — a pull-based interface. Pull is the better internal abstraction: you can build push on top of pull (C does this), but building pull on top of push is messier. Pull is the simplest wrapper for local models (sherpa-onnx, NeMo, Vosk) — high priority for local/private transcription. Pull gives control — caller decides timing, backpressure, when to drain (power users need this).

→ Details: *transcription-rs-b-streaming-engine.md (coming soon)*

### (C) StreamingTranscriptionSource (High-Level Adapter) — *optional*

**Problem:** The pull-based `StreamingTranscriptionEngine` (B) requires managing two loops: audio in and results out. App devs would need to manage threading themselves — easy to mess up. Most UI frameworks (Svelte, React, Tauri) work better with event/callback-based APIs. Without this, every app duplicates the same threading/callback logic.

**Solution:** `StreamingTranscriptionSource` — push audio, receive callbacks. Library owns the threads (audio processing + result delivery); app just pushes audio and handles callbacks. Built on top of (B), so all backends work automatically. Optional: devs who need more control can use (B) directly.

→ Details: *transcription-rs-c-high-level-adapter.md (coming soon)*

### (D) PushAdapter — *optional*

**Problem:** Most cloud streaming APIs (Deepgram, OpenAI, ElevenLabs, etc.) are push-based (WebSocket callbacks). They don't fit the pull-based `StreamingTranscriptionEngine` interface (B). Without an adapter, contributors would have to implement complex push→pull conversion for each backend. If push backends only exposed a push interface, power users couldn't use them in pull style.

**Solution:** `PushSource` — simpler 4-method interface that's basically just a thin wrapper over the 3rd-party API (start, send audio, finish, stop). `PushAdapter` wraps any `PushSource` and implements `StreamingTranscriptionEngine`. Contributors just wrap the natural API; adapter handles the tricky push→pull conversion once. All push backends work both ways: via (C) for most apps, or via (B) for power users.

→ Details: *transcription-rs-d-push-adapter.md (coming soon)*

See also: [Appendix](transcription-rs-appendix.md) (API survey, migration guide, implementation details)

## Architecture Diagrams

The following diagrams are labeled with sub-spec letters **(A)**–**(D)** to show how the specs relate to each other.

### Legend

```mermaid
graph LR
    Existing["[Existing]"] -- solid --> PlannedPull[Planned Pull]
    Existing -. dashed .-> PlannedPush[Planned Push]
    Renamed["NewName [OldName]"]

    classDef pullExisting fill:#cce5ff
    classDef pullPlanned fill:#cce5ff,stroke-dasharray: 5 5
    classDef pushPlanned fill:#e1f5e1,stroke-dasharray: 5 5
    class Existing,Renamed pullExisting
    class PlannedPull pullPlanned
    class PlannedPush pushPlanned
```

\* Links omitted for clarity — see similar items

### Batch Transcription (exists today)

```mermaid
graph TB
    subgraph Apps ["Consumer Apps"]
        Handy[Handy]
        Whispering[Whispering*]
    end

    subgraph Library ["transcription-rs Crate"]
        subgraph Core ["Core"]
            BatchEngine["BatchTranscriptionEngine [TranscriptionEngine] (interface)"]
        end

        subgraph Backends ["Backend Implementations — Contributors"]
            ParakeetEngine["[ParakeetEngine] (type)"]
            OpenAIEngine["[OpenAIEngine] (type)"]
            WhisperEngine["[WhisperEngine] (type)"]
        end
    end

    subgraph External ["3rd Party APIs — External"]
        NeMo[NeMo]
        OpenAI[OpenAI]
        WhisperCpp[whisper.cpp]
    end

    Handy -->|"<b>(A)</b> returns List&lt;Transcript&gt;"| BatchEngine

    BatchEngine --> ParakeetEngine
    BatchEngine --> OpenAIEngine
    BatchEngine --> WhisperEngine

    ParakeetEngine --> NeMo
    OpenAIEngine --> OpenAI
    WhisperEngine --> WhisperCpp

    classDef pullExisting fill:#cce5ff
    class ParakeetEngine,OpenAIEngine,WhisperEngine,BatchEngine pullExisting
```

### Streaming Transcription (planned)

```mermaid
graph TB
    subgraph Apps ["Consumer Apps"]
        Handy[Handy]
        Whispering[Whispering*]
    end

    subgraph Library ["transcription-rs Crate"]
        subgraph Core ["Core"]
            subgraph HighLevel ["<b>(C)</b> High-Level Adapter"]
                StreamingSource["StreamingTranscriptionSource (type)"]
            end

            subgraph LowLevel ["<b>(B)</b> Low-Level Interface"]
                StreamingEngine["StreamingTranscriptionEngine (interface)"]
            end

            subgraph PushHelper ["<b>(D)</b> Push Adapter"]
                PushAdapter["PushAdapter (type)"]
                PushSource["PushSource (interface)"]
            end
        end

        subgraph Backends ["Backend Implementations — Contributors"]
            ParakeetEngine["ParakeetEngine (type)"]
            OpenAISource["OpenAISource (type)"]
        end
    end

    subgraph External ["3rd Party APIs — External"]
        subgraph PullStreaming ["Pull Streaming (3)"]
            NeMo[NeMo]
            Vosk[Vosk*]
            Sherpa[sherpa-onnx*]
        end
        subgraph PushStreaming ["Push Streaming (9)"]
            Deepgram[Deepgram*]
            OpenAI[OpenAI Realtime]
            ElevenLabs[ElevenLabs*]
        end
    end

    Handy -.->|"<b>(A)</b> returns Transcript"| StreamingSource
    Handy -.->|"<b>(A)</b> returns Transcript"| StreamingEngine

    StreamingSource -.-> StreamingEngine

    StreamingEngine -.-> ParakeetEngine
    StreamingEngine -.-> PushAdapter
    
    PushAdapter -.-> PushSource
    PushSource -.-> OpenAISource

    ParakeetEngine -.-> NeMo
    OpenAISource -.-> OpenAI

    classDef pushPlanned fill:#e1f5e1,stroke-dasharray: 5 5
    classDef pullPlanned fill:#cce5ff,stroke-dasharray: 5 5
    class StreamingSource,PushSource,OpenAISource pushPlanned
    class StreamingEngine,PushAdapter,ParakeetEngine pullPlanned
```

### Combined View

```mermaid
graph TB
    subgraph Apps ["Consumer Apps"]
        Handy[Handy]
        Whispering[Whispering*]
    end

    subgraph Library ["transcription-rs Crate"]
        subgraph Core ["Core"]
            subgraph HighLevel ["<b>(C)</b> High-Level Adapter"]
                StreamingSource["StreamingTranscriptionSource (type)"]
            end

            subgraph LowLevel ["<b>(B)</b> Low-Level Interfaces"]
                StreamingEngine["StreamingTranscriptionEngine (interface)"]
                BatchEngine["BatchTranscriptionEngine [TranscriptionEngine] (interface)"]
            end

            subgraph PushHelper ["<b>(D)</b> Push Adapter"]
                PushAdapter["PushAdapter (type)"]
                PushSource["PushSource (interface)"]
            end
        end

        subgraph Backends ["Backend Implementations — Contributors"]
            ParakeetEngine["[ParakeetEngine] (type)"]
            OpenAISource["OpenAISource (type)"]
            OpenAIEngine["[OpenAIEngine] (type)"]
            WhisperEngine["[WhisperEngine] (type)"]
        end
    end

    subgraph External ["3rd Party APIs — External"]
        subgraph PullStreaming ["Pull Streaming (3)"]
            NeMo[NeMo]
            Vosk[Vosk*]
            Sherpa[sherpa-onnx*]
        end
        subgraph PushStreaming ["Push Streaming (9)"]
            Deepgram[Deepgram*]
            OpenAI[OpenAI]
            ElevenLabs[ElevenLabs*]
        end
        subgraph BatchOnly ["Batch Only (1)"]
            WhisperCpp[whisper.cpp]
        end
    end

    Handy -.->|"<b>(A)</b> returns Transcript"| StreamingSource
    Handy -.->|"<b>(A)</b> returns Transcript"| StreamingEngine
    Handy -->|"<b>(A)</b> returns List&lt;Transcript&gt;"| BatchEngine

    StreamingSource -.-> StreamingEngine

    StreamingEngine -.-> ParakeetEngine
    StreamingEngine -.-> PushAdapter
    
    PushAdapter -.-> PushSource
    PushSource -.-> OpenAISource
    
    BatchEngine --> ParakeetEngine
    BatchEngine --> OpenAIEngine
    BatchEngine --> WhisperEngine

    ParakeetEngine --> NeMo
    OpenAISource -.-> OpenAI
    OpenAIEngine --> OpenAI
    WhisperEngine --> WhisperCpp

    classDef pushPlanned fill:#e1f5e1,stroke-dasharray: 5 5
    classDef pullPlanned fill:#cce5ff,stroke-dasharray: 5 5
    classDef pushExisting fill:#e1f5e1
    classDef pullExisting fill:#cce5ff
    class StreamingSource,PushSource,OpenAISource pushPlanned
    class StreamingEngine,PushAdapter pullPlanned
    class ParakeetEngine,OpenAIEngine,WhisperEngine,BatchEngine pullExisting
```
