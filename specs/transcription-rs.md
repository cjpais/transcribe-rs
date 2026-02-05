# transcription-rs API Design

## Overview

**transcription-rs** unifies transcription APIs behind a common interface.

Realtime streaming transcription APIs come in two styles—pull (you call, get result) and push (results arrive via callbacks). App developers want both styles too:
- Most prefer push for simplicity (just receive callbacks)
- Some need pull for control (custom buffering, timing, multi-stream)

transcription-rs abstracts away the impedance mismatch: backends implement whichever style matches their API, apps consume whichever style they prefer.

## Sub-Specs

Each spec follows the format: one-sentence insight → simple code example → details. Letters correspond to diagram labels below.

| | Spec | Insight | Audience |
|-|------|---------|----------|
| **(A)** | Transcript Type | One unified result shape for partial, final, and batch | Everyone |
| **(B)** | Streaming Engine | Pull-based core interface — the common target for all backends | Backend implementors, power users |
| **(C)** | High-Level Adapter | Push audio, receive callbacks — library handles threading | App developers |
| **(D)** | Push Adapter | 4 methods instead of 7 for WebSocket backends | Contributors adding cloud APIs |

**Why expose the low-level API (B)?** It's the common target for backend implementors, so all backends benefit from the high-level adapter (C). Minimal extra work to expose, and some users need the control.

**Why the high-level adapter (C)?** Without it, every app duplicates threading logic — easy to mess up. The library owns the decode thread; apps just push audio and receive callbacks.

**Why the push adapter (D)?** Push-based backends (Deepgram, OpenAI, ElevenLabs) don't fit the pull interface. `PushSource` is simpler to implement (4 methods vs 7), and `PushAdapter` converts it to the common interface.

See also: [Appendix](transcription-rs-appendix.md) (API survey, migration guide, implementation details)

## Architecture Diagrams

### Legend

```mermaid
graph LR
    ExistingPull["Existing [OldName]"] -- solid --> PlannedPull[Planned Pull]
    ExistingPull -. dashed .-> PlannedPush[Planned Push]
    Omitted[Omitted*]

    classDef pullExisting fill:#cce5ff
    classDef pullPlanned fill:#cce5ff,stroke-dasharray: 5 5
    classDef pushPlanned fill:#e1f5e1,stroke-dasharray: 5 5
    class ExistingPull pullExisting
    class PlannedPull pullPlanned
    class PlannedPush pushPlanned
```

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

    Handy -->|"<b>(A)</b> List&lt;Transcript&gt;"| BatchEngine

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

    Handy -.->|"<b>(A)</b> Transcript"| StreamingSource
    Handy -.->|"<b>(A)</b> Transcript"| StreamingEngine

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

    Handy -.->|"<b>(A)</b> Transcript"| StreamingSource
    Handy -.->|"<b>(A)</b> Transcript"| StreamingEngine
    Handy -->|"<b>(A)</b> List&lt;Transcript&gt;"| BatchEngine

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
