# transcribe-rs

A Rust library for audio transcription supporting multiple engines including Whisper, Parakeet, and Moonshine.

This library was extracted from the [Handy](https://github.com/cjpais/handy) project to help other developers integrate transcription capabilities into their applications. We hope to support additional ASR models in the future and may expand to include features like microphone input and real-time transcription.

## Features

- **Multiple Transcription Engines**: Support for Whisper, Whisperfile, Parakeet, and Moonshine models
- **Cross-platform**: Works on macOS, Windows, and Linux with optimized backends
- **Hardware Acceleration**: Metal on macOS, Vulkan on Windows/Linux
- **Flexible API**: Common interface for different transcription engines
- **Multi-language Support**: Moonshine supports English, Arabic, Chinese, Japanese, Korean, Ukrainian, Vietnamese, and Spanish

## Parakeet Performance

Using the int8 quantized Parakeet model, performance benchmarks:

- **30x real time** on MBP M4 Max
- **20x real time** on Zen 3 (5700X)
- **5x real time** on Skylake (i5-6500)
- **5x real time** on Jetson Nano CPU


### Required Model Files

**Parakeet Model Directory Structure:**
```
models/parakeet-v0.3/
├── encoder-model.onnx           # Encoder model (FP32)
├── encoder-model.int8.onnx      # Encoder model (For quantized)
├── decoder_joint-model.onnx    # Decoder/joint model (FP32)
├── decoder_joint-model.int8.onnx # Decoder/joint model (For quantized)
├── nemo128.onnx                 # Audio preprocessor
├── vocab.txt                    # Vocabulary file
```

**Whisper Model:**
- Single GGML file (e.g., `whisper-medium-q4_1.bin`)

**Whisperfile:**
- Requires whisperfile binary and a Whisper GGML model
- Whisperfile manages a local server that handles transcription requests

**Moonshine Model Directory Structure:**
```
models/moonshine-tiny/
├── encoder_model.onnx          # Audio encoder
├── decoder_model_merged.onnx   # Decoder with KV cache support
└── tokenizer.json              # BPE tokenizer vocabulary
```

**Moonshine Model Variants:**
| Variant | Language | Model Folder |
|---------|----------|--------------|
| Tiny | English | moonshine-tiny |
| TinyAr | Arabic | moonshine-tiny-ar |
| TinyZh | Chinese | moonshine-tiny-zh |
| TinyJa | Japanese | moonshine-tiny-ja |
| TinyKo | Korean | moonshine-tiny-ko |
| TinyUk | Ukrainian | moonshine-tiny-uk |
| TinyVi | Vietnamese | moonshine-tiny-vi |
| Base | English | moonshine-base |
| BaseEs | Spanish | moonshine-base-es |

**Audio Requirements:**
- Format: WAV
- Sample Rate: 16 kHz
- Channels: Mono (1 channel)
- Bit Depth: 16-bit
- Encoding: PCM

## Model Downloads

- **Parakeet**: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main
- **Whisper**: https://huggingface.co/ggerganov/whisper.cpp/tree/main
- **Whisperfile Binary**: https://github.com/mozilla-ai/llamafile/releases/download/0.9.3/whisperfile-0.9.3
- **Moonshine**: https://huggingface.co/UsefulSensors/moonshine/tree/main/onnx/merged

## Usage

### Parakeet Engine
```rust
use transcribe_rs::{TranscriptionEngine, engines::parakeet::ParakeetEngine};
use std::path::PathBuf;

let mut engine = ParakeetEngine::new();
engine.load_model(&PathBuf::from("path/to/model"))?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
```

### Moonshine Engine
```rust
use transcribe_rs::{TranscriptionEngine, engines::moonshine::{MoonshineEngine, MoonshineModelParams, ModelVariant}};
use std::path::PathBuf;

let mut engine = MoonshineEngine::new();
engine.load_model_with_params(
    &PathBuf::from("path/to/model"),
    MoonshineModelParams::variant(ModelVariant::Tiny),
)?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
```

### Whisperfile Engine
```rust
use transcribe_rs::{TranscriptionEngine, engines::whisperfile::{WhisperfileEngine, WhisperfileModelParams}};
use std::path::PathBuf;

let mut engine = WhisperfileEngine::new(PathBuf::from("whisperfile-0.9.3"));
engine.load_model_with_params(
    &PathBuf::from("models/ggml-small.bin"),
    WhisperfileModelParams::default(),
)?;
let result = engine.transcribe_file(&PathBuf::from("audio.wav"), None)?;
println!("{}", result.text);
```

## Running the Examples

### Setup

1. **Create the models directory:**
   ```bash
   mkdir models
   ```

2. **Download models for the engine you want to use:**

   **For Parakeet:**
   ```bash
   cd models
   wget https://blob.handy.computer/parakeet-v3-int8.tar.gz
   tar -xzf parakeet-v3-int8.tar.gz
   rm parakeet-v3-int8.tar.gz
   cd ..
   ```

   **For Whisper:**
   ```bash
   cd models
   wget https://blob.handy.computer/whisper-medium-q4_1.bin
   cd ..
   ```

   **For Whisperfile:**

   First, download the whisperfile binary:
   ```bash
   wget https://github.com/mozilla-ai/llamafile/releases/download/0.9.3/whisperfile-0.9.3
   chmod +x whisperfile-0.9.3
   ```

   Then download a Whisper GGML model:
   ```bash
   cd models
   wget https://blob.handy.computer/ggml-small.bin
   cd ..
   ```

   **For Moonshine:**

   Download the required model files from [Huggingface](https://huggingface.co/UsefulSensors/moonshine/tree/main/onnx/merged).

   For the Tiny English model:
   ```bash
   mkdir -p models/moonshine-tiny
   cd models/moonshine-tiny
   wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/tiny/encoder_model.onnx
   wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/tiny/decoder_model_merged.onnx
   wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/tiny/tokenizer.json
   cd ../..
   ```

   For other variants (TinyAr, TinyZh, Base, etc.), replace `tiny` in the URLs with the appropriate variant folder name (e.g., `tiny-ar`, `tiny-zh`, `base`, `base-es`).

### Running the Examples

Each engine has its own example file:

```bash
# Run Parakeet example (recommended for performance)
cargo run --example parakeet

# Run Whisper example
cargo run --example whisper

# Run Whisperfile example
cargo run --example whisperfile

# Run Moonshine example
cargo run --example moonshine

# Run OpenAI API example
cargo run --example openai
```

Each example will:
- Load the specified model
- Transcribe a sample audio file
- Display timing information and transcription results
- Show real-time speedup factor

## Acknowledgments

- Big thanks to [istupakov](https://github.com/istupakov/onnx-asr) for the excellent ONNX implementation of Parakeet
- Thanks to NVIDIA for releasing the Parakeet model
- Thanks to the [whisper.cpp](https://github.com/ggerganov/whisper.cpp) project for the Whisper implementation
- Thanks to [Mozilla AI](https://github.com/mozilla-ai/llamafile) for the Whisperfile implementation
- Thanks to [UsefulSensors](https://github.com/usefulsensors) for the Moonshine models and ONNX exports
