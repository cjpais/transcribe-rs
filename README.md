# transcribe-rs

A Rust library for audio transcription supporting multiple engines including Whisper, Parakeet, and Moonshine.

This library was extracted from the [Handy](https://github.com/cjpais/handy) project to help other developers integrate transcription capabilities into their applications. We hope to support additional ASR models in the future and may expand to include features like microphone input and real-time transcription.

## Features

- **Multiple Transcription Engines**: Support for Whisper, Whisperfile, Parakeet, and Moonshine models
- **Cross-platform**: Works on macOS, Windows, and Linux with optimized backends
- **Hardware Acceleration**: Metal on macOS, Vulkan on Windows/Linux, CUDA on Windows x64 (opt-in)
- **Flexible API**: Common interface for different transcription engines
- **Multi-language Support**: Moonshine supports English, Arabic, Chinese, Japanese, Korean, Ukrainian, Vietnamese, and Spanish
- **Opt-in Dependencies**: Only compile and link the engines you need via Cargo features

## Installation

Add transcribe-rs to your `Cargo.toml` with the features you need:

```toml
[dependencies]
# Include only the engines you want to use
transcribe-rs = { version = "0.1.5", features = ["parakeet", "moonshine"] }

# Or enable all engines
transcribe-rs = { version = "0.1.5", features = ["all"] }
```

### Available Features

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `whisper` | OpenAI Whisper (local, GGML format) | whisper-rs with Metal/Vulkan |
| `parakeet` | NVIDIA Parakeet (ONNX) | ort, ndarray |
| `moonshine` | UsefulSensors Moonshine (ONNX) | ort, ndarray, tokenizers |
| `whisperfile` | Mozilla whisperfile server wrapper | reqwest |
| `openai` | OpenAI API (remote) | async-openai, tokio |
| `all` | All engines enabled | All of the above |
| `whisper-cuda` | Whisper with CUDA (Windows x64) | whisper-rs, CUDA Toolkit 12.0+ |

**Note**: By default, no features are enabled. You must explicitly choose which engines to include.

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

- **Parakeet**:
  - Pre-packaged int8 quantized model: https://blob.handy.computer/parakeet-v3-int8.tar.gz
  - Original model files: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/tree/main
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

Each engine has its own example file. You must specify the required feature when running:

```bash
# Run Parakeet example (recommended for performance)
cargo run --example parakeet --features parakeet

# Run Whisper example
cargo run --example whisper --features whisper

# Run Whisperfile example
cargo run --example whisperfile --features whisperfile

# Run Moonshine example
cargo run --example moonshine --features moonshine

# Run OpenAI API example
cargo run --example openai --features openai
```

Each example will:
- Load the specified model
- Transcribe a sample audio file
- Display timing information and transcription results
- Show real-time speedup factor

## Running Tests

### Running Individual Engine Tests

Tests are feature-gated and require you to specify which engine to test:

```bash
# Test a specific engine
cargo test --features parakeet
cargo test --features whisper
cargo test --features moonshine
cargo test --features whisperfile
cargo test --features openai

# Test multiple engines
cargo test --features "parakeet,moonshine"

# Test all engines
cargo test --all-features
```

### Local Development Shortcuts

The `.cargo/config.toml` file provides convenient aliases for local development:

```bash
# Run all tests with all features enabled
cargo test-all

# Check compilation with all features
cargo check-all

# Build with all features
cargo build-all
```

### Test Environment Setup

**For Whisperfile tests:**

The whisperfile tests require:
1. The whisperfile binary at `models/whisperfile-0.9.3` (or set `WHISPERFILE_BIN` env var)
2. A Whisper GGML model at `models/ggml-small.bin` (or set `WHISPERFILE_MODEL` env var)

```bash
# Download whisperfile binary
wget https://github.com/mozilla-ai/llamafile/releases/download/0.9.3/whisperfile-0.9.3
mv whisperfile-0.9.3 models/
chmod +x models/whisperfile-0.9.3

# Download a model
cd models
wget https://blob.handy.computer/ggml-small.bin
cd ..

# Run tests
cargo test --features whisperfile
```

**For Moonshine tests:**

Download the Moonshine base model:
```bash
mkdir -p models/moonshine-base
cd models/moonshine-base
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/base/encoder_model.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/base/decoder_model_merged.onnx
wget https://huggingface.co/UsefulSensors/moonshine/resolve/main/onnx/merged/base/tokenizer.json
cd ../..

# Run tests
cargo test --features moonshine
```

**For Parakeet tests:**

Download the int8 quantized Parakeet model:
```bash
cd models
wget https://blob.handy.computer/parakeet-v3-int8.tar.gz
tar -xzf parakeet-v3-int8.tar.gz
rm parakeet-v3-int8.tar.gz
cd ..

# Run tests
cargo test --features parakeet
```

**For Whisper tests:**

Whisper tests will skip if models are not available in the expected locations.

## Acknowledgments

- Big thanks to [istupakov](https://github.com/istupakov/onnx-asr) for the excellent ONNX implementation of Parakeet
- Thanks to NVIDIA for releasing the Parakeet model
- Thanks to the [whisper.cpp](https://github.com/ggerganov/whisper.cpp) project for the Whisper implementation
- Big thanks to [jart](http://github.com/jart) for [llamafile](https://github.com/mozilla-ai/llamafile). Thanks to [Mozilla AI](https://github.com/mozilla-ai) for maintaining the [Whisperfile](https://github.com/cjpais/whisperfile) implementation
- Thanks to [UsefulSensors](https://github.com/usefulsensors) for the Moonshine models and ONNX exports
