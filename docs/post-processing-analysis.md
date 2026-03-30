# ASR 后处理增强分析

## 1. 当前架构

transcribe-rs 使用 **ort (ONNX Runtime)** 静态编译，支持多种 ASR 引擎（Paraformer、SenseVoice、Zipformer 等），
后处理目前仅有 CT-Transformer 标点恢复（`punct.rs`）。

| 维度 | ort + ONNX（当前） | Vosk（已弃用） |
|------|-------------------|---------------|
| 链接方式 | Rust 静态编译，零动态库 | 需要 libvosk.dylib/so |
| 部署复杂度 | 单二进制 + 模型文件 | 二进制 + 动态库 + 模型 |
| 跨平台 | macOS/Linux/Windows 统一 | 每个平台单独编译动态库 |
| 推理性能 | ONNX 优化图，CPU 7ms/句 | 相当 |

**结论**：ort 静态编译路线已被验证，后续扩展应保持零动态库依赖。

---

## 2. 后处理方案对比

### 2.1 专用标点模型

| 模型 | 语言 | 大小 | 延迟 (CPU) | 能力 |
|------|------|------|-----------|------|
| CT-Transformer（当前） | 中英 | ~50MB (int8) | ~7ms/句 | 标点恢复 |
| punct_cap_seg_47lang | 47语言 | ~500MB | ~20ms/句 | 标点 + 大小写 + 分句 |

### 2.2 小型 LLM

| 模型 | 参数量 | 量化大小 | 延迟 (CPU) | 能力 |
|------|--------|---------|-----------|------|
| Qwen2.5-0.5B-Instruct | 0.5B | 280MB (q4_0) / 350MB (q4_k_m) | 2-5s/句 | 标点 + 纠错 + 语义修正 |
| Phi-4-mini | 3.8B | ~2.2GB (q4) | 10-30s/句 | 更强纠错，但太慢太大 |

### 2.3 对比结论

| 维度 | CT-Transformer | Qwen2.5-0.5B |
|------|---------------|---------------|
| 速度 | 极快 (~7ms) | 较慢 (~2-5s) |
| 内存 | ~100MB | ~280-350MB |
| 能力 | 仅标点 | 标点 + 纠错 + 语义 |
| 质量 | 标点准确率高 | 可纠正同音字、语法错误 |
| 适用场景 | 实时/批量 | 离线/高质量后处理 |

---

## 3. Rust 推理框架对比

| 框架 | 语言 | 链接方式 | GGUF 支持 | 成熟度 |
|------|------|---------|----------|--------|
| **candle** (0.9.2) | 纯 Rust | 静态，零 C 依赖 | 原生支持 | HuggingFace 官方 |
| llama-cpp-rs | Rust bindings → C++ | 静态链接 llama.cpp | 原生 | 成熟但引入 C++ |
| ort（现有） | Rust bindings → ONNX Runtime | 静态链接 | 不支持 GGUF | 项目已在用 |

**选择 candle 的理由**：
- 纯 Rust，与项目零动态库策略一致
- 原生支持 GGUF 量化格式（`candle_transformers::quantized`）
- 内置 Qwen2 架构支持（`quantized_qwen2::ModelWeights`）
- HuggingFace 官方维护，API 稳定
- 与现有 ort 依赖无冲突（candle CPU 路径纯 Rust）

---

## 4. 推荐架构：两层管线

```
ASR 原始文本
    │
    ▼
┌─────────────────────┐
│ 第一层：CT-Transformer │  ~7ms，始终运行
│ （标点恢复）            │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ 第二层：Qwen2.5-0.5B  │  ~2-5s，可选
│ （纠错 + 语义修正）     │
└─────────┬───────────┘
          │
          ▼
    最终输出文本
```

### 设计原则

1. **第一层始终运行**：CT-Transformer 速度极快，开销可忽略
2. **第二层可选启用**：通过 feature flag `llm-postprocess` 控制编译
3. **独立依赖**：candle 依赖与 ort 依赖完全隔离，互不影响
4. **渐进式增强**：不修改现有 `TranscriptionEngine` trait 和 `punct.rs`

---

## 5. 延迟 / 内存 / 大小对比

| 指标 | CT-Transformer | Qwen2.5-0.5B (q4_0) | Qwen2.5-0.5B (q4_k_m) | 两层合计 |
|------|---------------|---------------------|----------------------|---------|
| 模型文件 | ~50MB | ~280MB | ~350MB | ~330-400MB |
| 运行内存 | ~100MB | ~280MB | ~350MB | ~380-450MB |
| 推理延迟 | ~7ms | ~2-3s | ~2-5s | ~2-5s |
| 吞吐量 | ~30-60 tok/s | ~30-60 tok/s (M1) | ~30-60 tok/s (M1) | N/A |
| 编译产物增量 | 已含 | +5-8MB | +5-8MB | +5-8MB |

> 以上数据基于 Apple Silicon (M1/M2) CPU 推理，x86 平台可能略慢。

---

## 6. 实施路线

### Phase 1（当前）
- 创建独立 example 验证 candle + Qwen2.5-0.5B 可行性
- 不修改 lib.rs，不影响现有功能

### Phase 2（后续）
- 将 LLM 后处理封装为 `llm_postprocess.rs` 模块
- 集成到 `TranscriptionEngine` trait 的后处理管线
- 支持通过配置切换是否启用

### Phase 3（远期）
- 探索 candle Metal/CUDA 加速
- 评估 Qwen2.5-1.5B 或更大模型的效果
- 考虑流式后处理（逐句修正）
