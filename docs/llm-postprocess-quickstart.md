# LLM 后处理快速测试指南

## 环境准备

### 1. 下载模型文件

```bash
mkdir -p models/qwen2.5-0.5b

# 下载 GGUF 量化模型（约 350MB）
# 从 HuggingFace 下载 Qwen2.5-0.5B-Instruct 的 GGUF 版本
# 推荐 q4_k_m 量化（质量与大小平衡）
wget -O models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# 下载 tokenizer.json（GGUF 不含 tokenizer，需单独下载）
wget -O models/qwen2.5-0.5b/tokenizer.json \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json"
```

### 2. 验证文件

```bash
ls -lh models/qwen2.5-0.5b/
# 预期：
# qwen2.5-0.5b-instruct-q4_k_m.gguf  ~350MB
# tokenizer.json                       ~7MB
```

## 依赖说明

| 依赖 | 版本 | 用途 |
|------|------|------|
| candle-core | 0.9.2 | 张量运算 |
| candle-nn | 0.9.2 | 神经网络层 |
| candle-transformers | 0.9.2 | Qwen2 模型架构 + GGUF 加载 |
| tokenizers | 0.22 | HuggingFace tokenizer 加载 |

所有依赖均为纯 Rust（CPU 路径），无需安装任何 C/C++ 库。

## 编译

```bash
# 编译 example（首次编译 candle 约 2-3 分钟）
cargo build --example llm_postprocess --features llm-postprocess
```

## 运行

```bash
# 使用默认测试文本
cargo run --example llm_postprocess --features llm-postprocess

# 自定义输入
cargo run --example llm_postprocess --features llm-postprocess -- \
  --model models/qwen2.5-0.5b/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  --tokenizer models/qwen2.5-0.5b/tokenizer.json \
  --text "今天天气很好我们去公圆玩吧他说号的"
```

### 预期输出

```
原始文本: 今天天气很好我们去公圆玩吧他说号的
修正文本: 今天天气很好，我们去公园玩吧。他说好的。
生成 tokens: 25
耗时: 2.34s
速度: 10.7 tok/s
```

> 性能因硬件而异。Apple Silicon (M1/M2) CPU 约 30-60 tok/s，x86 可能更慢。

## 已知限制

1. **仅 CPU 推理**：当前未启用 Metal/CUDA 加速，纯 CPU 运行
2. **GGUF tokenizer 缺失**：candle 从 GGUF 文件只读取权重，tokenizer 需单独提供 `tokenizer.json`
3. **首次加载较慢**：模型加载约 1-2s（磁盘 I/O），后续推理约 2-5s/句
4. **中文为主**：Qwen2.5 对中文纠错效果好，英文效果取决于上下文
5. **非流式**：当前实现为完整生成后输出，非逐 token 流式
6. **max_tokens 固定**：默认最大生成 256 tokens，超长文本需分句处理

## 故障排查

| 问题 | 解决方案 |
|------|---------|
| 编译错误 `candle-core not found` | 确认使用 `--features llm-postprocess` |
| 运行时 `model file not found` | 检查模型路径是否正确 |
| 运行时 `tokenizer.json not found` | 需单独下载 tokenizer，见上方步骤 |
| 输出乱码 | 检查 GGUF 文件是否完整下载 |
| 内存不足 | q4_0 需 ~280MB RAM，q4_k_m 需 ~350MB |
