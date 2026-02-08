# Nemotron Finetuning - Agent Documentation

## Overview

**Pivot from Arcee model merging** (which was producing gibberish) to finetuning NVIDIA's Nemotron-3-Nano-30B-A3B for Triton kernel generation.

### Goal

Use Nemotron-3-Nano-30B-A3B as the base model for generating optimized Triton kernels from PyTorch code. The model will be:
1. Served via vLLM for inference
2. Used to generate reasoning traces on KernelBench problems
3. Eventually finetuned on verified traces for improved Triton generation

---

## Model: NVIDIA Nemotron-3-Nano-30B-A3B

### Architecture

| Property | Value |
|----------|-------|
| **Total Parameters** | 30B |
| **Active Parameters** | 3.5B per token |
| **Architecture** | Hybrid Mamba2-Transformer MoE (`NemotronH` / `NemotronHForCausalLM`) |
| **Model Type** | `nemotron_h` (custom, NOT Llama, NOT DeciLM) |
| **Total Layers** | 52 |
| **Mamba-2 Layers** | 23 (silu activation, 64 heads, head dim 64, SSM state 128) |
| **MoE Layers** | 23 (relu2 activation, intermediate size 1,856/expert) |
| **Attention Layers** | 6 (GQA: 32 query heads, 2 KV heads = 16:1 ratio, head dim 128) |
| **Experts per MoE Layer** | 128 routed + 1 shared (shared intermediate: 3,712) |
| **Experts Activated per Token** | 6 routed + 1 shared |
| **Hidden Size** | 2,688 |
| **Vocab Size** | 131,072 (much larger than Llama's 32K) |
| **Max Context Length** | 1M tokens |
| **Default Context Length** | 256k tokens |
| **Training Data** | 25T tokens |
| **Data Cutoff** | June 25, 2025 (pre-train), Nov 28, 2025 (post-train) |
| **Release Date** | December 15, 2025 |
| **License** | NVIDIA Open Model License (commercial OK) |

### Layer Interleaving Pattern

The 52 layers follow this specific pattern (M=Mamba, E=MoE, *=Attention):
```
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
```

### Why This Architecture is Special

- **Mamba-2 layers**: Linear-time sequence processing (not quadratic like attention). Great for long contexts.
- **MoE layers**: Only 3.5B of 30B params active per token = fast inference with large capacity.
- **Hybrid**: Combines Mamba's efficiency with Transformer attention's precision on key layers.
- **1M context**: Can handle very large PyTorch codebases as input.

### Chat Template

Uses **ChatML-style** `<|im_start|>` / `<|im_end|>` format with special tokens:

| Token | ID | Purpose |
|-------|-----|---------|
| `<\|im_start\|>` | 10 | Message start |
| `<\|im_end\|>` | 11 | Message end |
| `<think>` | 12 | Reasoning trace start |
| `</think>` | 13 | Reasoning trace end |
| `<tool_call>` | 14 | Tool call start |
| `</tool_call>` | 15 | Tool call end |

Supports `enable_thinking` parameter in `apply_chat_template()` to toggle reasoning on/off.

### Supported Languages

- Primary: English
- Also: German, Spanish, French, Italian, Japanese + 19 more
- **43 programming languages** in training data

### Benchmark Performance

| Benchmark | Score | Notes |
|-----------|-------|-------|
| MMLU-Pro | 78.3% | General knowledge |
| AIME25 (w/ tools) | 99.2% | Math reasoning with tools |
| LiveCodeBench | 68.3% | Code generation |
| MiniF2F pass@32 | 79.9% | Math reasoning |
| SWE-Bench | 38.8% | Agentic coding |
| RULER-100@1M | 86.3% | Long context |
| Arena-Hard-V2 | 67.7% | General chat |

### Model Variants

| Variant | VRAM Required | HuggingFace ID |
|---------|---------------|----------------|
| **BF16** | ~60GB | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| **FP8** | ~32GB | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |
| **NVFP4** | ~20GB | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` |

---

## vLLM Setup

### Requirements

- **vLLM >= 0.12.0** (required for Mamba-2 + MoE hybrid support)
- **GPU**: H100-80GB (BF16) or A100-80GB (BF16) or H100/A100 (FP8 with 32GB+)
- **`--trust-remote-code`**: Required (model has custom code)
- **`--async-scheduling`**: Recommended for lower latency

### Quick Start

```bash
# Install dependencies
uv sync

# BF16 on single H100 (default)
python serve.py

# FP8 on single GPU (lower VRAM)
python serve.py --dtype FP8

# With tool calling + reasoning parser
python serve.py --enable-tools

# Custom config
python serve.py --port 8080 --max-model-len 131072 --max-num-seqs 32
```

### Raw vLLM CLI (without wrapper)

```bash
# BF16
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name nemotron \
  --trust-remote-code \
  --async-scheduling \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 64 \
  --port 8000

# FP8 (set env vars first)
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
  --served-model-name nemotron \
  --trust-remote-code \
  --async-scheduling \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --port 8000
```

### Key vLLM Flags

| Flag | Purpose | Default |
|------|---------|---------|
| `--trust-remote-code` | **Required** - model has custom code | - |
| `--async-scheduling` | Reduces host overhead between decoding steps | off |
| `--tensor-parallel-size N` | Number of GPUs for tensor parallelism | 1 |
| `--max-model-len N` | Max context length (input + output tokens) | 262144 |
| `--max-num-seqs N` | Max concurrent sequences | 1024 |
| `--kv-cache-dtype fp8` | Use FP8 KV cache (for FP8 model) | auto |
| `--mamba-ssm-cache-dtype` | SSM cache dtype (`bfloat16` or `float32`) | bfloat16 |
| `--enable-auto-tool-choice` | Enable tool calling | off |
| `--tool-call-parser qwen3_coder` | Tool call format parser | - |
| `--reasoning-parser-plugin FILE` | Custom reasoning parser plugin | - |
| `--reasoning-parser nano_v3` | Reasoning parser name | - |

### Environment Variables

| Variable | When to Set | Value |
|----------|-------------|-------|
| `VLLM_USE_FLASHINFER_MOE_FP8` | FP8 models | `1` |
| `VLLM_FLASHINFER_MOE_BACKEND` | FP8 models | `throughput` |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | Context > 256k | `1` |
| `VLLM_ATTENTION_BACKEND` | Optimal attention | `FLASHINFER` |

### Reasoning Parser

The model uses a custom reasoning parser (`nano_v3_reasoning_parser.py`) for parsing `<think>...</think>` blocks. Download it:

```bash
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py
```

Or use `serve.py --enable-tools` which downloads it automatically.

---

## API Usage (OpenAI-Compatible)

### Basic Chat Completion (Reasoning ON)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="nemotron",
    messages=[{"role": "user", "content": "What is 25 * 37?"}],
    max_tokens=2048,
    temperature=1.0,  # Use 1.0 for reasoning tasks
    top_p=1.0,
)
print(response.choices[0].message.content)
```

### Reasoning OFF (faster, less accurate on hard tasks)

```python
response = client.chat.completions.create(
    model="nemotron",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

### Curl

```bash
# Reasoning ON
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Write a Triton kernel for ReLU"}],
    "max_tokens": 4096,
    "temperature": 1.0
  }'

# Reasoning OFF
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Hello"}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

---

## Generation Parameters

| Use Case | temperature | top_p | Notes |
|----------|-------------|-------|-------|
| Reasoning tasks (math, code) | 1.0 | 1.0 | Let the model think freely |
| Tool calling | 0.6 | 0.95 | More controlled |
| Non-reasoning / factual | 0.0 | - | `do_sample=False` |

### Reasoning Budget Control

For latency-sensitive apps, you can cap reasoning tokens:

```python
# First call: generate reasoning up to budget
response1 = client.chat.completions.create(
    model="nemotron", messages=messages, max_tokens=reasoning_budget
)
reasoning = response1.choices[0].message.content

# If no </think> tag, force-close it
if "</think>" not in reasoning:
    reasoning = f"{reasoning}.\n</think>\n\n"

# Second call: continue with remaining token budget
messages.append({"role": "assistant", "content": reasoning})
response2 = client.completions.create(
    model="nemotron", prompt=tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True),
    max_tokens=remaining_budget,
)
```

---

## Project Structure

```
nemotron-finetuning/
├── AGENTS.md              # This file
├── pyproject.toml         # uv project config (vllm, torch, transformers, openai)
├── serve.py               # vLLM server launcher with all flags configured
├── test_server.py         # Server health + inference tests
├── nano_v3_reasoning_parser.py  # Downloaded reasoning parser (auto-fetched)
└── main.py                # Entry point (placeholder)
```

---

## Hardware Requirements Summary

| Config | GPU | VRAM | Context |
|--------|-----|------|---------|
| BF16 + 256k ctx | 1x H100-80GB | ~60GB | 256k tokens |
| FP8 + 256k ctx | 1x H100-80GB | ~32GB | 256k tokens |
| BF16 + 1M ctx | Multi-GPU | >80GB | 1M tokens |
| NVFP4 + 256k ctx | 1x A100-40GB | ~20GB | 256k tokens |

---

## Integration with KernelBench Pipeline

This model replaces `gpt-oss-120b` in the trace generation pipeline:

```
KernelBench problem (PyTorch)
  → Nemotron-3-Nano generates <think>reasoning</think> + Triton kernel
  → Modal H100 verifies correctness + speedup
  → Keep verified traces for SFT dataset
```

The vLLM server exposes an OpenAI-compatible API, so the existing pipeline code just needs the `base_url` pointed to the Nemotron server.

---

## Known Limitations & Gotchas

### LoRA + MoE: vLLM Does NOT Support LoRA on Expert Layers

**This is critical for the finetuning phase.** vLLM's `fused_moe` kernels do not support LoRA adapters on MoE expert layers (`gate_proj`, `up_proj`, `down_proj`). The `merge_and_unload()` workaround from PEFT also corrupts MoE weights.

**Options for finetuned model serving:**

| Approach | Tool | Works? |
|----------|------|--------|
| LoRA on attention layers only | vLLM | Yes |
| LoRA on MoE expert layers | vLLM | **No** |
| LoRA on MoE expert layers | SGLang | Yes (>= 0.4.0) |
| Full SFT (no LoRA) | vLLM | Yes |
| PEFT inference (no merge) | transformers | Yes (slow) |

**Recommendation:** For base model inference (trace generation), vLLM is great. For serving the finetuned model with LoRA adapters on expert layers, use **SGLang** instead:

```bash
python3 -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --lora-paths "triton_adapter=./path/to/lora" \
  --enable-lora \
  --trust-remote-code \
  --tp 1 \
  --port 8000
```

### Other Gotchas

- **`--enforce-eager` flag**: Use if CUDA graph capture fails on startup (variable MoE layers can cause shape mismatches)
- **All 30B params loaded to VRAM**: Even though only 3.5B are *active* per token, all expert weights must reside in GPU memory
- **KV cache oversizing**: Variable GQA heads across layers may cause vLLM to over-allocate KV cache based on max heads
- **`--mamba-ssm-cache-dtype float32`**: Set for best accuracy (default `bfloat16` in vLLM 0.12.0)

### Community-Reported Issues

1. **HF Transformers is painfully slow** — KV cache (`NemotronHHybridDynamicCache`) not properly initialized by standard pipeline → ~1-2 tokens/sec on H200. NVIDIA says HF Transformers is "for prototyping only". Use vLLM/SGLang/TRT-LLM for production.
2. **Tool calling outputs Python booleans** — Model outputs `"True"`/`"False"` instead of JSON `true`/`false`. Chat template bug (values pass through Python's `str()` instead of `tojson`).
3. **Streaming broken with `enable_thinking=False`** — Known issue in vLLM integration.
4. **Needle-in-haystack retrieval is weak** — ~1/10 pass rate on simple retrieval from provided context vs ~8/10 for comparable models.
5. **Code generation quality mixed** — Some users report underwhelming results on basic programming despite strong benchmarks. Verify on your specific Triton generation use case.

---

## Throughput Notes

From NVIDIA benchmarks (8K input / 16K output, single H200):
- **3.3x higher throughput** than Qwen3-30B
- **2.2x higher throughput** than GPT-OSS-20B
- **4x throughput on B200** compared to FP8-H100 (with NVFP4)

---

## References

- [HuggingFace Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [vLLM Recipe](https://docs.vllm.ai/projects/recipes/en/latest/NVIDIA/Nemotron-3-Nano-30B-A3B.html)
- [vLLM Blog Post](https://blog.vllm.ai/2025/12/15/run-nvidia-nemotron-3-nano.html)
- [NVIDIA vLLM Cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/vllm_cookbook.ipynb)
- [NVIDIA Technical Report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-nvidia-nemotron-3-techniques-tools-and-data-that-make-it-efficient-and-accurate/)
- [HuggingFace Blog](https://huggingface.co/blog/nvidia/nemotron-3-nano-efficient-open-intelligent-models)
