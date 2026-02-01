# Arcee Mini Kernel - Setup Commands

## Prerequisites

Before running the commands, you need to set up the following:

### 1. Modal Authentication

You need to configure your Modal token to deploy the H100 benchmarking containers:

```bash
modal token set --token-id <YOUR_MODAL_TOKEN_ID> --token-secret <YOUR_MODAL_TOKEN_SECRET>
```

Get your Modal tokens from: https://modal.com/settings

To verify your Modal token is configured:
```bash
modal token info
```

### 2. Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```bash
# Required for Hugging Face datasets
HF_TOKEN=your_huggingface_token_here

# vLLM server configuration (defaults are usually fine)
VLLM_BASE_URL=http://localhost:8000/v1
MODEL_NAME=openai/gpt-oss-120b

# Generation settings (optional - these are the defaults)
MAX_TOKENS=16384
TEMPERATURE=0.7
REASONING_LEVEL=high

# Output configuration
OUTPUT_FILE=reasoning_traces.json
```

**Note:** The Modal token is set via the CLI command above, NOT in the `.env` file.

### 3. GPU Requirements

For the vLLM server:
- **2 GPUs required** (configured for tensor parallelism with `TENSOR_PARALLEL_SIZE=2`)
- The script uses `CUDA_VISIBLE_DEVICES=0,1`
- Ensure you have sufficient GPU memory (model is 120B parameters)

## Commands

### Option 1: Use the All-in-One Script (Recommended)

This script will:
1. Deploy Modal containers for H100 benchmarking
2. Start the vLLM server for gpt-oss-120b

```bash
bash run_gpt_oss.sh
```

**Note:** This script will run the vLLM server in the foreground. Keep this terminal open.

### Option 2: Run Steps Manually

#### Step 1: Deploy Modal Containers

```bash
uv run --no-sync modal deploy modal_app.py
```

This deploys the H100 benchmarking functions to Modal that will be used for validation.

#### Step 2: Start vLLM Server

In a separate terminal, start the vLLM server:

```bash
uv run --no-sync vllm serve openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-prefix-caching \
    --dtype auto \
    --reasoning-parser openai_gptoss \
    --tool-call-parser harmony
```

**Environment variables to set before running:**
```bash
export CUDA_VISIBLE_DEVICES=0,1
```

### Generate Reasoning Traces

Once the vLLM server is running (and Modal is deployed), in a **new terminal**, run the orchestrator:

```bash
uv run --no-sync python orchestrator.py
```

**Optional arguments:**
```bash
uv run --no-sync python orchestrator.py \
    --vllm-url http://localhost:8000/v1 \
    --output reasoning_traces.json \
    --kernelbook-samples 1500 \
    --kernelbench-samples 1000 \
    --batch-size 10 \
    --save-interval 10
```

## Summary of Required Values

| Value | How to Set | Where to Get It |
|-------|-----------|-----------------|
| `MODAL_TOKEN_ID` | `modal token set --token-id <ID>` | https://modal.com/settings |
| `MODAL_TOKEN_SECRET` | `modal token set --token-secret <SECRET>` | https://modal.com/settings |
| `HF_TOKEN` (optional) | Add to `.env` file | https://huggingface.co/settings/tokens |

## Workflow Summary

1. **Set Modal token** (one-time setup):
   ```bash
   modal token set --token-id <ID> --token-secret <SECRET>
   ```

2. **Start vLLM server** (keep this running):
   ```bash
   bash run_gpt_oss.sh
   # OR manually start the vLLM server
   ```

3. **Generate traces** (in a new terminal):
   ```bash
   uv run --no-sync python orchestrator.py
   ```

## Output

The reasoning traces will be saved to `reasoning_traces.json` (or the file specified by `--output`).

The orchestrator will:
- Load PyTorch code samples from KernelBook and KernelBench
- Send each to gpt-oss-120b for Triton kernel generation
- Validate correctness and measure speedup on Modal H100
- Save verified traces with reasoning, code, and benchmark results

## Troubleshooting

### "Modal not configured" error
Run: `modal token set --token-id <ID> --token-secret <SECRET>`

### GPU memory issues
- Reduce `--gpu-memory-utilization` (default: 0.9)
- Reduce `--max-model-len` (default: 16384)
- Ensure you have 2 GPUs available

### vLLM server connection errors
- Check that vLLM server is running: `curl http://localhost:8000/v1/models`
- Verify the URL in orchestrator matches: `--vllm-url http://localhost:8000/v1`
