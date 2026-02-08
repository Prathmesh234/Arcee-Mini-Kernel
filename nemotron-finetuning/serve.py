"""
Serve NVIDIA Nemotron-3-Nano-30B-A3B-BF16 via vLLM.

Usage:
    # BF16 (default, requires ~60GB VRAM - H100/A100 80GB)
    python serve.py

    # FP8 (requires ~32GB VRAM)
    python serve.py --dtype FP8

    # Custom port and max context
    python serve.py --port 8080 --max-model-len 131072

    # With tool calling + reasoning parser
    python serve.py --enable-tools
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODEL_BF16 = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_FP8 = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
REASONING_PARSER_URL = "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/resolve/main/nano_v3_reasoning_parser.py"
REASONING_PARSER_FILE = "nano_v3_reasoning_parser.py"


def download_reasoning_parser():
    """Download the custom reasoning parser if not present."""
    parser_path = Path(__file__).parent / REASONING_PARSER_FILE
    if parser_path.exists():
        print(f"Reasoning parser already exists: {parser_path}")
        return str(parser_path)

    print(f"Downloading reasoning parser from {REASONING_PARSER_URL}...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            filename="nano_v3_reasoning_parser.py",
            local_dir=str(parser_path.parent),
        )
        print(f"Downloaded reasoning parser: {downloaded}")
        return str(parser_path)
    except Exception as e:
        print(f"Failed to download via hf_hub: {e}")
        print("Falling back to wget...")
        subprocess.run(
            ["wget", "-O", str(parser_path), REASONING_PARSER_URL],
            check=True,
        )
        return str(parser_path)


def build_serve_command(args) -> list[str]:
    """Build the vllm serve command."""
    model = MODEL_FP8 if args.dtype == "FP8" else MODEL_BF16

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", "nemotron",
        "--trust-remote-code",
        "--async-scheduling",
        "--tensor-parallel-size", str(args.tp),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--port", str(args.port),
        "--host", args.host,
    ]

    # FP8-specific flags
    if args.dtype == "FP8":
        cmd.extend(["--kv-cache-dtype", "fp8"])

    # Tool calling + reasoning
    if args.enable_tools:
        parser_path = download_reasoning_parser()
        cmd.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", "qwen3_coder",
            "--reasoning-parser-plugin", parser_path,
            "--reasoning-parser", "nano_v3",
        ])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Serve Nemotron-3-Nano-30B-A3B via vLLM"
    )
    parser.add_argument(
        "--dtype", choices=["BF16", "FP8"], default="BF16",
        help="Model precision (BF16 ~60GB VRAM, FP8 ~32GB VRAM)"
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to serve on (default: 8000)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallel size / number of GPUs (default: 1)"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=262144,
        help="Max context length (default: 262144 = 256k)"
    )
    parser.add_argument(
        "--max-num-seqs", type=int, default=64,
        help="Max concurrent sequences (default: 64)"
    )
    parser.add_argument(
        "--enable-tools", action="store_true",
        help="Enable tool calling with reasoning parser"
    )
    args = parser.parse_args()

    # Set environment variables for FP8
    if args.dtype == "FP8":
        os.environ["VLLM_USE_FLASHINFER_MOE_FP8"] = "1"
        os.environ["VLLM_FLASHINFER_MOE_BACKEND"] = "throughput"

    cmd = build_serve_command(args)
    print(f"Launching vLLM server...")
    print(f"  Model: {'FP8' if args.dtype == 'FP8' else 'BF16'}")
    print(f"  Port: {args.port}")
    print(f"  TP: {args.tp}")
    print(f"  Max context: {args.max_model_len:,} tokens")
    print(f"  Max sequences: {args.max_num_seqs}")
    print(f"  Tools: {'enabled' if args.enable_tools else 'disabled'}")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
