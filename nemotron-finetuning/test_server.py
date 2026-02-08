"""
Test the Nemotron vLLM server is working correctly.

Usage:
    python test_server.py                    # Test with reasoning enabled (default)
    python test_server.py --no-thinking      # Test with reasoning disabled
    python test_server.py --url http://host:port  # Custom server URL
"""

import argparse
import json
import sys

from openai import OpenAI


def test_basic_completion(client: OpenAI, model: str, enable_thinking: bool = True):
    """Test a basic chat completion."""
    print("=" * 60)
    print(f"Test: Basic completion (thinking={'on' if enable_thinking else 'off'})")
    print("=" * 60)

    messages = [
        {"role": "user", "content": "What is 25 * 37? Show your work."},
    ]

    extra_body = {}
    if not enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=1.0 if enable_thinking else 0.0,
        top_p=1.0 if enable_thinking else None,
        extra_body=extra_body if extra_body else None,
    )

    choice = response.choices[0]
    print(f"Finish reason: {choice.finish_reason}")
    print(f"Content:\n{choice.message.content}")
    print(f"\nUsage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total")
    print()
    return True


def test_triton_generation(client: OpenAI, model: str):
    """Test Triton kernel generation (our actual use case)."""
    print("=" * 60)
    print("Test: Triton kernel generation")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": "You are an expert GPU programmer. Convert PyTorch operations to optimized Triton kernels.",
        },
        {
            "role": "user",
            "content": """Convert this PyTorch code to an optimized Triton kernel:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]
```

Provide a complete Triton kernel with a `triton_kernel_wrapper` function.""",
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
    )

    choice = response.choices[0]
    print(f"Finish reason: {choice.finish_reason}")
    print(f"Content:\n{choice.message.content[:2000]}...")
    print(f"\nUsage: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion = {response.usage.total_tokens} total")
    print()
    return True


def test_health(client: OpenAI):
    """Test server health via models endpoint."""
    print("=" * 60)
    print("Test: Server health check")
    print("=" * 60)

    models = client.models.list()
    for m in models.data:
        print(f"  Available model: {m.id}")
    print()
    return len(models.data) > 0


def main():
    parser = argparse.ArgumentParser(description="Test Nemotron vLLM server")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="Server URL")
    parser.add_argument("--model", default="nemotron", help="Model name")
    parser.add_argument("--no-thinking", action="store_true", help="Disable reasoning")
    args = parser.parse_args()

    client = OpenAI(base_url=args.url, api_key="EMPTY")

    tests = [
        ("Health check", lambda: test_health(client)),
        ("Basic completion", lambda: test_basic_completion(client, args.model, not args.no_thinking)),
        ("Triton generation", lambda: test_triton_generation(client, args.model)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"ERROR: {e}\n")
            results.append((name, f"ERROR: {e}"))

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
