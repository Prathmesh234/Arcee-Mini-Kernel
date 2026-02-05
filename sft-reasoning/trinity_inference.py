"""
Trinity-Mini Inference Client

Simple client to call the deployed Trinity server.
Make sure to run `modal deploy trinity_load_modal.py` first.

Usage:
    python trinity_inference.py "def relu(x): return torch.maximum(x, torch.zeros_like(x))"
    python trinity_inference.py --file input.py
"""

import argparse
import os

import requests
from dotenv import load_dotenv

load_dotenv()

# Default endpoint - set TRINITY_ENDPOINT in .env after running: modal deploy trinity_load_modal.py
DEFAULT_ENDPOINT = os.getenv("TRINITY_ENDPOINT")

if not DEFAULT_ENDPOINT:
    raise ValueError("TRINITY_ENDPOINT not set. Run 'modal deploy trinity_load_modal.py' and set the endpoint in .env")


def generate_triton(pytorch_code: str, endpoint: str = None, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    Convert PyTorch code to Triton kernel.

    Args:
        pytorch_code: The PyTorch code to convert
        endpoint: API endpoint URL (uses default if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated Triton kernel code
    """
    url = endpoint or DEFAULT_ENDPOINT

    payload = {
        "model": "trinity-triton-sft",
        "messages": [
            {
                "role": "user",
                "content": f"""Convert the following PyTorch code to an optimized Triton kernel:

```python
{pytorch_code}
```

Generate a complete Triton implementation that produces the same output as the PyTorch code.""",
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=300)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def chat(messages: list, endpoint: str = None, temperature: float = 0.7, max_tokens: int = 2048) -> str:
    """
    Send chat messages to the model.

    Args:
        messages: List of message dicts with 'role' and 'content'
        endpoint: API endpoint URL
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Model response
    """
    url = endpoint or DEFAULT_ENDPOINT

    payload = {
        "model": "trinity-triton-sft",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=300)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch code to Triton kernels")
    parser.add_argument("code", nargs="?", help="PyTorch code to convert")
    parser.add_argument("--file", "-f", help="Read PyTorch code from file")
    parser.add_argument("--endpoint", "-e", help="API endpoint URL")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", "-m", type=int, default=2048, help="Max tokens to generate")

    args = parser.parse_args()

    # Get code from file or argument
    if args.file:
        with open(args.file) as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        # Default example
        code = """def softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)"""

    print("=" * 60)
    print("INPUT:")
    print("=" * 60)
    print(code)
    print("\n" + "=" * 60)
    print("OUTPUT:")
    print("=" * 60)

    result = generate_triton(code, endpoint=args.endpoint, temperature=args.temperature, max_tokens=args.max_tokens)
    print(result)


if __name__ == "__main__":
    main()
