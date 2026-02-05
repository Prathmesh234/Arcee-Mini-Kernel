"""
Trinity-Mini Model Server on Modal

Deploys the LoRA-finetuned Trinity-Mini model on Modal.
Run this once to start the server, then use trinity_inference.py to make requests.

Usage:
    # Deploy the server
    modal deploy trinity_load_modal.py

    # The endpoints will be printed after deployment
"""

import os

import modal
from dotenv import load_dotenv

load_dotenv()

app = modal.App("trinity-triton-server")

# Volume with trained checkpoints
VOLUME_NAME = os.getenv("MODAL_VOLUME_NAME", "arcee-vol")
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

VOLUME_MOUNT = os.getenv("VOLUME_MOUNT", "/vol")
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/vol/models/trinity-triton-sft/checkpoint-40")
BASE_MODEL = os.getenv("BASE_MODEL", "arcee-ai/Trinity-Mini")
MODEL_NAME = os.getenv("MODEL_NAME", "trinity-triton-sft")

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40.0",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "fastapi[standard]",
        "pydantic>=2.0",
    )
    .env({"HF_HOME": "/vol/hf_cache"})
)


def load_model_and_tokenizer():
    """Load the PEFT model and tokenizer."""
    import json

    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    print(f"Loading model from {CHECKPOINT_PATH}...")

    with open(os.path.join(CHECKPOINT_PATH, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config.get("base_model_name_or_path", BASE_MODEL)
    print(f"Base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=False)
    config.pad_token_id = tokenizer.pad_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        config=config,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)

    print("Model loaded!")
    return model, tokenizer


def run_inference(model, tokenizer, messages, max_tokens=2048, temperature=0.7, top_p=0.95):
    """Run inference on the loaded model."""
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


@app.cls(
    gpu="A100-80GB",
    volumes={VOLUME_MOUNT: vol},
    image=image,
    timeout=600,
    scaledown_window=300,
    min_containers=0,
)
class TrinityServer:
    """FastAPI server with PEFT model."""

    @modal.enter()
    def setup(self):
        """Load model once when container starts."""
        self.model, self.tokenizer = load_model_and_tokenizer()

    @modal.fastapi_endpoint(method="POST")
    def v1_chat_completions(self, request: dict) -> dict:
        """OpenAI-compatible chat completions endpoint."""
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 2048)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p", 0.95)

        if not messages:
            return {"error": "messages is required"}

        response_text = run_inference(self.model, self.tokenizer, messages, max_tokens, temperature, top_p)

        return {
            "id": "chatcmpl-trinity",
            "object": "chat.completion",
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(response_text.split()),
                "total_tokens": 0,
            },
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "model": MODEL_NAME}
