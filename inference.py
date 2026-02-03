"""
AToM-FM Inference Script
========================
Load a fine-tuned model and run inference / interactive chat.

Usage:
    python inference.py --model_path ./models/final
    python inference.py --model_path ./models/final --interactive
    python inference.py --model_path ./models/final --prompt "Explain quantum computing"
"""

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils import load_config, print_gpu_info, setup_logging

logger = logging.getLogger(__name__)


def load_inference_model(model_path: str, config: dict):
    """Load the fine-tuned model for inference."""
    model_config = config["model"]

    # Quantization config (same as training)
    quant_cfg = model_config.get("quantization", {})
    bnb_config = None
    if quant_cfg.get("enabled", False):
        compute_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(
            quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )

    base_model_name = model_config["name"]

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """Generate a response for a given instruction."""
    if input_text.strip():
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()


def interactive_mode(model, tokenizer):
    """Run interactive chat loop."""
    print("\n" + "=" * 60)
    print("  AToM-FM Interactive Mode")
    print("  Type 'quit' or 'exit' to stop")
    print("  Type 'params' to adjust generation parameters")
    print("=" * 60)

    params = {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }

    while True:
        print()
        instruction = input("You: ").strip()

        if not instruction:
            continue
        if instruction.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if instruction.lower() == "params":
            print(f"Current params: {params}")
            for key in params:
                val = input(f"  {key} [{params[key]}]: ").strip()
                if val:
                    params[key] = type(params[key])(val)
            print(f"Updated params: {params}")
            continue

        response = generate_response(model, tokenizer, instruction, **params)
        print(f"\nAToM-FM: {response}")


def batch_inference(model, tokenizer, prompts: list[str], **kwargs) -> list[str]:
    """Run inference on a batch of prompts."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}...")
        response = generate_response(model, tokenizer, prompt, **kwargs)
        results.append(response)
    return results


def main():
    parser = argparse.ArgumentParser(description="AToM-FM Inference")
    parser.add_argument("--model_path", type=str, default="./models/final", help="Path to fine-tuned model")
    parser.add_argument("--config_dir", type=str, default="config", help="Config directory")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run")
    parser.add_argument("--input_text", type=str, default="", help="Optional input for the prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config_dir)

    print("=" * 60)
    print("  AToM-FM Inference")
    print("=" * 60)
    print_gpu_info()

    model, tokenizer = load_inference_model(args.model_path, config)
    print("Model loaded successfully!")

    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.prompt:
        response = generate_response(
            model, tokenizer, args.prompt, args.input_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nPrompt: {args.prompt}")
        if args.input_text:
            print(f"Input: {args.input_text}")
        print(f"\nResponse: {response}")
    else:
        # Demo prompts
        demo_prompts = [
            "Explain the concept of transfer learning in deep learning.",
            "Write a Python function that calculates the Fibonacci sequence.",
            "What are the key differences between GPT and BERT architectures?",
        ]
        print("\n--- Running demo prompts ---")
        for prompt in demo_prompts:
            print(f"\nInstruction: {prompt}")
            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            print(f"Response: {response}\n")
            print("-" * 40)


if __name__ == "__main__":
    main()
