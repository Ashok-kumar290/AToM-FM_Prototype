"""
AToM-FM Model Module
Handles Qwen model loading with quantization and LoRA adapter setup.
"""

import logging

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)


def get_quantization_config(model_config: dict) -> BitsAndBytesConfig | None:
    """Create BitsAndBytes quantization config for 4-bit QLoRA."""
    quant_cfg = model_config.get("quantization", {})

    if not quant_cfg.get("enabled", False):
        logger.info("Quantization disabled, loading in full precision")
        return None

    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtype_map.get(
        quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
        torch.bfloat16,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    logger.info(
        f"Quantization config: 4-bit={bnb_config.load_in_4bit}, "
        f"type={quant_cfg.get('bnb_4bit_quant_type', 'nf4')}, "
        f"compute_dtype={compute_dtype}"
    )

    return bnb_config


def get_lora_config(model_config: dict) -> LoraConfig:
    """Create LoRA adapter configuration."""
    lora_cfg = model_config.get("lora", {})

    config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("lora_alpha", 128),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    logger.info(
        f"LoRA config: r={config.r}, alpha={config.lora_alpha}, "
        f"dropout={config.lora_dropout}, targets={config.target_modules}"
    )

    return config


def load_tokenizer(model_config: dict) -> AutoTokenizer:
    """Load and configure the tokenizer."""
    tok_cfg = model_config.get("tokenizer", {})
    model_name = tok_cfg.get("name") or model_config["name"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = tok_cfg.get("padding_side", "right")

    logger.info(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"pad_token='{tokenizer.pad_token}', padding_side='{tokenizer.padding_side}'"
    )

    return tokenizer


def load_model(model_config: dict) -> AutoModelForCausalLM:
    """Load the base Qwen model with optional quantization."""
    model_name = model_config["name"]
    bnb_config = get_quantization_config(model_config)

    logger.info(f"Loading model: {model_name}")

    kwargs = {
        "pretrained_model_name_or_path": model_name,
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }

    if model_config.get("revision"):
        kwargs["revision"] = model_config["revision"]

    if bnb_config is not None:
        kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    # Prepare for k-bit training if quantized
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    logger.info(f"Model loaded: {model.__class__.__name__}, params={model.num_parameters():,}")

    return model


def apply_lora(model: AutoModelForCausalLM, model_config: dict) -> AutoModelForCausalLM:
    """Apply LoRA adapters to the model."""
    lora_config = get_lora_config(model_config)
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"LoRA applied: trainable={trainable:,} ({100 * trainable / total:.2f}%), "
        f"total={total:,}"
    )

    return model


def build_model_and_tokenizer(config: dict) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Full pipeline: load model + quantize + apply LoRA + load tokenizer."""
    model_config = config["model"]

    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config)
    model = apply_lora(model, model_config)

    return model, tokenizer


def print_model_summary(model):
    """Print a summary of trainable vs frozen parameters."""
    trainable = 0
    frozen = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()

    total = trainable + frozen
    print(f"{'='*50}")
    print(f"  Model Parameter Summary")
    print(f"{'='*50}")
    print(f"  Trainable:  {trainable:>12,}  ({100*trainable/total:.2f}%)")
    print(f"  Frozen:     {frozen:>12,}  ({100*frozen/total:.2f}%)")
    print(f"  Total:      {total:>12,}")
    print(f"{'='*50}")
