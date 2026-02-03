"""
AToM-FM Training Script
=======================
Main entry point for fine-tuning Qwen models with QLoRA.

Usage:
    python train.py
    python train.py --config_dir ./config
    python train.py --model Qwen/Qwen2.5-0.5B --epochs 1  # quick test
"""

import argparse
import logging

from src.dataset import prepare_datasets
from src.model import build_model_and_tokenizer, print_model_summary
from src.trainer import create_trainer, evaluate_model, train_and_save
from src.utils import (
    ensure_dirs,
    load_config,
    print_gpu_info,
    print_vram_usage,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AToM-FM: Fine-tune Qwen with QLoRA")
    parser.add_argument("--config_dir", type=str, default="config", help="Config directory")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--epochs", type=int, default=None, help="Override num epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    parser.add_argument("--max_length", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    return parser.parse_args()


def apply_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to config."""
    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["num_train_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.max_steps:
        config["training"]["max_steps"] = args.max_steps
    if args.lora_r:
        config["model"]["lora"]["r"] = args.lora_r
    if args.max_length:
        config["model"]["tokenizer"]["max_length"] = args.max_length
        config["sft"]["max_seq_length"] = args.max_length
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    return config


def main():
    args = parse_args()

    # Setup
    setup_logging()
    config = load_config(args.config_dir)
    config = apply_overrides(config, args)
    ensure_dirs(config)
    set_seed(config["training"].get("seed", 42))

    # Banner
    print("=" * 60)
    print("  AToM-FM: Adaptive Transformer of Multimodal FM")
    print("  Qwen Fine-Tuning with QLoRA")
    print("=" * 60)

    # GPU Info
    print("\n--- GPU Information ---")
    print_gpu_info()

    # Load model & tokenizer
    print("\n--- Loading Model ---")
    model, tokenizer = build_model_and_tokenizer(config)
    print_model_summary(model)
    print_vram_usage()

    # Load & format dataset
    print("\n--- Preparing Dataset ---")
    train_ds, eval_ds = prepare_datasets(config)
    print(f"Train: {len(train_ds)} samples | Eval: {len(eval_ds)} samples")
    print(f"\nSample formatted prompt:\n{train_ds[0]['text'][:500]}...")

    # Create trainer
    print("\n--- Creating Trainer ---")
    trainer = create_trainer(model, tokenizer, train_ds, eval_ds, config)

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")

    if args.eval_only:
        print("\n--- Running Evaluation Only ---")
        metrics = evaluate_model(trainer)
        print(f"Eval metrics: {metrics}")
    else:
        # Train
        print("\n--- Starting Training ---")
        print_vram_usage()
        metrics = train_and_save(trainer, config)
        print(f"\nTraining metrics: {metrics}")

        # Final eval
        print("\n--- Final Evaluation ---")
        eval_metrics = evaluate_model(trainer)
        print(f"Final eval metrics: {eval_metrics}")

    print_vram_usage()
    print("\n--- Done! ---")


if __name__ == "__main__":
    main()
