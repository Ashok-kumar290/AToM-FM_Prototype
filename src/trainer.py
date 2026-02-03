"""
AToM-FM Trainer Module
Configures and runs the SFTTrainer for Qwen fine-tuning.
"""

import logging
import os

from transformers import TrainingArguments
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def get_training_arguments(config: dict) -> TrainingArguments:
    """Build TrainingArguments from config."""
    t = config["training"]

    args = TrainingArguments(
        output_dir=t.get("output_dir", "./models/checkpoints"),
        num_train_epochs=t.get("num_train_epochs", 3),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 2e-4),
        weight_decay=t.get("weight_decay", 0.01),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        optim=t.get("optim", "paged_adamw_8bit"),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        tf32=t.get("tf32", True),
        max_grad_norm=t.get("max_grad_norm", 0.3),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_dir=t.get("logging_dir", "./logs"),
        logging_steps=t.get("logging_steps", 10),
        logging_first_step=t.get("logging_first_step", True),
        report_to=t.get("report_to", "tensorboard"),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 50),
        eval_accumulation_steps=t.get("eval_accumulation_steps", 4),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        seed=t.get("seed", 42),
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        remove_unused_columns=t.get("remove_unused_columns", False),
        group_by_length=t.get("group_by_length", True),
        disable_tqdm=t.get("disable_tqdm", False),
    )

    logger.info(
        f"Training args: epochs={args.num_train_epochs}, "
        f"batch={args.per_device_train_batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps}, "
        f"lr={args.learning_rate}, "
        f"effective_batch={args.per_device_train_batch_size * args.gradient_accumulation_steps}"
    )

    return args


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: dict,
) -> SFTTrainer:
    """Create and return a configured SFTTrainer."""
    training_args = get_training_arguments(config)
    sft_config = config.get("sft", {})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=sft_config.get("max_seq_length", 2048),
        packing=sft_config.get("packing", False),
        dataset_text_field=sft_config.get("dataset_text_field", "text"),
        neftune_noise_alpha=sft_config.get("neftune_noise_alpha", 5),
    )

    logger.info("SFTTrainer created successfully")
    return trainer


def train_and_save(trainer: SFTTrainer, config: dict) -> dict:
    """Run training and save the final model."""
    logger.info("Starting training...")
    result = trainer.train()

    # Log metrics
    metrics = result.metrics
    logger.info(f"Training complete. Metrics: {metrics}")

    # Save final model
    final_dir = config["training"].get("final_model_dir", "./models/final")
    os.makedirs(final_dir, exist_ok=True)

    trainer.save_model(final_dir)
    trainer.model.config.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")

    return metrics


def evaluate_model(trainer: SFTTrainer) -> dict:
    """Run evaluation and return metrics."""
    logger.info("Running evaluation...")
    metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {metrics}")
    return metrics
