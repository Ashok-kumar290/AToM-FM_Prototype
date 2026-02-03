"""
AToM-FM Trainer Module
Configures and runs the SFTTrainer for Qwen fine-tuning.
Updated for TRL 0.27+ API.
"""

import logging
import os

from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def get_sft_config(config: dict) -> SFTConfig:
    """Build SFTConfig from config (combines training args and SFT settings for TRL 0.27+)."""
    t = config["training"]
    sft = config.get("sft", {})

    # Calculate warmup steps from ratio if not directly specified
    # warmup_ratio is deprecated, so we'll convert to steps later if needed

    sft_config = SFTConfig(
        output_dir=t.get("output_dir", "./models/checkpoints"),
        num_train_epochs=t.get("num_train_epochs", 3),
        max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=t.get("learning_rate", 2e-4),
        weight_decay=t.get("weight_decay", 0.01),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(t.get("warmup_ratio", 0.03) * 1000),  # Approximate warmup steps
        optim=t.get("optim", "paged_adamw_8bit"),
        bf16=t.get("bf16", True),
        fp16=t.get("fp16", False),
        tf32=t.get("tf32", True),
        max_grad_norm=t.get("max_grad_norm", 0.3),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t.get("logging_steps", 10),
        logging_first_step=t.get("logging_first_step", True),
        report_to=t.get("report_to", "tensorboard"),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 50),
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
        # SFT-specific settings (now in SFTConfig)
        max_length=sft.get("max_seq_length", 2048),
        packing=sft.get("packing", False),
        dataset_text_field=sft.get("dataset_text_field", "text"),
        neftune_noise_alpha=sft.get("neftune_noise_alpha", 5),
    )

    logger.info(
        f"Training args: epochs={sft_config.num_train_epochs}, "
        f"batch={sft_config.per_device_train_batch_size}, "
        f"grad_accum={sft_config.gradient_accumulation_steps}, "
        f"lr={sft_config.learning_rate}, "
        f"effective_batch={sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}"
    )

    return sft_config


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: dict,
) -> SFTTrainer:
    """Create and return a configured SFTTrainer."""
    sft_config = get_sft_config(config)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
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
