"""
AToM-FM Dataset Module
Handles dataset loading, formatting, and preprocessing for Qwen fine-tuning.
"""

import logging
from typing import Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


def load_atom_dataset(config: dict) -> tuple[Dataset, Dataset]:
    """Load and split dataset based on config."""
    ds_config = config["dataset"]
    ds_name = ds_config["name"]

    if ds_name == "local":
        logger.info("Loading local dataset files...")
        train_ds = load_dataset("json", data_files=ds_config["train_file"], split="train")
        eval_ds = load_dataset("json", data_files=ds_config["eval_file"], split="train")
    else:
        logger.info(f"Loading dataset: {ds_name}")
        subset = ds_config.get("subset", None)
        
        # Get split notation from config
        split_train = ds_config.get("split_train", "train[:90%]")
        split_eval = ds_config.get("split_eval", "train[90%:]")
        
        # Load with split directly (handles percentage notation)
        if subset:
            train_ds = load_dataset(ds_name, subset, split=split_train)
            eval_ds = load_dataset(ds_name, subset, split=split_eval)
        else:
            train_ds = load_dataset(ds_name, split=split_train)
            eval_ds = load_dataset(ds_name, split=split_eval)

    if ds_config.get("shuffle", True):
        train_ds = train_ds.shuffle(seed=ds_config.get("shuffle_seed", 42))

    logger.info(f"Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")
    return train_ds, eval_ds


def format_instruction(sample: dict, config: dict) -> str:
    """Format a single sample using the prompt template."""
    ds_config = config["dataset"]

    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output_text = sample.get("output", "")

    if input_text and input_text.strip():
        template = ds_config["prompt_template"]
        text = template.format(
            instruction=instruction,
            input=input_text,
            output=output_text,
        )
    else:
        template = ds_config["prompt_template_no_input"]
        text = template.format(
            instruction=instruction,
            output=output_text,
        )

    return text


def format_dataset(dataset: Dataset, config: dict) -> Dataset:
    """Apply prompt formatting to the entire dataset."""

    def _format(sample):
        sample["text"] = format_instruction(sample, config)
        return sample

    formatted = dataset.map(_format, desc="Formatting prompts")
    logger.info(f"Formatted {len(formatted)} samples")
    return formatted


def prepare_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """Full pipeline: load -> format -> return train/eval datasets."""
    train_ds, eval_ds = load_atom_dataset(config)
    train_ds = format_dataset(train_ds, config)
    eval_ds = format_dataset(eval_ds, config)
    return train_ds, eval_ds


def create_custom_dataset(
    instructions: list[str],
    outputs: list[str],
    inputs: Optional[list[str]] = None,
    save_path: Optional[str] = None,
) -> Dataset:
    """Create a custom dataset from lists of instructions and outputs.

    Useful for creating small test datasets or domain-specific data.
    """
    data = {
        "instruction": instructions,
        "output": outputs,
        "input": inputs if inputs else [""] * len(instructions),
    }

    ds = Dataset.from_dict(data)

    if save_path:
        ds.to_json(save_path)
        logger.info(f"Dataset saved to {save_path}")

    return ds
