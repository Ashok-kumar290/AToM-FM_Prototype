"""
AToM-FM Utility Functions
"""

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_dir: str = "config") -> dict:
    """Load and merge model + training configs."""
    model_path = os.path.join(config_dir, "model_config.yaml")
    training_path = os.path.join(config_dir, "training_config.yaml")

    config = {}

    with open(model_path, "r") as f:
        config.update(yaml.safe_load(f))

    with open(training_path, "r") as f:
        config.update(yaml.safe_load(f))

    return config


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure logging to both console and file."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[console, file_handler])


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may reduce performance):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def print_gpu_info() -> dict:
    """Print and return GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_total_gb"] = round(props.total_mem / 1e9, 2)
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1e9, 2)
        info["vram_reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1e9, 2)
        info["cuda_version"] = torch.version.cuda

        print(f"GPU: {info['device_name']}")
        print(f"VRAM: {info['vram_total_gb']} GB total")
        print(f"CUDA: {info['cuda_version']}")
    else:
        print("No CUDA GPU available. Training will be very slow on CPU.")

    return info


def get_vram_usage() -> dict:
    """Get current VRAM usage statistics."""
    if not torch.cuda.is_available():
        return {"error": "No CUDA device"}

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_mem / 1e9
    free = total - allocated

    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "total_gb": round(total, 2),
        "free_gb": round(free, 2),
        "utilization_pct": round(100 * allocated / total, 1),
    }


def print_vram_usage():
    """Print formatted VRAM usage."""
    usage = get_vram_usage()
    if "error" in usage:
        print(usage["error"])
        return

    print(f"VRAM: {usage['allocated_gb']:.2f} / {usage['total_gb']:.2f} GB "
          f"({usage['utilization_pct']}%) | Free: {usage['free_gb']:.2f} GB")


def ensure_dirs(config: dict) -> None:
    """Create all required output directories."""
    dirs = [
        config.get("training", {}).get("output_dir", "models/checkpoints"),
        config.get("training", {}).get("final_model_dir", "models/final"),
        config.get("training", {}).get("logging_dir", "logs"),
        "data/raw",
        "data/processed",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
