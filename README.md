# AToM-FM: Adaptive Transformer of Multimodal Foundation Model

A prototype for fine-tuning **Qwen2.5** language models using **QLoRA** (4-bit quantization + LoRA adapters), optimized for consumer GPUs like the **NVIDIA RTX 4060 Ti**.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone https://github.com/Ashok-kumar290/AToM-FM_Prototype.git
cd AToM-FM_Prototype

# 2. Setup environment
chmod +x setup_environment.sh
./setup_environment.sh

# 3. Train (CLI)
python train.py

# 4. Inference
python inference.py --model_path ./models/final --interactive
```

## Project Structure

```
AToM-FM_Prototype/
├── config/
│   ├── model_config.yaml          # Model + LoRA + quantization settings
│   └── training_config.yaml       # Training hyperparameters + dataset config
├── src/
│   ├── __init__.py
│   ├── dataset.py                 # Dataset loading, formatting, preprocessing
│   ├── model.py                   # Model loading, quantization, LoRA setup
│   ├── trainer.py                 # SFTTrainer configuration and training loop
│   └── utils.py                   # Logging, seeds, GPU info, config loading
├── notebooks/
│   └── AToM_FM_Training.ipynb     # Full interactive Jupyter notebook
├── data/
│   ├── raw/                       # Raw dataset downloads
│   └── processed/                 # Formatted datasets
├── models/
│   ├── checkpoints/               # Training checkpoints
│   └── final/                     # Final saved model
├── logs/                          # Training logs (TensorBoard)
├── train.py                       # Main training entry point
├── inference.py                   # Inference and interactive chat
├── requirements.txt               # Python dependencies
├── setup_environment.sh           # One-click environment setup
└── README.md
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (8GB) | RTX 4060 Ti (8-16GB) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB free |
| CUDA | 11.8+ | 12.1+ |

## Model Options

| Model | Parameters | VRAM (QLoRA) | Quality | Speed |
|-------|-----------|-------------|---------|-------|
| `Qwen/Qwen2.5-0.5B` | 0.5B | ~2 GB | Basic | Fastest |
| `Qwen/Qwen2.5-1.5B` | 1.5B | ~4 GB | Good | Fast |
| `Qwen/Qwen2.5-3B` | 3B | ~6 GB | Better | Medium |
| `Qwen/Qwen2.5-7B` | 7B | ~8 GB | Best | Slower |

**Default:** `Qwen2.5-1.5B` — best balance of quality and speed for 8GB VRAM.

## Training Configuration

### Key Parameters (tuned for RTX 4060 Ti)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Quantization | 4-bit NF4 | Double quantization enabled |
| LoRA rank | 64 | ~1-3% trainable params |
| LoRA alpha | 128 | 2x rank |
| LoRA targets | All attention + MLP layers | q/k/v/o_proj + gate/up/down_proj |
| Batch size | 2 | Per device |
| Gradient accumulation | 8 | Effective batch = 16 |
| Learning rate | 2e-4 | With cosine schedule |
| Optimizer | Paged AdamW 8-bit | Saves ~30% optimizer memory |
| Gradient checkpointing | Enabled | Saves ~40% activation memory |
| Precision | bfloat16 | Native on Ampere+ GPUs |
| NEFTune noise | alpha=5 | Improves instruction following |

### Dataset Options

| Dataset | Samples | Type | Config Key |
|---------|---------|------|------------|
| `tatsu-lab/alpaca` | 52K | General instruction | Default |
| `Open-Orca/OpenOrca` | 4M | Chat/instruction | Use subset |
| `sahil2801/CodeAlpaca-20k` | 20K | Code generation | Code tasks |
| Custom JSONL | Any | Your domain | `name: "local"` |

## CLI Usage

```bash
# Full training with defaults
python train.py

# Quick test run (100 steps, smallest model)
python train.py --model Qwen/Qwen2.5-0.5B --max_steps 100

# Custom settings
python train.py --model Qwen/Qwen2.5-3B --epochs 2 --batch_size 1 --lora_r 32

# Evaluation only
python train.py --eval_only

# Inference - single prompt
python inference.py --prompt "Explain quantum computing"

# Inference - interactive mode
python inference.py --interactive

# Inference - custom generation params
python inference.py --interactive --temperature 0.8 --max_new_tokens 1024
```

## Jupyter Notebook

The notebook at `notebooks/AToM_FM_Training.ipynb` provides a full interactive workflow:

1. Environment verification and GPU check
2. Configuration (inline or YAML)
3. Model loading with QLoRA
4. Dataset inspection and token analysis
5. Training with live loss plotting
6. Evaluation metrics
7. Inference testing with comparison (base vs fine-tuned)
8. Model merging and export
9. Custom dataset creation

```bash
cd notebooks
jupyter lab AToM_FM_Training.ipynb
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors, try these in order:

1. Reduce batch size: `per_device_train_batch_size: 1`
2. Reduce sequence length: `max_seq_length: 1024` (or 512)
3. Reduce LoRA rank: `r: 32` (or 16)
4. Use a smaller model: `Qwen/Qwen2.5-0.5B`
5. Ensure gradient checkpointing is enabled (it is by default)

### bitsandbytes Issues

```bash
# If bitsandbytes fails to find CUDA:
pip install bitsandbytes --prefer-binary
# Or build from source:
pip install bitsandbytes --no-binary bitsandbytes
```

### Flash Attention (Optional Speedup)

```bash
pip install flash-attn --no-build-isolation
```

## License

MIT License - see [LICENSE](LICENSE)
