#!/bin/bash
# ============================================
# AToM-FM Environment Setup Script
# Target: RTX 4060 Ti with CUDA
# ============================================

set -e

echo "================================================"
echo "  AToM-FM Prototype - Environment Setup"
echo "================================================"

# --- Check NVIDIA GPU ---
echo ""
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi

# --- Create Conda Environment (recommended) ---
echo ""
echo "[2/6] Creating conda environment..."
if command -v conda &> /dev/null; then
    conda create -n atom-fm python=3.11 -y
    echo "Activate with: conda activate atom-fm"
    # If running in conda context:
    # conda activate atom-fm
else
    echo "Conda not found. Using system Python."
    echo "Recommend installing Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    python3 -m venv venv
    echo "Activate with: source venv/bin/activate"
fi

# --- Install PyTorch with CUDA ---
echo ""
echo "[3/6] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- Install Requirements ---
echo ""
echo "[4/6] Installing project requirements..."
pip install -r requirements.txt

# --- Install bitsandbytes (CUDA) ---
echo ""
echo "[5/6] Installing bitsandbytes for quantization..."
pip install bitsandbytes --prefer-binary

# --- Verify Installation ---
echo ""
echo "[6/6] Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')

import transformers
print(f'Transformers version: {transformers.__version__}')

import peft
print(f'PEFT version: {peft.__version__}')

import bitsandbytes
print(f'Bitsandbytes version: {bitsandbytes.__version__}')
"

echo ""
echo "================================================"
echo "  Setup Complete! Ready for AToM-FM training."
echo "================================================"
