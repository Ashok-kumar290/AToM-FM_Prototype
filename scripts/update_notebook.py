import json
import os

notebook_path = r'c:\Users\Harsha\Documents\AToM-FM_Prototype\notebooks\AToM_FM_Training.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Add Full Dependency Installation and W&B login cell
# We'll use %pip to ensure it's installed in the notebook's environment
setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- AToM-FM Environment Setup ---\n",
        "# Run this cell to ensure all dependencies are installed for the notebook kernel\n",
        "%pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n",
        "%pip install transformers datasets accelerate peft trl bitsandbytes wandb pyyaml\n",
        "\n",
        "# Login to Weights & Biases\n",
        "import wandb\n",
        "wandb.login(key=\"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\")"
    ]
}

# Insert at the very beginning of Section 1 (after the intro markdown)
nb['cells'].insert(3, setup_cell)

# 2. Update the introduction markdown to reflect the 3B model and packing
intro_source = nb['cells'][0]['source']
for i, line in enumerate(intro_source):
    if "Qwen/Qwen2.5-1.5B" in line:
        intro_source[i] = line.replace("Qwen2.5-1.5B (default)", "Qwen2.5-3B (Optimized for Accuracy)")
    if "tatsu-lab/alpaca" in line:
        intro_source[i] = line + "  \\n**Optimization:** Sequence Packing enabled (~2x faster)"

# 3. Save it back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with W&B login and optimized settings.")
