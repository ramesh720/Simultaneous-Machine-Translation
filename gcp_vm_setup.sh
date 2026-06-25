#!/usr/bin/env bash
# =============================================================================
# VM Setup Script — Run this INSIDE the GCP VM after SSH
#
# Usage:
#   cd Simultaneous-Machine-Translation
#   bash gcp_vm_setup.sh
# =============================================================================
set -euo pipefail

echo "=============================================="
echo "  Setting up Simultaneous MT environment"
echo "=============================================="

# --- Step 1: Verify GPU ---
echo "[1/7] Checking GPU..."
nvidia-smi || { echo "ERROR: GPU not detected. Wait 2-3 min for driver install, then retry."; exit 1; }
echo ""

# --- Step 2: Install Python dependencies ---
echo "[2/7] Installing Python packages..."
pip install --upgrade pip
pip install -r waitk_finetune/requirements.txt
pip install unbabel-comet fire fastapi uvicorn python-multipart sse-starlette aiofiles
echo ""

# --- Step 3: HuggingFace login ---
echo "[3/7] HuggingFace authentication..."
echo "You need a HuggingFace token to download datasets and models."
echo "Get one at: https://huggingface.co/settings/tokens"
echo ""
if ! huggingface-cli whoami &>/dev/null; then
    read -rp "Enter your HuggingFace token: " HF_TOKEN
    huggingface-cli login --token "${HF_TOKEN}"
fi
echo ""

# --- Step 4: Create dev branch ---
echo "[4/7] Setting up git branch..."
git checkout -b dev/gcp-multilang 2>/dev/null || git checkout dev/gcp-multilang
echo ""

# --- Step 5: Prepare IN22 evaluation data (all 4 languages) ---
echo "[5/7] Preparing IN22 evaluation data..."
python prepare_in22_data.py --out_dir ./eval_data --langs te,hi,gu,ta
echo ""

# --- Step 6: Prepare multi-language training data ---
echo "[6/7] Preparing multi-language training data..."
python prepare_multilang_data.py --out_path ./multilang_train.tsv --langs te,hi,gu,ta --max_per_direction 50000
echo ""

# --- Step 7: Verify setup ---
echo "[7/7] Verifying setup..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import transformers, peft, pytorch_lightning
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'PyTorch Lightning: {pytorch_lightning.__version__}')
"
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Train:  bash run_full_pipeline.sh"
echo "  2. Or step by step:"
echo "     cd waitk_finetune"
echo "     python train.py --config configs/config_multilang.yaml"
echo ""
