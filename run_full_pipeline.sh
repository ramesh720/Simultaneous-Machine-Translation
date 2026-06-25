#!/usr/bin/env bash
# =============================================================================
# Master Pipeline — Run this on GCP VM to execute everything end-to-end
#
# Usage:
#   bash run_full_pipeline.sh            # Full pipeline
#   bash run_full_pipeline.sh --skip-train  # Skip training, use existing model
#   bash run_full_pipeline.sh --frontend-only  # Just start the frontend
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SKIP_TRAIN=false
FRONTEND_ONLY=false
ADAPTER_PATH="waitk_finetune/checkpoints_multilang/final"

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --frontend-only) FRONTEND_ONLY=true ;;
    esac
done

echo "=============================================="
echo "  Simultaneous MT — Full Pipeline"
echo "=============================================="
echo ""

# --- Step 0: Verify GPU ---
echo "[0/6] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ "$FRONTEND_ONLY" = true ]; then
    echo "Skipping to frontend..."
    echo ""
    echo "[6/6] Starting frontend server..."
    EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "localhost")
    echo ""
    echo "=============================================="
    echo "  Frontend accessible at:"
    echo "  http://${EXTERNAL_IP}:8080"
    echo "=============================================="
    echo ""
    cd frontend
    ADAPTER_ARG=""
    if [ -d "../${ADAPTER_PATH}" ]; then
        ADAPTER_ARG="--adapter ../${ADAPTER_PATH}"
    fi
    python server.py ${ADAPTER_ARG} --port 8080
    exit 0
fi

# --- Step 1: Prepare multi-language data ---
if [ ! -f "multilang_train.tsv" ]; then
    echo "[1/6] Preparing multi-language training data..."
    python prepare_multilang_data.py --out_path multilang_train.tsv --langs te,hi,gu,ta --max_per_direction 50000
    echo ""
else
    echo "[1/6] Training data already exists (multilang_train.tsv)"
fi

# --- Step 2: Prepare IN22 evaluation data ---
if [ ! -d "eval_data" ] || [ ! -f "eval_data/in22_conv_te_en.json" ]; then
    echo "[2/6] Preparing IN22 evaluation data..."
    python prepare_in22_data.py --out_dir ./eval_data --langs te,hi,gu,ta
    echo ""
else
    echo "[2/6] Evaluation data already exists (eval_data/)"
fi

# --- Step 3: Train ---
if [ "$SKIP_TRAIN" = false ]; then
    echo "[3/6] Training Wait-K model with SimulMask (this takes ~3-4 hours on L4)..."
    cd waitk_finetune
    python train.py --config configs/config_multilang.yaml
    cd ..
    echo ""
    echo "  Training complete! Adapter saved to ${ADAPTER_PATH}"
else
    echo "[3/6] Skipping training (--skip-train)"
fi

# --- Step 4: Evaluate ---
echo "[4/6] Running evaluation (all 4 languages, both directions)..."
if [ -d "${ADAPTER_PATH}" ]; then
    ADAPTER_FLAG="--adapter ${ADAPTER_PATH}"
else
    ADAPTER_FLAG=""
    echo "  (No adapter found, evaluating base model)"
fi

# Single GPU evaluation (no accelerate needed)
python eval_simulmt.py \
    --data_dir eval_data \
    --out_dir eval_results \
    --langs te hi gu ta \
    --datasets conv gen \
    --directions x2e e2x \
    --policies full waitk \
    --k 1 3 5 7 \
    --batch-size 4 \
    --limit 100 \
    ${ADAPTER_FLAG} \
    || echo "  [warn] Evaluation had issues but continuing..."

echo ""

# --- Step 5: Generate plots ---
echo "[5/6] Generating quality-latency plots..."
if [ -f "eval_results/metrics_summary.json" ]; then
    python plot_results.py --summary eval_results/metrics_summary.json || echo "  [warn] Plot generation had issues"
fi
echo ""

# --- Step 6: Start frontend ---
echo "[6/6] Starting frontend server..."
EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "localhost")
echo ""
echo "=============================================="
echo "  Pipeline Complete!"
echo ""
echo "  Frontend accessible at:"
echo "  http://${EXTERNAL_IP}:8080"
echo ""
echo "  Press Ctrl-C to stop the server."
echo "=============================================="
echo ""

cd frontend
ADAPTER_ARG=""
if [ -d "../${ADAPTER_PATH}" ]; then
    ADAPTER_ARG="--adapter ../${ADAPTER_PATH}"
fi
python server.py ${ADAPTER_ARG} --port 8080
