#!/usr/bin/env bash
# Prepare IN22 data (both directions, 4 langs) and run the full vs wait-k
# simultaneous-MT evaluation on all available GPUs.
set -euo pipefail
cd "$(dirname "$0")"

NUM_GPUS=${NUM_GPUS:-4}
DATA_DIR=${DATA_DIR:-eval_data}
OUT_DIR=${OUT_DIR:-eval_results}

# 1) Download / build the test sets (needs a HF token with IN22 access).
if [ ! -f "${DATA_DIR}/in22_conv_te_en.json" ]; then
  python prepare_in22_data.py --out_dir "${DATA_DIR}" --langs te,hi,gu,ta
fi

# 2) Evaluate: full + wait-{1,3,5,7}, Indic<->English, conv + gen.
accelerate launch --num_processes "${NUM_GPUS}" eval_simulmt.py \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --langs te hi gu ta \
  --datasets conv gen \
  --directions x2e e2x \
  --policies full waitk \
  --k 1 3 5 7 \
  --batch-size 8 \
  "$@"




# full comparison: fixed + adaptive, all langs/directions/datasets, 4 GPUs
#accelerate launch --num_processes 4 eval_adaptive.py \
#    --langs te hi gu ta --datasets conv gen --directions x2e e2x \
#    --policies full waitk la conf --k 1 3 5 7 --n 2 3 --tau 0.4 0.6 0.8

# smoke test first (recommended)
#accelerate launch --num_processes 4 eval_adaptive.py \
#    --langs te --datasets conv --directions x2e \
#    --policies la conf --n 2 --tau 0.6 --limit 40