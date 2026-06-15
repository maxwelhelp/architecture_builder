#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${UNIVERSAL_MODEL:-google/bert_uncased_L-2_H-128_A-2}"
SCRIPT="experiments/universal_program/run_universal_matrix_program_v1.py"
BACKUP="$(mktemp)"
cp "$SCRIPT" "$BACKUP"
restore_script() {
  cp "$BACKUP" "$SCRIPT" || true
  rm -f "$BACKUP" || true
}
trap restore_script EXIT

# Runtime safety patch: candidate primitive mixing must use separate labels for
# candidate index and batch index. Wrong: cb,bnod->bncd. Correct: co,bnod->bncd.
python - <<'PY'
from pathlib import Path
p = Path("experiments/universal_program/run_universal_matrix_program_v1.py")
s = p.read_text(encoding="utf-8")
s = s.replace('torch.einsum("cb,bnod->bncd", mix, base)', 'torch.einsum("co,bnod->bncd", mix, base)')
p.write_text(s, encoding="utf-8")
PY

python "$SCRIPT" \
  --model-name "$MODEL_NAME" \
  --layer-idx 0 \
  --device cuda \
  --amp bf16 \
  --seq-len 64 \
  --targets block,delta,norm \
  --evidence-cells 40 \
  --task-cells 12 \
  --memory-cells 6 \
  --global-cells 3 \
  --init-layers 4 \
  --max-layers 8 \
  --init-blocks-per-layer 4 \
  --max-blocks-per-layer 8 \
  --max-primitives 24 \
  --category-temp 1.4 \
  --primitive-temp 1.2 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --steps-per-epoch 120 \
  --val-steps 20 \
  --epochs 45 \
  --lr 7e-4 \
  --weight-decay 0.01 \
  --grad-clip 0.7 \
  --lambda-cos 0.25 \
  --controller-min-delta 0.0015 \
  --controller-patience 2 \
  --rollback-patience 1 \
  --out-dir ./runs/universal_matrix_program_v1_hf_45ep
