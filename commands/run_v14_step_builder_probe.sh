#!/usr/bin/env bash
set -euo pipefail

OUT="./runs/v14_step_builder_probe_L${V14_LAYERS:-4}_B${V14_BLOCKS:-4}_S${V14_STEPS_PER_BLOCK:-4}"
mkdir -p "$OUT"

python experiments/v14_step_builder/run_probe.py \
  --device cuda \
  --dim "${V14_DIM:-128}" \
  --layers "${V14_LAYERS:-4}" \
  --blocks "${V14_BLOCKS:-4}" \
  --steps-per-block "${V14_STEPS_PER_BLOCK:-4}" \
  --memory-cells "${V14_MEMORY_CELLS:-6}" \
  --tokens "${V14_TOKENS:-64}" \
  --classes "${V14_CLASSES:-10}" \
  --batch "${V14_BATCH:-64}" \
  --steps "${V14_STEPS:-220}" \
  --lr "${V14_LR:-4e-4}" \
  --weight-decay "${V14_WEIGHT_DECAY:-0.02}" \
  --dropout "${V14_DROPOUT:-0.08}" \
  --read-temp "${V14_READ_TEMP:-1.10}" \
  --op-temp "${V14_OP_TEMP:-1.20}" \
  --write-bias "${V14_WRITE_BIAS:--0.35}" \
  --topk-ops "${V14_TOPK_OPS:-3}" \
  --hard-topk-after "${V14_HARD_TOPK_AFTER:-120}" \
  --gate-target "${V14_GATE_TARGET:-0.72}" \
  --logit-target "${V14_LOGIT_TARGET:-8.0}" \
  --lambda-gate-budget "${V14_LAMBDA_GATE_BUDGET:-0.01}" \
  --lambda-logit-budget "${V14_LAMBDA_LOGIT_BUDGET:-0.001}" \
  --lambda-op-entropy-floor "${V14_LAMBDA_OP_ENTROPY_FLOOR:-0.003}" \
  --grad-clip "${V14_GRAD_CLIP:-0.7}" \
  --log-every "${V14_LOG_EVERY:-20}" \
  --controller-every "${V14_CONTROLLER_EVERY:-25}" \
  --archive-skills \
  --out-dir "$OUT" 2>&1 | tee "$OUT/console.log"

echo "DONE: $OUT"
echo "Review: cat $OUT/step_builder_report.md"
echo "Push: bash commands/push_run_report.sh $OUT v14_step_builder_probe 'Add v14 StepBuilder probe report'"
