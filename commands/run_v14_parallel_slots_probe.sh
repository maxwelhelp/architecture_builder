#!/usr/bin/env bash
set -euo pipefail

MODE="${V14_MODE:-parallel_refine}"
TOPK="${V14_TOPK:-3}"
REFINE="${V14_REFINE_ITERS:-2}"
PROF="${V14_PROFILER:-0}"
GRAD="${V14_GRAD_HEALTH:-0}"
MLOSS="${V14_MATRIX_LOSSES:-0}"
EXTRA=()
if [[ "$PROF" == "1" || "$PROF" == "true" || "$PROF" == "yes" ]]; then
  EXTRA+=(--torch-profiler --profile-wait "${V14_PROFILE_WAIT:-5}" --profile-warmup "${V14_PROFILE_WARMUP:-5}" --profile-active "${V14_PROFILE_ACTIVE:-10}")
fi
if [[ "$GRAD" == "1" || "$GRAD" == "true" || "$GRAD" == "yes" ]]; then
  EXTRA+=(--grad-health)
fi
if [[ "$MLOSS" == "1" || "$MLOSS" == "true" || "$MLOSS" == "yes" ]]; then
  EXTRA+=(
    --lambda-write-budget "${V14_LAMBDA_WRITE_BUDGET:-0.04}"
    --write-target "${V14_WRITE_TARGET:-0.72}"
    --lambda-op-entropy-floor "${V14_LAMBDA_OP_ENTROPY_FLOOR:-0.015}"
    --op-min-entropy "${V14_OP_MIN_ENTROPY:-0.35}"
    --lambda-op-usage-balance "${V14_LAMBDA_OP_USAGE_BALANCE:-0.006}"
    --lambda-slot-diversity "${V14_LAMBDA_SLOT_DIVERSITY:-0.002}"
  )
fi

python experiments/v14_parallel_slots/run_probe.py \
  --mode "$MODE" \
  --device cuda \
  --dim 128 \
  --layers 4 \
  --blocks 4 \
  --tokens 64 \
  --classes 10 \
  --memory-cells 6 \
  --refine-iters "$REFINE" \
  --topk "$TOPK" \
  --warmup-steps "${V14_WARMUP_STEPS:-20}" \
  --steps "${V14_STEPS:-120}" \
  --batch "${V14_BATCH:-64}" \
  --lr "${V14_LR:-5e-4}" \
  --log-every "${V14_LOG_EVERY:-20}" \
  --speed-tail "${V14_SPEED_TAIL:-30}" \
  --out-dir "./runs/v14_parallel_slots_probe_${MODE}_topk${TOPK}_refine${REFINE}" \
  "${EXTRA[@]}"
