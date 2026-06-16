#!/usr/bin/env bash
set -euo pipefail

MODE="${V14_MODE:-parallel_refine}"
TOPK="${V14_TOPK:-3}"
REFINE="${V14_REFINE_ITERS:-2}"
PROF="${V14_PROFILER:-0}"
EXTRA=()
if [[ "$PROF" == "1" || "$PROF" == "true" || "$PROF" == "yes" ]]; then
  EXTRA+=(--torch-profiler --profile-wait "${V14_PROFILE_WAIT:-5}" --profile-warmup "${V14_PROFILE_WARMUP:-5}" --profile-active "${V14_PROFILE_ACTIVE:-10}")
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
  --warmup-steps 20 \
  --steps "${V14_STEPS:-120}" \
  --batch "${V14_BATCH:-64}" \
  --lr 5e-4 \
  --log-every "${V14_LOG_EVERY:-20}" \
  --speed-tail "${V14_SPEED_TAIL:-30}" \
  --out-dir "./runs/v14_parallel_slots_probe_${MODE}_topk${TOPK}_refine${REFINE}" \
  "${EXTRA[@]}"
