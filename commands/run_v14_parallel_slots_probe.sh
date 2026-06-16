#!/usr/bin/env bash
set -euo pipefail

MODE="${V14_MODE:-parallel_refine}"
TOPK="${V14_TOPK:-3}"
REFINE="${V14_REFINE_ITERS:-2}"

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
  --steps 120 \
  --batch 64 \
  --lr 5e-4 \
  --out-dir "./runs/v14_parallel_slots_probe_${MODE}_topk${TOPK}_refine${REFINE}"
