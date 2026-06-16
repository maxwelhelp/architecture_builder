#!/usr/bin/env bash
set -euo pipefail

# Small v14 sweep for speed/health. Defaults are intentionally short.
# Override with: V14_STEPS=200 V14_BATCH=128 bash commands/run_v14_parallel_slots_sweep.sh

STEPS="${V14_STEPS:-120}"
BATCH="${V14_BATCH:-64}"
LOG_EVERY="${V14_LOG_EVERY:-120}"

run_one() {
  local mode="$1"
  local topk="$2"
  local refine="$3"
  echo "=== mode=$mode topk=$topk refine=$refine ==="
  V14_MODE="$mode" V14_TOPK="$topk" V14_REFINE_ITERS="$refine" V14_STEPS="$STEPS" V14_BATCH="$BATCH" V14_LOG_EVERY="$LOG_EVERY" \
    bash commands/run_v14_parallel_slots_probe.sh
}

run_one parallel_refine 1 1
run_one parallel_refine 2 1
run_one parallel_refine 3 1
run_one parallel_refine 2 2
run_one parallel_refine 3 2
run_one hybrid_groups 2 2
run_one hybrid_groups 3 2
run_one sequential_exact 3 2

python tools/compare_v14_probe_reports.py \
  runs/v14_parallel_slots_probe_parallel_refine_topk1_refine1 \
  runs/v14_parallel_slots_probe_parallel_refine_topk2_refine1 \
  runs/v14_parallel_slots_probe_parallel_refine_topk3_refine1 \
  runs/v14_parallel_slots_probe_parallel_refine_topk2_refine2 \
  runs/v14_parallel_slots_probe_parallel_refine_topk3_refine2 \
  runs/v14_parallel_slots_probe_hybrid_groups_topk2_refine2 \
  runs/v14_parallel_slots_probe_hybrid_groups_topk3_refine2 \
  runs/v14_parallel_slots_probe_sequential_exact_topk3_refine2 \
  --out runs/v14_parallel_slots_sweep_compare.md

cat runs/v14_parallel_slots_sweep_compare.md
