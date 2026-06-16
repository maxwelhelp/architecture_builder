#!/usr/bin/env bash
set -euo pipefail

# Full v14 probe with matrix auxiliary losses, gradient health, console log, and health report.
# Defaults are deliberately light: matrix auxiliary losses should guide the program,
# not dominate CE after the synthetic task is already solved.
# Override example:
#   V14_MODE=hybrid_groups V14_STEPS=240 bash commands/run_v14_matrixloss_full_probe.sh

MODE="${V14_MODE:-parallel_refine}"
TOPK="${V14_TOPK:-3}"
REFINE="${V14_REFINE_ITERS:-2}"
RUN_DIR="./runs/v14_parallel_slots_probe_${MODE}_topk${TOPK}_refine${REFINE}"
mkdir -p "$RUN_DIR"

V14_GRAD_HEALTH=1 \
V14_MATRIX_LOSSES=1 \
V14_MODE="$MODE" \
V14_TOPK="$TOPK" \
V14_REFINE_ITERS="$REFINE" \
V14_STEPS="${V14_STEPS:-160}" \
V14_LOG_EVERY="${V14_LOG_EVERY:-20}" \
V14_BATCH="${V14_BATCH:-64}" \
V14_LR="${V14_LR:-5e-4}" \
V14_LAMBDA_WRITE_BUDGET="${V14_LAMBDA_WRITE_BUDGET:-0.02}" \
V14_WRITE_TARGET="${V14_WRITE_TARGET:-0.72}" \
V14_LAMBDA_OP_ENTROPY_FLOOR="${V14_LAMBDA_OP_ENTROPY_FLOOR:-0.006}" \
V14_OP_MIN_ENTROPY="${V14_OP_MIN_ENTROPY:-0.30}" \
V14_LAMBDA_OP_USAGE_BALANCE="${V14_LAMBDA_OP_USAGE_BALANCE:-0.0015}" \
V14_LAMBDA_SLOT_DIVERSITY="${V14_LAMBDA_SLOT_DIVERSITY:-0.0005}" \
  bash commands/run_v14_parallel_slots_probe.sh 2>&1 | tee "$RUN_DIR/console.log"

python tools/analyze_v14_probe_health.py "$RUN_DIR" 2>&1 | tee "$RUN_DIR/health_analyzer_stdout.log"

echo "DONE: $RUN_DIR"
echo "Review: cat $RUN_DIR/health_report.md"
echo "Push: bash commands/push_run_report.sh $RUN_DIR v14_${MODE}_topk${TOPK}_refine${REFINE}_matrixloss"
