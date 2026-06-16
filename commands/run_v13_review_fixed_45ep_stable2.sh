#!/usr/bin/env bash
set -euo pipefail

OUT="./runs/v13_review_fixed_45ep_stable2"
mkdir -p "$OUT"

bash commands/apply_v13_all_fixes.sh

python sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py \
  --data-root ./data/speechcommands \
  --device cuda \
  --amp "${V13_AMP:-fp16}" \
  --classes yes,no,up,down,left,right,on,off,stop,go \
  --train-limit 12000 \
  --val-limit 2000 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --workers 4 \
  --pin-memory \
  --dim 128 \
  --n-mels 64 \
  --hop-length 160 \
  --evidence-cells 48 \
  --topology grid \
  --num-layers 4 \
  --blocks-per-layer 4 \
  --skip-mode on \
  --block-cells 2 \
  --global-cells 3 \
  --memory-cells 6 \
  --gate-floor 0.10 \
  --late-gate-floor 0.16 \
  --late-gate-floor-end 0.04 \
  --late-gate-floor-decay-start 6 \
  --late-gate-floor-decay-epochs 12 \
  --late-write-target 0.30 \
  --late-dyn-gate-target 0.10 \
  --weak-class-late-read-target 0.35 \
  --role-signal-scale 0.25 \
  --signal-op-bias-scale 0.08 \
  --category-temp 1.20 \
  --primitive-temp 1.15 \
  --lock-last-role \
  --stage-gate-bias-init 0.45 \
  --class-stage-bias-std 0.35 \
  --task-attn-temp 0.85 \
  --evidence-attn-temp 1.0 \
  --operator-temp 1.45 \
  --dropout 0.11 \
  --evidence-dropout 0.16 \
  --lr "${V13_LR:-3.8e-4}" \
  --lr-scheduler cosine \
  --warmup-epochs 5 \
  --min-lr-ratio 0.12 \
  --weight-decay 0.018 \
  --head-weight-decay 0.07 \
  --grad-clip 0.60 \
  --lambda-read-entropy 0.0 \
  --lambda-read-diversity 0.040 \
  --lambda-stage-load-balance 0.004 \
  --lambda-evidence-source-balance 0.018 \
  --lambda-operator-balance 0.010 \
  --lambda-global-diversity 0.008 \
  --lambda-memory-diversity 0.010 \
  --lambda-operator-entropy-per-block 0.0020 \
  --lambda-block-diversity 0.034 \
  --lambda-route-diversity 0.026 \
  --lambda-adapter-diversity 0.008 \
  --lambda-phase-role 0.014 \
  --lambda-global-skip-penalty 0.004 \
  --lambda-late-write 0.006 \
  --lambda-late-dyn-gate 0.003 \
  --lambda-weak-class-late-read 0.016 \
  --lambda-role-diversity 0.010 \
  --lambda-role-anchor 0.010 \
  --lambda-category-diversity 0.006 \
  --lambda-category-entropy 0.0010 \
  --lambda-gate-budget 0.10 \
  --lambda-signal-op-norm 0.035 \
  --decomp-every 20 \
  --speed-log \
  --epochs "${V13_EPOCHS:-45}" \
  --out-dir "$OUT" 2>&1 | tee "$OUT/console.log"

LATEST_ANALYSIS="$(ls -1 "$OUT"/seq_analysis_epoch_*.json 2>/dev/null | sort | tail -1 || true)"
if [[ -n "$LATEST_ANALYSIS" ]]; then
  python tools/export_program_pseudocode_v2.py "$LATEST_ANALYSIS" --out "$OUT/program_pseudocode_v2.md" || true
fi
python tools/analyze_v13_health_report.py "$OUT" || true

echo "DONE: $OUT"
echo "Push: bash commands/push_run_report.sh $OUT v13_review_fixed_45ep_stable2 'Add stable2 v13 review fixed 45ep report'"
