#!/usr/bin/env bash
set -euo pipefail

OUT="./runs/v13_review_fixed_45ep_tuned"
mkdir -p "$OUT"

# Apply/smoke patch first so the run cannot accidentally use the old logic.
bash commands/apply_v13_review_fixes.sh

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
  --gate-floor 0.13 \
  --late-gate-floor 0.20 \
  --late-gate-floor-end 0.06 \
  --late-gate-floor-decay-start 8 \
  --late-gate-floor-decay-epochs 14 \
  --late-write-target 0.36 \
  --late-dyn-gate-target 0.12 \
  --weak-class-late-read-target 0.35 \
  --role-signal-scale 0.35 \
  --signal-op-bias-scale 0.14 \
  --category-temp 1.15 \
  --primitive-temp 1.10 \
  --lock-last-role \
  --stage-gate-bias-init 0.75 \
  --class-stage-bias-std 0.35 \
  --task-attn-temp 0.85 \
  --evidence-attn-temp 1.0 \
  --operator-temp 1.30 \
  --dropout 0.08 \
  --evidence-dropout 0.14 \
  --lr "${V13_LR:-4.5e-4}" \
  --lr-scheduler cosine \
  --warmup-epochs 4 \
  --min-lr-ratio 0.10 \
  --weight-decay 0.012 \
  --head-weight-decay 0.05 \
  --grad-clip 0.65 \
  --lambda-read-entropy 0.0 \
  --lambda-read-diversity 0.035 \
  --lambda-stage-load-balance 0.003 \
  --lambda-evidence-source-balance 0.016 \
  --lambda-operator-balance 0.012 \
  --lambda-global-diversity 0.006 \
  --lambda-memory-diversity 0.006 \
  --lambda-operator-entropy-per-block 0.0015 \
  --lambda-block-diversity 0.024 \
  --lambda-route-diversity 0.018 \
  --lambda-adapter-diversity 0.006 \
  --lambda-phase-role 0.018 \
  --lambda-global-skip-penalty 0.003 \
  --lambda-late-write 0.014 \
  --lambda-late-dyn-gate 0.006 \
  --lambda-weak-class-late-read 0.018 \
  --lambda-role-diversity 0.008 \
  --lambda-role-anchor 0.010 \
  --lambda-category-diversity 0.005 \
  --lambda-category-entropy 0.0012 \
  --decomp-every 20 \
  --speed-log \
  --epochs "${V13_EPOCHS:-45}" \
  --out-dir "$OUT" 2>&1 | tee "$OUT/console.log"

LATEST_ANALYSIS="$(ls -1 "$OUT"/seq_analysis_epoch_*.json 2>/dev/null | sort | tail -1 || true)"
if [[ -n "$LATEST_ANALYSIS" ]]; then
  python tools/export_program_pseudocode.py "$LATEST_ANALYSIS" --out "$OUT/program_pseudocode.md" || true
fi
python tools/analyze_v13_health_report.py "$OUT" || true

echo "DONE: $OUT"
echo "Review: cat $OUT/health_diagnosis.md"
echo "Push: bash commands/push_run_report.sh $OUT v13_review_fixed_45ep_tuned 'Add tuned v13 review fixed 45ep report'"
