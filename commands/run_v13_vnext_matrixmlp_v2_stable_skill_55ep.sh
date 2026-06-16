#!/usr/bin/env bash
set -euo pipefail

# v2 goal:
# - keep the strong 64%+ behavior from v13_vnext_matrixmlp_45ep;
# - reuse its architecture_memory as a soft prior, not a checkpoint;
# - reduce overconfidence/gate saturation with lower LR, lower stage gate bias,
#   stronger head decay, slightly higher dropout, and lower gate floors.

python sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py \
  --data-root ./data/speechcommands \
  --device cuda \
  --amp fp16 \
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
  --dropout 0.08 \
  --evidence-dropout 0.12 \
  --gate-floor 0.10 \
  --late-gate-floor 0.14 \
  --late-gate-floor-end 0.08 \
  --late-gate-floor-decay-start 8 \
  --late-gate-floor-decay-epochs 14 \
  --late-write-target 0.40 \
  --late-dyn-gate-target 0.18 \
  --weak-class-late-read-target 0.35 \
  --role-signal-scale 0.42 \
  --signal-op-bias-scale 0.16 \
  --category-temp 1.30 \
  --primitive-temp 1.25 \
  --lock-last-role \
  --stage-gate-bias-init 0.55 \
  --class-stage-bias-std 0.30 \
  --task-attn-temp 0.95 \
  --evidence-attn-temp 1.05 \
  --operator-temp 1.35 \
  --lr 4e-4 \
  --lr-scheduler cosine \
  --warmup-epochs 4 \
  --min-lr-ratio 0.10 \
  --weight-decay 0.012 \
  --head-weight-decay 0.06 \
  --grad-clip 0.65 \
  --lambda-read-entropy 0.0 \
  --lambda-read-diversity 0.035 \
  --lambda-stage-load-balance 0.004 \
  --lambda-evidence-source-balance 0.018 \
  --lambda-operator-balance 0.010 \
  --lambda-global-diversity 0.008 \
  --lambda-memory-diversity 0.006 \
  --lambda-operator-entropy-per-block 0.0012 \
  --lambda-block-diversity 0.020 \
  --lambda-route-diversity 0.014 \
  --lambda-adapter-diversity 0.008 \
  --lambda-phase-role 0.035 \
  --lambda-global-skip-penalty 0.004 \
  --lambda-late-write 0.020 \
  --lambda-late-dyn-gate 0.016 \
  --lambda-weak-class-late-read 0.020 \
  --lambda-role-diversity 0.014 \
  --lambda-role-anchor 0.012 \
  --lambda-category-diversity 0.006 \
  --lambda-category-entropy 0.001 \
  --architecture-memory ./reports/v13_vnext_matrixmlp_45ep/architecture_memory.jsonl \
  --architecture-memory-strength 0.05 \
  --decomp-every 25 \
  --speed-log \
  --epochs 55 \
  --out-dir ./runs/v13_vnext_matrixmlp_v2_stable_skill_55ep
