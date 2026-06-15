#!/usr/bin/env bash
set -euo pipefail

python experiments/dropin_transformer/run_multihead_dropin_probe.py \
  --device cuda \
  --amp bf16 \
  --dim 128 \
  --seq-len 64 \
  --teacher-heads 4 \
  --evidence-cells 40 \
  --task-cells 12 \
  --num-layers 4 \
  --blocks-per-layer 4 \
  --memory-cells 6 \
  --global-cells 3 \
  --category-temp 1.4 \
  --primitive-temp 1.2 \
  --targets attention,ffn,block \
  --weight-attention 1.0 \
  --weight-ffn 1.0 \
  --weight-block 1.2 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --steps-per-epoch 120 \
  --val-steps 20 \
  --epochs 40 \
  --lr 7e-4 \
  --weight-decay 0.01 \
  --grad-clip 0.7 \
  --lambda-cos 0.25 \
  --controller \
  --controller-patience 2 \
  --rollback-patience 2 \
  --controller-min-delta 0.002 \
  --out-dir ./runs/multihead_dropin_probe_v1_40ep
