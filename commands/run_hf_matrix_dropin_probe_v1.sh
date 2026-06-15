#!/usr/bin/env bash
set -euo pipefail

python experiments/hf_dropin/run_hf_matrix_dropin_probe.py \
  --model-name prajjwal1/bert-tiny \
  --layer-idx 0 \
  --device cuda \
  --amp bf16 \
  --seq-len 64 \
  --evidence-cells 40 \
  --task-cells 12 \
  --num-layers 4 \
  --blocks-per-layer 4 \
  --memory-cells 6 \
  --global-cells 3 \
  --category-temp 1.4 \
  --primitive-temp 1.2 \
  --targets block,delta,norm \
  --batch-size 64 \
  --eval-batch-size 128 \
  --steps-per-epoch 120 \
  --val-steps 20 \
  --epochs 40 \
  --lr 7e-4 \
  --weight-decay 0.01 \
  --grad-clip 0.7 \
  --lambda-cos 0.25 \
  --controller \
  --controller-patience 2 \
  --rollback-patience 1 \
  --controller-min-delta 0.0015 \
  --out-dir ./runs/hf_matrix_dropin_probe_v1_40ep
