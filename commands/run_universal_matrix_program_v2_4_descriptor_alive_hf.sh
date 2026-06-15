#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${UNIVERSAL_MODEL:-google/bert_uncased_L-2_H-128_A-2}"

python experiments/universal_program/run_universal_matrix_program_v2_4_descriptor_alive.py \
  --model-name "$MODEL_NAME" \
  --layer-idx 0 \
  --device cuda \
  --amp bf16 \
  --seq-len 64 \
  --targets block,delta,norm \
  --evidence-cells 40 \
  --task-cells 12 \
  --memory-cells 6 \
  --global-cells 3 \
  --init-layers 4 \
  --max-layers 8 \
  --init-blocks-per-layer 4 \
  --max-blocks-per-layer 8 \
  --init-steps-per-block 1 \
  --max-steps-per-block 4 \
  --max-primitives 24 \
  --category-temp 1.4 \
  --primitive-temp 1.2 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --steps-per-epoch 120 \
  --val-steps 20 \
  --epochs 45 \
  --lr 7e-4 \
  --weight-decay 0.01 \
  --grad-clip 0.7 \
  --lambda-cos 0.25 \
  --controller-min-delta 0.0015 \
  --controller-patience 2 \
  --rollback-patience 1 \
  --pending-epochs 2 \
  --weak-action-reward 0.00025 \
  --out-dir ./runs/universal_matrix_program_v2_4_descriptor_alive_hf_45ep
