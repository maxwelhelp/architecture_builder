#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${UNIVERSAL_MODEL:-google/bert_uncased_L-2_H-128_A-2}"

# Quality anchor based on the last known strong core: v2.2 StepPlanner + UniversalProgramV21.
# This intentionally avoids v2.5/v2.6 alive/cost/skill losses because those reduced score.
# Purpose: recover quality first, then add efficiency/skills only after a strong baseline is restored.

python experiments/universal_program/run_universal_matrix_program_v2_2_step_planner.py \
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
  --steps-per-epoch 160 \
  --val-steps 24 \
  --epochs 70 \
  --lr 7e-4 \
  --weight-decay 0.01 \
  --grad-clip 0.7 \
  --lambda-cos 0.25 \
  --controller-min-delta 0.0012 \
  --controller-patience 3 \
  --rollback-patience 1 \
  --pending-epochs 2 \
  --weak-action-reward 0.00025 \
  --out-dir ./runs/universal_matrix_program_v2_2_quality_anchor_hf_70ep
