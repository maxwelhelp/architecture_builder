#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${UNIVERSAL_MODEL:-google/bert_uncased_L-2_H-128_A-2}"

# v2.6d quality-first recovery:
# Use v2.6 trainable-prior code, but make it behave like the good v2.2/v2.4
# for most of training. Architecture losses are effectively off until very late.
# This tests whether the quality drop came from over-regularization, not from core capacity.

python experiments/universal_program/run_universal_matrix_program_v2_6_loss_driven_skills.py \
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
  --controller-min-delta 0.0010 \
  --controller-patience 4 \
  --rollback-patience 1 \
  --pending-epochs 2 \
  --weak-action-reward 0.00020 \
  --cleanup-start-frac 0.90 \
  --late-growth-start-frac 0.96 \
  --late-growth-penalty 0.08 \
  --descriptor-action-boost 0.00 \
  --soft-prune-strength 0.05 \
  --alive-init-logit 2.2 \
  --alive-floor 0.05 \
  --lambda-op-cost 0.00005 \
  --lambda-alive-sparsity 0.00001 \
  --lambda-prior-cost 0.00 \
  --cost-start-frac 0.94 \
  --cost-end-frac 1.00 \
  --alive-start-frac 0.96 \
  --alive-end-frac 1.00 \
  --lambda-semantic-diversity 0.00010 \
  --semantic-start-frac 0.80 \
  --semantic-end-frac 1.00 \
  --min-semantic-mass 0.08 \
  --max-residual-only-ratio 0.95 \
  --lambda-skill-loss 0.00000 \
  --skill-start-frac 0.95 \
  --skill-end-frac 1.00 \
  --lambda-category-consistency 0.00000 \
  --consistency-start-frac 0.95 \
  --consistency-end-frac 1.00 \
  --candidate-primitive-cost 0.70 \
  --out-dir ./runs/universal_matrix_program_v2_6d_quality_first_hf_70ep
