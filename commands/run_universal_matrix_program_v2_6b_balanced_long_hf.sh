#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${UNIVERSAL_MODEL:-google/bert_uncased_L-2_H-128_A-2}"
SKILL_BANK="${SKILL_BANK:-./skills/skill_bank.jsonl}"

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
  --epochs 60 \
  --lr 7e-4 \
  --weight-decay 0.01 \
  --grad-clip 0.7 \
  --lambda-cos 0.25 \
  --controller-min-delta 0.0012 \
  --controller-patience 3 \
  --rollback-patience 1 \
  --pending-epochs 2 \
  --weak-action-reward 0.00025 \
  --cleanup-start-frac 0.68 \
  --late-growth-start-frac 0.78 \
  --late-growth-penalty 0.35 \
  --descriptor-action-boost 0.08 \
  --soft-prune-strength 0.22 \
  --alive-init-logit 2.2 \
  --alive-floor 0.05 \
  --lambda-op-cost 0.0006 \
  --lambda-alive-sparsity 0.00010 \
  --lambda-prior-cost 0.08 \
  --cost-start-frac 0.70 \
  --cost-end-frac 0.98 \
  --alive-start-frac 0.75 \
  --alive-end-frac 1.00 \
  --lambda-semantic-diversity 0.004 \
  --semantic-start-frac 0.20 \
  --semantic-end-frac 0.70 \
  --min-semantic-mass 0.18 \
  --max-residual-only-ratio 0.82 \
  --use-skill-loss \
  --skill-bank "$SKILL_BANK" \
  --lambda-skill-loss 0.002 \
  --skill-start-frac 0.12 \
  --skill-end-frac 0.60 \
  --lambda-category-consistency 0.002 \
  --consistency-start-frac 0.18 \
  --consistency-end-frac 0.80 \
  --candidate-primitive-cost 0.70 \
  --out-dir ./runs/universal_matrix_program_v2_6b_balanced_long_hf_60ep
