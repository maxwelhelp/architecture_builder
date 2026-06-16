#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${UNIVERSAL_MODEL:-google/bert_uncased_L-2_H-128_A-2}"

# v2.6c recovery:
# - do NOT read skills/skill_bank.jsonl, because recent bad runs can pollute it
# - keep weak default skill-prior only
# - delay cost/alive pressure until quality is already learned
# - keep growth possible longer
# - use weak semantic/category losses so they guide but do not dominate task loss

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
  --cleanup-start-frac 0.78 \
  --late-growth-start-frac 0.90 \
  --late-growth-penalty 0.18 \
  --descriptor-action-boost 0.02 \
  --soft-prune-strength 0.12 \
  --alive-init-logit 2.2 \
  --alive-floor 0.05 \
  --lambda-op-cost 0.00025 \
  --lambda-alive-sparsity 0.00004 \
  --lambda-prior-cost 0.02 \
  --cost-start-frac 0.82 \
  --cost-end-frac 1.00 \
  --alive-start-frac 0.86 \
  --alive-end-frac 1.00 \
  --lambda-semantic-diversity 0.0015 \
  --semantic-start-frac 0.35 \
  --semantic-end-frac 0.85 \
  --min-semantic-mass 0.14 \
  --max-residual-only-ratio 0.88 \
  --lambda-skill-loss 0.0008 \
  --skill-start-frac 0.20 \
  --skill-end-frac 0.80 \
  --lambda-category-consistency 0.0005 \
  --consistency-start-frac 0.35 \
  --consistency-end-frac 0.95 \
  --candidate-primitive-cost 0.70 \
  --out-dir ./runs/universal_matrix_program_v2_6c_quality_recovery_hf_70ep
