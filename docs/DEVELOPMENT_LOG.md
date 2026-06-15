# Development Log

## 2026-06-16 — universal_matrix_program_v2_steps result

Run: `universal_matrix_program_v2_steps_hf_45ep`

Final metrics:

```text
block_cos = 0.97543
block_mse = 0.08999
delta_cos = 0.93178
delta_mse = 0.08999
norm_cos  = 0.98851
norm_mse  = 0.02284
```

What worked:

- ordered step slots ran without breaking training
- controller activated steps, depth and width automatically
- final architecture used all 8 layer slots
- L1 widened to 7 blocks
- L1B0 grew to 4 steps
- no candidate primitives were activated

What this means:

- steps are technically valid and differentiable
- but v2 policy is biased: it over-targets L1B0, then depth, then L1 width
- logs were too coarse: `operator_mix_by_layer` hides actual per-step primitive order

Next fix:

- v2.1 balanced growth
- per-step operator/category/gate logs
- head-loss attribution via head block reads
- action memory with pending reward/cooldown
- growth action scoring for step/depth/width/primitive
- rollback/reject harmful growth

## 2026-06-16 — universal_matrix_program_v2_1_balanced added

Files:

```text
experiments/universal_program/run_universal_matrix_program_v2_1_balanced.py
commands/run_universal_matrix_program_v2_1_balanced_hf.sh
docs/V2_1_BALANCED_STEPS_AND_GROWTH_PLAN.md
```

Main changes:

- `TraceStepCore`: logs per layer/block/step operator mixes
- `TraceMatrixHead`: returns block read pressure per head
- `BalancedGrowthController`: scores add_step/add_depth/add_width/add_primitive instead of always trying add_step first
- pending action buffer evaluates growth after 2 epochs
- weak or harmful growth gets cooldown / rollback

Success criteria:

```text
block_cos >= v2
delta_cos > v2
fewer repeated add_step on the same block
clear per-step primitive order in analysis_epoch_XXX.json
candidate primitives only if old primitives + steps/depth/width are insufficient
```
