# Next Diagnostics and Matrix Attention Plan

## Goal

Make the universal matrix program easier to understand and improve:

1. choose steps/depth/width/primitives more intelligently
2. add learnable fast matrix attention families
3. add enough diagnostics to see what the program actually learned
4. export the learned matrix program into readable pseudocode

## Smarter step/primitive choice

Steps are the main mechanism for ordered programs.

```text
primitive = what operation to apply
step      = when / in what order to apply it
```

So the controller should not just add steps. It must evaluate whether a block really needs more ordering capacity.

Add step when:

```text
operator entropy is high
several primitives are useful at the same time
previous add_step on similar target helped
step_count is low
head read pressure points to this block
```

Do not add step when:

```text
same target got weak reward recently
step_count already high
operator mix is collapsed to one primitive
block is not read by failing heads
```

Add primitive only when:

```text
old primitives + extra steps + depth/width did not solve plateau
operator entropy is low
residual pattern repeats on heldout
candidate is not duplicate
```

## Matrix attention families to add

All attention must stay matrix-space native and fast.

### 1. Cross-space attention

Already used:

```text
Mechanism -> Evidence
Mechanism -> Task
Mechanism -> Memory
Head -> Blocks
```

Improve logging and specialize by space type.

### 2. Local/window attention

For sequence/particle/audio tasks:

```text
Token_i reads Token_{i-r...i+r}
```

Fast because it is banded/local, not full all-to-all.

### 3. Low-rank / landmark attention

Use evidence/global landmarks:

```text
Token -> EvidenceLandmarks -> Blocks
```

Good when sequence is long.

### 4. Memory key/value attention

Memory should have separate key/value geometry:

```text
query @ memory_keys -> memory_values
```

Useful if memory_read dominates.

### 5. Route attention

A routing matrix chooses which block/step should read which space.

```text
Head error -> route logits -> target blocks/steps
```

This should help controller target growth.

### 6. Pairwise relation attention

For particles/objects:

```text
relative pair features -> relation summary -> Mechanism
```

Should be local/top-k, not dense O(N^2) by default.

## Diagnostics to add

### Core quality

```text
score
block_mse/cos
delta_mse/cos
norm_mse/cos
best_score
reward
```

### Architecture health

```text
active_layer_ids
active_blocks
active_steps
active_primitives
layer_alive/block_alive/step_alive/primitive_alive later
```

### Per-step program

```text
operator_mix_by_layer_block_step
category_mix_by_layer_block_step
gate_by_layer_block_step
route_entropy_by_layer_block_step
```

### Head attribution

```text
head_block_read
head_error_pressure_by_block
head_error_pressure_by_layer
```

### Growth/action memory

```text
action
target_layer
target_block
score_before
score_after_1_epoch
score_after_2_epoch
reward
accepted/rejected
cooldown
pending
```

### Redundancy / pruning

```text
block_cosine_by_layer
step_similarity
primitive_similarity
unused_step_rate
low_gate_steps
```

## Matrix-program to pseudocode exporter

For each final report, generate a readable program.

Input:

```text
analysis_epoch_XXX.json
```

Output:

```text
reports/<run>/program_pseudocode.md
reports/<run>/program_summary.json
```

Example:

```text
L0 B0 S0:
  category: extract
  ops: 0.55 evidence_read + 0.44 residual_repair
  gate: 0.62

L1 B0 S0:
  category: compare/repair
  ops: route_read + residual_repair

L1 B0 S1:
  category: memory
  ops: memory_read + aggregate_merge
```

Also export high-level pattern:

```text
extract -> repair/compare -> route/memory -> delta/repair -> norm
```

## Pattern mining

Detect repeated useful step chains:

```text
route_read -> memory_read -> delta_route
normalize -> residual_repair
local_route -> aggregate_merge
```

If repeated and heldout-useful, store as macro primitive candidate.

## Next implementation order

1. Run v2.1 balanced.
2. Analyze whether action scoring reduces blind add_step repeats.
3. Add pseudocode exporter script.
4. Add head_error_pressure_by_block logs.
5. Add matrix attention variants one at a time:
   - memory key/value attention
   - local/window attention
   - route attention
6. Add prune/recycle gates after action memory is stable.
