# V2.1 Balanced Steps and Growth Plan

## Why

`universal_matrix_program_v2_steps_hf_45ep` proved that step slots work, but the controller is still biased:

- it filled `L1B0` to 4 steps
- then activated all depth slots
- then widened layer 1
- it did not activate candidate primitives

Final result:

```text
block_cos = 0.97543
delta_cos = 0.93178
norm_cos  = 0.98851
```

This is a useful diagnostic run, but not yet the final universal policy.

## Key distinction

Primitive = operation vocabulary.

```text
evidence_read
route_read
memory_read
delta_route
residual_repair
```

Step = ordered execution slot inside a block.

```text
step0 chooses primitive mix
step1 chooses primitive mix
step2 chooses primitive mix
```

The goal is not to add steps blindly. The goal is to let steps learn ordered primitive programs.

## Required v2.1 changes

### 1. Per-step logging

Current logs average operator/category mixes by layer, hiding the actual step program.

Add:

```text
operator_mix_by_layer_block_step
category_mix_by_layer_block_step
gate_by_layer_block_step
route_entropy_by_layer_block_step
```

### 2. Head-loss attribution to blocks

Use head read weights to assign error pressure:

```text
block_error_pressure[b] = sum_head head_error * head_read_weight[head,b]
```

This prevents always targeting L1B0.

### 3. Action memory

For every growth action store:

```text
action
target_layer
target_block
score_before
score_after_1_epoch
score_after_2_epoch
reward
accepted/rejected
```

If reward is weak or negative:

```text
cooldown(action,target)
rollback or deactivate slot
```

### 4. Balanced action scoring

Do not always try `add_step` first.

Compute scores:

```text
step_score = multi_op_pressure + operator_entropy + previous_step_reward - step_count_penalty - cooldown

depth_score = layer_transition_pressure + delta_gap - active_layers_penalty - cooldown

width_score = head_conflict + block_cosine + read_concentration - active_width_penalty - cooldown

primitive_score = residual_plateau + low_operator_entropy + repeated_residual_pattern - duplicate_penalty - cooldown
```

Pick the best action.

### 5. Add-step conditions

Allow `add_step` only when:

```text
operator mix uses multiple useful primitives
existing step improved score recently OR step_count is still low
not on cooldown
step_count < max_steps
```

Stop repeating `add_step` on the same block if reward is weak.

### 6. Prune / recycle bad growth

Every activated layer/block/step/primitive gets an `alive_logit` or at least an action record.

If it hurts:

```text
rollback absolute best
cooldown action
deactivate slot or reduce alive gate
```

If it is unused:

```text
alive gate -> 0
slot can be recycled later
```

### 7. Differentiable soft deactivation

Add:

```text
layer_alive[layer]
block_alive[layer,block]
step_alive[layer,block,step]
primitive_alive[primitive]
```

Forward uses sigmoid gates. Hard deactivate only between epochs.

### 8. Primitive mining only after old mechanisms fail

Do not add candidate primitive too early.

Trigger if:

```text
add_step failed
add_depth failed
add_width failed
plateau remains
operator entropy low
residual repeats on heldout
```

Then:

```text
collect residual
align basis
fit residual operator
project away existing primitives
deduplicate
activate candidate primitive slot
validate heldout
```

### 9. Latent/space embeddings for operation choice

Use embeddings to bias operation selection:

```text
space_type_emb
head_target_emb
health_signal_emb
step_pos_emb
block_role_emb
```

Primitive choice should see:

```text
current block state
step position
target head error
space type
health signals
```

### 10. Vectorization strategy

Keep correctness first.

Can vectorize:

```text
blocks in same layer/step
primitive bank evaluation
candidate primitive mixes
head reads
health maps
```

Cannot fully parallelize sequential dependency unless using linear/product approximations:

```text
step s+1 depends on h_s
layer l+1 depends on layer l
```

Possible future optimization:

```text
topological tree levels
parallel scan for purely linear product steps
torch.compile / CUDA graphs
masked preallocated tensors [L,B,S,P,D]
```

## Next implementation target

Create:

```text
universal_matrix_program_v2_1_balanced.py
commands/run_universal_matrix_program_v2_1_balanced_hf.sh
```

Main goal:

- same HF task
- step/depth/width/primitive chosen by balanced scores
- per-step logs
- action memory/cooldown
- pruning/recycling hooks

Success criteria:

```text
block_cos >= v2
delta_cos > v2
fewer blind add_step repeats
clear per-step primitive order
```
