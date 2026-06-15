# V2.4 Descriptor-Aware Architecture Planner

## Goal

Use semantic descriptors for primitives/components so the architecture space helps the controller choose operations, steps, and cleanup actions.

This version is a practical incremental step after v2.3.

## What is implemented now

File:

```text
experiments/universal_program/run_universal_matrix_program_v2_4_descriptor_alive.py
```

Command:

```text
commands/run_universal_matrix_program_v2_4_descriptor_alive_hf.sh
```

### 1. Primitive descriptors

Each base primitive has a small semantic profile:

```text
evidence_read: extract/read/early
route_read: route/read/middle
memory_read: memory/read/middle
memory_write: memory/write/late
delta_route: delta/route/repair
residual_repair: repair/delta
aggregate_merge: aggregate/memory/late
```

### 2. Category descriptors

Categories are mapped to descriptor fields:

```text
extract -> extract/read/early
memory -> memory/read/write
repair -> repair/delta/normalize
aggregate -> aggregate/global/late
```

### 3. Descriptor-aware need vector

From current health:

```text
delta_gap
norm pain
head pressure
operator entropy
category consistency
step count
```

controller builds need fields:

```text
delta
repair
memory
route
extract
aggregate
normalize
read/write
```

### 4. Descriptor repair action

New action:

```text
descriptor_repair
```

It nudges `op_prior` and `cat_prior` toward primitives/categories matching current need.

Example:

```text
high delta pain -> delta_route/residual_repair
high memory pressure -> memory_read/memory_write
low category consistency -> repair_category/descriptor_repair
```

### 5. Alive/proxy diagnostics

Full differentiable `step_alive/block_alive/primitive_alive` inside forward is not implemented yet because it requires a deeper core rewrite. v2.4 adds cleanup/alive proxy diagnostics:

```text
alive_proxy.op_prior_energy
alive_proxy.cat_prior_energy
```

and keeps soft cleanup through priors + pending validation.

## What remains for true alive gates

Needed in a future v2.5 core rewrite:

```text
step_alive_logits[L,B,S]
block_alive_logits[L,B]
primitive_alive_logits[P]
```

Forward should use:

```text
h_next = h + step_alive * block_alive * gate * update
primitive_logits += log(sigmoid(primitive_alive))
```

Then prune/recycle can be partly gradient-driven.

## Why not make all growth fully differentiable immediately?

Full soft growth would require computing all inactive layers/blocks/steps/primitives all the time. This is expensive and can make interpretation muddy.

Current compromise:

```text
preallocated slots
soft priors/descriptors/action scores
pending validation
hard activation between epochs
```

Future compromise:

```text
inactive slots have alive near 0
cheap masked forward
controller can raise alive softly
hard commit only after reward
```

## What to inspect in v2.4

Stdout fields:

```text
desc=...
cons=...
repair=...
prune=...
```

Good signs:

```text
action=descriptor_repair
category consistency increases
program_pseudocode categories match operations better
score does not drop below v2.2/v2.3
```

Baseline v2.2:

```text
score     0.12968
block_cos 0.97567
delta_cos 0.93279
norm_cos  0.98884
```

## Next after v2.4

If descriptor repair helps:

```text
v2.5 true alive gates in core forward
```

If score drops:

```text
reduce descriptor prior strength
keep descriptors for logging only
```

If consistency improves but score same:

```text
keep v2.4 as interpretation-cleanup branch
```
