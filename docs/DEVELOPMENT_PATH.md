# Development Path

## Goal

Build a universal matrix-program backend where task-specific input adapters and heads can change, while the internal program learns reusable matrix skills.

## Current rule

Do not optimize only for score. Track four things together:

```text
quality
program logic
cost/efficiency
transferable skills
```

## What we learned

### v2.2 step planner

Best early universal result.

```text
score around 0.12968
logical program: evidence -> repair -> memory/route -> aggregation
```

Problem:

```text
no true forward-level cleanup
```

### v2.4 descriptor planner

Added primitive/category descriptors.

Good:

```text
program stayed logical
skills export works
```

Problem:

```text
descriptor_repair was mostly diagnostic, not dominant
```

### v2.5 alive cost

Added forward-level soft gates and cost pressure.

Good:

```text
step/block/primitive soft gates work
cost is measurable and trainable
late growth guard works
descriptor_repair appears late
```

Problem:

```text
cost pressure was too strong and too early
program collapsed toward identity + residual_repair
score dropped to about 0.12619
```

## v2.5b purpose

Keep the good part of v2.5, but prevent cheap-program collapse.

Changes:

```text
lower cost penalty
scheduled cost ramp
scheduled alive sparsity
semantic diversity anti-collapse
stronger late growth guard
more cleanup after late phase
```

Expected result:

```text
score closer to v2.2/v2.4
program keeps evidence/route/memory/aggregate mass
cost/alive remain measurable
late add_width/add_depth reduced
```

## Success criteria for v2.5b

Good run if:

```text
score >= 0.1285
block_cos >= 0.9752
delta_cos >= 0.9320
norm_cos >= 0.9886
semantic_mass >= 0.18
residual_only_ratio <= 0.82
late add_width/add_depth mostly absent after epoch 35
program_pseudocode is richer than identity/residual only
```

## Next after v2.5b

If v2.5b works:

```text
v2.6 baseline comparison
```

Compare against:

```text
linear
low-rank linear
small MLP
tiny attention block
matrix program v2.5b
```

If v2.5b still collapses:

```text
reduce cost more
increase semantic diversity only late
make stage-aware primitive cost inside core
```

If v2.5b is stable:

```text
add MemoryKV primitive
add ablation trace exporter
start skill-prior cold-start experiment
```
