# Dynamic Program Builder Core

This document captures the core architectural problem and the intended direction of the system.

The current v13 line is a useful differentiable prototype, but it is still mostly a fixed soft architecture. It can learn weights inside a predefined topology, but it does not yet behave like a living program builder that detects weak regions, grows useful structure, replaces failed steps, prunes dead parts, and archives reusable algorithmic fragments.

The long-term goal is not just a bigger model. The goal is a system that learns small effective programs from primitives and uses them to build larger task-specific architectures.

## Root problem

A fixed soft matrix-program can overfit or plateau for architectural reasons, not only because of dataset size or regularization.

Main failure mode:

```text
fixed topology
+ many soft primitive mixtures
+ no real step-level growth/replacement
+ no reliable pruning
= train can improve while validation stalls or falls
```

This means the model may be using capacity as a soft neural soup instead of discovering compact reusable algorithms.

The system must therefore move from:

```text
learn weights inside a fixed architecture
```

to:

```text
learn, test, replace, grow, prune, archive, and reuse small matrix programs
```

## Why gradient alone is not enough

We see gradients for primitives, blocks, routes, and writes, but raw gradient is ambiguous.

Low gradient can mean:

```text
- the component is useless;
- the component is already solved;
- the component is gated off and receives no signal;
- the component is starved by routing or class-read decisions.
```

High gradient can mean:

```text
- the component is important;
- the component is unstable;
- the component is overloaded;
- the component is hurting the loss.
```

Therefore the controller must track utility, not just gradient.

## UtilityTracker

Every layer, block, step, primitive, route, and register write should have a utility record.

Minimum fields:

```text
gate_mass          how often the element actually writes or participates
op_mass            which primitives it uses
class_read_mass    whether task/class heads read it
grad_norm          whether gradient reaches it
update_norm        whether it changes state
write_norm         whether it writes meaningful deltas
val_delta_proxy    whether changes correlate with validation improvement
cost               estimated compute cost
age                how long the element has trained
status             newborn / learning / useful / overloaded / dead / replace_candidate / prune_candidate / frozen
```

A practical utility score can start as:

```text
utility = gate_mass
        * class_read_mass
        * healthy_update_norm
        * healthy_grad_norm
        * val_or_train_help_proxy
        / compute_cost
```

This score should not be treated as perfect truth. It is a decision signal for probation, growth, replacement, pruning, and freezing.

## StepPlanner is the heart of the system

A block should not be only one large soft update. A useful block should be able to build a small algorithm made of steps.

Target structure:

```text
Layer
  Block
    Step
      read_source
      primitive/category
      selected primitive or top-k primitives
      write target
      write gate
```

Example mini-program:

```text
S0: read evidence/time/onset
S1: compare with memory or class-pair pressure
S2: repair residual / normalize / suppress common mode
S3: write compact useful state or keep previous state
```

Steps are the most important reusable unit. A layer or block is useful only if it can assemble small effective algorithms from these steps.

## Growth policy

Growth should not happen simply because validation is flat. The reason for the plateau matters.

### If train and validation are both bad or stagnant

The system likely lacks capacity or structure.

Allowed actions:

```text
- add a step inside an overloaded block;
- add a sibling block in an overloaded layer;
- add memory/global cells if registers are saturated;
- insert a preparation layer before a weak layer.
```

### If train improves but validation falls or stalls

This is overfitting or soft-soup behavior.

Preferred actions:

```text
- harden primitive selection;
- increase cost pressure;
- prune dead or expensive branches;
- freeze stable useful skills;
- reduce late write pressure when safe/noop primitives are selected;
- avoid adding more capacity until compactness improves.
```

### Add step

Add a step inside a block when:

```text
block is useful
but overloaded
and op_gates are high-entropy
and class_read_mass is high
and gradient/update pressure remains high
```

Interpretation:

```text
the block is trying to do too much in one operation
```

### Add width

Add a sibling block when:

```text
layer is overloaded
multiple classes/routes depend on the same block
block diversity is low
category/role separation is weak
```

Interpretation:

```text
the layer needs more specialized organs
```

### Add depth

Insert a layer before a weak layer when:

```text
the weak layer has high gradient and high write pressure
but its outputs are not useful to validation/class read
```

Interpretation:

```text
the layer needs prepared input rather than more internal capacity
```

## Prune policy

Pruning should be probation-based, not instant deletion.

Soft prune protocol:

```text
1. mark element as prune_candidate;
2. reduce its gate/route/primitive prior;
3. continue training for a grace window;
4. if validation and utility do not fall, remove or freeze it;
5. if validation falls, restore it.
```

Dead element signal:

```text
low gate_mass
+ low class_read_mass
+ low grad_norm
+ low update/write_norm
+ low validation usefulness
```

Expensive-unused primitives should be pruned or moved to top-k-only execution first.

## Replace policy

Weak parts should often be replaced rather than immediately removed.

Replacement protocol:

```text
1. detect weak block/step/primitive bundle;
2. create a candidate replacement next to it;
3. initialize candidate from a strong related skill, or mutate the current program;
4. run old and new in parallel through a router/gate;
5. give the candidate a grace period;
6. keep the better one by validation/utility/cost;
7. prune or freeze the loser.
```

This avoids destroying useful partial programs before the replacement has time to receive gradient.

## SkillArchive

Stable useful programs should be archived as reusable skills.

A skill should record:

```text
role
category path
step sequence
read sources
selected primitives
write behavior
utility
cost
validation context
failure modes
```

Example:

```json
{
  "role": "repair",
  "steps": [
    ["read_task", "class_pair_contrast"],
    ["read_memory", "residual_refdelta"],
    ["write", "normalize_or_keep_prev"]
  ],
  "cost": 2.3,
  "utility": 0.18,
  "status": "reusable"
}
```

New blocks/steps should not always start random. They should be initialized from archived skills plus small mutation noise.

## Anti-overfitting behavior

When overfitting appears, the system should prefer compactification over growth.

Actions:

```text
- lower entropy pressure late in training;
- move from soft mix to top-k primitive execution;
- increase cost penalty on expensive primitives;
- reduce or disable unused memory/global writes;
- prune low-utility branches;
- freeze high-utility stable skills;
- make late_write_loss safe-aware so noop/identity/keep_prev are not punished for correctly avoiding writes.
```

Growth should happen only when the utility tracker points to a bottleneck, not merely because validation plateaued.

## Architecture controller loop

A future controller should run periodically, for example every 1-2 epochs:

```text
observe metrics and utilities
classify state:
  learning / plateau / overfit / bottleneck / dead capacity
choose action:
  do nothing / harden / prune / replace / add step / add block / insert layer
apply action with grace period
compare old vs new by validation, utility, and cost
archive stable useful skills
```

The controller should be conservative. Too frequent structure edits prevent gradients from having enough time to train new parts.

Recommended timing:

```text
newborn grace:       1-2 epochs
evaluation window:   2-3 epochs
growth/prune action: not more often than every 1-2 epochs
```

## Priority implementation roadmap

### v14.0: Step slots without growth

Add explicit step dimension inside blocks:

```text
H[layer, block, step, dim]
```

Goal: verify that steps specialize into read/compare/repair/write patterns.

### v14.1: UtilityTracker

Track per-layer, per-block, per-step, per-primitive utility.

Goal: identify dead, useful, overloaded, and overfit-prone parts.

### v14.2: Soft prune

Implement probation-based pruning without growth.

Goal: prove the system can remove useless elements without hurting validation.

### v14.3: Add step

Allow overloaded blocks to grow one extra step.

Goal: convert soft primitive soup into explicit mini-programs.

### v14.4: Add sibling block

Allow overloaded layers to add a new specialized block.

Goal: grow width only when the layer needs more specialized organs.

### v14.5: Insert preparation layer

Allow insertion of a layer before a weak layer.

Goal: grow depth only when the next layer needs prepared input.

### v14.6: Replace and SkillArchive

Add candidate replacement and reusable skill initialization.

Goal: build a system that improves programs over time instead of only training fixed weights.

## Core principle

The system should not only learn parameters.

It should learn small programs, measure their utility, replace weak parts, prune dead parts, grow only at real bottlenecks, and archive reusable skills.

This is the core difference between a soft neural mixture and a true dynamic matrix-program architecture builder.
