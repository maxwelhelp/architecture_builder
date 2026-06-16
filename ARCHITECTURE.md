# Architecture Builder — living architecture source of truth

This is the main file to update when the architecture changes.  The README is a short entry point; this file is the detailed design contract.

The core goal is not to hand-pick a fixed architecture.  The goal is to define a differentiable space of small matrix programs, train it end-to-end, then sharpen, prune, export, and archive the useful programs.

## Current core idea

A useful architecture should be a hierarchy of differentiable program choices:

```text
Input / Evidence / Task signal
  -> Layer
    -> Block
      -> Step
        -> Primitive slots
        -> Primitive-to-primitive transitions
      -> Step-to-step transitions
    -> Block/layer transitions
  -> Output read
  -> Loss
```

The important point is that gradients flow through all of these choices:

```text
loss
  -> output read
  -> layer/block routes
  -> step read transitions
  -> primitive transition gates
  -> primitive gates
  -> primitive parameters
  -> input projections / memory / global registers
```

This makes the architecture closer to a trainable butterfly/circuit than to a manually assembled network.

## What must be logged

Every experiment must log the following levels with explicit addresses.

### 1. Operations inside a step

Address format:

```text
L{layer}.B{block}.S{step}.P{primitive_slot}
```

Log:

```text
top primitives
primitive probabilities
primitive entropy
primitive cost
nontrivial primitive mass
safe/noop mass
```

### 2. Operations between primitives inside a step

Address format:

```text
L{layer}.B{block}.S{step}.P{k}->P{k+1}
```

Transition basis:

```text
keep             output = previous value
replace          output = new primitive output
residual         output = previous + gated new
norm_residual    output = Norm(previous + new)
product_gate     output = previous * sigmoid(new)
compare_mix      output = learned(previous - new, previous * new, context)
gated_sum        output = weighted sum of primitive outputs
serial           output = next primitive consumes previous primitive output
```

### 3. Operations between steps

Address format:

```text
L{layer}.B{block}.S{step}.read
```

Step read basis:

```text
state
input / evidence
previous step
all previous steps
layer route context
global register
memory register
task signal
```

Strict sequential mode should mask impossible reads, for example `S0 -> prev_step` and `S0 -> all_prev_steps`.

### 4. Operations on input/output

Input projection basis:

```text
raw/evidence projection
local/time projection
frequency projection
task-conditioned projection
memory-conditioned projection
low-rank sketch projection
```

Output read basis:

```text
class reads all slots
class reads final steps
class reads aggregator slots
class reads memory/global
class reads input + repair slots
```

Output read must log which class reads which slots and whether those slots are useful or mostly noop.

### 5. Operations between layers/blocks

Address format:

```text
L{layer}.B{block}.route
```

Layer/block transition basis:

```text
input
previous same block
previous layer mean
previous layer selected blocks
global mean
memory mean
skip / keep
aggregate layer
split specialists
repair before next layer
```

### 6. Which parts are needed

Address format:

```text
L{layer}.B{block}.S{step}.utility
```

Minimum utility fields:

```text
write_gate
class_read_mass
primitive cost
safe/noop mass
nontrivial op mass
global_write_gate
memory_write_gate
grad_norm when available
update_norm
utility_proxy
status: newborn / learning / useful / overloaded / dead / prune_candidate / replace_candidate / frozen
```

## Grammar banks

The architecture should not only define primitive banks.  It must define grammar banks.

### PrimitiveBank

Initial useful primitive basis:

```text
noop
identity
keep_state
small_refine
mlp
matrix_mlp
compare
memory_read
global_read
normalize
suppress
```

Future primitives:

```text
local_mix
global_mix
lowrank_matrix_mlp
memory_write
energy_count
class_pair_contrast
residual_refdelta
```

### PrimitiveTransitionBank

Initial basis:

```text
keep
replace
residual
norm_residual
product_gate
compare_mix
gated_sum
serial
```

The current StepProgram v1 implements most of this basis, but `gated_sum` and explicit `serial` should be made visible in logs as named transition modes rather than implicit behavior.

### StepOutputTransitionBank

A step should not always update state by one fixed residual formula.  It should choose how to write:

```text
write_residual       state <- Norm(state + gate * update)
replace_state        state <- update
gated_keep_write     state <- keep_gate * state + write_gate * update
write_to_memory_only state unchanged, memory updated
write_to_global_only state unchanged, global updated
normalize_write      state <- Norm(state + update)
suppress_write       state <- state - gate * suppress(update)
keep_state           state unchanged
```

This is a required next addition.

### RegisterTransitionBank

Memory/global registers should have their own grammar:

```text
read_only
write_only
read_then_write
ema_update
append_like_update
replace_cell
suppress_memory_noise
keep_register
```

Register writes must be tied to selected primitive/step-output mass.  A step that did not choose a memory/global operation should not strongly mutate registers.

### LayerTransitionBank

Layer transitions should be explicit:

```text
pass_same_block
mix_blocks
aggregate_layer
split_specialists
repair_before_next
skip_layer
keep_layer
```

### FreeLearnedTransition

All manually defined grammar is only a starting basis.  The model must be allowed to discover a new learned combination not explicitly listed.

Recommended formula:

```text
transition_output = basis_mix(known_transition_bank)
                  + free_gate * learned_free_transition(context)
```

Training schedule:

```text
early:  free_gate small, learn readable basis
middle: free_gate can open if it improves task loss
late:   top-k / cost / utility decides whether free path stays
export: keep free path only if useful and compact
```

This prevents conflict between our starting basis and newly discovered combinations.

## Good combinations

Examples that should be considered normal and useful:

```text
read input -> small_refine -> normalize
read previous step -> compare -> residual
read memory -> compare -> suppress
read global -> matrix_mlp -> normalize
identity -> small_refine -> keep_state
S0 extract -> S1 compare -> S2 repair -> S3 write/keep
L0 local extraction -> L1 interaction -> L2 memory/global repair -> L3 aggregate/class read
```

## Suspicious or absurd combinations

These should not all be hard-forbidden.  Most should be detected and softly penalized by utility/cost rules.

### Expensive operation thrown away

```text
noop -> expensive matrix_mlp -> keep_state
```

Fix:

```text
cost penalty + low utility prune
```

### Impossible previous-step read

```text
S0 reads prev_step / all_prev_steps
```

Fix:

```text
hard causal mask in sequential mode
```

### Empty memory dependency

```text
early layer heavily reads memory before memory has useful content
```

Fix:

```text
soft prior against early memory overuse, not a hard ban
```

### Suppression without evidence

```text
suppress before reading input/prev/global context
```

Fix:

```text
utility penalty if suppress has low class-read/write usefulness
```

### Output reads mostly noop slots

```text
class_read high on slots with high noop/keep mass and low write/update norm
```

Fix:

```text
log as suspicious; penalize high class_read * high noop_mass when validation does not improve
```

### Product gate saturation

```text
product_gate dominates and gradients vanish
```

Fix:

```text
LayerNorm before product, bounded/tanh gate branch, cost/utility penalty if low usefulness
```

## Constraint policy

Hard-mask only physically impossible or shape-invalid choices:

```text
S0 reads previous step in strict sequential mode
future step read in strict sequential mode
future layer read in strict sequential mode
invalid route dimensions
```

Soft-penalize suspicious choices:

```text
expensive op with low utility
memory write without memory mass
output read from noop slots
large write with low class-read utility
```

Leave freedom for unusual but potentially useful combinations:

```text
compare before normalize
early global use
early memory read
product-gate chains
nonstandard primitive order
```

## Execution modes

### sequential_exact

Current safe baseline:

```text
for layer
  for step
    for primitive_slot
```

Blocks are still vectorized as `[B, N, D]`.  The loops are only over the small program axes `L`, `S`, and `K`.

### parallel_primitives

Compute primitive slots in parallel from a shared context, then use a transition matrix to combine them.

Goal:

```text
reduce primitive-slot loop while keeping readable primitive-transition logs
```

### parallel_refine

Initialize all step slots at once and run 1-2 refinement passes:

```text
H0 = init all steps
H1 = parallel_step_update(H0)
H2 = parallel_refine(H1)
```

Goal:

```text
faster approximate program search with more parallelism
```

### hybrid

Parallelize inside groups, keep a small number of causal boundaries between groups.

Goal:

```text
balance honesty and speed
```

## Current active implementation

### StepProgram v1

File:

```text
experiments/step_program/run_step_program_speechcommands_v1.py
```

Purpose:

```text
clean baseline for differentiable step-program grammar
```

Current implemented levels:

```text
primitive gates inside step
primitive-to-primitive transition gates
step-read gates
layer/block route gates
register writes
output class read
utility proxy logging
```

Known issues / next additions:

```text
- add explicit StepOutputTransitionBank;
- add explicit RegisterTransitionBank logs;
- add explicit LayerTransitionBank logs;
- add FreeLearnedTransition path;
- add hard masks for impossible reads;
- add suspicious-combination metrics;
- add execution-mode flag: sequential_exact / parallel_primitives / parallel_refine / hybrid.
```

## Inactive / rejected / moved-to-archive registry

When something does not work, do not leave it as active design.  Move it here with a short reason.

### v13 huge-script growth-by-patching

Status:

```text
inactive as main direction
```

Reason:

```text
v13 is useful as a baseline, but continued patching makes the architecture hard to reason about.  New reusable logic must go into small step-program / matrix-program modules.
```

### Fixed soft primitive soup without hardening

Status:

```text
rejected as final form
```

Reason:

```text
soft mixtures help early gradient flow, but if they remain soft forever they can overfit and fail to become readable compact programs.
```

### Growth before StepPlanner / UtilityTracker

Status:

```text
rejected as first step
```

Reason:

```text
without step-level utility, growth can add capacity to the wrong place and worsen overfitting.
```

## Roadmap

### Immediate

```text
1. stabilize StepProgram v1;
2. add StepOutputTransitionBank;
3. add masks for impossible step reads;
4. add suspicious-combination report;
5. add FreeLearnedTransition with scheduled free_gate.
```

### Next

```text
6. add RegisterTransitionBank;
7. add LayerTransitionBank;
8. add execution-mode parallel_primitives;
9. compare speed/accuracy/readability vs sequential_exact.
```

### Later

```text
10. add UtilityTracker with gradients/update norms;
11. add soft prune;
12. add top-k execution;
13. add SkillArchive;
14. add conservative grow/replace controller.
```
