# Universal Matrix Program — Logic Map

This file is the project map. Keep it updated when the architecture changes.

## Core idea

We are building a universal differentiable matrix-program environment.

Task-specific parts:

```text
InputAdapter: task input -> TokenMatrix / EvidenceMatrix
OutputHeads: matrix spaces -> task outputs/losses
```

Shared universal part:

```text
UniversalMatrixProgram:
  TokenMatrix
  EvidenceMatrix
  TaskMatrix
  MechanismMatrix
  MemoryMatrix
  GlobalMatrix
  MatrixHeads
  GrowthController
  SkillMemory later
```

The goal is not to hardcode a network. The goal is to let the backend assemble and refine a matrix program from differentiable operations and health/growth signals.

## Spaces

### TokenMatrix

Source depends on task:

```text
HF hidden states
speech/audio tokens
particle/object tokens
image patches
sequence tokens
```

Role:

```text
raw structured state for the current task
```

### EvidenceMatrix

Built from TokenMatrix by task adapter:

```text
chunks
local summaries
global summaries
delta summaries
energy/stat summaries
future: local/window/landmark summaries
```

Role:

```text
compressed evidence/readable features for MechanismMatrix
```

### TaskMatrix

Task and head-specific pressure:

```text
target type
head type
confusion/weak class info later
loss/error signal embeddings later
```

Role:

```text
tells the universal backend what kind of computation is needed
```

### MechanismMatrix

Main program space:

```text
layers -> blocks -> steps -> primitive mix
```

Role:

```text
where the computation program is assembled
```

### MemoryMatrix

Persistent-in-run memory/register space:

```text
memory cells
future: key/value memory geometry
```

Role:

```text
carry reusable internal state across layers/steps
```

### GlobalMatrix

Global state/registers:

```text
global summaries
event/global context
```

Role:

```text
global context for program and heads
```

### MatrixHeads

Task output heads, must stay matrix-native:

```text
block head
delta head
norm head
future: task-specific prediction heads
```

For HF drop-in:

```text
delta = MatrixHead(Token, Blocks, Evidence, Memory, Global)
block = x + delta
norm  = MatrixHead(block, Blocks, Evidence, Memory, Global)
```

## Program hierarchy

```text
Layer
  Block
    Step
      Primitive mix
```

### Primitive

A primitive is what operation is available.

Examples:

```text
evidence_read
route_read
memory_read
delta_route
residual_repair
normalize
aggregate_merge
```

### Step

A step is when/how primitives are applied.

```text
S0: primitive mix -> update h
S1: primitive mix -> update h
S2: primitive mix -> update h
```

Steps solve ordering:

```text
route_read -> memory_read -> delta_route -> residual_repair
```

### Block

A block is an addressable submatrix/branch inside a layer.

Role:

```text
parallel specialization
```

### Layer

A layer is an ordered program stage.

Role:

```text
sequential stage of computation
```

## Growth actions

Growth is discrete between epochs. Active computation remains differentiable.

### add_step

Use when:

```text
old primitives are useful but need ordering
operator entropy high
head-read pressure points to the block
previous similar add_step helped
```

### add_depth

Use when:

```text
whole transition stage is overloaded
delta/head loss remains high
current layers cannot express global sequence of stages
```

### add_width

Use when:

```text
multiple heads/tasks compete for same blocks
block cosine high
read pressure concentrated
```

### add_primitive

Use only when:

```text
old primitives + steps + depth + width do not solve plateau
operator entropy low
residual pattern repeats
candidate is not duplicate
heldout improves
```

### prune/recycle

Use when:

```text
step/block has low gate
low gradient
low head-read pressure
bad action reward
```

Future behavior:

```text
soft alive gate -> hard deactivate between epochs -> slot recycle
```

## Controller signal flow

```text
Losses
  block/delta/norm mse/cos
  future task-specific losses
      ↓
Head attribution
  head_block_read
  head_error_pressure_by_block
      ↓
Step/program health
  operator entropy
  category entropy
  route entropy
  gate
  gradient health
  block cosine
  action reward
      ↓
Planner attention over all active layer/block/step slots
      ↓
Action scoring
  add_step
  add_depth
  add_width
  add_primitive
  prune_step
      ↓
Pending evaluation
      ↓
keep / cooldown / rollback / recycle
```

## Diagnostics and reports

Every serious run should publish:

```text
metrics.csv
final_report.json
analysis_epoch_XXX.json
controller_state.jsonl
program_pseudocode.md
```

Required analysis fields:

```text
active_layer_ids
active_blocks
active_steps
active_primitives
operator_mix_by_layer_block_step
category_mix_by_layer_block_step
gate_by_layer_block_step
route_entropy_by_layer_block_step
head_block_read
planner_attention_top
action_history
cooldown
pending
grad_health_last_batch
```

## Current known weak spots

### 1. Primitive mining is not complete yet

Current candidate primitives are learned mixes of old primitives. Full residual mining still needs:

```text
collect residual
align basis
fit residual operator
project out existing primitives
deduplicate
heldout validation
activate candidate slot
```

### 2. Pruning/recycling is early

`prune_step` exists as controller action in v2.2, but full alive gates and slot recycling are not implemented yet.

### 3. Attention families are still basic

Needed matrix-native attention families:

```text
memory key/value attention
local/window attention
landmark/low-rank attention
route attention for growth targeting
pairwise relation attention for particle/object tasks
```

### 4. Skills are planned, not implemented

Need skill memory before cross-task transfer.

## What counts as progress

A version is better if it improves at least one of:

```text
higher heldout quality
more logical program pseudocode
less blind growth repetition
better action reward/cooldown behavior
clearer transferable skills
faster runtime without losing logic
```

## Non-negotiable design rules

```text
1. Keep computation matrix-space native.
2. Do not turn the backend into a normal MLP/Transformer with new names.
3. Weak priors are allowed; hardcoded specialization is not.
4. Growth location must come from health signals.
5. New primitives must be validated on heldout and deduplicated.
6. Reports must explain what program was assembled.
```
