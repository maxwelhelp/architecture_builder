# Skill Memory and Transfer Plan

## Goal

Make useful learned architecture/program patterns reusable across tasks, even when the input adapter and output heads change.

This is not the same as loading a full checkpoint.

A skill should be a portable reusable matrix-program capsule:

```text
step sequence
primitive mix pattern
category/role pattern
route/memory pattern
growth action outcome
pseudocode explanation
validation statistics
```

## Why skills matter

When we switch task:

```text
SpeechCommands -> HF hidden transition -> particle prediction -> image patches
```

we do not want to relearn every useful internal pattern from zero.

Instead, new task starts with:

```text
fresh input adapter
fresh output heads
shared skill memory priors
```

## What can transfer safely

### 1. Step-program motifs

Example:

```text
route_read -> memory_read -> delta_route -> residual_repair
```

Stored as:

```text
ordered list of top primitives per step
categories per step
gates/read usage
where it helped
```

### 2. Primitive priors

Example:

```text
for delta-like tasks:
  prefer delta_route + residual_repair + normalize
```

### 3. Growth policy priors

Example:

```text
if delta gap high and block has high op entropy:
  add_step helped in prior runs
```

### 4. Matrix attention patterns

Example:

```text
memory KV attention helped when memory_read saturated
local window attention helped on sequence/local tasks
```

### 5. Pseudocode patterns

Human-readable pattern:

```text
extract -> compare/repair -> route/memory -> delta/repair -> norm
```

This is useful for analysis and for initializing weak role/category priors.

## What should not transfer blindly

### Dense weights

Dense weights only transfer if geometry matches:

```text
same dim
same space type
same task family or aligned basis
```

Otherwise they can hurt.

### Task-specific heads

Output heads are task-specific. Do not transfer final classifier/prediction heads blindly.

### Input adapters

Input adapters depend on raw modality and must be task-specific.

## Skill capsule format

Store in JSONL:

```json
{
  "skill_id": "route_memory_delta_v1",
  "status": "candidate|promoted|rejected",
  "source_run": "universal_matrix_program_v2_2_step_planner_hf_45ep",
  "task_family": "hf_hidden_transition",
  "space_signature": {
    "dim": 128,
    "spaces": ["Token", "Evidence", "Mechanism", "Memory", "Global"],
    "input_type": "hidden_state",
    "target_heads": ["block", "delta", "norm"]
  },
  "program": [
    {"step": 0, "category": "compare", "ops": {"route_read": 0.55, "residual_repair": 0.25}},
    {"step": 1, "category": "memory", "ops": {"memory_read": 0.60, "aggregate_merge": 0.20}},
    {"step": 2, "category": "repair", "ops": {"delta_route": 0.65, "residual_repair": 0.25}}
  ],
  "metrics": {
    "heldout_delta_cos_gain": 0.01,
    "heldout_score_gain": 0.002,
    "reward_mean": 0.001
  },
  "reuse_policy": {
    "apply_when": ["delta_gap_high", "operator_entropy_high", "memory_read_useful"],
    "avoid_when": ["route_entropy_collapsed", "heldout_reward_negative"]
  }
}
```

## Skill lifecycle

### Candidate

A pattern is found in one run.

```text
repeatable inside a run
positive local action reward
visible in pseudocode
```

### Promoted

A pattern helps across:

```text
multiple seeds
multiple heldout batches
or multiple task families
```

### Rejected

A pattern:

```text
helps train but hurts heldout
is duplicate
is too task-specific
is unstable after transfer
```

## How to use skills in a new task

### Step 1: task signature

Compute:

```text
input modality
TokenMatrix shape
feature dim
target heads
loss family
local/global structure
```

### Step 2: retrieve skills

Find compatible skills by:

```text
task family
space signature
head type
needed pressure signal
```

### Step 3: warm-start weak priors only

Apply to:

```text
cat_prior
op_prior
role prior
controller action prior
candidate macro primitive prior
```

Do not force the skill.

### Step 4: validate quickly

Run short probe:

```text
with skill priors
without skill priors
```

Promote only if heldout improves.

## Skill memory files

Planned files:

```text
skills/skill_bank.jsonl
skills/rejected_skills.jsonl
skills/skill_metrics.csv
skills/README.md
```

Do not store heavy checkpoints in skill bank.

## Relation to checkpoints

Two modes:

### No checkpoint mode

Use only skill priors:

```text
fresh model weights
skill priors loaded into controller/category/op priors
```

This tests whether skills transfer as knowledge rather than memorized weights.

### Compatible checkpoint mode

If dims/spaces match:

```text
load selected primitive/step weights
load skill priors
freeze/soft-start with low alpha
```

## What to measure

Second-run learning should improve:

```text
faster early score
fewer bad growth actions
better initial program structure
same or better final heldout
clearer pseudocode
```

Compare:

```text
cold_start
skill_prior_start
checkpoint_start
```

## Risks

### Negative transfer

A skill from one task may hurt another.

Mitigation:

```text
weak priors only
short validation
cooldown/reject bad skills
```

### Skill overfitting

A pattern may just memorize one teacher/task.

Mitigation:

```text
promote only across seeds/tasks
```

### Skill bank bloat

Too many near-duplicates.

Mitigation:

```text
deduplicate by functional similarity and pseudocode similarity
```

## Implementation phases

### Phase 1: passive extraction

After each run:

```text
export program_pseudocode.md
extract top step chains
write candidate skills to skills/skill_bank.jsonl
```

### Phase 2: skill prior loading

New run can pass:

```text
--skill-bank skills/skill_bank.jsonl
--skill-mode priors_only
```

### Phase 3: A/B validation

Run:

```text
cold_start vs skill_prior_start
```

### Phase 4: cross-task transfer

Use skills from HF hidden transition in:

```text
sequence block probe
speech matrix task
particle adapter task
```

## Success criteria

A skill system is useful if:

```text
second run learns faster without checkpoint
controller makes fewer bad growth actions
useful step programs appear earlier
heldout quality does not drop
skills can be explained in pseudocode
```
