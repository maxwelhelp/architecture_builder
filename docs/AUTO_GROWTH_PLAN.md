# Auto Growth / Drop-in Improvement Plan

## Current result

`hf_matrix_dropin_probe_v1_40ep` successfully approximates one hidden transition of `google/bert_uncased_L-2_H-128_A-2`:

```text
hidden_states[0] -> hidden_states[1]

block_cos ~= 0.9882
norm_cos  ~= 0.9882
delta_cos ~= 0.9284
```

The assembled program is logical:

```text
L0: extract / evidence_read
L1: compare+repair / evidence_read + residual_repair
L2: memory+repair / memory_read + route_read
L3: aggregate as transition / delta_route + residual_repair
```

Main bottleneck is `delta`, not `block` or `norm`.

## Success interpretation

This is not yet a full competitive replacement for a production Transformer layer, because it is trained by distillation on one hidden transition and not evaluated inside downstream tasks. But it is a strong proof that the matrix backend can approximate a real HF layer transition while assembling an interpretable matrix program.

## Core rule

Keep all growth matrix-space native:

- add rows/blocks to MechanismMatrix, not arbitrary new MLP modules
- add MatrixSequenceHead operators, not plain output MLPs
- add primitives as matrix operators between Token/Evidence/Task/Mechanism/Memory/Global spaces
- use soft routing/gating first; only later harden

## Growth signals

### Add depth when

- `block_cos` and `norm_cos` are high, but `delta_cos` is stuck lower
- categories are logical by layer
- route entropy is not collapsed
- final layer mostly uses delta/repair operators

Interpretation: the program has the right ingredients but needs more sequential stages.

Recommended next run:

```text
num_layers = 6
blocks_per_layer = 4
```

Insert new layers between current L2 and L3:

```text
L0 extract
L1 compare/repair
L2 memory/route
NEW L3 route/compare
NEW L4 delta/repair
L5 aggregate/write
```

### Add width when

- several heads/targets compete for the same blocks
- block usage is too concentrated
- block cosine is high inside a layer
- attention/ffn/block targets need different branches

Recommended run:

```text
num_layers = 4 or 6
blocks_per_layer = 6
```

Add blocks inside the failing layer, not everywhere first.

### Add primitives when

- depth/width do not improve plateau
- operator mix is dominated by 1-2 primitives
- category is correct but target error remains high

For HF hidden transition, candidate matrix primitives:

- `residual_stream_write`: controlled residual delta write
- `layernorm_affine_approx`: matrix-space normalization/scale approximation
- `token_pair_compare`: pairwise token relation summary
- `local_window_route`: local token neighborhood route
- `kv_memory_read`: memory read with separate key/value geometry
- `prefix_summary`: prefix/global causal summary
- `headwise_channel_mix`: grouped channel mixing analogous to attention heads

## Where to insert growth

Do not randomly append capacity.

### If L3 delta is bottleneck

Insert before final aggregate/write:

```text
old:
L2 memory/route -> L3 delta/repair

new:
L2 memory/route -> L3 route/compare -> L4 delta/repair -> L5 aggregate/write
```

### If L2 memory is overloaded

Widen L2 first:

```text
L2 blocks: memory_read branch, route_read branch, repair branch, suppress branch, compare branch, reserve branch
```

### If L0/L1 evidence dominates too much

Add specialized evidence branches:

```text
early evidence
late evidence
delta evidence
global evidence
local evidence
```

## Controller fixes

### Absolute best vs significant best

Controller must track two bests:

- `absolute_best_state`: update on any score improvement
- `significant_best_state`: update only when `score > best + min_delta`

Rollback should restore `absolute_best_state`, not stale significant best.

### Pending-effect buffer

Do not judge an action immediately unless the score drops sharply.

```text
pending_epochs = 1 or 2
if score improves after pending -> keep action
if score drops beyond threshold -> rollback + cooldown action
if score is flat -> try next action
```

### Action cooldown

If an action was harmful, put it on cooldown for a few epochs/runs.

```text
bad_action[action] += 1
cooldown[action] = 2 or 3
```

## Space embeddings / operation selection

Yes, the backend should have explicit embeddings that describe spaces and positions.

Add:

- `space_type_emb`: Token/Evidence/Task/Mechanism/Memory/Global/Head
- `layer_pos_emb`: layer index / depth role
- `block_role_emb`: extract/compare/memory/repair/aggregate/suppress/transform prior
- `target_head_emb`: block/delta/norm/attention/ffn
- `health_signal_emb`: gradient norm, error, entropy, saturation, route usage

Use these embeddings to bias:

- category logits
- primitive logits
- route logits
- growth action choice

This lets the space help choose operations instead of choosing from a flat pile.

## Next recommended experiments

### v2 depth6

```text
num_layers=6
blocks_per_layer=4
```

Goal: improve delta head.

### v3 width6

```text
num_layers=4 or 6
blocks_per_layer=6
```

Goal: reduce target/head conflict.

### v4 primitive expansion

Add only after v2/v3 confirm primitive bottleneck.

## What would count as competitive

For this probe:

```text
block_cos >= 0.99
delta_cos >= 0.95
norm_cos  >= 0.99
```

For real replacement:

1. Distill hidden transition well.
2. Insert as adapter:

```text
y = original_layer(x) + alpha * MatrixDropIn(x)
```

3. Verify downstream task does not degrade.
4. Try partial replacement:

```text
y = MatrixDropIn(x)
```

Only after downstream validation can it be called competitive with the baseline layer.
