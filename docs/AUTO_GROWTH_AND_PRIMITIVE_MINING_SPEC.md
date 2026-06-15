# Auto Growth and Primitive Mining Spec

## Important correction

Do not hardcode `insert after L2` as a permanent rule.

The phrase from the analysis meant only this: in the current HF drop-in run, the health-map showed that the late delta path is the bottleneck, so the next manual depth experiment should insert capacity before the final transition/write stage.

The real system must choose the growth location from signals.

## Target behavior

The controller should act only near plateau or strong slowdown:

```text
normal growth:
  train weights, no structure mutation

plateau / slow growth:
  try one soft architecture action

bad action:
  rollback to absolute best arch-state
  cooldown that action
  try a different action later

persistent plateau:
  recommend or perform local growth
```

## No hardcoded specialization

Initial role priors are allowed as weak starting bias, but final specialization must come from gradients, metrics, route usage, operator usage, and head errors.

Allowed:

```text
weak prior: early layers like extract, late layers like aggregate
soft temperature and priors
controller mutates priors only when plateau appears
```

Not allowed:

```text
force L2 to always be memory
force L3 to always be delta
always append layers to the end
always widen all layers
```

## Health-map per layer/block/head

Every epoch or every N epochs compute:

### Per head

```text
block_mse, block_cos
delta_mse, delta_cos
norm_mse, norm_cos
head_error_slope
```

### Per layer

```text
category_entropy
operator_entropy
route_entropy
gate_mean
update_norm
grad_norm
block_cosine
memory_read_usage
route_read_usage
delta_route_usage
residual_repair_usage
```

### Per block

```text
read usage by head
operator mix
route mix
grad norm
activation/update norm
cosine to other blocks in same layer
```

## Growth decision table

### Add depth

Trigger:

```text
score plateau
block/norm good
delta bad or slow
late layer uses delta_route/residual_repair heavily
route entropy is not collapsed
```

Where:

```text
insert BEFORE the most overloaded late transition layer
not always at the end
```

Example for current HF result:

```text
L0 extract
L1 repair/compare
L2 memory/route
L3 delta/repair  <-- overloaded transition
```

Growth proposal:

```text
L0 extract
L1 repair/compare
L2 memory/route
NEW route/compare layer
NEW delta/repair layer
L_final aggregate/write
```

### Add width

Trigger:

```text
one layer has high block usage concentration
multiple heads read same few blocks
block cosine high inside layer
operator entropy low inside overloaded layer
```

Where:

```text
widen only the overloaded layer first
```

Examples:

```text
L2 overloaded by memory+route -> add L2 memory branch and route branch
L3 overloaded by delta+repair -> add L3 delta branch and repair branch
```

### Add memory cells

Trigger:

```text
memory_read high
memory category high
memory cells have high cosine/saturation
memory writes are high but delta still bad
```

Where:

```text
increase MemoryMatrix cells
optionally add memory-specific route keys
```

### Add route branch

Trigger:

```text
route entropy collapsed
same route used by many blocks/heads
route_read high but improvement slow
```

Where:

```text
add route category or route primitive in the overloaded layer
```

### Add primitive

Trigger:

```text
depth/width did not solve plateau
category choice is logical
operator mix is dominated by existing primitive(s)
residuals show repeated unexplained structure
```

Where:

```text
add candidate primitive only to the category/layer where residual points
start with low prior and gate
validate before promotion
```

## Primitive mining from residuals

Use ideas from `symbolic_operator_lab_v5_joint_programs.py`:

- project target on current dictionary
- compute residual
- mine repeated residual directions with SVD/PCA
- add candidate with `add_external_operator`
- accept only if projection and functional validation improve

For HF drop-in:

```text
residual_block = y - y_hat
residual_delta = (y - x) - delta_hat
residual_norm  = norm(y) - norm_hat
```

Convert residuals into candidate matrix operators by fitting local linear maps:

```text
min_A ||R - X A^T||
A_candidate = solve(X -> R)
```

Then remove known dictionary projection:

```text
A_residual = A_candidate - project_on_existing_ops(A_candidate)
```

Then mine repeated candidates:

```text
PCA/SVD over residual operator cloud
```

## Deduplication rules

A mined primitive is rejected if:

```text
cosine with existing primitive > 0.97
functional output correlation > 0.97
same category+same behavior already exists
rank/profile is near duplicate
improves train but not heldout
```

A candidate is accepted as `candidate` if:

```text
heldout head MSE improves
cos improves
operator is compact/reusable
not duplicate
not just dense random overfit
```

A candidate becomes `promoted` if it helps across:

```text
multiple batches
multiple heldout seeds
or multiple heads/layers
```

## Alignment before mining

Do not mine raw residuals blindly. If hidden basis rotated, align first:

```text
1. collect X and target residual R
2. optional PCA/whitening of X
3. fit A in aligned basis
4. mine residual operator in aligned basis
5. map back to model basis
```

This follows the earlier rule: alignment before residual mining.

## Primitive families to mine/generate

Do not create arbitrary dense operators first. Generate candidate families:

### Residual/delta family

```text
residual_stream_write
delta_update
refdelta2/refdelta3
matrix_refdelta
```

### Normalization family

```text
layernorm_affine_approx
mean_center
variance_scale
channel_group_norm
```

### Route/token family

```text
local_window_route
content_route
token_pair_compare
prefix_summary
suffix_summary
```

### Memory family

```text
kv_memory_read
memory_write_gate
copy_gate
ema_memory_update
```

### Spectral/rank family

```text
low_rank_projector
rank_compress
DCT/lowpass-like projection if geometry supports it
```

## Using the uploaded symbolic lab

The uploaded lab already contains the exact concepts we need:

- additive dictionary decode
- ordered product grammar where step order matters
- functional validation, not only matrix error
- masked operator prediction
- joint formula + program learner
- residual mining to turn repeated residual into a new operator
- external operator injection

Adaptation plan:

```text
OperatorLibrary -> MatrixPrimitiveLibrary
SparseMatrixDecoder -> PrimitiveProjectionDecoder
ProductOperatorProgram -> ordered primitive pair/step evaluator
Residual mining -> candidate primitive generator
functional_error_matrix -> hidden-transition functional validation
```

## Auto-growth controller state

Track three bests:

```text
absolute_best_state: update on any score improvement
significant_best_state: update only if improvement > min_delta
candidate_state: current mutation under pending evaluation
```

Rollback must restore `absolute_best_state`, not stale significant best.

## Growth actions

Controller actions:

```text
soft_explore_categories
boost_l2_memory
boost_l2_repair
boost_l1_compare
boost_late_delta
reduce_priors
add_depth_candidate
add_width_candidate
add_memory_candidate
mine_primitive_candidate
```

Actual module growth should happen only at epoch boundaries, after saving current/best state.

## Safe implementation order

### Phase 1: logging only

Add health-map and growth recommendation. No dynamic mutation.

### Phase 2: soft mutation only

Only priors/temps/actions. Rollback enabled.

### Phase 3: static growth variants

Run generated configs:

```text
4x4 baseline
6x4 depth
4x6 width
6x6 depth+width
```

### Phase 4: candidate slots

Pre-allocate inactive layers/blocks/primitive slots. Growth activates a slot instead of changing Python module shape. This keeps optimizer state stable.

### Phase 5: primitive mining

Add mined candidates into inactive primitive slots with low prior and validation gates.

## Key principle

Growth is discrete and not differentiable. Computation inside active architecture remains differentiable. This is acceptable: growth happens between epochs, not inside a batch.
