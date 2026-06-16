# Matrix Program Logic vNext

This note records the corrected architecture direction after v13/v2.x debugging.

## Core thesis

The architecture should not be a flat bag of primitives. It should be a differentiable program builder:

```text
state + task/error signals + evidence + memory/global
  -> layer planner
  -> block role planner
  -> step planner
  -> primitive/category planner
  -> read/transform/write
  -> next state
```

The important point is not that the model is conscious. The important point is that the planner receives gradient from task loss. Bad primitive/step choices increase loss; useful choices reduce loss; gradients move role/category/primitive/router logits.

## Current v13 state

Already present:

- RolePlanner
- CategoryPlanner
- PrimitivePlanner
- operator candidates
- route/task/time/freq/onset/energy/global/memory attention
- global and memory registers
- task/error pressure features
- matrix-style operator mixing through `einsum(op_gates, candidates)`

This means v13 is a real differentiable matrix assembly, not just a pretty report.

## v13 vNext patch policy

The safe patch path is:

```bash
bash commands/apply_v13_vnext_matrix_mlp_patch_and_smoke.sh
```

This applies:

- correctness/stability patch
- live role regularization path
- fp32 softmax for fp16/P40
- attention channel reports
- topology smoke tests
- MatrixMLP primitive
- entropy/balance schedule

## Current gaps and fixes

### 1. Too much soft soup

All operator candidates are computed and softly mixed. This is good early in training, but late training should become sharper:

```text
early: soft all-operator exploration
middle: weaker balance/entropy pressure
late: sharper category/primitive program
export: fixed program, do not compute all candidates
```

The vNext patch adds an entropy/balance schedule in `train_one_epoch`:

```text
operator_balance: high early, decays later
operator_entropy_per_block: high early, weaker later
category_entropy: near-zero early, active later for sharper categories
```

This keeps gradient flowing through all candidates first, then prevents the final report from becoming “every block does a little bit of everything”.

### 2. Step hierarchy is still missing in v13

A block should not only select one primitive mixture. It should be able to execute several ordered steps:

```text
S0: read evidence/time/freq
S1: compare with memory/global
S2: repair residual / normalize / suppress
S3: write useful state / aggregate
```

Planned structure:

```text
LayerPlanner      -> phase: extract / compare / memory / repair / aggregate
BlockPlanner      -> block role
StepPlanner       -> ordered step roles
PrimitivePlanner  -> top-k primitives inside the step
WritePlanner      -> how much to write and where
```

This is not in the v13 patch yet. It is the main v14 task.

### 3. Step embeddings are needed

Add in v14:

```python
self.step_addr = nn.Parameter(torch.randn(max_steps, dim) * 0.04)
```

Each step planner input should include:

```text
h + layer_addr + block_addr + step_addr + role_addr + task/error/evidence/memory context
```

Embeddings are not decoration. They are addresses: where am I, which step am I, which role am I, which primitive am I near?

### 4. Safe no-op primitives are required

Every writable block/step needs safe options:

- noop
- identity
- keep_prev
- small_refine

Without this, a block can damage state simply because the architecture forces it to write. This is v14 work.

### 5. MatrixMLP is added as a separate primitive

Plain MLP is mostly channel-wise:

```text
[B, N, D] -> [B, N, D]
```

The vNext patch adds `matrix_mlp` as a new operator candidate, not a replacement for `mlp`. It mixes both block/state axis and channel axis:

```text
mm_score = Q(h) @ K(block_seed + route_ctx + memory_ctx)^T
mm_ctx   = softmax(mm_score) @ V(block_seed + route_ctx + memory_ctx)
Y        = MatrixMLPOut([h, mm_ctx, task_ctx, global_ctx + mem_ctx])
```

This is still matrix-native and can be optimized later through grouped block/state mixers.

## Implementation order

1. Correctness and reports first:
   - live role regularization
   - fp32 softmax for fp16/P40
   - full attention-channel reports
   - topology smoke tests

2. Quality anchor:
   - keep v2.2/v2.4 quality-first behavior
   - do not reintroduce cost/skill/alive until quality is stable

3. v13 vNext patch:
   - MatrixMLP candidate
   - entropy/balance schedule
   - preserve soft-gradient exploration at the start

4. v14 architecture:
   - StepPlanner + step embeddings
   - noop/keep_prev/small_refine primitive
   - per-step report
   - late top-k primitive execution/export

5. Efficiency:
   - profile after correctness
   - vectorize/group operator candidates
   - only then consider custom CUDA for P40

## P40 kernel policy

Do not start with custom CUDA kernels. First profile the corrected code.

Possible later kernel targets:

- fused weighted operator sum: `einsum(op_gates, candidates)`
- fused cell write: gate + delta + residual norm
- grouped block/state mixer for MatrixMLP

Triton is not a good default for Tesla P40. Prefer PyTorch vectorization first, then a minimal C++/CUDA extension only for proven bottlenecks.
