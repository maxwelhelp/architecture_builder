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

## Current gaps

### 1. Too much soft soup

All operator candidates are computed and softly mixed. This is good early in training, but late training should become sharper:

```text
early: soft all-operator exploration
middle: top-k / sparse primitive selection
late: hard-ish top-2/top-3 program
export: fixed program, do not compute all candidates
```

Otherwise blocks may become:

```text
0.12 mlp + 0.09 memory + 0.07 repair + ...
```

which can work but is less readable and less transferable.

### 2. Step hierarchy is missing in v13

A block should not only select one primitive mixture. It should be able to execute several ordered steps:

```text
S0: read evidence/time/freq
S1: compare with memory/global
S2: repair residual / normalize / suppress
S3: write useful state / aggregate
```

Planned structure:

```text
LayerPlanner   -> phase: extract / compare / memory / repair / aggregate
BlockPlanner   -> block role
StepPlanner    -> ordered step roles
PrimitivePlanner -> top-k primitives inside the step
WritePlanner   -> how much to write and where
```

### 3. Step embeddings are needed

Add later:

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

Without this, a block can damage state simply because the architecture forces it to write.

### 5. MatrixMLP should replace plain per-block MLP as a stronger primitive

Plain MLP is mostly channel-wise:

```text
[B, N, D] -> [B, N, D]
```

MatrixMLP should mix both block/state axis and channel axis:

```text
X = A_blocks @ X
X = X @ W_channels
Y = gate * update + residual
```

Useful variants:

- low-rank block mixer
- butterfly / structured mixer
- block-diagonal channel mixer
- gated residual MatrixMLP

This is still matrix-native and can be made faster than attention for small block counts.

## Implementation order

1. Correctness and reports first:
   - live role regularization
   - fp32 softmax for fp16/P40
   - full attention-channel reports
   - topology smoke tests

2. Quality anchor:
   - keep v2.2/v2.4 quality-first behavior
   - do not reintroduce cost/skill/alive until quality is stable

3. v14 architecture:
   - add StepPlanner + step embeddings
   - add MatrixMLP candidate
   - add noop/keep_prev primitive
   - add late top-k primitive selection schedule

4. Efficiency:
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
