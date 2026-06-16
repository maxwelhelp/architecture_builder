# Architecture Builder

Private research repo for matrix-program architecture experiments.

This repository is used for fast experimental development, report publishing, and architecture-memory tracking. Heavy local data and checkpoints must stay out of git.

## One-paragraph project goal

Build a differentiable matrix-program builder that can assemble useful neural-network substructures from roles, categories, primitives, matrix memories, and learned signals. The long-term direction is not a single fixed model, but a system that can search, compare, archive, and reuse good architecture fragments across tasks.

## Living architecture source of truth

The main design contract is:

```text
ARCHITECTURE.md
```

Update that file whenever the system logic changes. It contains the current grammar, logging contract, roadmap, and inactive/rejected ideas. The README is only a short entry point.

## Core principle / root problem

The main system problem is documented here:

```text
docs/DYNAMIC_PROGRAM_BUILDER_CORE.md
```

Short version:

```text
v13 is a useful fixed soft differentiable prototype, but the real system must become a dynamic program builder:
StepPlanner + UtilityTracker + Grow/Prune/Replace Controller + SkillArchive.
```

If training plateaus or overfits, the first question should not be only "which lambda should change?". The first question should be whether the architecture can detect weak steps/blocks, add capacity only at real bottlenecks, prune useless branches, replace failed mini-programs, and archive reusable primitive-step skills.

## Current active lines

### 1. SpeechCommands matrix architecture

Active legacy/large script:

```text
sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py
```

Main command:

```bash
bash commands/run_v13_15ep.sh
```

Current corrected v13/vNext ideas:

- live RolePlanner gradients, not detached report-only role mixes;
- honest attention reports: time/freq/onset/energy/global/memory channels;
- topology-aware reports for grid and merge topologies;
- fp32 softmax before cast for fp16/P40 stability;
- MatrixMLP added as a primitive candidate;
- entropy/balance schedule: early soft exploration, later sharper readable program.

### 2. Modular vNext core

New reusable code should go here:

```text
src/matrix_program/
```

Current modules:

```text
src/matrix_program/operators.py      operation metadata, families, costs, role hints
src/matrix_program/shadow_topk.py    warmup/top-k/shadow planning helpers
src/matrix_program/archive.py        lightweight architecture snapshot archive
```

Rule: do not add new complex logic to the huge v13 script unless it is only a local comparison. New reusable ideas should be small modules under `src/matrix_program/` or small experiment folders.

### 3. HF drop-in / universal program probes

Existing probes live under:

```text
experiments/hf_dropin/
experiments/universal_program/
experiments/attention_replacement/
experiments/shadow_topk/
```

These are used to test ideas such as replacing a hidden layer/block/head with a matrix-program module before moving the idea into SpeechCommands or a larger model.

## Architecture map

The intended architecture is not a flat bag of primitives. It is a differentiable program builder:

```text
input / evidence / task signal
  -> EvidenceMatrix / TaskMatrix / GlobalMatrix / MemoryMatrix
  -> SignalBus
  -> LayerPlanner
  -> Block RolePlanner
  -> CategoryPlanner
  -> PrimitivePlanner
  -> Matrix-program backend
  -> write gates / memory updates / class read
  -> task head
```

Current v13 decision chain:

```text
role -> category -> primitive -> matrix execution -> write -> class read
```

Planned v14 decision chain:

```text
layer -> block -> step -> primitive -> matrix_mlp/read/write
```

The important rule: early training should keep gradients flowing through many choices. Later training can sharpen/top-k the choices for readability and speed.

## ShadowTopK / counterfactual branch search

The next major idea is cheap branch search:

```text
h -> cheap sketch P(h) -> score all branches -> full compute top-k only
```

Training phases:

```text
warmup:  compute all branches fully; gradient reaches all primitives
top-k:   cheap sketch scores all; full compute only k selected branches
shadow:  sometimes compute one unselected branch for router learning only
deploy:  fixed selected program; do not compute unselected branches
```

Prefer block-level or step-level top-k, not per-token top-k, because per-token top-k creates scatter/gather overhead and can be slower on Tesla P40.

## MatrixMLP policy

Plain MLP is mostly channel-wise:

```text
[B, N, D] -> [B, N, D]
```

MatrixMLP should mix a state/block/sequence axis and then a channel axis:

```text
score = Q(x) @ K(context)^T
ctx   = softmax(score) @ V(context)
out   = MLP([x, ctx, task/memory/global context])
```

Useful variants:

```text
matrix_mlp_local
matrix_mlp_global
matrix_mlp_lowrank
matrix_mlp_memory
```

Safe primitives should also exist:

```text
noop
identity
keep_prev
small_refine
normalize
suppress
```

## Reports and logs

Heavy outputs stay local in `runs/`. Lightweight summaries go to `reports/`.

Important report files:

```text
reports/<run_name>/README.md
reports/<run_name>/metrics.csv
reports/<run_name>/final_report.json
reports/<run_name>/seq_analysis_epoch_*.json
reports/<run_name>/analysis_epoch_*.json
reports/<run_name>/architecture_memory.jsonl
reports/<run_name>/program_pseudocode.md
```

How to publish a report:

```bash
bash commands/push_latest_report.sh \
  ./runs/<run_dir> \
  <report_name> \
  "Add <report_name> report"
```

`push_latest_report.sh` stages only safe lightweight files and now does `git pull --rebase` before `git push`.

## Git sync rules

Never delete local datasets or local run outputs from git cleanup commands. Heavy local data and checkpoints are intentionally kept outside normal report commits.

Before editing from another machine, run:

```bash
git pull --rebase
```

Before publishing run summaries, use the report publishing command instead of manually staging full local run directories.
