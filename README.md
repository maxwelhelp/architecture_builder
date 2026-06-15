# Architecture Builder

Private research repo for the matrix architecture builder experiments.

## Current active line

Current active script is `sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py`.

The project direction is no longer plain blockless water. The current design is:

```text
EvidenceMatrix / TaskMatrix / MechanismMatrix / GlobalMatrix / MemoryMatrix
        ↓
SignalBus
        ↓
RolePlanner
        ↓
CategoryPlanner
        ↓
PrimitivePlanner
        ↓
Matrix program backend
        ↓
Task head
```

## Important sync rule: do not delete dataset

Datasets and runs are intentionally ignored by git:

```text
data/
runs/
checkpoints/
artifacts/
```

Safe sync:

```bash
git pull --rebase
```

Safe local update from this repo into your working project:

```bash
rsync -av --exclude data --exclude runs --exclude checkpoints --exclude artifacts ./ /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/Functional\ Matrix\ Grower/
```

Do **not** use this in the project folder unless you know what you are doing:

```bash
git clean -fdx
```

`git clean -fdx` can delete ignored local folders such as datasets and runs.

## Project layout

```text
src/current/        latest active implementation
old/                best historical versions / baselines
docs/               design docs, plan, notes
commands/           ready-to-run commands
reports/            lightweight report summaries only
```

## Main experiment command

See:

```text
commands/run_v13_15ep.sh
```

## What to compare

Primary metrics:

```text
best_val_acc
class_acc.go / no / down / left / right
role_mix_by_layer
category_mix_by_layer
operator_program_by_block
route_usage_by_block
gate_write_by_block
matrix_decomposition
speed_forward_ms / speed_backward_ms
```

## Current known direction

v13 adds categorized primitives so the model does not select from a flat operator pile. The intended decision chain is:

```text
role → category → primitive → matrix execution
```

Next likely work:

```text
v13.1 gradient usefulness analyzer
v13.2 route category planner
v13.3 architecture memory ranking across runs
```
