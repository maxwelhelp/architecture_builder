# Development Plan

## Core rule

Do not return to all-to-all blockless water for this task. The current line is causal matrix spaces with structured signal-driven assembly.

## Active architecture target

```text
EvidenceMatrix
TaskMatrix
MechanismMatrix
GlobalMatrix
MemoryMatrix
SignalBus
RolePlanner
CategoryPlanner
PrimitivePlanner
Matrix backend
Task head
```

## v13 immediate goals

- Fix categorized primitive path stability.
- Verify `category_mix_by_layer` and `category_program` are written correctly.
- Check whether weak classes such as `go` move into `memory/repair` categories.
- Compare v13 vs v12/v11.2 on 15 epochs.

## Next versions

### v13.1 Gradient usefulness analyzer

Add diagnostics, not training-time heavy logic yet:

- grad norm by block
- grad norm by stage state
- attention vs gradient mismatch
- operator usefulness proxy
- route usefulness proxy

Goal: see which routes/operators are truly useful, not only highly attended.

### v13.2 Route Category Planner

Current route attention can still be too similar across blocks. Add route categories:

- same-column causal route
- cross-branch compare route
- base skip route
- global skip route
- memory route
- repair route
- merge route

Goal: route selection should become as structured as primitive selection.

### v13.3 Architecture Memory Ranking

Use `architecture_memory.jsonl` across runs:

- rank best historical configurations
- extract best role/category/operator patterns
- warn about repeated failed patterns
- provide soft priors for next run

No validation leakage inside the same run. Use memory only as prior for new runs.

### v14 Growth / capacity planner

Only after v13 is stable:

- add route/rank/block growth signals
- use mechanism health: block cosine, write norm, route entropy, operator entropy
- optionally add nullspace/rank-aware growth later

## Stop conditions

Do not add new attention heads if the real problem is route/category specialization.
Do not add more operators if the current operators are not being used or analyzed.
Do not optimize speed until the logic is stable unless training becomes impossible.
