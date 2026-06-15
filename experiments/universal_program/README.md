# Universal Matrix Program v1

Goal: one universal matrix-program backend for many tasks.

Task-specific parts:

```text
InputAdapter: raw task input -> TokenMatrix / EvidenceMatrix
OutputHeads: MatrixHeads -> task outputs
```

Shared core:

```text
UniversalMatrixCore
  preallocated layer slots
  preallocated block slots
  preallocated primitive slots
  Token/Evidence/Task/Mechanism/Memory/Global spaces
  matrix-space attention and operator mix
  automatic health-map controller
```

Important: growth is not hardcoded. The controller chooses growth type and location from health-map signals.

```text
plateau -> health-map -> local growth action
```

Growth actions:

- soft prior mutation
- insert/activate depth slot at selected layer order position
- activate block slot in overloaded layer
- activate candidate primitive slot in overloaded layer
- rollback to absolute best state on harmful actions

The first runnable task is HF hidden transition distillation:

```text
hidden_states[layer] -> hidden_states[layer+1]
```

Command:

```bash
bash commands/run_universal_matrix_program_v1_hf.sh
```
