# v14 StepBuilder

v13 is the fixed soft-program baseline. v14 StepBuilder is the first explicit `Layer -> Block -> Step` system.

## Goal

The core idea is that a block should not be one large primitive soup. A block should be a mini algorithm:

```text
Step 0: read evidence / previous state / memory
Step 1: transform / compare / normalize
Step 2: repair / suppress / route
Step 3: write / keep / aggregate
```

This makes skills transferable: the reusable object is a step program, not only weights.

## Files

- `src/matrix_program/step_builder.py`
  - explicit `[Layer, Block, Step]` slots
  - read-source planner
  - primitive planner
  - write gate
  - safe ops: `noop`, `identity`, `keep_prev`, `small_refine`

- `src/matrix_program/step_utility.py`
  - per-step utility summary
  - detects dead, overloaded, saturated, useful steps

- `src/matrix_program/skill_archive.py`
  - JSONL archive for reusable step skills

- `src/matrix_program/grow_prune_controller.py`
  - conservative grow/prune/replace policy
  - returns proposed actions, does not mutate architecture yet

- `experiments/v14_step_builder/run_probe.py`
  - runnable synthetic probe
  - logs utility, gates, op entropy, controller decisions, skills

- `commands/run_v14_step_builder_probe.sh`
  - one-command probe

## Controller states

Each step can be labelled as:

```text
learning
useful/skill_candidate
dead/prune_candidate
overloaded/add_step_candidate
gate_saturated
```

The controller uses train/val history plus utility report:

```text
overfit -> compress / prune / freeze / gate budget
plateau without overfit -> grow / add step / replace weak step
low utility -> mark prune candidate
useful stable step -> archive skill
```

## Next stages

1. Run synthetic StepBuilder probe.
2. Check that steps specialize and utility is not all saturated.
3. Add real SpeechCommands frontend adapter.
4. Add shadow replacement: old step and candidate replacement run in parallel for a grace period.
5. Add actual architecture mutation after controller decisions.
6. Transfer archived skills into new blocks/layers.

## Run

```bash
bash commands/run_v14_step_builder_probe.sh
cat runs/v14_step_builder_probe_L4_B4_S4/step_builder_report.md
```

Push report:

```bash
bash commands/push_run_report.sh \
  runs/v14_step_builder_probe_L4_B4_S4 \
  v14_step_builder_probe \
  "Add v14 StepBuilder probe report"
```
