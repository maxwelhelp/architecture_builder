# Drop-in Transformer Probe

Goal: test the matrix backend as a drop-in replacement for parts of a Transformer-like block.

## Targets

One shared matrix backend can be trained against several teacher targets:

- `attention`: replace/imitate frozen MultiheadAttention output.
- `ffn`: replace/imitate frozen FFN output.
- `block`: replace/imitate full TransformerBlock output.

This is more realistic than a single-head probe because a real drop-in module may need multiple task heads.

## Principle

Only the input and output heads change. The core remains matrix-space based:

```text
SequenceEvidenceMatrix
TaskMatrix
MechanismMatrix
GlobalMatrix
MemoryMatrix
Role/Category/Primitive planner
```

## Main command

```bash
bash commands/run_multihead_dropin_probe_v1.sh
```

## What to inspect

- metrics.csv: per-head MSE/cos and total score
- controller_state.jsonl: explore/exploit/rollback decisions
- analysis_epoch_XXX.json: category/role/operator program

## Extension questions

- If attention head is easy but block head is hard, add depth.
- If all heads plateau but categories are diverse, add primitives.
- If categories collapse early, increase exploration temperature.
- If memory category is weak, add/boost memory routes/primitives.
