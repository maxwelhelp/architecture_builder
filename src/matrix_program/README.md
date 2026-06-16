# matrix_program modular core

This directory is the new small-file architecture layer.

Why it exists:

- the old v13 file is too large to maintain through small GitHub edits;
- patch scripts are fragile;
- new ideas such as ShadowTopK, MatrixMLP families, and architecture archives need reusable components.

Current modules:

- `operators.py` — operation metadata and costs.
- `shadow_topk.py` — warmup/top-k/shadow planning helpers.
- `archive.py` — lightweight architecture snapshot archive.

Target design:

```text
input/task/evidence/memory
  -> planner
  -> shadow_topk scorer
  -> selected matrix primitives
  -> report/archive
```

Rules:

1. Keep files small.
2. No more giant one-file experiments for new logic.
3. Heavy run outputs stay in `runs/` and are ignored.
4. Reports go through `commands/push_latest_report.sh`.
5. Reusable skills/snapshots go into JSONL archives, not checkpoints.
