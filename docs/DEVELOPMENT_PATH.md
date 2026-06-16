# Development Path

## Goal

Build a universal matrix-program backend where task-specific input adapters and heads can change, while the internal program learns reusable matrix skills.

## Current rule

Do not optimize only for score. Track four things together:

```text
quality
program logic
cost/efficiency
transferable skills
```

## What we learned

### v2.2 step planner

Best early universal result.

```text
score around 0.12968
logical program: evidence -> repair -> memory/route -> aggregation
```

Problem:

```text
no true forward-level cleanup
```

### v2.4 descriptor planner

Added primitive/category descriptors.

Good:

```text
program stayed logical
skills export works
```

Problem:

```text
descriptor_repair was mostly diagnostic, not dominant
```

### v2.5 alive cost

Added forward-level soft gates and cost pressure.

Good:

```text
step/block/primitive soft gates work
cost is measurable and trainable
late growth guard works
descriptor_repair appears late
```

Problem:

```text
cost pressure was too strong and too early
program collapsed toward identity + residual_repair
score dropped to about 0.12619
```

### v2.5b scheduled cost

Purpose:

```text
keep v2.5 forward-level self-optimization
but prevent cheap identity/residual-only collapse
```

Changes:

```text
lower cost penalty
scheduled cost ramp
scheduled alive sparsity
semantic diversity anti-collapse
stronger late growth guard
more cleanup after late phase
```

Expected result:

```text
score closer to v2.2/v2.4
program keeps evidence/route/memory/aggregate mass
cost/alive remain measurable
late add_width/add_depth reduced
```

### v2.6 loss-driven skills

Purpose:

```text
turn skills/semantic/category consistency into real losses,
not only table/controller diagnostics
```

Important correction:

```text
semantic and skill losses must be computed on trainable priors/logits,
not only on detached report tensors
```

Implemented signals:

```text
skill_loss:
  nudges op_prior/cat_prior toward reusable skill motifs from skills/skill_bank.jsonl

semantic_loss:
  prevents identity/residual-only collapse by requiring useful semantic op mass

category_consistency_loss:
  aligns cat_prior with compatible primitive families

scheduled cost/alive:
  keeps v2.5 self-optimization but weaker and later
```

Skill is not just a checkpoint:

```text
symbolic skill = step/op/category pattern
prior skill = weak op_prior/cat_prior target
weight skill = optional compatible weights, not committed to repo
```

## Success criteria for v2.6

Good run if:

```text
score >= 0.1285
block_cos >= 0.9752
delta_cos >= 0.9320
norm_cos >= 0.9886
semantic_mass >= 0.20
residual_only_ratio <= 0.80
skill_loss decreases
category consistency improves
program_pseudocode is richer than identity/residual only
late add_width/add_depth mostly absent after epoch 35
```

## Next after v2.6

If v2.6 works:

```text
v2.7 baseline comparison
```

Compare against:

```text
linear
low-rank linear
small MLP
tiny attention block
matrix program v2.6
```

If v2.6 still collapses:

```text
reduce cost more
reduce skill loss if it over-forces old motifs
increase semantic diversity only after early quality is stable
make stage-aware primitive cost inside core
```

If v2.6 is stable:

```text
add MemoryKV primitive
add ablation trace exporter
start skill-prior cold-start experiment on another model/task
build program compiler for fast runtime
```

## Safe workflow

Before starting a new run:

```bash
git status -sb
git pull --rebase
```

After a run:

```bash
python tools/export_program_pseudocode.py RUN/analysis_epoch_045.json --out RUN/program_pseudocode.md
python tools/extract_skill_candidates.py RUN/program_pseudocode.md --source-run RUN_NAME --out ./skills/skill_bank.jsonl
bash commands/push_latest_report.sh RUN RUN_NAME "Add RUN_NAME report"
git add skills tools docs commands experiments
git commit -m "Add RUN_NAME skills" || true
git push
```
