# ShadowTopK Matrix Program

Goal: test cheap counterfactual branch search before we wire it into the large v13 SpeechCommands file.

Core idea:

```text
h -> cheap sketch P(h) -> score all branches -> full compute top-k only
```

Training phases:

1. Warmup: compute all operators fully so every branch receives normal task-gradient.
2. Shadow scoring: cheap sketch scores all branches while full compute still includes enough alternatives.
3. Top-k execution: compute only k selected branches, usually k=1,2,3.
4. Shadow probe: sometimes compute one unselected branch only for router/utility training; it does not affect the main output.

Why block-level top-k, not token-level top-k:

- per-token top-k creates scatter/gather overhead in PyTorch;
- Tesla P40 benefits more from dense batched ops;
- block-level top-k keeps the execution vectorized.

MatrixMLP variants planned:

- matrix_mlp_local
- matrix_mlp_global
- matrix_mlp_lowrank
- matrix_mlp_memory
- noop / keep_prev / small_refine for safe writes

Recommended next experiment:

```bash
bash commands/run_shadow_topk_hf_v1.sh
```

Report fields to inspect:

- selected_ops_by_block
- branch_scores
- shadow_probe_wins
- estimated_speedup_vs_all
- val block/delta/norm mse/cos
