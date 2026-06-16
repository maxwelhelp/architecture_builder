# v14 parallel slots experiment

This experiment tests the v14 idea without touching the strong v13 baseline.

Core tensor:

```text
H [B, L, N, D]
```

Modes:

```text
sequential_exact  slot baseline with layer-by-layer update
parallel_refine   all layers/blocks update together for K refinement iterations
hybrid_groups     early group then late group, compromise between both
```

Implemented mechanisms:

- lower-triangular layer read, so no direct ready-made future output is read;
- top-down aggregator feedback from late slots to early slots;
- global bus;
- memory read;
- safe primitives: noop, identity, small_refine, normalize, suppress;
- torch ShadowTopK dense prototype: hard top-k weights with soft gradient.

Important limitation:

`dense_topk_weighted_sum` still receives precomputed candidates, so this first v14 probe tests semantics, not final speed. True speed-up needs lazy branch execution after the candidate quality is validated.
