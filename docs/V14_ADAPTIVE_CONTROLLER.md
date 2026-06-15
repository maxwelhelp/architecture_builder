# V14 Adaptive Architecture Controller

## Why

`v13_soft_explore_25ep` became the current best SpeechCommands baseline:

```text
best_val_acc = 64.35% @ epoch 24
final_val_acc = 64.30% @ epoch 25
```

This showed that softer exploration is useful. The next step is not a fixed epoch schedule, but an adaptive controller.

## Core idea

The gradient trains weights. The controller adjusts architecture priors softly.

```text
metrics + mechanism health
        ↓
ArchitectureState
        ↓
controller mode/action
        ↓
small prior/temp changes
        ↓
next training epoch
```

## Modes

- EXPLORE: soften categories/primitives when growth is slow or categories collapsed early.
- EXPLOIT: sharpen current best path when score improves.
- REPAIR: boost memory/repair/compare when a specific class or pattern is weak.
- ROLLBACK: restore best architecture state if a change hurts score.
- WAIT: keep a short pending-effect buffer before judging an action.

## Signals

- validation score slope
- train/validation gap
- weak class pressure
- category collapse
- route collapse
- block cosine
- late write health
- token/block entropy

## Rule

Do not hard switch the architecture. Apply only small soft deltas to role/category/operator/route priors and temperatures. Keep rollback.
