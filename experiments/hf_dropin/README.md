# HF Matrix Drop-in Probe

Goal: test the matrix backend as a drop-in replacement for one hidden layer of a small HuggingFace model.

## Teacher

A frozen HF model provides hidden states:

```text
x = hidden_states[layer]
y = hidden_states[layer + 1]
```

## Student

```text
x -> SequenceEvidenceMatrix -> MatrixCore -> MatrixSequenceHeads -> y_hat
```

All heads are matrix heads, not plain MLP heads:

- `block`: predicts the next hidden state
- `delta`: predicts `y - x`
- `norm`: predicts normalized next hidden state

## Controller

The controller acts only on plateau / strong slowdown:

- EXPLORE: soften categories / try new priors
- EXPLOIT: sharpen after new best
- ROLLBACK: restore best architecture state on harmful action
- WAIT: pending-effect buffer before judging a recent action

Bad actions get cooldown/blacklist so it does not repeat the same failed mutation forever.

## Command

```bash
bash commands/run_hf_matrix_dropin_probe_v1.sh
```

## What to inspect

- `metrics.csv`: block/delta/norm MSE and cosine
- `controller_state.jsonl`: action decisions
- `analysis_epoch_XXX.json`: role/category/operator program
