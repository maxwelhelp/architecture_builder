# v13 health diagnosis

Best val: **0.56% @ epoch 13**
Last val: **0.55%**, train: **0.56%**

> Note: `rolediv/phase/latew/routediv/blkdiv/...` are loss/proxy values, not direct component activity. Near-zero can mean the regularizer is satisfied.

## Last epoch health
- `train_loss`: `1.21838`
- `val_loss`: `1.31294`
- `logit_norm`: `6.70723`
- `rolediv_loss`: `0`
- `phase_loss`: `0.178892`
- `latew_loss`: `0`
- `ldyn_loss`: `0`
- `routediv_loss`: `0.79466`
- `blkdiv_loss`: `0.412445`
- `catdiv_loss`: `0.97664`
- `catent_loss`: `0.37918`
- `sigop`: `9.50552`
- `mean_effective_gate`: `0.829793`
- `mean_dynamic_gate`: `0.806406`
- `program_blocks`: `16`
- `role_layers`: `4`
- `category_layers`: `4`
- `route_blocks`: `16`
- `attention_channel_items`: `96`

## Issues / blockers
- **MED `operator_signal_strong`** — sigop 9.51. Fix: check whether signal-op prior dominates learned role/step choices

## Top confusions
- true `right` → pred `left`: n=88 rate=0.4656084656084656
- true `no` → pred `go`: n=60 rate=0.2702702702702703
- true `down` → pred `go`: n=57 rate=0.26635514018691586
- true `yes` → pred `left`: n=46 rate=0.21296296296296297
- true `stop` → pred `up`: n=38 rate=0.2
- true `on` → pred `up`: n=35 rate=0.16990291262135923
- true `go` → pred `up`: n=32 rate=0.16666666666666666
- true `down` → pred `up`: n=35 rate=0.16355140186915887

## Next-version recommendation
- Keep this as a strong baseline if best >= 64%.
- Main risk is saturated writes/logits and train-val gap, not necessarily missing role gradients.
- Next architecture fix should add safe no-op/keep primitives, per-block health logging, and StepPlanner/ShadowTopK.