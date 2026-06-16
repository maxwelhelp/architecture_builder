# v13 health diagnosis

Best val: **0.60% @ epoch 19**
Last val: **0.59%**, train: **0.65%**

> Note: `rolediv/phase/latew/routediv/blkdiv/...` are loss/proxy values, not direct component activity. Near-zero can mean the regularizer is satisfied.

## Last epoch health
- `train_loss`: `0.97284`
- `val_loss`: `1.17053`
- `logit_norm`: `8.02914`
- `rolediv_loss`: `0`
- `phase_loss`: `0.165394`
- `latew_loss`: `0`
- `ldyn_loss`: `0`
- `routediv_loss`: `0.350463`
- `blkdiv_loss`: `0.184028`
- `catdiv_loss`: `0.830369`
- `catent_loss`: `0.352364`
- `sigop`: `11.3806`
- `mean_effective_gate`: `0.784573`
- `mean_dynamic_gate`: `0.798585`
- `program_blocks`: `16`
- `role_layers`: `4`
- `category_layers`: `4`
- `route_blocks`: `16`
- `attention_channel_items`: `96`

## Issues / blockers
- **MED `operator_signal_strong`** — sigop 11.38. Fix: check whether signal-op prior dominates learned role/step choices

## Top confusions
- true `down` → pred `on`: n=45 rate=0.2102803738317757
- true `no` → pred `go`: n=40 rate=0.18018018018018017
- true `left` → pred `off`: n=31 rate=0.1657754010695187
- true `go` → pred `up`: n=31 rate=0.16145833333333334
- true `left` → pred `right`: n=30 rate=0.16042780748663102
- true `right` → pred `left`: n=26 rate=0.13756613756613756
- true `on` → pred `up`: n=27 rate=0.13106796116504854
- true `down` → pred `up`: n=27 rate=0.1261682242990654

## Next-version recommendation
- Keep this as a strong baseline if best >= 64%.
- Main risk is saturated writes/logits and train-val gap, not necessarily missing role gradients.
- Next architecture fix should add safe no-op/keep primitives, per-block health logging, and StepPlanner/ShadowTopK.