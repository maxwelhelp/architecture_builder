# v13 health diagnosis

Best val: **0.62% @ epoch 26**
Last val: **0.60%**, train: **0.85%**

> Note: `rolediv/phase/latew/routediv/blkdiv/...` are loss/proxy values, not direct component activity. Near-zero can mean the regularizer is satisfied.

## Last epoch health
- `train_loss`: `0.434291`
- `val_loss`: `1.39323`
- `logit_norm`: `10.9214`
- `rolediv_loss`: `0`
- `phase_loss`: `0.188857`
- `latew_loss`: `0`
- `ldyn_loss`: `0`
- `routediv_loss`: `0.0748217`
- `blkdiv_loss`: `0.0151539`
- `catdiv_loss`: `0.794159`
- `catent_loss`: `0.249139`
- `sigop`: `17.9016`
- `mean_effective_gate`: `0.995497`
- `mean_dynamic_gate`: `0.997195`
- `program_blocks`: `16`
- `role_layers`: `4`
- `category_layers`: `4`
- `route_blocks`: `16`
- `attention_channel_items`: `96`

## Issues / blockers
- **MED `val_loss_high`** — train_loss 0.4343, val_loss 1.3932. Fix: reduce overconfidence; inspect persistent class confusions
- **HIGH `logit_norm_saturation`** — logit_norm 10.92. Fix: lower head LR or raise head weight decay; add logit/confidence penalty in v14
- **MED `operator_signal_strong`** — sigop 17.90. Fix: check whether signal-op prior dominates learned role/step choices
- **HIGH `stage_gates_saturated`** — mean effective_stage_gate 0.995. Fix: add safe noop/keep_prev/small_refine primitives, reduce stage_gate_bias_init/gate_floor, add write-budget logging
- **HIGH `dynamic_gates_saturated`** — mean dynamic_stage_gate 0.997. Fix: reduce late floors and add explicit write/no-write choices

## Top confusions
- true `go` → pred `no`: n=38 rate=0.19791666666666666
- true `down` → pred `go`: n=40 rate=0.18691588785046728
- true `right` → pred `left`: n=31 rate=0.164021164021164
- true `no` → pred `go`: n=36 rate=0.16216216216216217
- true `go` → pred `down`: n=25 rate=0.13020833333333334
- true `off` → pred `up`: n=20 rate=0.10526315789473684
- true `down` → pred `no`: n=21 rate=0.09813084112149532
- true `on` → pred `up`: n=19 rate=0.09223300970873786

## Next-version recommendation
- Keep this as a strong baseline if best >= 64%.
- Main risk is saturated writes/logits and train-val gap, not necessarily missing role gradients.
- Next architecture fix should add safe no-op/keep primitives, per-block health logging, and StepPlanner/ShadowTopK.