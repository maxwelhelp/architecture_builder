# Matrix Program Pseudocode

Source: `runs/universal_matrix_program_v2_5_alive_cost_hf_45ep/analysis_epoch_045.json`

## Metrics

- `block_cos` = 0.974761
- `block_mse` = 0.0918599
- `delta_cos` = 0.930292
- `delta_mse` = 0.0918599
- `norm_cos` = 0.988598
- `norm_mse` = 0.0226751

## Controller

- `mode`: WAIT
- `action`: noop
- `target_layer`: -1
- `target_block`: -1
- `target_step`: -1
- `reason`: normal
- `growth_detail`: none

## Learned Program


### Layer order 0 / layer_id 0

**L0.B0**
- S0: category `0.55*extract + 0.12*transform` | ops `0.94*identity + 0.06*residual_repair` | gate=0.988, routeH=0.712

**L0.B1**
- S0: category `0.51*extract + 0.11*memory` | ops `0.51*identity + 0.49*residual_repair` | gate=0.992, routeH=0.775

**L0.B2**
- S0: category `0.44*extract + 0.10*compare` | ops `0.83*residual_repair + 0.17*identity` | gate=0.992, routeH=0.709

**L0.B3**
- S0: category `0.50*extract + 0.11*transform` | ops `0.59*residual_repair + 0.41*identity` | gate=0.992, routeH=0.839

**L0.B4**
- S0: category `0.54*extract + 0.12*transform` | ops `0.87*identity + 0.13*residual_repair` | gate=0.996, routeH=0.724


### Layer order 1 / layer_id 1

**L1.B0**
- S0: category `0.23*compare + 0.23*repair` | ops `0.93*residual_repair + 0.07*identity` | gate=0.949, routeH=0.008

**L1.B1**
- S0: category `0.26*compare + 0.26*repair` | ops `1.00*residual_repair` | gate=0.980, routeH=0.077

**L1.B2**
- S0: category `0.25*compare + 0.25*repair` | ops `1.00*residual_repair` | gate=0.988, routeH=0.149

**L1.B3**
- S0: category `0.22*compare + 0.22*repair` | ops `1.00*residual_repair` | gate=0.988, routeH=0.265

**L1.B4**
- S0: category `0.19*compare + 0.19*repair` | ops `0.95*residual_repair + 0.05*identity` | gate=0.988, routeH=0.014


### Layer order 2 / layer_id 4

**L4.B0**
- S0: category `0.33*memory + 0.15*transform` | ops `0.73*identity + 0.27*residual_repair` | gate=0.969, routeH=1.433

**L4.B1**
- S0: category `0.29*repair + 0.19*memory` | ops `0.82*residual_repair + 0.18*identity` | gate=0.996, routeH=1.529

**L4.B2**
- S0: category `0.24*repair + 0.23*memory` | ops `0.87*residual_repair + 0.13*identity` | gate=0.996, routeH=1.749

**L4.B3**
- S0: category `0.25*memory + 0.21*repair` | ops `0.73*residual_repair + 0.27*identity` | gate=0.996, routeH=1.353

**L4.B4**
- S0: category `0.39*memory + 0.18*transform` | ops `0.93*identity + 0.07*residual_repair` | gate=0.980, routeH=2.150


### Layer order 3 / layer_id 5

**L5.B0**
- S0: category `0.36*memory + 0.17*transform` | ops `0.82*identity + 0.18*residual_repair` | gate=0.973, routeH=1.612

**L5.B1**
- S0: category `0.29*memory + 0.17*repair` | ops `0.88*residual_repair + 0.11*identity` | gate=0.574, routeH=1.248

**L5.B2**
- S0: category `0.25*memory + 0.23*repair` | ops `0.81*residual_repair + 0.19*identity` | gate=0.617, routeH=0.412

**L5.B3**
- S0: category `0.24*memory + 0.23*repair` | ops `0.91*residual_repair + 0.08*identity` | gate=0.699, routeH=1.453

**L5.B4**
- S0: category `0.37*memory + 0.18*transform` | ops `0.94*identity + 0.06*residual_repair` | gate=0.984, routeH=2.093


### Layer order 4 / layer_id 2

**L2.B0**
- S0: category `0.36*memory + 0.17*transform` | ops `0.87*identity + 0.13*residual_repair` | gate=0.961, routeH=2.042

**L2.B1**
- S0: category `0.26*repair + 0.21*memory` | ops `0.95*residual_repair + 0.05*identity` | gate=0.996, routeH=0.174

**L2.B2**
- S0: category `0.25*repair + 0.23*memory` | ops `0.95*residual_repair + 0.05*identity` | gate=0.996, routeH=0.060

**L2.B3**
- S0: category `0.28*memory + 0.18*repair` | ops `0.90*residual_repair + 0.10*identity` | gate=1.000, routeH=0.764

**L2.B4**
- S0: category `0.38*memory + 0.18*transform` | ops `0.90*identity + 0.10*residual_repair` | gate=0.988, routeH=1.992


### Layer order 5 / layer_id 3

**L3.B0**
- S0: category `0.27*memory + 0.26*transform` | ops `1.00*identity` | gate=0.914, routeH=1.994

**L3.B1**
- S0: category `0.34*aggregate + 0.14*repair` | ops `0.95*residual_repair + 0.04*identity` | gate=0.574, routeH=1.544

**L3.B2**
- S0: category `0.28*aggregate + 0.16*memory` | ops `0.94*residual_repair + 0.06*identity` | gate=0.691, routeH=0.458

**L3.B3**
- S0: category `0.31*aggregate + 0.15*transform` | ops `0.83*residual_repair + 0.17*identity` | gate=0.969, routeH=1.496


## High-level pattern

```text
identity/residual_repair -> residual_repair/identity -> identity/residual_repair -> identity/residual_repair -> identity/residual_repair -> identity/residual_repair
```

## Head block read summary

- `delta` top blocks: #2:0.075, #8:0.054, #25:0.054, #7:0.051, #17:0.051, #18:0.045, #6:0.045, #24:0.044
- `norm` top blocks: #2:0.061, #5:0.058, #22:0.054, #17:0.052, #25:0.051, #3:0.051, #8:0.049, #10:0.046
