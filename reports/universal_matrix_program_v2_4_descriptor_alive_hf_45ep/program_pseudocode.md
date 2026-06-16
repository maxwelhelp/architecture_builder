# Matrix Program Pseudocode

Source: `runs/universal_matrix_program_v2_4_descriptor_alive_hf_45ep/analysis_epoch_045.json`

## Metrics

- `block_cos` = 0.9755
- `block_mse` = 0.0891866
- `delta_cos` = 0.932402
- `delta_mse` = 0.0891866
- `norm_cos` = 0.988695
- `norm_mse` = 0.0224821

## Controller

- `mode`: EXPLOIT
- `action`: sharpen
- `target_layer`: -1
- `target_block`: -1
- `target_step`: -1
- `reason`: significant_best
- `growth_detail`: none

## Learned Program


### Layer order 0 / layer_id 0

**L0.B0**
- S0: category `0.51*extract + 0.11*transform` | ops `0.82*evidence_read + 0.18*residual_repair` | gate=0.848, routeH=1.605

**L0.B1**
- S0: category `0.37*extract + 0.14*compare` | ops `0.92*residual_repair + 0.08*evidence_read` | gate=0.992, routeH=1.242

**L0.B2**
- S0: category `0.44*extract` | ops `0.70*evidence_read + 0.30*residual_repair` | gate=0.832, routeH=1.542

**L0.B3**
- S0: category `0.51*extract + 0.11*transform` | ops `0.60*residual_repair + 0.40*evidence_read` | gate=0.973, routeH=1.397

**L0.B4**
- S0: category `0.50*extract + 0.11*transform` | ops `0.77*residual_repair + 0.23*evidence_read` | gate=0.961, routeH=1.603


### Layer order 1 / layer_id 1

**L1.B0**
- S0: category `0.21*compare + 0.21*repair` | ops `0.85*residual_repair + 0.15*delta_route` | gate=0.883, routeH=1.345

**L1.B1**
- S0: category `0.25*repair + 0.25*compare` | ops `0.99*residual_repair` | gate=0.969, routeH=0.179

**L1.B2**
- S0: category `0.23*compare + 0.23*repair` | ops `0.89*residual_repair + 0.07*evidence_read + 0.05*compare_mul` | gate=0.508, routeH=2.045

**L1.B3**
- S0: category `0.27*compare + 0.27*repair` | ops `0.96*residual_repair` | gate=0.734, routeH=1.251

**L1.B4**
- S0: category `0.24*compare + 0.24*repair` | ops `0.95*residual_repair + 0.04*compare_mul` | gate=0.473, routeH=1.917


### Layer order 2 / layer_id 2

**L2.B0**
- S0: category `0.21*memory + 0.17*repair` | ops `0.47*identity + 0.20*evidence_read + 0.14*memory_write + 0.06*delta_route` | gate=0.902, routeH=1.886

**L2.B1**
- S0: category `0.23*repair + 0.21*memory` | ops `0.92*route_read + 0.06*memory_read` | gate=0.969, routeH=1.018

**L2.B2**
- S0: category `0.26*memory + 0.22*repair` | ops `0.69*evidence_read + 0.18*residual_repair + 0.08*route_read` | gate=0.902, routeH=1.405

**L2.B3**
- S0: category `0.29*repair + 0.18*memory` | ops `0.60*memory_read + 0.20*route_read + 0.10*task_read + 0.07*aggregate_merge` | gate=0.980, routeH=1.868

**L2.B4**
- S0: category `0.26*repair + 0.19*memory` | ops `0.42*aggregate_merge + 0.26*residual_repair + 0.19*route_read + 0.09*memory_read` | gate=0.945, routeH=1.825


### Layer order 3 / layer_id 5

**L5.B0**
- S0: category `0.39*aggregate + 0.13*suppress` | ops `0.93*memory_write` | gate=0.949, routeH=0.958
- S1: category `0.43*aggregate + 0.13*suppress` | ops `0.97*memory_write` | gate=0.973, routeH=1.134

**L5.B1**
- S0: category `0.33*aggregate + 0.13*memory` | ops `0.88*evidence_read + 0.11*residual_repair` | gate=0.762, routeH=0.864

**L5.B2**
- S0: category `0.30*aggregate + 0.17*transform` | ops `0.97*compare_mul` | gate=0.840, routeH=1.515

**L5.B3**
- S0: category `0.43*aggregate + 0.12*memory` | ops `0.82*residual_repair + 0.14*evidence_read` | gate=0.969, routeH=1.404

**L5.B4**
- S0: category `0.44*aggregate + 0.11*suppress` | ops `0.80*residual_repair + 0.08*memory_write` | gate=0.840, routeH=1.601
- S1: category `0.44*aggregate + 0.11*suppress` | ops `0.41*memory_read + 0.19*route_read + 0.17*aggregate_merge + 0.13*residual_repair` | gate=0.938, routeH=1.496


### Layer order 4 / layer_id 4

**L4.B0**
- S0: category `0.45*aggregate + 0.12*suppress` | ops `0.62*memory_write + 0.35*evidence_read` | gate=0.949, routeH=2.004
- S1: category `0.47*aggregate + 0.13*memory` | ops `0.44*evidence_read + 0.34*memory_write + 0.15*residual_repair` | gate=0.945, routeH=2.025

**L4.B1**
- S0: category `0.32*aggregate + 0.13*memory` | ops `0.84*memory_read + 0.07*residual_repair + 0.06*route_read` | gate=0.984, routeH=0.759

**L4.B2**
- S0: category `0.30*aggregate + 0.15*transform` | ops `0.63*evidence_read + 0.37*compare_mul` | gate=0.494, routeH=1.603

**L4.B3**
- S0: category `0.43*aggregate + 0.12*memory` | ops `0.41*memory_read + 0.40*route_read + 0.09*evidence_read + 0.05*task_read` | gate=0.977, routeH=1.564

**L4.B4**
- S0: category `0.44*aggregate + 0.12*memory` | ops `0.63*evidence_read + 0.12*residual_repair + 0.09*memory_write + 0.08*memory_read` | gate=0.879, routeH=1.830
- S1: category `0.47*aggregate + 0.12*memory` | ops `0.55*memory_read + 0.28*route_read + 0.09*aggregate_merge` | gate=0.961, routeH=1.886


### Layer order 5 / layer_id 3

**L3.B0**
- S0: category `0.46*aggregate + 0.12*suppress` | ops `0.66*memory_write + 0.08*evidence_read + 0.08*memory_read + 0.07*residual_repair` | gate=0.984, routeH=2.053
- S1: category `0.45*aggregate + 0.11*suppress` | ops `0.60*memory_write + 0.25*evidence_read + 0.04*identity + 0.04*residual_repair` | gate=0.977, routeH=2.046

**L3.B1**
- S0: category `0.32*aggregate + 0.13*transform` | ops `0.57*residual_repair + 0.35*evidence_read + 0.06*route_read` | gate=0.988, routeH=1.354

**L3.B2**
- S0: category `0.31*aggregate + 0.15*transform` | ops `0.90*evidence_read + 0.07*residual_repair` | gate=0.930, routeH=1.635

**L3.B3**
- S0: category `0.40*aggregate + 0.12*suppress` | ops `0.97*evidence_read` | gate=0.660, routeH=1.154


## High-level pattern

```text
evidence_read/residual_repair -> residual_repair/delta_route/evidence_read -> evidence_read/route_read/memory_read -> memory_write/residual_repair/evidence_read -> evidence_read/memory_read/memory_write -> evidence_read/memory_write/residual_repair
```

## Head block read summary

- `delta` top blocks: #1:0.075, #2:0.055, #3:0.051, #0:0.050, #5:0.046, #7:0.045, #13:0.041, #10:0.036
- `norm` top blocks: #2:0.071, #7:0.068, #1:0.061, #0:0.052, #17:0.046, #22:0.046, #15:0.046, #5:0.044
