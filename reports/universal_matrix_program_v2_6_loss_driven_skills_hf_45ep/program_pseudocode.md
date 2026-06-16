# Matrix Program Pseudocode

Source: `runs/universal_matrix_program_v2_6_loss_driven_skills_hf_45ep/analysis_epoch_045.json`

## Metrics

- `block_cos` = 0.974628
- `block_mse` = 0.093229
- `delta_cos` = 0.929226
- `delta_mse` = 0.093229
- `norm_cos` = 0.988564
- `norm_mse` = 0.0227418

## Controller

- `mode`: EXPLORE
- `action`: descriptor_repair
- `target_layer`: 2
- `target_block`: 2
- `target_step`: 1
- `reason`: plateau_cleanup_planner
- `growth_detail`: descriptor_repair_L2B2_steps_1_ops_residual_repair,delta_route,normalize

## Learned Program


### Layer order 0 / layer_id 0

**L0.B0**
- S0: category `0.42*extract + 0.10*memory` | ops `1.00*evidence_read` | gate=0.996, routeH=0.830

**L0.B1**
- S0: category `0.45*extract` | ops `1.00*evidence_read` | gate=0.996, routeH=0.693

**L0.B2**
- S0: category `0.47*extract + 0.10*transform` | ops `1.00*evidence_read` | gate=0.984, routeH=0.695

**L0.B3**
- S0: category `0.38*extract + 0.11*compare` | ops `1.00*evidence_read` | gate=0.992, routeH=0.753

**L0.B4**
- S0: category `0.40*extract + 0.11*memory` | ops `1.00*evidence_read` | gate=1.000, routeH=0.826


### Layer order 1 / layer_id 1

**L1.B0**
- S0: category `0.18*compare + 0.18*repair` | ops `1.00*evidence_read` | gate=0.918, routeH=1.313

**L1.B1**
- S0: category `0.23*compare + 0.23*repair` | ops `1.00*evidence_read` | gate=0.926, routeH=1.340

**L1.B2**
- S0: category `0.20*compare + 0.20*repair` | ops `1.00*evidence_read` | gate=0.992, routeH=1.264

**L1.B3**
- S0: category `0.25*compare + 0.25*repair` | ops `1.00*evidence_read` | gate=0.965, routeH=2.054

**L1.B4**
- S0: category `0.24*compare + 0.24*repair` | ops `0.90*evidence_read + 0.10*residual_repair` | gate=0.773, routeH=1.286


### Layer order 2 / layer_id 4

**L4.B0**
- S0: category `0.23*repair + 0.21*memory` | ops `0.94*evidence_read + 0.04*residual_repair` | gate=0.996, routeH=1.762

**L4.B1**
- S0: category `0.25*memory + 0.23*repair` | ops `0.97*evidence_read` | gate=1.000, routeH=0.199

**L4.B2**
- S0: category `0.32*repair + 0.16*suppress` | ops `0.98*residual_repair` | gate=1.000, routeH=0.124

**L4.B3**
- S0: category `0.25*memory + 0.17*repair` | ops `0.92*evidence_read + 0.07*residual_repair` | gate=0.992, routeH=1.266


### Layer order 3 / layer_id 2

**L2.B0**
- S0: category `0.26*repair + 0.18*memory` | ops `0.85*residual_repair + 0.15*delta_route` | gate=0.988, routeH=0.504

**L2.B1**
- S0: category `0.24*repair + 0.24*memory` | ops `0.67*residual_repair + 0.32*evidence_read` | gate=0.988, routeH=0.102

**L2.B2**
- S0: category `0.29*repair + 0.18*memory` | ops `0.72*evidence_read + 0.19*residual_repair + 0.08*memory_write` | gate=0.781, routeH=0.793

**L2.B3**
- S0: category `0.25*memory + 0.20*repair` | ops `0.71*delta_route + 0.28*residual_repair` | gate=0.996, routeH=0.129


### Layer order 4 / layer_id 3

**L3.B0**
- S0: category `0.45*aggregate + 0.14*transform` | ops `0.91*memory_write + 0.06*evidence_read` | gate=0.984, routeH=1.484

**L3.B1**
- S0: category `0.26*aggregate + 0.14*compare` | ops `0.54*delta_route + 0.43*residual_repair` | gate=0.977, routeH=0.064

**L3.B2**
- S0: category `0.41*aggregate + 0.14*memory` | ops `0.91*memory_write + 0.07*delta_route` | gate=0.977, routeH=1.314

**L3.B3**
- S0: category `0.34*aggregate + 0.12*compare` | ops `0.88*evidence_read + 0.10*memory_write` | gate=0.582, routeH=0.974


## High-level pattern

```text
evidence_read -> evidence_read/residual_repair -> evidence_read/residual_repair -> residual_repair/delta_route/evidence_read -> memory_write/evidence_read/delta_route
```

## Head block read summary

- `delta` top blocks: #7:0.097, #14:0.083, #18:0.076, #19:0.070, #17:0.056, #12:0.055, #8:0.055, #13:0.055
- `norm` top blocks: #9:0.113, #7:0.112, #18:0.079, #2:0.071, #14:0.056, #8:0.054, #5:0.051, #4:0.049
