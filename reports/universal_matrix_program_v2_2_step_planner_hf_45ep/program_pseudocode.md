# Matrix Program Pseudocode

Source: `runs/universal_matrix_program_v2_2_step_planner_hf_45ep/analysis_epoch_045.json`

## Metrics

- `block_cos` = 0.975671
- `block_mse` = 0.0887007
- `delta_cos` = 0.932788
- `delta_mse` = 0.0887007
- `norm_cos` = 0.988837
- `norm_mse` = 0.0222029

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
- S0: category `0.50*extract + 0.11*transform` | ops `0.83*evidence_read + 0.17*residual_repair` | gate=0.855, routeH=1.473

**L0.B1**
- S0: category `0.38*extract + 0.14*compare` | ops `0.88*residual_repair + 0.12*evidence_read` | gate=0.992, routeH=1.031

**L0.B2**
- S0: category `0.45*extract` | ops `0.75*evidence_read + 0.25*residual_repair` | gate=0.852, routeH=1.607

**L0.B3**
- S0: category `0.51*extract + 0.11*transform` | ops `0.64*evidence_read + 0.36*residual_repair` | gate=0.957, routeH=1.272


### Layer order 1 / layer_id 1

**L1.B0**
- S0: category `0.21*compare + 0.21*repair` | ops `0.98*residual_repair` | gate=0.473, routeH=1.754

**L1.B1**
- S0: category `0.25*repair + 0.25*compare` | ops `1.00*residual_repair` | gate=0.941, routeH=0.644

**L1.B2**
- S0: category `0.23*compare + 0.23*repair` | ops `0.97*residual_repair` | gate=0.773, routeH=2.007

**L1.B3**
- S0: category `0.27*compare + 0.27*repair` | ops `0.94*residual_repair + 0.06*evidence_read` | gate=0.742, routeH=1.952

**L1.B4**
- S0: category `0.25*compare + 0.25*repair` | ops `0.99*residual_repair` | gate=0.805, routeH=2.059


### Layer order 2 / layer_id 2

**L2.B0**
- S0: category `0.22*memory + 0.17*repair` | ops `0.89*evidence_read + 0.10*residual_repair` | gate=0.719, routeH=1.762

**L2.B1**
- S0: category `0.23*repair + 0.23*memory` | ops `0.85*evidence_read + 0.11*residual_repair` | gate=0.656, routeH=1.505

**L2.B2**
- S0: category `0.26*memory + 0.23*repair` | ops `0.97*evidence_read` | gate=0.479, routeH=1.608

**L2.B3**
- S0: category `0.29*repair + 0.18*memory` | ops `0.84*residual_repair + 0.14*evidence_read` | gate=0.945, routeH=1.534

**L2.B4**
- S0: category `0.25*repair + 0.20*memory` | ops `0.78*evidence_read + 0.16*residual_repair + 0.04*compare_mul` | gate=0.523, routeH=2.133


### Layer order 3 / layer_id 5

**L5.B0**
- S0: category `0.38*aggregate + 0.13*suppress` | ops `0.56*delta_route + 0.26*residual_repair + 0.08*identity + 0.05*memory_write` | gate=0.836, routeH=1.955
- S1: category `0.40*aggregate + 0.13*suppress` | ops `0.37*memory_write + 0.37*identity + 0.10*evidence_read + 0.07*route_read` | gate=0.902, routeH=1.883

**L5.B1**
- S0: category `0.33*aggregate + 0.13*memory` | ops `0.57*memory_read + 0.33*route_read + 0.06*aggregate_merge` | gate=0.984, routeH=1.358

**L5.B2**
- S0: category `0.29*aggregate + 0.18*transform` | ops `0.90*evidence_read` | gate=0.910, routeH=1.474

**L5.B3**
- S0: category `0.42*aggregate + 0.12*memory` | ops `0.69*memory_read + 0.18*route_read + 0.05*aggregate_merge` | gate=0.973, routeH=1.478

**L5.B4**
- S0: category `0.45*aggregate + 0.11*suppress` | ops `0.93*aggregate_merge` | gate=0.914, routeH=2.101
- S1: category `0.45*aggregate + 0.12*memory` | ops `0.38*residual_repair + 0.34*delta_route + 0.07*identity + 0.07*evidence_read` | gate=0.840, routeH=2.059


### Layer order 4 / layer_id 4

**L4.B0**
- S0: category `0.47*aggregate + 0.11*suppress` | ops `1.00*memory_write` | gate=0.883, routeH=0.713
- S1: category `0.49*aggregate + 0.12*memory` | ops `1.00*memory_write` | gate=0.945, routeH=0.975

**L4.B1**
- S0: category `0.30*aggregate + 0.13*memory` | ops `0.64*residual_repair + 0.33*evidence_read` | gate=0.965, routeH=0.995

**L4.B2**
- S0: category `0.30*aggregate + 0.15*transform` | ops `0.71*compare_mul + 0.24*identity` | gate=0.836, routeH=1.600

**L4.B3**
- S0: category `0.43*aggregate + 0.12*memory` | ops `0.82*evidence_read + 0.14*memory_write` | gate=0.494, routeH=1.649

**L4.B4**
- S0: category `0.46*aggregate + 0.12*memory` | ops `0.92*memory_write` | gate=0.898, routeH=1.764


### Layer order 5 / layer_id 3

**L3.B0**
- S0: category `0.45*aggregate + 0.11*suppress` | ops `0.87*memory_write + 0.11*evidence_read` | gate=0.977, routeH=1.845
- S1: category `0.43*aggregate + 0.12*suppress` | ops `0.49*memory_write + 0.48*evidence_read` | gate=0.828, routeH=1.877

**L3.B1**
- S0: category `0.30*aggregate + 0.15*compare` | ops `0.85*route_read + 0.12*memory_read` | gate=0.977, routeH=0.494

**L3.B2**
- S0: category `0.30*aggregate + 0.15*transform` | ops `0.64*compare_mul + 0.26*evidence_read + 0.08*identity` | gate=0.812, routeH=1.098

**L3.B3**
- S0: category `0.42*aggregate + 0.12*suppress` | ops `0.50*memory_read + 0.17*route_read + 0.09*evidence_read + 0.08*aggregate_merge` | gate=0.977, routeH=1.566
- S1: category `0.42*aggregate + 0.12*suppress` | ops `0.78*evidence_read + 0.09*residual_repair + 0.05*memory_write + 0.05*identity` | gate=0.645, routeH=1.577


## High-level pattern

```text
evidence_read/residual_repair -> residual_repair/evidence_read -> evidence_read/residual_repair -> delta_route/residual_repair/memory_read -> memory_write/evidence_read/residual_repair -> evidence_read/memory_write/route_read
```

## Head block read summary

- `delta` top blocks: #1:0.070, #2:0.069, #0:0.051, #3:0.048, #14:0.048, #17:0.047, #7:0.046, #9:0.044
- `norm` top blocks: #2:0.080, #1:0.061, #6:0.058, #4:0.054, #0:0.053, #11:0.052, #19:0.051, #14:0.051
