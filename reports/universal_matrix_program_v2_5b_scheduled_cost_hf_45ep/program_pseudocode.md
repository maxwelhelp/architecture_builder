# Matrix Program Pseudocode

Source: `runs/universal_matrix_program_v2_5b_scheduled_cost_hf_45ep/analysis_epoch_045.json`

## Metrics

- `block_cos` = 0.974788
- `block_mse` = 0.0924558
- `delta_cos` = 0.929831
- `delta_mse` = 0.0924558
- `norm_cos` = 0.988499
- `norm_mse` = 0.0228703

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
- S0: category `0.43*extract + 0.10*memory` | ops `1.00*evidence_read` | gate=0.996, routeH=1.098

**L0.B1**
- S0: category `0.46*extract` | ops `1.00*evidence_read` | gate=0.996, routeH=1.101

**L0.B2**
- S0: category `0.41*extract + 0.11*repair` | ops `1.00*evidence_read` | gate=0.996, routeH=1.105

**L0.B3**
- S0: category `0.39*extract + 0.11*memory` | ops `1.00*evidence_read` | gate=1.000, routeH=1.098

**L0.B4**
- S0: category `0.41*extract + 0.11*memory` | ops `1.00*evidence_read` | gate=0.996, routeH=1.099


### Layer order 1 / layer_id 1

**L1.B0**
- S0: category `0.20*compare + 0.20*repair` | ops `0.99*evidence_read` | gate=0.996, routeH=1.119

**L1.B1**
- S0: category `0.24*compare + 0.23*repair` | ops `1.00*evidence_read` | gate=0.996, routeH=1.849

**L1.B2**
- S0: category `0.24*compare + 0.23*repair` | ops `0.97*evidence_read` | gate=1.000, routeH=1.086

**L1.B3**
- S0: category `0.21*compare + 0.21*repair` | ops `1.00*evidence_read` | gate=1.000, routeH=1.110

**L1.B4**
- S0: category `0.24*compare + 0.24*repair` | ops `0.70*evidence_read + 0.29*residual_repair` | gate=0.988, routeH=1.747


### Layer order 2 / layer_id 4

**L4.B0**
- S0: category `0.26*repair + 0.19*memory` | ops `0.48*evidence_read + 0.46*residual_repair` | gate=1.000, routeH=1.584

**L4.B1**
- S0: category `0.24*memory + 0.22*repair` | ops `0.87*evidence_read + 0.11*residual_repair` | gate=1.000, routeH=2.018

**L4.B2**
- S0: category `0.33*repair + 0.16*suppress` | ops `0.80*residual_repair + 0.19*delta_route` | gate=1.000, routeH=0.258

**L4.B3**
- S0: category `0.23*memory + 0.18*repair` | ops `0.87*evidence_read + 0.08*residual_repair + 0.04*task_read` | gate=1.000, routeH=1.572


### Layer order 3 / layer_id 2

**L2.B0**
- S0: category `0.23*memory + 0.19*repair` | ops `0.72*residual_repair + 0.24*delta_route` | gate=0.996, routeH=1.808

**L2.B1**
- S0: category `0.24*memory + 0.23*repair` | ops `0.91*residual_repair + 0.07*evidence_read` | gate=1.000, routeH=0.099

**L2.B2**
- S0: category `0.29*repair + 0.17*memory` | ops `0.67*evidence_read + 0.27*residual_repair + 0.05*memory_read` | gate=0.996, routeH=0.091

**L2.B3**
- S0: category `0.23*memory + 0.22*repair` | ops `0.72*residual_repair + 0.26*delta_route` | gate=1.000, routeH=0.279


### Layer order 4 / layer_id 3

**L3.B0**
- S0: category `0.38*aggregate + 0.14*transform` | ops `0.76*delta_route + 0.16*memory_read` | gate=0.992, routeH=1.210

**L3.B1**
- S0: category `0.26*aggregate + 0.18*compare` | ops `0.85*residual_repair + 0.14*evidence_read` | gate=1.000, routeH=0.003

**L3.B2**
- S0: category `0.31*aggregate + 0.15*transform` | ops `0.96*evidence_read` | gate=1.000, routeH=0.699

**L3.B3**
- S0: category `0.33*aggregate + 0.13*repair` | ops `0.94*memory_read` | gate=0.996, routeH=1.417


## High-level pattern

```text
evidence_read -> evidence_read/residual_repair -> residual_repair/evidence_read/delta_route -> residual_repair/delta_route/evidence_read -> memory_read/evidence_read/delta_route
```

## Head block read summary

- `delta` top blocks: #7:0.088, #20:0.080, #12:0.076, #14:0.066, #18:0.063, #17:0.059, #21:0.048, #15:0.045
- `norm` top blocks: #7:0.111, #9:0.103, #2:0.081, #5:0.072, #0:0.068, #4:0.057, #10:0.056, #8:0.052
