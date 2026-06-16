# v14 probe health

## Tail means
- `loss`: `0.00175485`
- `ce_loss`: `0.000627246`
- `matrix_aux_loss`: `0.0011276`
- `acc`: `1`
- `write_gate_mean`: `0.721682`
- `write_gate_min`: `0.645184`
- `write_gate_max`: `0.787422`
- `op_entropy`: `1.09734`
- `op_usage_entropy`: `1.09853`
- `update_norm_mean`: `12.6827`
- `update_norm_max`: `13.4382`
- `grad_slot_norm_mean`: `5.1081e-07`
- `grad_slot_norm_max`: `5.4317e-07`
- `grad_layer_entropy`: `1.38629`
- `grad_block_entropy`: `1.38629`
- `grad_dead_slot_frac`: `0`
- `loss_write_budget`: `4.28588e-06`
- `loss_op_entropy_floor`: `0`
- `loss_op_usage_balance`: `0.434613`
- `loss_slot_diversity`: `0.951194`
- `step_ms`: `39.6496`
- `fwd_ms`: `13.9489`
- `bwd_ms`: `23.9916`
- `max_cuda_mem_mb`: `69.1367`

## Issues
- **MED `aux_loss_too_strong`** — aux 0.0011276 vs ce 0.000627246. Fix: lower matrix auxiliary lambdas