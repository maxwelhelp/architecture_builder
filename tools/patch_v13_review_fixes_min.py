#!/usr/bin/env python3
from pathlib import Path

p = Path('sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py')
text = p.read_text(encoding='utf-8')
orig = text
changes = []

def rep(old, new, name):
    global text
    if old in text:
        text = text.replace(old, new, 1)
        changes.append(name)

old = '''    OP_NAMES = [
        "mlp", "matrix_mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",
        "global_summary", "memory_read", "memory_write", "ema_memory", "class_pair_memory", "residual_refdelta",
        "block_compare", "class_pair_contrast", "late_read_repair", "suppress", "normalize", "energy_count",
    ]'''
new = '''    OP_NAMES = [
        "noop", "identity", "keep_prev", "small_refine",
        "mlp", "matrix_mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",
        "global_summary", "memory_read", "memory_write", "ema_memory", "class_pair_memory", "residual_refdelta",
        "block_compare", "class_pair_contrast", "late_read_repair", "suppress", "normalize", "energy_count",
    ]'''
if '"noop", "identity", "keep_prev", "small_refine"' not in text:
    rep(old, new, 'safe_op_names')

old = '            candidates = torch.stack([u_mlp,u_matrix_mlp,u_bilin,u_time,u_freq,u_delta,u_short,u_offset,u_global,u_mem_read,u_mem_write,u_ema_memory,u_class_pair_memory,u_refdelta,u_compare,u_pair,u_late_read_repair,u_suppress,u_norm,u_energy], dim=2)'
new = '''            u_noop = torch.zeros_like(h)
            u_identity = h
            u_keep_prev = block_seed
            u_small_refine = 0.10 * u_mlp
            candidates = torch.stack([u_noop,u_identity,u_keep_prev,u_small_refine,u_mlp,u_matrix_mlp,u_bilin,u_time,u_freq,u_delta,u_short,u_offset,u_global,u_mem_read,u_mem_write,u_ema_memory,u_class_pair_memory,u_refdelta,u_compare,u_pair,u_late_read_repair,u_suppress,u_norm,u_energy], dim=2)'''
if 'u_noop = torch.zeros_like(h)' not in text:
    rep(old, new, 'safe_candidates')

old = '''            op_gates = (category_gates.unsqueeze(-1) * primitive_gates).sum(dim=2)  # [B,N,O]
            op_gates = op_gates / op_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            op_scale = 1.0 + 0.20 * torch.tanh(self.block_op_scale[layer, active_block_ids].to(device=device, dtype=dtype)).view(1,N,self.num_ops,1)'''
new = '''            op_gates = (category_gates.unsqueeze(-1) * primitive_gates).sum(dim=2)  # [B,N,O]
            op_gates = op_gates / op_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            def _op_mass(names: Tuple[str, ...]) -> torch.Tensor:
                idxs = [self.OP_NAMES.index(nm) for nm in names if nm in self.OP_NAMES]
                if not idxs:
                    return torch.zeros(B, N, device=device, dtype=dtype)
                return op_gates.index_select(-1, torch.tensor(idxs, device=device)).sum(dim=-1).clamp(0.0, 1.0)
            safe_write_mass = _op_mass(("noop", "identity", "keep_prev"))
            global_write_mass = _op_mass(("global_summary", "energy_count"))
            mem_write_mass = _op_mass(("memory_write", "ema_memory", "class_pair_memory"))

            op_scale = 1.0 + 0.20 * torch.tanh(self.block_op_scale[layer, active_block_ids].to(device=device, dtype=dtype)).view(1,N,self.num_ops,1)'''
if 'safe_write_mass = _op_mass' not in text:
    rep(old, new, 'primitive_write_masses')

old = '''            eff_gate = phase_floor + (1.0 - phase_floor) * dyn_gate
            update_expand = update[:, :, None, :].expand(-1,-1,self.block_cells,-1)'''
new = '''            eff_gate = phase_floor + (1.0 - phase_floor) * dyn_gate
            eff_gate = eff_gate * (1.0 - 0.75 * safe_write_mass).clamp(0.10, 1.0)
            update_expand = update[:, :, None, :].expand(-1,-1,self.block_cells,-1)'''
if 'safe_write_mass).clamp(0.10, 1.0)' not in text:
    rep(old, new, 'safe_mass_cell_gate')

if 'global_write_mass.unsqueeze' not in text:
    rep('            gw_gate = 0.15 * torch.sigmoid(self.global_write_gate(write_in))', '            gw_gate = 0.15 * torch.sigmoid(self.global_write_gate(write_in))\n            gw_gate = gw_gate * (0.05 + 0.95 * global_write_mass.unsqueeze(-1))', 'global_write_mass_gate')
if 'mem_write_mass.unsqueeze' not in text:
    rep('            mw_gate = 0.20 * torch.sigmoid(self.memory_write_gate(torch.cat([summaries, task_ctx, mem_ctx], dim=-1)))', '            mw_gate = 0.20 * torch.sigmoid(self.memory_write_gate(torch.cat([summaries, task_ctx, mem_ctx], dim=-1)))\n            mw_gate = mw_gate * (0.05 + 0.95 * mem_write_mass.unsqueeze(-1))', 'memory_write_mass_gate')

rep('lambda_operator_entropy_per_block_eff = args.lambda_operator_entropy_per_block * max(0.20, explore_w)', 'lambda_operator_entropy_per_block_eff = args.lambda_operator_entropy_per_block * max(0.05, explore_w)', 'late_entropy_005')
rep('p.add_argument("--amp", type=str, default="bf16", choices=["fp16", "bf16", "fp32", "off"])', 'p.add_argument("--amp", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "off"])', 'amp_fp16_default')

old = '''        if lambda_addr > 0:
            addr = torch.cat([
                self.layer_addr.float(), self.block_addr.float(), self.cell_addr.float(),
                self.global_addr.float(), self.memory_addr.float(), self.head_addr.float(), self.operator_addr.float()
            ], dim=0)
            addr = F.normalize(addr, dim=-1)
            sim = addr @ addr.T
            eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
            reg = reg + float(lambda_addr) * sim.masked_select(~eye).pow(2).mean()
        return reg'''
new = '''        if lambda_addr > 0:
            addr = torch.cat([
                self.layer_addr.float(), self.block_addr.float(), self.cell_addr.float(),
                self.global_addr.float(), self.memory_addr.float(), self.head_addr.float(), self.operator_addr.float()
            ], dim=0)
            addr = F.normalize(addr, dim=-1)
            sim = addr @ addr.T
            eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
            reg = reg + float(lambda_addr) * sim.masked_select(~eye).pow(2).mean()
        if lambda_stage > 0:
            terms = [q.float().pow(2).mean() for q in self.stage_gate_net.parameters()]
            terms += [self.class_stage_bias.float().pow(2).mean(), self.class_layer_read_bias.float().pow(2).mean()]
            reg = reg + float(lambda_stage) * torch.stack(terms).mean()
        return reg'''
if 'if lambda_stage > 0:' not in text:
    rep(old, new, 'lambda_stage_real')

if text == orig:
    print('No changes; already patched or patterns changed.')
else:
    bak = p.with_suffix(p.suffix + '.bak_before_review_fixes_min')
    if not bak.exists():
        bak.write_text(orig, encoding='utf-8')
    p.write_text(text, encoding='utf-8')
    print('Applied:', ', '.join(changes))
    print('Backup:', bak)
