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

# 1) Reports must use the real OP_NAMES after safe ops were prepended.
old = '''        op_names = [
            "mlp", "matrix_mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",
            "global_summary", "memory_read", "memory_write", "ema_memory", "class_pair_memory",
            "residual_refdelta", "block_compare", "class_pair_contrast", "late_read_repair", "suppress", "normalize", "energy_count",
        ]'''
new = '        op_names = list(SequentialMatrixCellsCore.OP_NAMES)'
rep(old, new, 'summary_uses_core_op_names')

# 2) Program role labels should understand safe primitives.
old = '''            top = ops[0]["op"]
            op_set = {o["op"] for o in ops[:3]}
            if "global_summary" in op_set or "energy_count" in op_set:'''
new = '''            top = ops[0]["op"]
            op_set = {o["op"] for o in ops[:3]}
            if top in ("noop", "identity", "keep_prev"):
                return "safe/noop_keep"
            if top == "small_refine":
                return "safe/small_refine"
            if "global_summary" in op_set or "energy_count" in op_set:'''
if 'safe/noop_keep' not in text:
    rep(old, new, 'role_from_ops_safe')

# 3) Safe ops should be reachable through priors, not only fallback.
repls = [
('        target(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "matrix_mlp", "normalize"))',
 '        target(1, ("small_refine", "identity", "block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "matrix_mlp", "normalize"))', 'phase_l1_safe'),
('        target(2, ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "suppress", "normalize", "residual_refdelta", "late_read_repair"))',
 '        target(2, ("keep_prev", "small_refine", "memory_read", "memory_write", "ema_memory", "class_pair_memory", "suppress", "normalize", "residual_refdelta", "late_read_repair"))', 'phase_l2_safe'),
('        target(3, ("global_summary", "energy_count", "suppress", "memory_write", "ema_memory", "block_compare"))',
 '        target(3, ("noop", "keep_prev", "global_summary", "energy_count", "suppress", "memory_write", "ema_memory", "block_compare"))', 'phase_l3_safe'),
('        role_ops("transform", ("matrix_mlp", "bilinear", "normalize", "residual_refdelta", "time", "freq"))',
 '        role_ops("transform", ("identity", "small_refine", "matrix_mlp", "bilinear", "normalize", "residual_refdelta", "time", "freq"))', 'role_transform_safe'),
('        role_ops("repair", ("residual_refdelta", "suppress", "normalize", "class_pair_contrast", "late_read_repair", "class_pair_memory"))',
 '        role_ops("repair", ("noop", "identity", "keep_prev", "small_refine", "residual_refdelta", "suppress", "normalize", "class_pair_contrast", "late_read_repair", "class_pair_memory"))', 'role_repair_safe'),
('        role_ops("suppress", ("suppress", "normalize", "block_compare", "global_summary"))',
 '        role_ops("suppress", ("noop", "identity", "keep_prev", "suppress", "normalize", "block_compare", "global_summary"))', 'role_suppress_safe'),
('        category_ops("repair", ("residual_refdelta", "class_pair_contrast", "class_pair_memory", "late_read_repair", "suppress", "normalize"))',
 '        category_ops("repair", ("noop", "identity", "keep_prev", "small_refine", "residual_refdelta", "class_pair_contrast", "class_pair_memory", "late_read_repair", "suppress", "normalize"))', 'cat_repair_safe'),
('        category_ops("suppress", ("suppress", "normalize", "block_compare"))',
 '        category_ops("suppress", ("noop", "identity", "keep_prev", "suppress", "normalize", "block_compare"))', 'cat_suppress_safe'),
('        category_ops("transform", ("mlp", "matrix_mlp", "bilinear", "normalize", "time", "freq", "offset", "residual_refdelta"))',
 '        category_ops("transform", ("identity", "small_refine", "mlp", "matrix_mlp", "bilinear", "normalize", "time", "freq", "offset", "residual_refdelta"))', 'cat_transform_safe'),
]
for old, new, name in repls:
    if old in text:
        text = text.replace(old, new, 1)
        changes.append(name)

# 4) Late write should not fight selected safe ops.
old = '''        late_write_losses = []
        if live_write_delta_norm_by_block.numel() > 0:
            off = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                if layer >= 1:
                    wn = live_write_delta_norm_by_block[:, off:off+n].mean(dim=1)
                    late_write_losses.append(F.relu(self.late_write_target - wn).pow(2).mean())
                off += n
        late_write_loss = torch.stack(late_write_losses).mean() if late_write_losses else torch.zeros((), device=device)'''
new = '''        late_write_losses = []
        safe_idxs_for_late = [self.OP_NAMES.index(nm) for nm in ("noop", "identity", "keep_prev") if nm in self.OP_NAMES]
        safe_mass_for_late = None
        if operator_gates.numel() > 0 and safe_idxs_for_late:
            safe_mass_for_late = operator_gates.index_select(-1, torch.tensor(safe_idxs_for_late, device=device)).sum(dim=-1).float().clamp(0.0, 1.0)
        if live_write_delta_norm_by_block.numel() > 0:
            off = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                if layer >= 1:
                    wn = live_write_delta_norm_by_block[:, off:off+n].mean(dim=1)
                    if safe_mass_for_late is not None and off + n <= safe_mass_for_late.shape[1]:
                        safe = safe_mass_for_late[:, off:off+n].mean(dim=1)
                    else:
                        safe = torch.zeros_like(wn)
                    target = float(self.late_write_target) * (1.0 - safe)
                    late_write_losses.append(F.relu(target - wn).pow(2).mean())
                off += n
        late_write_loss = torch.stack(late_write_losses).mean() if late_write_losses else torch.zeros((), device=device)'''
if 'safe_idxs_for_late' not in text:
    rep(old, new, 'late_write_safe_aware')

if text == orig:
    print('No changes; report/safe logic already patched or patterns changed.')
else:
    bak = p.with_suffix(p.suffix + '.bak_before_report_safe_logic')
    if not bak.exists():
        bak.write_text(orig, encoding='utf-8')
    p.write_text(text, encoding='utf-8')
    print('Applied:', ', '.join(changes))
    print('Backup:', bak)
