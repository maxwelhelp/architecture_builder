#!/usr/bin/env python3
"""Patch StepProgram v1 with two safe grammar improvements.

1. Hard-mask physically impossible step reads:
   S0 cannot read prev_step or all_prev_steps in sequential_exact mode.

2. Add suspicious-combination reporting:
   - high class read on noop/identity/keep_state-heavy slots;
   - expensive primitive mass with low utility;
   - memory/global writes without memory/global primitive mass.

This is intentionally conservative: it does not forbid unusual combinations,
it only masks impossible ones and logs suspicious ones.
"""
from pathlib import Path

path = Path("experiments/step_program/run_step_program_speechcommands_v1.py")
text = path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Hard mask impossible reads for S0.
# ---------------------------------------------------------------------------
old = '''                read_logits = self.step_read_net(torch.cat([state, input_ctx, layer_route_ctx, global_ctx, memory_ctx], dim=-1))
                read_gates = torch.softmax((read_logits / max(1e-4, read_temp)).float(), dim=-1).to(dtype)
'''
new = '''                read_logits = self.step_read_net(torch.cat([state, input_ctx, layer_route_ctx, global_ctx, memory_ctx], dim=-1))
                # Hard-mask physically impossible reads in strict sequential mode:
                # S0 has no previous step and no all-prev-steps state.
                if s == 0:
                    read_logits = read_logits.clone()
                    read_logits[..., 2] = -1.0e4  # prev_step
                    read_logits[..., 3] = -1.0e4  # all_prev_steps
                read_gates = torch.softmax((read_logits / max(1e-4, read_temp)).float(), dim=-1).to(dtype)
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 2. Add suspicious_combinations key to program report.
# ---------------------------------------------------------------------------
old = '''            "5_which_parts_are_needed_utility": [],
            "output_class_read": [],
        }
'''
new = '''            "5_which_parts_are_needed_utility": [],
            "suspicious_combinations": [],
            "output_class_read": [],
        }
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# 3. Add suspicious metrics inside per-step report.
# ---------------------------------------------------------------------------
old = '''                    slot_index = 1 + ((l * model.S + s) * model.N + b)
                    levels["5_which_parts_are_needed_utility"].append({
                        "address": step_addr,
                        "write_gate": float(wg[l, b, s]),
                        "global_write_gate": float(gw[l, b, s]),
                        "memory_write_gate": float(mw[l, b, s]),
                        "class_read_mass": float(slot_use[slot_index]) if slot_index < slot_use.numel() else 0.0,
                        "utility_proxy": float(wg[l, b, s] * (slot_use[slot_index] if slot_index < slot_use.numel() else 0.0)),
                    })
'''
new = '''                    slot_index = 1 + ((l * model.S + s) * model.N + b)
                    class_mass = float(slot_use[slot_index]) if slot_index < slot_use.numel() else 0.0
                    safe_mass = float(pg[l, b, s, :, :3].mean().clamp(0.0, 1.0))
                    expensive_mass = float(pg[l, b, s, :, [4, 5, 6]].sum(dim=-1).mean().clamp(0.0, 1.0))
                    memory_prim_mass = float(pg[l, b, s, :, 7].mean().clamp(0.0, 1.0))
                    global_prim_mass = float(pg[l, b, s, :, 8].mean().clamp(0.0, 1.0))
                    utility_proxy = float(wg[l, b, s] * class_mass)
                    levels["5_which_parts_are_needed_utility"].append({
                        "address": step_addr,
                        "write_gate": float(wg[l, b, s]),
                        "global_write_gate": float(gw[l, b, s]),
                        "memory_write_gate": float(mw[l, b, s]),
                        "class_read_mass": class_mass,
                        "safe_mass": safe_mass,
                        "expensive_mass": expensive_mass,
                        "memory_primitive_mass": memory_prim_mass,
                        "global_primitive_mass": global_prim_mass,
                        "utility_proxy": utility_proxy,
                    })
                    if class_mass > 0.025 and safe_mass > 0.55:
                        levels["suspicious_combinations"].append({
                            "address": step_addr,
                            "kind": "class_reads_safe_slot",
                            "class_read_mass": class_mass,
                            "safe_mass": safe_mass,
                            "why": "output reads a slot that mostly selected noop/identity/keep_state",
                        })
                    if expensive_mass > 0.50 and utility_proxy < 0.002:
                        levels["suspicious_combinations"].append({
                            "address": step_addr,
                            "kind": "expensive_low_utility",
                            "expensive_mass": expensive_mass,
                            "utility_proxy": utility_proxy,
                            "why": "expensive primitives are selected but the slot is barely written/read",
                        })
                    if float(mw[l, b, s]) > 0.02 and memory_prim_mass < 0.08:
                        levels["suspicious_combinations"].append({
                            "address": step_addr,
                            "kind": "memory_write_without_memory_mass",
                            "memory_write_gate": float(mw[l, b, s]),
                            "memory_primitive_mass": memory_prim_mass,
                            "why": "memory register is written without enough memory primitive mass",
                        })
                    if float(gw[l, b, s]) > 0.02 and global_prim_mass < 0.08:
                        levels["suspicious_combinations"].append({
                            "address": step_addr,
                            "kind": "global_write_without_global_mass",
                            "global_write_gate": float(gw[l, b, s]),
                            "global_primitive_mass": global_prim_mass,
                            "why": "global register is written without enough global primitive mass",
                        })
'''
if old in text and new not in text:
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print(f"patched {path}")
