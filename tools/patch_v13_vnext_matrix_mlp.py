#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

TARGET = Path("sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        print(f"SKIP {name}: pattern not found or already patched")
        return text
    print(f"PATCH {name}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # ------------------------------------------------------------------
    # 0) Old patch safety: never duplicate argparse flag.
    # ------------------------------------------------------------------
    if "--no-export-architecture-memory" not in text:
        text = replace_once(
            text,
            '    p.add_argument("--export-architecture-memory", action="store_true", default=True)',
            '    p.add_argument("--export-architecture-memory", action="store_true", default=True)\n    p.add_argument("--no-export-architecture-memory", dest="export_architecture_memory", action="store_false")',
            "idempotent no-export flag",
        )
    else:
        print("SKIP no-export flag: already present")

    # ------------------------------------------------------------------
    # 1) MatrixMLP primitive: matrix-native MLP that mixes block/state axis and channel axis.
    # ------------------------------------------------------------------
    if '"matrix_mlp"' not in text:
        text = replace_once(
            text,
            '        "mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",',
            '        "mlp", "matrix_mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",',
            "OP_NAMES add matrix_mlp",
        )
    else:
        print("SKIP OP_NAMES matrix_mlp: already present")

    text = replace_once(
        text,
        "        self.mlp_op = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))\n        self.bilin_h = nn.Linear(dim, dim, bias=False)",
        "        self.mlp_op = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))\n        # MatrixMLP primitive: unlike channel-only MLP, this mixes the active block axis\n        # through a learned block-to-block matrix and then applies a channel MLP.\n        self.matrix_mlp_q = nn.Linear(dim, dim, bias=False)\n        self.matrix_mlp_k = nn.Linear(dim, dim, bias=False)\n        self.matrix_mlp_v = nn.Linear(dim, dim, bias=False)\n        self.matrix_mlp_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))\n        self.bilin_h = nn.Linear(dim, dim, bias=False)",
        "MatrixMLP modules",
    )

    # Priors/roles/categories: make matrix_mlp reachable but not dominant.
    text = replace_once(
        text,
        '                bump(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "normalize"), 0.24)',
        '                bump(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "matrix_mlp", "normalize"), 0.24)',
        "layer1 prior matrix_mlp",
    )
    text = replace_once(
        text,
        '                bump(2, ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "suppress", "normalize", "residual_refdelta", "late_read_repair"), 0.24)',
        '                bump(2, ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "matrix_mlp", "suppress", "normalize", "residual_refdelta", "late_read_repair"), 0.24)',
        "layer2 prior matrix_mlp",
    )
    text = replace_once(
        text,
        '        target(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "normalize"))',
        '        target(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "matrix_mlp", "normalize"))',
        "phase target matrix_mlp L1",
    )
    text = replace_once(
        text,
        '        role_ops("transform", ("bilinear", "normalize", "residual_refdelta", "time", "freq"))',
        '        role_ops("transform", ("matrix_mlp", "bilinear", "normalize", "residual_refdelta", "time", "freq"))',
        "role transform matrix_mlp",
    )
    text = replace_once(
        text,
        '        role_ops("memory", ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "residual_refdelta", "normalize"))',
        '        role_ops("memory", ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "matrix_mlp", "residual_refdelta", "normalize"))',
        "role memory matrix_mlp",
    )
    text = replace_once(
        text,
        '        category_ops("transform", ("mlp", "bilinear", "normalize", "time", "freq", "offset", "residual_refdelta"))',
        '        category_ops("transform", ("mlp", "matrix_mlp", "bilinear", "normalize", "time", "freq", "offset", "residual_refdelta"))',
        "category transform matrix_mlp",
    )
    text = replace_once(
        text,
        '        category_ops("memory", ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "global_summary"))',
        '        category_ops("memory", ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "global_summary", "matrix_mlp"))',
        "category memory matrix_mlp",
    )

    # Candidate compute and stack. This keeps soft gradient through all candidates early.
    text = replace_once(
        text,
        "            # Operator candidates.\n            u_mlp = self.mlp_op(torch.cat([h, ev_mix, task_ctx, global_ctx, mem_ctx], dim=-1))\n            u_bilin = self.bilin_out(self.bilin_h(h) * self.bilin_e(ev_mix) * torch.sigmoid(self.bilin_t(task_ctx)))",
        "            # Operator candidates.\n            u_mlp = self.mlp_op(torch.cat([h, ev_mix, task_ctx, global_ctx, mem_ctx], dim=-1))\n            # MatrixMLP: block/state-axis mixer + channel mixer. It is still fully differentiable:\n            # early training gives gradient to all blocks/primitives; later entropy schedule can sharpen.\n            mm_src = block_seed + 0.50 * route_ctx + 0.25 * mem_ctx\n            mm_q = self.matrix_mlp_q(h + task_ctx)\n            mm_k = self.matrix_mlp_k(mm_src)\n            mm_v = self.matrix_mlp_v(mm_src)\n            mm_score = torch.einsum('bnd,bmd->bnm', mm_q, mm_k) * scale\n            mm_attn = torch.softmax(mm_score.float(), dim=-1).to(dtype)\n            mm_ctx = torch.einsum('bnm,bmd->bnd', mm_attn, mm_v)\n            u_matrix_mlp = self.matrix_mlp_out(torch.cat([h, mm_ctx, task_ctx, global_ctx + mem_ctx], dim=-1))\n            u_bilin = self.bilin_out(self.bilin_h(h) * self.bilin_e(ev_mix) * torch.sigmoid(self.bilin_t(task_ctx)))",
        "MatrixMLP candidate",
    )
    text = replace_once(
        text,
        "            candidates = torch.stack([u_mlp,u_bilin,u_time,u_freq,u_delta,u_short,u_offset,u_global,u_mem_read,u_mem_write,u_ema_memory,u_class_pair_memory,u_refdelta,u_compare,u_pair,u_late_read_repair,u_suppress,u_norm,u_energy], dim=2)",
        "            candidates = torch.stack([u_mlp,u_matrix_mlp,u_bilin,u_time,u_freq,u_delta,u_short,u_offset,u_global,u_mem_read,u_mem_write,u_ema_memory,u_class_pair_memory,u_refdelta,u_compare,u_pair,u_late_read_repair,u_suppress,u_norm,u_energy], dim=2)",
        "MatrixMLP candidate stack",
    )

    # ------------------------------------------------------------------
    # 2) fp16/P40 stability: all remaining major softmaxes use fp32 logits.
    # ------------------------------------------------------------------
    fp32_repls = [
        (
            "            A_route = torch.softmax(torch.einsum('bnd,brd->bnr', q_route, K_route) * scale + route_prior, dim=-1).to(dtype)\n            route_ctx = torch.einsum('bnr,brd->bnd', A_route, V_route)",
            "            score_route = torch.einsum('bnd,brd->bnr', q_route, K_route) * scale + route_prior\n            A_route = torch.softmax(score_route.float(), dim=-1).to(dtype)\n            route_ctx = torch.einsum('bnr,brd->bnd', A_route, V_route)",
            "fp32 A_route",
        ),
        (
            "            A_time = torch.softmax(torch.einsum('bnd,btd->bnt', q_time, K_time) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)",
            "            score_time = torch.einsum('bnd,btd->bnt', q_time, K_time) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_time = torch.softmax(score_time.float(), dim=-1).to(dtype)",
            "fp32 A_time",
        ),
        (
            "            A_freq = torch.softmax(torch.einsum('bnd,bfd->bnf', q_freq, K_freq) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)",
            "            score_freq = torch.einsum('bnd,bfd->bnf', q_freq, K_freq) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_freq = torch.softmax(score_freq.float(), dim=-1).to(dtype)",
            "fp32 A_freq",
        ),
        (
            "            A_onset = torch.softmax(torch.einsum('bnd,bsd->bns', q_onset, K_ev) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)",
            "            score_onset = torch.einsum('bnd,bsd->bns', q_onset, K_ev) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_onset = torch.softmax(score_onset.float(), dim=-1).to(dtype)",
            "fp32 A_onset",
        ),
        (
            "            A_energy = torch.softmax(torch.einsum('bnd,bqd->bnq', q_energy, K_energy) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)",
            "            score_energy = torch.einsum('bnd,bqd->bnq', q_energy, K_energy) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_energy = torch.softmax(score_energy.float(), dim=-1).to(dtype)",
            "fp32 A_energy",
        ),
        (
            "            A_ev = 0.5 * A_onset + 0.5 * torch.softmax(torch.einsum('bnd,bsd->bns', q_base, K_ev) * scale, dim=-1).to(dtype)",
            "            score_ev_raw = torch.einsum('bnd,bsd->bns', q_base, K_ev) * scale\n            A_ev = 0.5 * A_onset + 0.5 * torch.softmax(score_ev_raw.float(), dim=-1).to(dtype)",
            "fp32 raw evidence",
        ),
        (
            "            A_global = torch.softmax(torch.einsum('bnd,bgd->bng', q_global, K_global) * scale, dim=-1).to(dtype)",
            "            score_global = torch.einsum('bnd,bgd->bng', q_global, K_global) * scale\n            A_global = torch.softmax(score_global.float(), dim=-1).to(dtype)",
            "fp32 A_global",
        ),
        (
            "            A_mem = torch.softmax(torch.einsum('bnd,bmd->bnm', q_mem, K_mem) * scale, dim=-1).to(dtype)",
            "            score_mem = torch.einsum('bnd,bmd->bnm', q_mem, K_mem) * scale\n            A_mem = torch.softmax(score_mem.float(), dim=-1).to(dtype)",
            "fp32 A_mem",
        ),
        (
            "            gw = torch.softmax(torch.einsum('bnd,bgd->bng', gw_q, K_global2) * scale, dim=-1).to(dtype)",
            "            score_gw = torch.einsum('bnd,bgd->bng', gw_q, K_global2) * scale\n            gw = torch.softmax(score_gw.float(), dim=-1).to(dtype)",
            "fp32 global write",
        ),
        (
            "            mw = torch.softmax(torch.einsum('bnd,bmd->bnm', mw_q, K_mem2) * scale, dim=-1).to(dtype)",
            "            score_mw = torch.einsum('bnd,bmd->bnm', mw_q, K_mem2) * scale\n            mw = torch.softmax(score_mw.float(), dim=-1).to(dtype)",
            "fp32 memory write",
        ),
    ]
    for old, new, name in fp32_repls:
        text = replace_once(text, old, new, name)

    # ------------------------------------------------------------------
    # 3) Entropy/balance schedule: soft gradients early, sharper program late.
    # ------------------------------------------------------------------
    text = replace_once(
        text,
        "            category_entropy = aux[\"seq\"].category_entropy_loss\n            loss = (",
        "            category_entropy = aux[\"seq\"].category_entropy_loss\n            # vNext schedule: early soft exploration, late sharper readable primitive program.\n            # This preserves gradient from all candidates early, then stops forcing every block to mix everything.\n            denom_epochs = max(1, int(getattr(args, \"epochs\", 1)) - 1)\n            train_progress = min(1.0, max(0.0, (float(epoch) - 1.0) / float(denom_epochs)))\n            explore_w = max(0.0, 1.0 - train_progress / 0.45)\n            sharpen_w = min(1.0, max(0.0, (train_progress - 0.25) / 0.55))\n            lambda_operator_balance_eff = args.lambda_operator_balance * explore_w\n            lambda_operator_entropy_per_block_eff = args.lambda_operator_entropy_per_block * max(0.20, explore_w)\n            lambda_category_entropy_eff = args.lambda_category_entropy * sharpen_w\n            loss = (",
        "entropy schedule vars",
    )
    text = replace_once(
        text,
        "                + args.lambda_operator_balance * op_lb",
        "                + lambda_operator_balance_eff * op_lb",
        "operator balance schedule",
    )
    text = replace_once(
        text,
        "                + args.lambda_operator_entropy_per_block * op_entropy_block",
        "                + lambda_operator_entropy_per_block_eff * op_entropy_block",
        "operator entropy schedule",
    )
    text = replace_once(
        text,
        "                + args.lambda_category_entropy * category_entropy",
        "                + lambda_category_entropy_eff * category_entropy",
        "category entropy schedule",
    )

    if text == original:
        print("No changes made.")
        return
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_v13_vnext_matrix_mlp")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backup written: {backup}")
    TARGET.write_text(text, encoding="utf-8")
    print("Done. Run:")
    print("  python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")
    print("  python tools/smoke_v13_topologies.py")


if __name__ == "__main__":
    main()
