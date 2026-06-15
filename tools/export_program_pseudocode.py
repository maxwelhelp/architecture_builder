#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_OPS = [
    "identity", "evidence_read", "task_read", "route_read", "delta_route",
    "compare_mul", "memory_read", "memory_write", "global_summary",
    "suppress_global", "normalize", "energy_count", "residual_repair", "aggregate_merge",
]
CATS = ["extract", "compare", "memory", "repair", "aggregate", "suppress", "transform"]


def topk(vals: List[float], names: List[str], k: int = 3, minv: float = 0.05) -> List[Tuple[str, float]]:
    pairs = sorted([(names[i] if i < len(names) else f"op_{i}", float(v)) for i, v in enumerate(vals)], key=lambda x: x[1], reverse=True)
    return [(n, v) for n, v in pairs[:k] if v >= minv]


def fmt_mix(pairs: List[Tuple[str, float]]) -> str:
    if not pairs:
        return "<weak/no dominant op>"
    return " + ".join([f"{v:.2f}*{n}" for n, v in pairs])


def export(analysis_path: Path, out_path: Path | None = None) -> str:
    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    mp: Dict[str, Any] = data.get("matrix_program", {})
    primitive_names = mp.get("primitive_names") or BASE_OPS
    active_layers = mp.get("active_layer_ids", [])
    ops_by_lbs = mp.get("operator_mix_by_layer_block_step", [])
    cats_by_lbs = mp.get("category_mix_by_layer_block_step", [])
    gates_by_lbs = mp.get("gate_by_layer_block_step", [])
    routes_by_lbs = mp.get("route_entropy_by_layer_block_step", [])
    head_read = mp.get("head_block_read", {})
    ctrl = data.get("controller", {})
    val = data.get("val", {})

    lines: List[str] = []
    lines.append(f"# Matrix Program Pseudocode\n")
    lines.append(f"Source: `{analysis_path}`\n")
    lines.append("## Metrics\n")
    for k in sorted(val):
        if k.endswith("_mse") or k.endswith("_cos"):
            lines.append(f"- `{k}` = {val[k]:.6g}")
    lines.append("\n## Controller\n")
    for k in ["mode", "action", "target_layer", "target_block", "target_step", "reason", "growth_detail"]:
        if k in ctrl:
            lines.append(f"- `{k}`: {ctrl[k]}")

    lines.append("\n## Learned Program\n")
    high_level = []
    for order, layer_id in enumerate(active_layers):
        layer_ops = ops_by_lbs[order] if order < len(ops_by_lbs) else []
        layer_cats = cats_by_lbs[order] if order < len(cats_by_lbs) else []
        lines.append(f"\n### Layer order {order} / layer_id {layer_id}\n")
        layer_top_ops = []
        for block_id, block_steps in enumerate(layer_ops):
            lines.append(f"**L{layer_id}.B{block_id}**")
            cat_steps = layer_cats[block_id] if block_id < len(layer_cats) else []
            gate_steps = gates_by_lbs[order][block_id] if order < len(gates_by_lbs) and block_id < len(gates_by_lbs[order]) else []
            route_steps = routes_by_lbs[order][block_id] if order < len(routes_by_lbs) and block_id < len(routes_by_lbs[order]) else []
            for step_id, op_vals in enumerate(block_steps):
                cat_vals = cat_steps[step_id] if step_id < len(cat_steps) else []
                op_top = topk(op_vals, primitive_names, k=4, minv=0.04)
                cat_top = topk(cat_vals, CATS, k=2, minv=0.10)
                gate = gate_steps[step_id] if step_id < len(gate_steps) else None
                route = route_steps[step_id] if step_id < len(route_steps) else None
                layer_top_ops.extend([n for n, _ in op_top[:2]])
                extra = []
                if gate is not None: extra.append(f"gate={float(gate):.3f}")
                if route is not None: extra.append(f"routeH={float(route):.3f}")
                extra_s = ", ".join(extra)
                lines.append(f"- S{step_id}: category `{fmt_mix(cat_top)}` | ops `{fmt_mix(op_top)}`" + (f" | {extra_s}" if extra_s else ""))
            lines.append("")
        if layer_top_ops:
            counts: Dict[str, int] = {}
            for op in layer_top_ops:
                counts[op] = counts.get(op, 0) + 1
            summary = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
            high_level.append("/".join([x[0] for x in summary]))

    lines.append("\n## High-level pattern\n")
    if high_level:
        lines.append("```text")
        lines.append(" -> ".join(high_level))
        lines.append("```")
    else:
        lines.append("No clear high-level pattern found.")

    if head_read:
        lines.append("\n## Head block read summary\n")
        for head, vals in head_read.items():
            pairs = sorted([(i, float(v)) for i, v in enumerate(vals)], key=lambda x: x[1], reverse=True)[:8]
            lines.append(f"- `{head}` top blocks: " + ", ".join([f"#{i}:{v:.3f}" for i, v in pairs]))

    text = "\n".join(lines) + "\n"
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("analysis", help="Path to analysis_epoch_XXX.json")
    p.add_argument("--out", default=None, help="Output markdown path. Defaults to stdout only.")
    args = p.parse_args()
    text = export(Path(args.analysis), Path(args.out) if args.out else None)
    if not args.out:
        print(text)


if __name__ == "__main__":
    main()
