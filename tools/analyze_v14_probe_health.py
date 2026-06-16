#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def mean(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = [f(r.get(key)) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def analyze(report: Dict[str, Any], tail_n: int = 20) -> Dict[str, Any]:
    rows = report.get("rows", [])
    tail = rows[-tail_n:]
    issues = []

    def add(level: str, item: str, why: str, fix: str) -> None:
        issues.append({"level": level, "item": item, "why": why, "fix": fix})

    m = {k: mean(tail, k) for k in [
        "loss", "ce_loss", "matrix_aux_loss", "acc", "write_gate_mean", "write_gate_min", "write_gate_max",
        "op_entropy", "op_usage_entropy", "update_norm_mean", "update_norm_max",
        "grad_slot_norm_mean", "grad_slot_norm_max", "grad_layer_entropy", "grad_block_entropy", "grad_dead_slot_frac",
        "loss_write_budget", "loss_op_entropy_floor", "loss_op_usage_balance", "loss_slot_diversity",
        "step_ms", "fwd_ms", "bwd_ms", "max_cuda_mem_mb",
    ]}

    if m.get("write_gate_mean") is not None:
        if m["write_gate_mean"] > 0.9:
            add("HIGH", "write_gate_saturation", f"mean write gate {m['write_gate_mean']:.3f}", "raise safe/noop usage, lower write target, add write budget or lower gate bias")
        elif m["write_gate_mean"] < 0.25:
            add("MED", "write_gate_too_closed", f"mean write gate {m['write_gate_mean']:.3f}", "raise write target or reduce suppress/noop prior")
    if m.get("op_entropy") is not None and m["op_entropy"] < 0.15:
        add("MED", "op_choice_too_hard", f"op entropy {m['op_entropy']:.3f}", "increase entropy floor during warmup or use topk=3 before topk=1")
    if m.get("grad_dead_slot_frac") is not None and m["grad_dead_slot_frac"] > 0.25:
        add("HIGH", "dead_gradient_slots", f"dead slot frac {m['grad_dead_slot_frac']:.3f}", "increase topdown/global feedback or add per-slot readout/load balance")
    if m.get("grad_layer_entropy") is not None and m["grad_layer_entropy"] < 0.75:
        add("HIGH", "gradient_layer_collapse", f"grad layer entropy {m['grad_layer_entropy']:.3f}", "add layer role balance or inspect layer-specific gates")
    if m.get("grad_block_entropy") is not None and m["grad_block_entropy"] < 0.75:
        add("HIGH", "gradient_block_collapse", f"grad block entropy {m['grad_block_entropy']:.3f}", "add block role balance or increase block diversity")
    if m.get("matrix_aux_loss") is not None and m.get("ce_loss") is not None and m["matrix_aux_loss"] > 0.25 * max(m["ce_loss"], 1e-8):
        add("MED", "aux_loss_too_strong", f"aux {m['matrix_aux_loss']:.6g} vs ce {m['ce_loss']:.6g}", "lower matrix auxiliary lambdas")
    if m.get("update_norm_max") is not None and m.get("update_norm_mean") is not None and m["update_norm_max"] > 8 * max(m["update_norm_mean"], 1e-8):
        add("MED", "spiky_update_norm", f"update max {m['update_norm_max']:.3g} mean {m['update_norm_mean']:.3g}", "log per-slot update map and consider update norm clip")

    return {"ok": True, "means": m, "issues": issues, "args": report.get("args", {})}


def write_md(path: Path, result: Dict[str, Any]) -> None:
    lines = ["# v14 probe health", ""]
    m = result["means"]
    lines.append("## Tail means")
    for k, v in m.items():
        if v is not None:
            lines.append(f"- `{k}`: `{v:.6g}`")
    lines.append("")
    lines.append("## Issues")
    if not result["issues"]:
        lines.append("- No major health blocker detected in the report tail.")
    for it in result["issues"]:
        lines.append(f"- **{it['level']} `{it['item']}`** — {it['why']}. Fix: {it['fix']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tail", type=int, default=20)
    args = ap.parse_args()
    run = Path(args.run_dir)
    report = json.loads((run / "probe_report.json").read_text(encoding="utf-8"))
    result = analyze(report, tail_n=args.tail)
    (run / "health_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(run / "health_report.md", result)
    print(json.dumps({"ok": True, "run": str(run), "issues": len(result["issues"]), "health_md": str(run / "health_report.md")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
