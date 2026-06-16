#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load(path: Path) -> Dict[str, Any]:
    return json.loads((path / "probe_report.json").read_text(encoding="utf-8"))


def metric(report: Dict[str, Any], key: str) -> float:
    ss = report.get("speed_summary", {})
    if key in ss:
        return float(ss[key])
    rows = report.get("rows", [])
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / max(1, len(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="./runs/v14_parallel_slots_probe_compare.md")
    args = ap.parse_args()
    reports = []
    for d in args.dirs:
        p = Path(d)
        r = load(p)
        a = r.get("args", {})
        reports.append({
            "dir": str(p),
            "mode": a.get("mode", p.name),
            "topk": a.get("topk"),
            "refine": a.get("refine_iters"),
            "mean_step_ms": metric(r, "mean_step_ms"),
            "mean_fwd_ms": metric(r, "mean_fwd_ms"),
            "mean_bwd_ms": metric(r, "mean_bwd_ms"),
            "mean_opt_ms": metric(r, "mean_opt_ms"),
            "max_cuda_mem_mb": metric(r, "max_cuda_mem_mb"),
            "last_loss": float(r.get("rows", [{}])[-1].get("loss", 0.0)),
            "last_acc": float(r.get("rows", [{}])[-1].get("acc", 0.0)),
            "last_write_gate": r.get("rows", [{}])[-1].get("write_gate_mean"),
            "last_op_entropy": r.get("rows", [{}])[-1].get("op_entropy"),
        })
    best = min(reports, key=lambda x: x["mean_step_ms"])
    lines = ["# v14 parallel slots probe comparison", ""]
    lines.append("| mode | topk | refine | step ms | fwd ms | bwd ms | opt ms | max MB | last acc | last loss | write gate | op entropy |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in reports:
        wg = "" if r["last_write_gate"] is None else f"{float(r['last_write_gate']):.4f}"
        ent = "" if r["last_op_entropy"] is None else f"{float(r['last_op_entropy']):.4f}"
        lines.append(f"| {r['mode']} | {r['topk']} | {r['refine']} | {r['mean_step_ms']:.3f} | {r['mean_fwd_ms']:.3f} | {r['mean_bwd_ms']:.3f} | {r['mean_opt_ms']:.3f} | {r['max_cuda_mem_mb']:.1f} | {r['last_acc']:.3f} | {r['last_loss']:.6f} | {wg} | {ent} |")
    lines.append("")
    lines.append(f"Fastest by mean_step_ms: **{best['mode']}** at `{best['mean_step_ms']:.3f} ms/step`.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Ignore step 1 in raw logs: CUDA/module warmup dominates it.")
    lines.append("- `sequential_exact` performs one update per layer, so it is expected to be slower and use more memory.")
    lines.append("- Current top-k is dense-prototype top-k: candidates are still precomputed. Real speedup requires lazy heavy-branch execution.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "fastest": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
