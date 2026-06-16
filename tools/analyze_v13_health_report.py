#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def last_existing(pattern_dir: Path, prefix: str) -> Path | None:
    files = sorted(pattern_dir.glob(prefix))
    return files[-1] if files else None


def top_confusions(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    return analysis.get("top_confusions", [])[:8]


def mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def diagnose(rows: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
    if not rows:
        return {"ok": False, "reason": "no metrics.csv rows"}
    best_row = max(rows, key=lambda r: f(r.get("val_acc")))
    last = rows[-1]
    best_acc = f(best_row.get("val_acc"))
    last_acc = f(last.get("val_acc"))
    train_acc = f(last.get("train_acc"))
    val_loss = f(last.get("val_loss"))
    train_loss = f(last.get("train_loss"))
    logit_norm = f(last.get("logits_cell_norm", last.get("logit_norm")))

    # These are mostly loss/proxy values printed by v13, not raw activation strengths.
    # Near-zero can mean the regularizer is satisfied, not necessarily that the component is dead.
    rolediv_loss = f(last.get("role_diversity", last.get("rolediv")))
    phase_loss = f(last.get("phase_role", last.get("phase")))
    latew_loss = f(last.get("late_write", last.get("latew")))
    ldyn_loss = f(last.get("late_dyn_gate", last.get("ldyn")))
    routediv_loss = f(last.get("route_diversity", last.get("routediv")))
    blkdiv_loss = f(last.get("block_diversity", last.get("blkdiv")))
    catdiv_loss = f(last.get("category_diversity", last.get("catdiv")))
    catent_loss = f(last.get("category_entropy", last.get("catent")))
    sigop = f(last.get("signal_op_bias_norm", last.get("sigop")))

    issues: List[Dict[str, Any]] = []
    def add(level: str, item: str, why: str, fix: str) -> None:
        issues.append({"level": level, "item": item, "why": why, "fix": fix})

    if train_acc - last_acc > 18:
        add("HIGH", "train_val_gap", f"train_acc {train_acc:.2f}% vs val_acc {last_acc:.2f}%", "use stronger confidence/head regularization, keep best checkpoint, and avoid interpreting last-epoch loss as quality")
    if val_loss > train_loss * 2.5:
        add("MED", "val_loss_high", f"train_loss {train_loss:.4f}, val_loss {val_loss:.4f}", "reduce overconfidence; inspect persistent class confusions")
    if logit_norm > 10:
        add("HIGH", "logit_norm_saturation", f"logit_norm {logit_norm:.2f}", "lower head LR or raise head weight decay; add logit/confidence penalty in v14")
    if sigop > 8:
        add("MED", "operator_signal_strong", f"sigop {sigop:.2f}", "check whether signal-op prior dominates learned role/step choices")
    if catdiv_loss > 0.80 and catent_loss < 0.20:
        add("INFO", "sharp_category_program", f"catdiv_loss {catdiv_loss:.3f}, catent_loss {catent_loss:.3f}", "categories are sharp; inspect actual category_mix_by_layer before changing loss")

    seq = analysis.get("seq_space", {}) if analysis else {}
    gate = seq.get("effective_stage_gate") or []
    dyn = seq.get("dynamic_stage_gate") or []
    if gate:
        gm = mean([f(x) for x in gate])
        if gm > 0.95:
            add("HIGH", "stage_gates_saturated", f"mean effective_stage_gate {gm:.3f}", "add safe noop/keep_prev/small_refine primitives, reduce stage_gate_bias_init/gate_floor, add write-budget logging")
    else:
        gm = None
    if dyn:
        dm = mean([f(x) for x in dyn])
        if dm > 0.95:
            add("HIGH", "dynamic_gates_saturated", f"mean dynamic_stage_gate {dm:.3f}", "reduce late floors and add explicit write/no-write choices")
    else:
        dm = None

    # Raw program/report fields. Their presence is more important than scalar losses.
    program = seq.get("operator_program_by_block", [])
    role_mix = seq.get("role_mix_by_layer", [])
    cat_mix = seq.get("category_mix_by_layer", [])
    route_usage = seq.get("route_usage_by_block", [])
    attention_channels = seq.get("attention_channels_by_stage", [])

    if not program:
        add("HIGH", "missing_operator_program_report", "operator_program_by_block is missing", "run with fixed report code and publish seq_analysis")
    if not role_mix:
        add("MED", "missing_role_mix_report", "role_mix_by_layer is missing", "add/report role mix per layer before judging role health")
    if not attention_channels:
        add("MED", "missing_attention_channel_report", "attention_channels_by_stage is missing", "apply v13 report/profiler fixes before next report")

    return {
        "ok": True,
        "best": {"epoch": int(f(best_row.get("epoch"))), "val_acc": best_acc},
        "last": {
            "epoch": int(f(last.get("epoch"))), "train_acc": train_acc, "val_acc": last_acc,
            "train_loss": train_loss, "val_loss": val_loss, "logit_norm": logit_norm,
            "rolediv_loss": rolediv_loss, "phase_loss": phase_loss, "latew_loss": latew_loss, "ldyn_loss": ldyn_loss,
            "routediv_loss": routediv_loss, "blkdiv_loss": blkdiv_loss, "catdiv_loss": catdiv_loss, "catent_loss": catent_loss, "sigop": sigop,
            "mean_effective_gate": gm,
            "mean_dynamic_gate": dm,
            "program_blocks": len(program),
            "role_layers": len(role_mix),
            "category_layers": len(cat_mix),
            "route_blocks": len(route_usage),
            "attention_channel_items": len(attention_channels),
        },
        "issues": issues,
        "top_confusions": top_confusions(analysis),
        "note": "Printed rolediv/phase/latew/etc are loss/proxy values. Use detailed seq_space fields to judge actual component usage.",
    }


def write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# v13 health diagnosis", ""]
    if not report.get("ok"):
        lines.append(f"Not OK: {report.get('reason')}")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    best = report["best"]; last = report["last"]
    lines.append(f"Best val: **{best['val_acc']:.2f}% @ epoch {best['epoch']}**")
    lines.append(f"Last val: **{last['val_acc']:.2f}%**, train: **{last['train_acc']:.2f}%**")
    lines.append("")
    lines.append("> Note: `rolediv/phase/latew/routediv/blkdiv/...` are loss/proxy values, not direct component activity. Near-zero can mean the regularizer is satisfied.")
    lines.append("")
    lines.append("## Last epoch health")
    for k in ["train_loss", "val_loss", "logit_norm", "rolediv_loss", "phase_loss", "latew_loss", "ldyn_loss", "routediv_loss", "blkdiv_loss", "catdiv_loss", "catent_loss", "sigop", "mean_effective_gate", "mean_dynamic_gate", "program_blocks", "role_layers", "category_layers", "route_blocks", "attention_channel_items"]:
        v = last.get(k)
        if isinstance(v, float):
            lines.append(f"- `{k}`: `{v:.6g}`")
        elif v is not None:
            lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Issues / blockers")
    for it in report.get("issues", []):
        lines.append(f"- **{it['level']} `{it['item']}`** — {it['why']}. Fix: {it['fix']}")
    lines.append("")
    lines.append("## Top confusions")
    for c in report.get("top_confusions", []):
        lines.append(f"- true `{c.get('true')}` → pred `{c.get('pred')}`: n={c.get('n')} rate={c.get('rate_of_true')}")
    lines.append("")
    lines.append("## Next-version recommendation")
    lines.append("- Keep this as a strong baseline if best >= 64%.")
    lines.append("- Main risk is saturated writes/logits and train-val gap, not necessarily missing role gradients.")
    lines.append("- Next architecture fix should add safe no-op/keep primitives, per-block health logging, and StepPlanner/ShadowTopK.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_or_report_dir")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    root = Path(args.run_or_report_dir)
    metrics = load_metrics(root / "metrics.csv")
    analysis_path = last_existing(root, "seq_analysis_epoch_*.json") or last_existing(root, "analysis_epoch_*.json") or (root / "final_report.json")
    analysis = load_json(analysis_path) if analysis_path else {}
    report = diagnose(metrics, analysis)
    out_dir = Path(args.out_dir) if args.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "health_diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(out_dir / "health_diagnosis.md", report)
    print(json.dumps({"ok": report.get("ok"), "best": report.get("best"), "last": report.get("last"), "issues": len(report.get("issues", [])), "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
