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


def metric_series(rows: List[Dict[str, Any]], key: str) -> List[float]:
    return [f(r.get(key)) for r in rows if r.get(key) not in (None, "")]


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
    rolediv = f(last.get("role_diversity", last.get("rolediv")))
    phase = f(last.get("phase_role", last.get("phase")))
    latew = f(last.get("late_write", last.get("latew")))
    ldyn = f(last.get("late_dyn_gate", last.get("ldyn")))
    routediv = f(last.get("route_diversity", last.get("routediv")))
    blkdiv = f(last.get("block_diversity", last.get("blkdiv")))
    catdiv = f(last.get("category_diversity", last.get("catdiv")))
    catent = f(last.get("category_entropy", last.get("catent")))
    sigop = f(last.get("signal_op_bias_norm", last.get("sigop")))

    issues: List[Dict[str, Any]] = []
    def add(level: str, item: str, why: str, fix: str) -> None:
        issues.append({"level": level, "item": item, "why": why, "fix": fix})

    if train_acc - last_acc > 18:
        add("HIGH", "overfit_or_confidence_gap", f"train_acc {train_acc:.2f}% vs val_acc {last_acc:.2f}%", "lower LR/weight decay schedule, add confidence/logit penalty, keep best checkpoint/early stop")
    if val_loss > train_loss * 2.5:
        add("MED", "val_loss_high", f"train_loss {train_loss:.4f}, val_loss {val_loss:.4f}", "reduce overconfidence and check class confusions")
    if logit_norm > 10:
        add("HIGH", "logit_norm_saturation", f"logit_norm {logit_norm:.2f}", "add/enable logit norm penalty or lower head LR; monitor confidence")
    if rolediv <= 1e-4:
        add("HIGH", "role_planner_dead_or_collapsed", f"role_diversity {rolediv:.6f}", "force/seed role priors, weaken category dominance, add per-layer role health loss")
    if phase < 0.05:
        add("HIGH", "phase_program_dead", f"phase_role {phase:.4f}", "add StepPlanner or stronger phase anchors; log phase per layer/block")
    if latew <= 1e-4 and ldyn <= 1e-4:
        add("HIGH", "late_write_dead", f"late_write {latew:.6f}, late_dyn {ldyn:.6f}", "add safe primitives/noop and make late-write target affect live gates; log per-block write")
    if routediv < 0.05:
        add("MED", "route_collapse", f"route_diversity {routediv:.4f}", "increase/extend route diversity schedule or use top-k route exploration")
    if blkdiv < 0.05:
        add("MED", "block_collapse", f"block_diversity {blkdiv:.4f}", "add block role/step diversity or anti-collapse only before late sharpening")
    if catdiv > 0.85 and catent < 0.15:
        add("OK", "categories_are_alive", f"catdiv {catdiv:.3f}, catent {catent:.3f}", "keep category planner; do not over-penalize it")
    if sigop > 8:
        add("MED", "operator_signal_strong", f"sigop {sigop:.2f}", "check if operator prior dominates learned roles; add per-op contribution report")

    seq = analysis.get("seq_space", {}) if analysis else {}
    gate = seq.get("effective_stage_gate") or []
    dyn = seq.get("dynamic_stage_gate") or []
    if gate:
        gm = mean([f(x) for x in gate])
        if gm > 0.95:
            add("HIGH", "stage_gates_saturated", f"mean effective_stage_gate {gm:.3f}", "add keep/noop primitives; reduce gate floors; add write budget or gate entropy")
    if dyn:
        dm = mean([f(x) for x in dyn])
    else:
        dm = 0.0

    return {
        "ok": True,
        "best": {"epoch": int(f(best_row.get("epoch"))), "val_acc": best_acc},
        "last": {
            "epoch": int(f(last.get("epoch"))), "train_acc": train_acc, "val_acc": last_acc,
            "train_loss": train_loss, "val_loss": val_loss, "logit_norm": logit_norm,
            "rolediv": rolediv, "phase": phase, "latew": latew, "ldyn": ldyn,
            "routediv": routediv, "blkdiv": blkdiv, "catdiv": catdiv, "catent": catent, "sigop": sigop,
            "mean_effective_gate": mean([f(x) for x in gate]) if gate else None,
            "mean_dynamic_gate": dm if dyn else None,
        },
        "issues": issues,
        "top_confusions": top_confusions(analysis),
    }


def write_md(path: Path, report: Dict[str, Any]) -> None:
    lines = []
    lines.append("# v13 health diagnosis")
    lines.append("")
    if not report.get("ok"):
        lines.append(f"Not OK: {report.get('reason')}")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    best = report["best"]; last = report["last"]
    lines.append(f"Best val: **{best['val_acc']:.2f}% @ epoch {best['epoch']}**")
    lines.append(f"Last val: **{last['val_acc']:.2f}%**, train: **{last['train_acc']:.2f}%**")
    lines.append("")
    lines.append("## Last epoch health")
    for k in ["train_loss", "val_loss", "logit_norm", "rolediv", "phase", "latew", "ldyn", "routediv", "blkdiv", "catdiv", "catent", "sigop", "mean_effective_gate", "mean_dynamic_gate"]:
        v = last.get(k)
        if v is not None:
            lines.append(f"- `{k}`: `{v:.6g}`")
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
    lines.append("- Keep the run as a strong baseline if best >= 64%.")
    lines.append("- Do not over-focus on more epochs; main blockers are role/phase/late-write collapse and gate saturation.")
    lines.append("- Next version should add safe no-op/keep primitives, per-block health logging, and StepPlanner/ShadowTopK rather than only changing LR.")
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
