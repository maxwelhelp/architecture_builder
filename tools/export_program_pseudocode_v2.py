#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def top(items: Iterable[Dict[str, Any]], key: str = "weight", n: int = 3) -> List[Dict[str, Any]]:
    return sorted(list(items or []), key=lambda x: float(x.get(key, 0.0)), reverse=True)[:n]


def op_desc(op: str) -> str:
    return {
        "noop": "ничего не менять",
        "identity": "пропустить текущее состояние",
        "keep_prev": "сохранить предыдущее состояние блока",
        "small_refine": "малое уточнение без разрушения",
        "short_onset": "короткие onset/атаки",
        "delta": "изменения во времени",
        "offset": "смещение/раннее-позднее отличие",
        "freq": "частотная проекция",
        "time": "временная проекция",
        "residual_refdelta": "ремонт residual/delta",
        "normalize": "нормализация состояния",
        "suppress": "подавление общего/шумового режима",
        "ema_memory": "EMA-память",
        "memory_read": "чтение памяти",
        "memory_write": "запись памяти",
        "global_summary": "глобальная сводка",
        "energy_count": "учёт энергии",
        "class_pair_contrast": "контраст пар классов",
        "block_compare": "сравнение блоков",
        "matrix_mlp": "матричный MLP mixer",
        "bilinear": "билинейное смешивание",
        "mlp": "канальный MLP",
    }.get(op, op)


def extract_blocks(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    for root in (obj.get("val"), obj.get("train"), obj):
        if not isinstance(root, dict):
            continue
        seq = root.get("seq", root)
        if isinstance(seq, dict) and isinstance(seq.get("operator_program_by_block"), list):
            return seq["operator_program_by_block"]
        if isinstance(seq, dict) and isinstance(seq.get("program_summary_by_layer"), list):
            out: List[Dict[str, Any]] = []
            for layer in seq["program_summary_by_layer"]:
                out.extend(layer.get("blocks", []))
            if out:
                return out
    return []


def infer_pattern(blocks: List[Dict[str, Any]]) -> str:
    roles = [str(b.get("role", "")) for b in blocks]
    parts = []
    if any("extract" in r for r in roles[:4]):
        parts.append("ранний слой извлекает onset/delta/time/freq признаки")
    if any("repair" in r for r in roles):
        parts.append("средние блоки делают residual repair / normalize / small_refine")
    if any("suppress" in r for r in roles):
        parts.append("есть suppress/common-mode removal для подавления шума/общего режима")
    if any(any(o.get("op") in ("noop", "identity", "keep_prev", "small_refine") for o in b.get("operator_program", [])) for b in blocks):
        parts.append("safe ops реально используются, поэтому блоки могут не писать разрушительный update")
    if any(any(o.get("op") in ("memory_read", "memory_write", "ema_memory") for o in b.get("operator_program", [])) for b in blocks):
        parts.append("память участвует как EMA/read/write компонент")
    return "; ".join(parts) + "." if parts else "No clear high-level pattern found."


def render_md(obj: Dict[str, Any], source: str) -> str:
    blocks = extract_blocks(obj)
    lines: List[str] = ["# Matrix Program Pseudocode", "", f"Source: `{source}`", ""]
    if obj.get("epoch") is not None:
        lines += [f"Epoch: `{obj.get('epoch')}`", ""]
    train = obj.get("train", {}) if isinstance(obj.get("train"), dict) else {}
    val = obj.get("val", {}) if isinstance(obj.get("val"), dict) else {}
    lines.append("## Metrics")
    if train:
        lines.append(f"- train: loss `{train.get('loss')}` acc `{train.get('acc')}`")
    if val:
        lines.append(f"- val: loss `{val.get('loss')}` acc `{val.get('acc')}`")
    lines.append("")
    lines.append("## Learned Program")
    lines.append("")
    if not blocks:
        lines.append("No operator_program_by_block found in this analysis file.")
        return "\n".join(lines) + "\n"

    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for b in blocks:
        by_layer[int(b.get("layer", 0))].append(b)
    for layer in sorted(by_layer):
        lines.append(f"### Layer {layer}")
        for b in sorted(by_layer[layer], key=lambda x: int(x.get("block", x.get("stage", 0)))):
            name = b.get("name", f"L{layer}.B{b.get('block', '?')}")
            role = b.get("role", "unknown")
            gate = b.get("effective_gate")
            ops = top(b.get("operator_program", []), n=5)
            cats = top(b.get("category_program", []), key="weight", n=3)
            lines.append(f"- **{name}** role=`{role}` gate=`{gate}`")
            if cats:
                lines.append("  - categories: " + ", ".join([f"{c.get('category')}={float(c.get('weight', 0)):.3f}" for c in cats]))
            if ops:
                lines.append("  - ops: " + ", ".join([f"{o.get('op')}={float(o.get('weight', 0)):.3f}" for o in ops]))
                lines.append("  - смысл: " + "; ".join([op_desc(str(o.get("op"))) for o in ops[:3]]))
            task = top(b.get("task_patterns", []), n=3)
            if task:
                lines.append("  - task focus: " + ", ".join([f"{t.get('name', t.get('cell'))}={float(t.get('weight', 0)):.3f}" for t in task]))
        lines.append("")
    lines += ["## High-level pattern", "", infer_pattern(blocks), "", "## Practical read", "", "Это soft program. Если top-ops стабильны на heldout и между эпохами, их можно переводить в hard/top-k программу."]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src = Path(args.analysis_json)
    Path(args.out).write_text(render_md(load_json(src), str(src)), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
