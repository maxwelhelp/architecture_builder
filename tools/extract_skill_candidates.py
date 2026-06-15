#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, re, uuid
from pathlib import Path
from typing import Any, Dict, List

STEP_RE = re.compile(r"^- S(?P<step>\d+): category `(?P<category>[^`]*)` \| ops `(?P<ops>[^`]*)`(?: \| (?P<extra>.*))?$")
BLOCK_RE = re.compile(r"^\*\*L(?P<layer>-?\d+)\.B(?P<block>\d+)\*\*$")
METRIC_RE = re.compile(r"^- `(?P<name>[^`]*)` = (?P<value>[-+0-9.eE]+)")


def parse_mix(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s or "<" in s:
        return out
    for part in s.split(" + "):
        part = part.strip()
        m = re.match(r"(?P<w>[-+0-9.]+)\*(?P<n>.+)$", part)
        if m:
            out[m.group("n").strip()] = float(m.group("w"))
    return out


def parse(path: Path) -> Dict[str, Any]:
    metrics: Dict[str, float] = {}
    skills: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = METRIC_RE.match(line)
        if m:
            metrics[m.group("name")] = float(m.group("value"))
            continue
        m = BLOCK_RE.match(line)
        if m:
            if current and len(current["program"]) >= 1:
                skills.append(current)
            current = {"layer": int(m.group("layer")), "block": int(m.group("block")), "program": []}
            continue
        m = STEP_RE.match(line)
        if m and current is not None:
            current["program"].append({
                "step": int(m.group("step")),
                "category_mix": parse_mix(m.group("category")),
                "op_mix": parse_mix(m.group("ops")),
                "raw_extra": m.group("extra") or "",
            })
    if current and len(current["program"]) >= 1:
        skills.append(current)
    return {"metrics": metrics, "skills": skills}


def classify_skill(program: List[Dict[str, Any]]) -> str:
    ops = []
    for step in program:
        ops.extend(step.get("op_mix", {}).keys())
    joined = " ".join(ops)
    if "delta_route" in joined and ("memory_write" in joined or "memory_read" in joined):
        return "delta_memory_program"
    if "route_read" in joined and "memory_read" in joined:
        return "route_memory_program"
    if "evidence_read" in joined and "residual_repair" in joined:
        return "evidence_repair_program"
    if "compare_mul" in joined:
        return "compare_program"
    if "memory_write" in joined:
        return "memory_write_program"
    return "generic_matrix_program"


def build_candidates(parsed: Dict[str, Any], source_run: str) -> List[Dict[str, Any]]:
    out = []
    metrics = parsed["metrics"]
    for s in parsed["skills"]:
        prog = s["program"]
        # Keep useful one-step motifs too, but multi-step motifs get higher value.
        op_names = set()
        for step in prog:
            op_names.update(step.get("op_mix", {}).keys())
        value = 0.3 + 0.2 * (len(prog) > 1) + 0.1 * ("delta_route" in op_names) + 0.1 * ("memory_read" in op_names or "memory_write" in op_names) + 0.1 * ("route_read" in op_names)
        if value < 0.35:
            continue
        out.append({
            "skill_id": f"candidate_{classify_skill(prog)}_{uuid.uuid4().hex[:8]}",
            "status": "candidate",
            "source_run": source_run,
            "source_layer": s["layer"],
            "source_block": s["block"],
            "skill_type": classify_skill(prog),
            "program": prog,
            "metrics": {
                "block_cos": metrics.get("block_cos"),
                "delta_cos": metrics.get("delta_cos"),
                "norm_cos": metrics.get("norm_cos"),
                "estimated_value": round(value, 3),
            },
            "reuse_policy": {
                "apply_as": "weak_priors_only",
                "apply_when": ["matching_space_signature", "head_pressure_matches_program_ops"],
                "avoid_when": ["heldout_reward_negative", "duplicate_skill", "task_geometry_mismatch"],
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pseudocode", help="program_pseudocode.md")
    ap.add_argument("--source-run", default=None)
    ap.add_argument("--out", default="skills/skill_bank.jsonl")
    args = ap.parse_args()
    src = Path(args.pseudocode)
    source_run = args.source_run or src.parent.name
    parsed = parse(src)
    candidates = build_candidates(parsed, source_run)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(candidates)} candidate skills to {out}")


if __name__ == "__main__":
    main()
