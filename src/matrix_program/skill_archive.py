from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class StepSkill:
    name: str
    role: str
    steps: List[Dict[str, Any]]
    utility: float
    cost: float
    stability: float
    source: str = "unknown"
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class SkillArchive:
    """Small append-only JSONL archive for reusable step mini-algorithms."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, skill: StepSkill) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(skill), ensure_ascii=False) + "\n")

    def load(self) -> List[StepSkill]:
        if not self.path.exists():
            return []
        out: List[StepSkill] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            out.append(StepSkill(**data))
        return out

    def top(self, role: str | None = None, limit: int = 8) -> List[StepSkill]:
        skills = self.load()
        if role is not None:
            skills = [s for s in skills if s.role == role]
        skills.sort(key=lambda s: (s.utility / max(s.cost, 1e-6), s.stability), reverse=True)
        return skills[:limit]


def extract_skill_candidates(utility_report: Dict[str, Any], *, source: str, limit: int = 8) -> List[StepSkill]:
    per = list(utility_report.get("per_step", []))
    useful = [x for x in per if str(x.get("status", "")).startswith("useful")]
    useful.sort(key=lambda x: float(x.get("utility", 0.0)), reverse=True)
    skills: List[StepSkill] = []
    for x in useful[:limit]:
        li, bi, si = int(x["layer"]), int(x["block"]), int(x["step"])
        name = f"L{li}B{bi}S{si}_utility_{float(x['utility']):.3f}"
        role = "step_useful"
        skills.append(StepSkill(
            name=name,
            role=role,
            steps=[{
                "layer": li, "block": bi, "step": si,
                "gate": float(x.get("gate", 0.0)),
                "update_norm": float(x.get("update_norm", 0.0)),
                "op_entropy": float(x.get("op_entropy", 0.0)),
                "read_entropy": float(x.get("read_entropy", 0.0)),
            }],
            utility=float(x.get("utility", 0.0)),
            cost=1.0,
            stability=0.5,
            source=source,
            tags=["auto", "step_builder"],
        ))
    return skills


def archive_skill_candidates(path: str | Path, utility_report: Dict[str, Any], *, source: str, limit: int = 8) -> int:
    archive = SkillArchive(path)
    skills = extract_skill_candidates(utility_report, source=source, limit=limit)
    for s in skills:
        archive.append(s)
    return len(skills)
