from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ArchitectureSnapshot:
    run_name: str
    epoch: int
    score: float
    status: str
    selected_ops: List[str] = field(default_factory=list)
    role_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


def append_snapshot(path: str | Path, snapshot: ArchitectureSnapshot) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(snapshot), ensure_ascii=False) + "\n")


def load_snapshots(path: str | Path) -> List[ArchitectureSnapshot]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[ArchitectureSnapshot] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            out.append(ArchitectureSnapshot(**data))
    return out


def best_snapshots(path: str | Path, limit: int = 20) -> List[ArchitectureSnapshot]:
    snaps = load_snapshots(path)
    snaps.sort(key=lambda s: float(s.score), reverse=True)
    return snaps[: int(limit)]
