from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class ShadowTopKConfig:
    """Phased cheap-search config.

    warmup: full soft execution for gradient to all branches.
    topk: full compute only k selected branches after warmup.
    shadow: occasionally evaluate one unselected branch for router learning only.
    """

    warmup_epochs: int = 5
    topk: int = 3
    sketch_dim: int = 32
    shadow_prob: float = 0.20
    utility_weight: float = 0.002
    selected_cost_weight: float = 0.0005
    temperature: float = 1.0


def phase_for_epoch(epoch: int, cfg: ShadowTopKConfig) -> str:
    return "warmup_all" if int(epoch) <= int(cfg.warmup_epochs) else "topk"


def selected_indices(scores: Sequence[float], cfg: ShadowTopKConfig, epoch: int) -> List[int]:
    """Small pure-Python helper for reports/tests.

    The torch implementation should mirror this but stay vectorized.
    """

    if phase_for_epoch(epoch, cfg) == "warmup_all":
        return list(range(len(scores)))
    k = max(1, min(int(cfg.topk), len(scores)))
    return sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]


def estimate_speedup(op_names: Sequence[str], selected: Iterable[int], heavy_ops: Iterable[str]) -> float:
    heavy = set(heavy_ops)

    def cost(name: str) -> float:
        return 3.0 if name in heavy else 1.0

    all_cost = sum(cost(n) for n in op_names)
    sel_cost = sum(cost(op_names[i]) for i in selected)
    return float(all_cost / max(1e-6, sel_cost))


def branch_report(op_names: Sequence[str], scores: Sequence[float], selected: Sequence[int], heavy_ops: Iterable[str]) -> Dict[str, object]:
    return {
        "selected": [op_names[i] for i in selected],
        "selected_idx": list(map(int, selected)),
        "scores": {op_names[i]: float(scores[i]) for i in range(len(op_names))},
        "estimated_speedup_vs_all": estimate_speedup(op_names, selected, heavy_ops),
    }
