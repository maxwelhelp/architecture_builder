from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class OperationSpec:
    """Lightweight metadata for planner/search decisions.

    The executable implementation can live elsewhere. This registry is for cheap
    branch scoring, cost penalties, reports, and archive matching.
    """

    name: str
    family: str
    cost: float
    role_hint: str
    is_heavy: bool = False
    deployable: bool = True


def default_operation_specs() -> List[OperationSpec]:
    return [
        OperationSpec("noop", "safe", 0.01, "keep", False, True),
        OperationSpec("identity", "safe", 0.03, "keep", False, True),
        OperationSpec("linear", "channel", 0.16, "transform", False, True),
        OperationSpec("gated_linear", "channel", 0.20, "transform", False, True),
        OperationSpec("local_mix", "sequence", 0.22, "extract", False, True),
        OperationSpec("global_mix", "sequence", 0.42, "aggregate", True, True),
        OperationSpec("memory_read", "memory", 0.44, "memory", True, True),
        OperationSpec("memory_write", "memory", 0.38, "memory", True, True),
        OperationSpec("matrix_mlp_local", "matrix_mlp", 0.45, "transform", True, True),
        OperationSpec("matrix_mlp_global", "matrix_mlp", 0.60, "aggregate", True, True),
        OperationSpec("matrix_mlp_lowrank", "matrix_mlp", 0.30, "repair", False, True),
        OperationSpec("residual_repair", "repair", 0.22, "repair", False, True),
        OperationSpec("normalize", "safe", 0.05, "repair", False, True),
        OperationSpec("suppress", "safe", 0.08, "repair", False, True),
        OperationSpec("shadow_probe", "training_only", 0.05, "search", False, False),
    ]


def names(specs: Iterable[OperationSpec]) -> List[str]:
    return [s.name for s in specs]


def costs(specs: Iterable[OperationSpec]) -> List[float]:
    return [float(s.cost) for s in specs]
