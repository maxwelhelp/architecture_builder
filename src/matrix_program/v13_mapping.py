from __future__ import annotations

from typing import Dict, List

# Adapter between the current v13 monolithic executor names and the modular
# vNext registry names. This prevents archive/ShadowTopK reports from pretending
# that both systems already use identical primitive names.

V13_TO_MODULAR: Dict[str, List[str]] = {
    "mlp": ["linear", "gated_linear"],
    "matrix_mlp": ["matrix_mlp_local", "matrix_mlp_global", "matrix_mlp_lowrank"],
    "bilinear": ["gated_linear"],
    "time": ["local_mix"],
    "freq": ["local_mix"],
    "delta": ["residual_repair"],
    "short_onset": ["local_mix"],
    "offset": ["local_mix"],
    "global_summary": ["global_mix"],
    "memory_read": ["memory_read"],
    "memory_write": ["memory_write"],
    "ema_memory": ["memory_write"],
    "class_pair_memory": ["memory_read", "memory_write"],
    "residual_refdelta": ["residual_repair"],
    "block_compare": ["matrix_mlp_lowrank"],
    "class_pair_contrast": ["matrix_mlp_lowrank"],
    "late_read_repair": ["residual_repair"],
    "suppress": ["suppress"],
    "normalize": ["normalize"],
    "energy_count": ["global_mix"],
}

MODULAR_TO_V13_HINT: Dict[str, List[str]] = {}
for old, news in V13_TO_MODULAR.items():
    for new in news:
        MODULAR_TO_V13_HINT.setdefault(new, []).append(old)


def map_v13_op(name: str) -> List[str]:
    return V13_TO_MODULAR.get(str(name), [str(name)])
