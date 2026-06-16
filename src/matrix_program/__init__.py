"""Modular matrix-program components.

This package is the new home for reusable architecture-builder pieces.
Keep files small and composable so future experiments do not require patching one huge script.
"""

from .operators import OperationSpec, default_operation_specs
from .shadow_topk import ShadowTopKConfig
from .archive import ArchitectureSnapshot, append_snapshot

__all__ = [
    "OperationSpec",
    "default_operation_specs",
    "ShadowTopKConfig",
    "ArchitectureSnapshot",
    "append_snapshot",
]
