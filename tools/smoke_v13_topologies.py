#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder import (
    SeqAccumulator,
    SequentialMatrixCellsCore,
)


def main() -> None:
    classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    for top in ["grid", "grid_3x3x3x3", "two_squares_merge", "two_by_two_by_two_merge"]:
        print("TEST", top, flush=True)
        m = SequentialMatrixCellsCore(
            dim=32,
            num_classes=len(classes),
            evidence_cells=48,
            topology=top,
            block_cells=2,
            global_cells=2,
            memory_cells=3,
            category_temp=1.2,
            primitive_temp=1.2,
            operator_temp=1.2,
        )
        x = torch.randn(2, 48, 32)
        logits, aux = m(x)
        acc = SeqAccumulator(
            num_classes=len(classes),
            evidence_cells=48,
            chain_depth=m.chain_depth,
            num_layers=m.num_layers,
            blocks_per_layer=m.blocks_per_layer,
            decomp_every=1,
            topology_plan=m.topology_plan,
        )
        acc.add(aux)
        s = acc.summary(classes)
        assert logits.shape == (2, len(classes)), logits.shape
        assert len(s["operator_program_by_block"]) == m.chain_depth, (
            top,
            len(s["operator_program_by_block"]),
            m.chain_depth,
        )
        assert "attention_channels_by_stage" in s, "missing attention_channels_by_stage"
        assert len(s["gate_write_by_block"]) == m.chain_depth, (
            top,
            len(s["gate_write_by_block"]),
            m.chain_depth,
        )
        print(
            top,
            "OK",
            "logits",
            tuple(logits.shape),
            "chain_depth",
            m.chain_depth,
            "program_blocks",
            len(s["operator_program_by_block"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
