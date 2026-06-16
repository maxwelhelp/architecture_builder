#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

TARGET = Path("sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")


def replace_all(text: str, old: str, new: str, name: str) -> str:
    n = text.count(old)
    if n == 0:
        print(f"SKIP {name}: not found or already patched")
        return text
    print(f"PATCH {name}: {n} occurrence(s)")
    return text.replace(old, new)


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        print(f"SKIP {name}: not found or already patched")
        return text
    print(f"PATCH {name}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # After adding matrix_mlp, summary must use the module-level OP_NAMES,
    # because SeqAccumulator does not own self.OP_NAMES. Guard out-of-range
    # indices so reports never crash even if a future candidate is added.
    text = replace_all(
        text,
        'op_names[int(j)]',
        'OP_NAMES[int(j)] if int(j) < len(OP_NAMES) else f"op_{int(j)}"',
        "summary op_names guard",
    )
    text = replace_all(
        text,
        'self.OP_NAMES[int(j)] if int(j) < len(self.OP_NAMES) else f"op_{int(j)}"',
        'OP_NAMES[int(j)] if int(j) < len(OP_NAMES) else f"op_{int(j)}"',
        "repair accidental self.OP_NAMES guard",
    )

    # Some patched files still miss topology_plan in train_one_epoch accumulator,
    # while evaluate already has it. Patch the train accumulator call as well.
    text = replace_once(
        text,
        'acc = SeqAccumulator(num_classes=len(classes), evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=args.decomp_every)',
        'acc = SeqAccumulator(num_classes=len(classes), evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=args.decomp_every, topology_plan=model.core.topology_plan)',
        "train accumulator topology_plan",
    )

    # Make old smoke-test import work even before pulling the fixed smoke script.
    smoke = Path("tools/smoke_v13_topologies.py")
    if smoke.exists():
        sm = smoke.read_text(encoding="utf-8")
        if "ROOT = Path(__file__).resolve().parents[1]" not in sm:
            sm = sm.replace(
                "import torch\n\nfrom sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder import (",
                "import sys\nfrom pathlib import Path\n\nimport torch\n\nROOT = Path(__file__).resolve().parents[1]\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\n\nfrom sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder import (",
            )
            smoke.write_text(sm, encoding="utf-8")
            print("PATCH smoke sys.path")

    if text == original:
        print("No main-file changes made.")
        return
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_matrix_mlp_summary_repair")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backup written: {backup}")
    TARGET.write_text(text, encoding="utf-8")
    print("Done. Run:")
    print("  python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")
    print("  PYTHONPATH=\"$PWD\" python tools/smoke_v13_topologies.py")


if __name__ == "__main__":
    main()
