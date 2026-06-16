#!/usr/bin/env python3
from __future__ import annotations

import re
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


def ensure_matrix_mlp_in_local_op_names(text: str) -> str:
    # Patch local summary/report lists like: op_names = ["mlp", "bilinear", ...]
    # The main class may not expose module-level OP_NAMES, so summary must use local op_names.
    pattern = re.compile(r'(op_names\s*=\s*\[[^\]]*?"mlp"\s*,\s*)"bilinear"', re.S)
    n = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal n
        block_start = m.group(0)
        # Avoid double insertion if this specific list already has matrix_mlp nearby.
        list_prefix = m.group(1)
        if '"matrix_mlp"' in list_prefix:
            return block_start
        n += 1
        return m.group(1) + '"matrix_mlp", "bilinear"'

    text2 = pattern.sub(repl, text)
    print(f"PATCH local op_names matrix_mlp: {n} list(s)")
    return text2


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # Use local summary op_names, not OP_NAMES/self.OP_NAMES. SeqAccumulator does not own self.OP_NAMES,
    # and this file does not reliably expose global OP_NAMES.
    text = replace_all(
        text,
        'self.OP_NAMES[int(j)] if int(j) < len(self.OP_NAMES) else f"op_{int(j)}"',
        'op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}"',
        "repair accidental self.OP_NAMES guard",
    )
    text = replace_all(
        text,
        'OP_NAMES[int(j)] if int(j) < len(OP_NAMES) else f"op_{int(j)}"',
        'op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}"',
        "repair accidental global OP_NAMES guard",
    )
    text = replace_all(
        text,
        'op_names[int(j)]',
        'op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}"',
        "summary op_names bounds guard",
    )
    # Undo double-guard if a previous repair was run more than once.
    text = text.replace(
        'op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}" if int(j) < len(op_names) else f"op_{int(j)}"',
        'op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}"',
    )
    text = ensure_matrix_mlp_in_local_op_names(text)

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
