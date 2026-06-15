#!/usr/bin/env python3
"""Hotfix for v13 summary crash.

Fixes:
    UnboundLocalError: category_gates_by_stage is not associated with a value

Reason:
    summary() used category_gates_by_stage in operator_program_by_block before
    the list was constructed.

Usage:
    python tools/fix_v13_category_gates_order.py sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ANCHOR = '''        operator_program_by_block = []\n        for i in range(op_gates.shape[0]):\n'''

INSERT = '''        category_names = ["extract", "compare", "memory", "repair", "aggregate", "suppress", "transform"]\n        cat_gates = self.category_gates / c if self.category_gates is not None else torch.zeros(self.chain_depth, len(category_names))\n        category_gates_by_stage = []\n        for i in range(cat_gates.shape[0]):\n            vals, idxs = torch.topk(cat_gates[i], k=min(4, cat_gates.shape[1]))\n            item = self.block_label(i)\n            item["categories"] = [{"category": category_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]\n            category_gates_by_stage.append(item)\n\n        operator_program_by_block = []\n        for i in range(op_gates.shape[0]):\n'''


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: fix_v13_category_gates_order.py <v13_file.py>")
        return 2
    path = Path(sys.argv[1])
    text = path.read_text(encoding="utf-8")
    if ANCHOR not in text:
        raise SystemExit("anchor not found; file may already be changed")
    pre = text[: text.find(ANCHOR)]
    if "category_gates_by_stage = []" in pre[-2500:]:
        print("already patched")
    else:
        text = text.replace(ANCHOR, INSERT, 1)
        path.write_text(text, encoding="utf-8")
        print(f"patched: {path}")
    py_compile.compile(str(path), doraise=True)
    print("compile OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
