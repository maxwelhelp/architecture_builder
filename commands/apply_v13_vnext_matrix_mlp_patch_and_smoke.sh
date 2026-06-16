#!/usr/bin/env bash
set -euo pipefail

# Correctness/stability patch first.
python tools/patch_v13_stability_v2.py

# vNext architecture patch: matrix_mlp primitive + fp32 softmax + entropy schedule.
python tools/patch_v13_vnext_matrix_mlp.py

python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py
python -m py_compile tools/patch_v13_stability_v2.py
python -m py_compile tools/patch_v13_vnext_matrix_mlp.py

COUNT=$(grep -c -- "--no-export-architecture-memory" sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py || true)
if [[ "$COUNT" != "1" ]]; then
  echo "ERROR: expected exactly one --no-export-architecture-memory flag, got $COUNT" >&2
  exit 1
fi

python tools/smoke_v13_topologies.py

echo "v13 vNext MatrixMLP patch + smoke OK"
