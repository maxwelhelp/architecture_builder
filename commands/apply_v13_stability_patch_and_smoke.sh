#!/usr/bin/env bash
set -euo pipefail

python tools/patch_v13_stability_v2.py
python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py

COUNT=$(grep -c -- "--no-export-architecture-memory" sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py || true)
if [[ "$COUNT" != "1" ]]; then
  echo "ERROR: expected exactly one --no-export-architecture-memory flag, got $COUNT" >&2
  exit 1
fi

python tools/smoke_v13_topologies.py

echo "v13 stability patch + smoke OK"
