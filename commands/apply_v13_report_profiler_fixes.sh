#!/usr/bin/env bash
set -euo pipefail

python tools/fix_v13_report_profiler_v1.py
python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py
PYTHONPATH="$PWD" python tools/smoke_v13_topologies.py

echo "v13 report/profiler fixes OK"
