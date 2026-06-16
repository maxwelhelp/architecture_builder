#!/usr/bin/env bash
set -euo pipefail

python tools/patch_v13_review_fixes_min.py
python tools/patch_v13_eval_prof_bug.py
python tools/patch_v13_report_safe_logic.py
python tools/patch_v13_saturation_losses.py
python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py
PYTHONPATH="$PWD" python tools/smoke_v13_topologies.py

echo "OK: all v13 fixes applied and smoke-tested."
