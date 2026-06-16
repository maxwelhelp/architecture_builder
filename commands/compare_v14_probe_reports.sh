#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-./runs/v14_parallel_slots_probe_compare.md}"
shift || true

if [[ "$#" -eq 0 ]]; then
  set -- \
    runs/v14_parallel_slots_probe_parallel_refine_topk3_refine2 \
    runs/v14_parallel_slots_probe_sequential_exact_topk3_refine2 \
    runs/v14_parallel_slots_probe_hybrid_groups_topk3_refine2
fi

python tools/compare_v14_probe_reports.py "$@" --out "$OUT"
cat "$OUT"
