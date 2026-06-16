#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-./reports/v13_vnext_matrixmlp_45ep}"
python tools/analyze_v13_health_report.py "$DIR"
