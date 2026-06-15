#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-./runs/v13_categorized_signal_bus_builder_15ep}"
NAME="${2:-$(basename "$RUN_DIR")}" 
DEST="reports/$NAME"

mkdir -p "$DEST"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  fi
}

copy_if_exists "$RUN_DIR/metrics.csv" "$DEST/metrics.csv"
copy_if_exists "$RUN_DIR/final_report.json" "$DEST/final_report.json"
copy_if_exists "$RUN_DIR/architecture_memory.jsonl" "$DEST/architecture_memory.jsonl"

# Copy best/latest seq analysis only, not full heavy folder.
latest_analysis=""
if compgen -G "$RUN_DIR/seq_analysis_epoch_*.json" > /dev/null; then
  latest_analysis="$(ls "$RUN_DIR"/seq_analysis_epoch_*.json | sort | tail -n 1)"
  cp "$latest_analysis" "$DEST/$(basename "$latest_analysis")"
fi

cat > "$DEST/README.md" <<EOF
# $NAME

Copied from local run directory:

\`\`\`text
$RUN_DIR
\`\`\`

Included lightweight files only:

- metrics.csv
- final_report.json
- architecture_memory.jsonl if present
- latest seq_analysis_epoch_XXX.json if present

No checkpoints, weights, datasets, audio, or full run folder are committed.
EOF

git add "$DEST" docs/IDEAS.csv docs/VERSION_STATS.csv docs/PLAN.md README.md .gitignore commands || true

echo "Prepared report files in $DEST"
echo "Review with: git status"
echo "Commit with: git commit -m 'Add $NAME report' && git push"
