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

copy_latest_pattern() {
  local pattern="$1"
  if compgen -G "$pattern" > /dev/null; then
    local latest
    latest="$(ls $pattern | sort | tail -n 1)"
    cp "$latest" "$DEST/$(basename "$latest")"
  fi
}

copy_if_exists "$RUN_DIR/metrics.csv" "$DEST/metrics.csv"
copy_if_exists "$RUN_DIR/final_report.json" "$DEST/final_report.json"
copy_if_exists "$RUN_DIR/architecture_memory.jsonl" "$DEST/architecture_memory.jsonl"
copy_if_exists "$RUN_DIR/controller_state.jsonl" "$DEST/controller_state.jsonl"

# Copy only the latest lightweight analysis file, not full heavy run folders.
copy_latest_pattern "$RUN_DIR/seq_analysis_epoch_*.json"
copy_latest_pattern "$RUN_DIR/analysis_epoch_*.json"

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
- controller_state.jsonl if present
- latest seq_analysis_epoch_XXX.json if present
- latest analysis_epoch_XXX.json if present

No checkpoints, weights, datasets, audio, or full run folder are committed.
EOF

git add "$DEST" docs/IDEAS.csv docs/VERSION_STATS.csv docs/PLAN.md README.md .gitignore commands || true

echo "Prepared report files in $DEST"
echo "Review with: git status"
echo "Commit with: git commit -m 'Add $NAME report' && git push"
