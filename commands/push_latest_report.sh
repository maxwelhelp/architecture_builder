#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-./runs/v13_categorized_signal_bus_builder_15ep}"
NAME="${2:-$(basename "$RUN_DIR")}" 
MSG="${3:-Add $NAME report}"

bash commands/publish_latest_report.sh "$RUN_DIR" "$NAME"

# Safety: never commit heavy artifacts even if somebody force-added them.
heavy_staged="$(git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|npy|npz|zip|wav|flac|log)$' || true)"
if [[ -n "$heavy_staged" ]]; then
  echo "ERROR: heavy files are staged. Unstage them first:" >&2
  echo "$heavy_staged" >&2
  echo "Run for each file: git reset HEAD -- <file>" >&2
  exit 1
fi

# Stage only safe project/report files.
git add reports docs README.md .gitignore commands tools old src || true

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

git commit -m "$MSG"
git push

echo "Pushed report: $NAME"
