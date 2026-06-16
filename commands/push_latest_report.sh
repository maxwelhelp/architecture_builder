#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-./runs/v13_categorized_signal_bus_builder_15ep}"
NAME="${2:-$(basename "$RUN_DIR")}" 
MSG="${3:-Add $NAME report}"
BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "ERROR: detached HEAD. Resolve rebase/checkout branch before publishing report." >&2
  exit 1
fi

bash commands/publish_latest_report.sh "$RUN_DIR" "$NAME"

# Safety: never commit heavy artifacts even if somebody force-added them.
heavy_staged="$(git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|npy|npz|zip|wav|flac|log)$' || true)"
if [[ -n "$heavy_staged" ]]; then
  echo "ERROR: heavy files are staged. Unstage them first:" >&2
  echo "$heavy_staged" >&2
  echo "Run for each file: git reset HEAD -- <file>" >&2
  exit 1
fi

# Stage only safe project/report files. Some directories may not exist yet.
git add reports docs README.md .gitignore commands tools old src experiments || true
if [[ -d skills ]]; then
  git add skills || true
fi

if git diff --cached --quiet; then
  echo "Nothing new to commit. Will still sync/push local commits if branch is ahead."
else
  git commit -m "$MSG"
fi

# Avoid the common 'fetch first' push rejection by rebasing after local commit.
# If there is a real conflict, Git will stop and tell the user which file to resolve.
git pull --rebase origin "$BRANCH"
git push origin "$BRANCH"

echo "Pushed report: $NAME"
