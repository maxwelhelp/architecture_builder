#!/usr/bin/env bash
set -euo pipefail

BRANCH="$(git branch --show-current)"
if [[ -z "$BRANCH" ]]; then
  echo "ERROR: detached HEAD. Run git status and finish/abort rebase first." >&2
  exit 1
fi

git status -sb
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has uncommitted changes. Commit/stash them first." >&2
  exit 1
fi

git pull --rebase origin "$BRANCH"
git push origin "$BRANCH"
git status -sb
