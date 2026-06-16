#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: bash commands/push_run_report.sh <run_dir> <report_name> [commit_message]}"
REPORT_NAME="${2:?usage: bash commands/push_run_report.sh <run_dir> <report_name> [commit_message]}"
MSG="${3:-Add ${REPORT_NAME} report}"
REPORT_DIR="reports/${REPORT_NAME}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "ERROR: run dir not found: $RUN_DIR" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

# Always copy compact files. Large raw logs are useful but optional.
for f in \
  probe_report.json \
  health_report.json \
  health_report.md \
  health_analyzer_stdout.log \
  console.log \
  profiler_table.txt \
  README.md; do
  if [[ -f "$RUN_DIR/$f" ]]; then
    cp "$RUN_DIR/$f" "$REPORT_DIR/$f"
  fi
  if [[ -f "$RUN_DIR/profiles/$f" ]]; then
    mkdir -p "$REPORT_DIR/profiles"
    cp "$RUN_DIR/profiles/$f" "$REPORT_DIR/profiles/$f"
  fi
done

# Copy tensorboard profiler traces only if explicitly requested; they can be huge.
if [[ "${INCLUDE_PROFILE_TRACES:-0}" == "1" && -d "$RUN_DIR/profiles" ]]; then
  mkdir -p "$REPORT_DIR/profiles"
  cp -a "$RUN_DIR/profiles/." "$REPORT_DIR/profiles/"
fi

cat > "$REPORT_DIR/README.md" <<EOF
# ${REPORT_NAME}

Source run: \`${RUN_DIR}\`

Included:
- \`probe_report.json\`
- \`health_report.md/json\` if available
- \`console.log\` if captured
- profiler table if available

Quick review:

\`\`\`bash
cat ${REPORT_DIR}/health_report.md
python tools/analyze_v14_probe_health.py ${REPORT_DIR}
\`\`\`
EOF

git add "$REPORT_DIR"
git commit -m "$MSG" || true
git pull --rebase origin main
git push origin main

echo "Pushed report: $REPORT_DIR"
