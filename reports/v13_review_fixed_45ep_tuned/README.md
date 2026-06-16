# v13_review_fixed_45ep_tuned

Source run: `./runs/v13_review_fixed_45ep_tuned`

Included when available:
- `metrics.csv`, `console.log`, `final_report.json`
- `health_diagnosis.md/json` or `health_report.md/json`
- `program_pseudocode.md`, `program_pseudocode_v2.md`
- last seq/analysis JSON snapshots
- profiler table if available

Quick v13 review:

```bash
cat reports/v13_review_fixed_45ep_tuned/health_diagnosis.md
cat reports/v13_review_fixed_45ep_tuned/program_pseudocode_v2.md
```

Quick v14 review:

```bash
cat reports/v13_review_fixed_45ep_tuned/health_report.md
python tools/analyze_v14_probe_health.py reports/v13_review_fixed_45ep_tuned
```
