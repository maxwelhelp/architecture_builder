#!/usr/bin/env python3
from pathlib import Path

p = Path('sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py')
text = p.read_text(encoding='utf-8')
orig = text
changes = []

def rep(old: str, new: str, name: str):
    global text
    if old in text:
        text = text.replace(old, new, 1)
        changes.append(name)

# 1) Close/export torch profiler in train_one_epoch before return.
old = '''    return {
        "loss": total_loss / max(1, total),'''
new = '''    if prof is not None:
        prof.__exit__(None, None, None)
        try:
            profile_dir = ensure_dir(Path(args.out_dir) / "profiles")
            sort_key = "cuda_time_total" if device.startswith("cuda") else "cpu_time_total"
            table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
            profile_path = profile_dir / f"profile_epoch_{int(epoch):03d}.txt"
            profile_path.write_text(table, encoding="utf-8")
            print(f"torch_profiler: wrote {profile_path}", flush=True)
        except Exception as e:
            print(f"torch_profiler: failed to export profile table: {e}", flush=True)

    return {
        "loss": total_loss / max(1, total),'''
if 'torch_profiler: wrote {profile_path}' not in text:
    rep(old, new, 'train_profiler_close_export')

# 2) Return effective lambdas from train_one_epoch so seq_analysis records the real schedule.
old = '''        "n": total,
        "seq": acc.summary(classes),
        "speed": {'''
new = '''        "n": total,
        "seq": acc.summary(classes),
        "lambda_operator_balance_eff": float(lambda_operator_balance_eff) if 'lambda_operator_balance_eff' in locals() else 0.0,
        "lambda_operator_entropy_per_block_eff": float(lambda_operator_entropy_per_block_eff) if 'lambda_operator_entropy_per_block_eff' in locals() else 0.0,
        "lambda_category_entropy_eff": float(lambda_category_entropy_eff) if 'lambda_category_entropy_eff' in locals() else 0.0,
        "explore_w": float(explore_w) if 'explore_w' in locals() else 0.0,
        "sharpen_w": float(sharpen_w) if 'sharpen_w' in locals() else 0.0,
        "speed": {'''
if 'lambda_operator_balance_eff' not in text[text.find('def train_one_epoch'):text.find('def save_checkpoint')]:
    rep(old, new, 'train_effective_lambda_return')

if text == orig:
    print('No changes; profiler/lambda logs already patched or patterns changed.')
else:
    bak = p.with_suffix(p.suffix + '.bak_before_profiler_lambda_logs')
    if not bak.exists():
        bak.write_text(orig, encoding='utf-8')
    p.write_text(text, encoding='utf-8')
    print('Applied:', ', '.join(changes))
    print('Backup:', bak)
