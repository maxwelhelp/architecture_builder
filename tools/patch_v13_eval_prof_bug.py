#!/usr/bin/env python3
from pathlib import Path

p = Path('sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py')
text = p.read_text(encoding='utf-8')
orig = text

# A profiler export block was accidentally left inside evaluate(), where `prof`
# and `epoch` are not defined. train_one_epoch() already owns profiler export.
start = text.find('    if prof is not None:\n        prof.__exit__(None, None, None)')
if start != -1:
    ret = text.find('    return {\n        "loss": total_loss / max(1, total),', start)
    if ret != -1:
        text = text[:start] + text[ret:]

if text == orig:
    print('No evaluate profiler bug block found; already fixed or pattern changed.')
else:
    bak = p.with_suffix(p.suffix + '.bak_before_eval_prof_fix')
    if not bak.exists():
        bak.write_text(orig, encoding='utf-8')
    p.write_text(text, encoding='utf-8')
    print('Fixed evaluate profiler NameError block.')
    print('Backup:', bak)
