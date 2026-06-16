#!/usr/bin/env python3
from pathlib import Path

p = Path('sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py')
text = p.read_text(encoding='utf-8')
orig = text

# Robustly remove any profiler-close/export block that accidentally lives inside evaluate().
# In evaluate(), prof and epoch are not defined, so even `if prof is not None:` crashes.
def_start = text.find('def evaluate(')
if def_start == -1:
    raise SystemExit('def evaluate not found')
next_def = text.find('\ndef ', def_start + 1)
if next_def == -1:
    next_def = len(text)
segment = text[def_start:next_def]

changed = False
while True:
    start = segment.find('    if prof is not None:')
    if start == -1:
        break
    # Prefer cutting until the evaluate return block. Fallback: cut until next top-level 4-space return.
    ret = segment.find('    return {', start)
    if ret == -1:
        raise SystemExit('found evaluate profiler block, but no evaluate return after it')
    segment = segment[:start] + segment[ret:]
    changed = True

if changed:
    text = text[:def_start] + segment + text[next_def:]

if text == orig:
    print('No evaluate profiler bug block found; already fixed.')
else:
    bak = p.with_suffix(p.suffix + '.bak_before_eval_prof_fix')
    if not bak.exists():
        bak.write_text(orig, encoding='utf-8')
    p.write_text(text, encoding='utf-8')
    print('Fixed evaluate profiler NameError block robustly.')
    print('Backup:', bak)
