#!/usr/bin/env python3
"""Patch StepProgram v1 with per-address gradient analytics.

This does not add new architecture logic.  It only logs whether the current
program choices actually receive task-loss gradient.

Logged compressed scores use:

    grad_x_gate = mean(abs(d task_loss / d gate) * gate)

for:

    L.B.S.P primitive gates
    L.B.S.Pk->P{k+1} primitive-transition gates
    L.B.S.read step-read gates
    L.B.route layer/block route gates
    L.B.S.write_gate step write gates
    class.read(slot) output reads

With fp16 GradScaler, intermediate tensor grads are scaled, so the report divides
by the current scaler value.
"""
from pathlib import Path

path = Path("experiments/step_program/run_step_program_speechcommands_v1.py")
text = path.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Insert helper functions before Train / eval section.
# ---------------------------------------------------------------------------
marker = '''

# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
'''
helper = r'''

def _unravel_index_flat(idx: int, shape: Sequence[int]) -> Tuple[int, ...]:
    coords: List[int] = []
    for size in reversed(tuple(int(s) for s in shape)):
        coords.append(idx % max(1, size))
        idx //= max(1, size)
    return tuple(reversed(coords))


def _top_score_items(score: torch.Tensor, names_fn, topk: int) -> List[Dict]:
    score = torch.nan_to_num(score.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if score.numel() == 0:
        return []
    flat = score.flatten()
    k = min(int(topk), int(flat.numel()))
    vals, idxs = torch.topk(flat, k=k)
    out: List[Dict] = []
    shape = tuple(score.shape)
    for v, idx in zip(vals.tolist(), idxs.tolist()):
        coords = _unravel_index_flat(int(idx), shape)
        item = names_fn(coords)
        item["grad_x_gate"] = float(v)
        out.append(item)
    return out


def make_grad_report(model: StepProgramNet, aux: StepProgramAux, classes: Sequence[str], grad_scale: float, topk: int = 30) -> Dict:
    """Return compressed per-address gradient utility report.

    This is intentionally top-k/summary only.  Dumping every scalar gradient would
    be too large and slow.  The report answers: which exact addresses are being
    pulled by task loss right now?
    """
    scale = max(float(grad_scale), 1.0)

    def gxg(t: torch.Tensor) -> Optional[torch.Tensor]:
        if t is None or t.grad is None:
            return None
        g = torch.nan_to_num((t.grad.detach().float() / scale).abs(), nan=0.0, posinf=0.0, neginf=0.0)
        v = torch.nan_to_num(t.detach().float().abs(), nan=0.0, posinf=0.0, neginf=0.0)
        return (g * v).mean(dim=0)

    pg = gxg(aux.primitive_gates)                 # [L,N,S,K,O]
    tg = gxg(aux.primitive_transition_gates)      # [L,N,S,K-1,T]
    rg = gxg(aux.step_read_gates)                 # [L,N,S,R]
    lr = gxg(aux.layer_route_gates)               # [L,N,LR]
    wg = gxg(aux.step_write_gates)                # [L,N,S]
    cr = gxg(aux.class_read)                      # [C,T]

    report: Dict[str, object] = {
        "definition": "grad_x_gate = mean(abs(d task_loss / d gate) * gate), divided by GradScaler scale when fp16 is active",
        "topk": int(topk),
        "summary": {},
        "top": {},
    }

    if pg is not None:
        report["summary"]["primitive_grad_x_gate_mean"] = float(pg.mean())
        report["summary"]["primitive_grad_x_gate_max"] = float(pg.max())
        report["top"]["primitive_gates"] = _top_score_items(
            pg,
            lambda c: {
                "address": f"L{c[0]}.B{c[1]}.S{c[2]}.P{c[3]}.{model.PRIMITIVE_NAMES[c[4]]}",
                "layer": int(c[0]), "block": int(c[1]), "step": int(c[2]), "primitive_slot": int(c[3]),
                "primitive": model.PRIMITIVE_NAMES[c[4]],
            },
            topk,
        )
    if tg is not None:
        report["summary"]["primitive_transition_grad_x_gate_mean"] = float(tg.mean())
        report["summary"]["primitive_transition_grad_x_gate_max"] = float(tg.max())
        report["top"]["primitive_transitions"] = _top_score_items(
            tg,
            lambda c: {
                "address": f"L{c[0]}.B{c[1]}.S{c[2]}.P{c[3]}->P{c[3]+1}.{model.PRIM_TRANSITION_NAMES[c[4]]}",
                "layer": int(c[0]), "block": int(c[1]), "step": int(c[2]), "primitive_slot": int(c[3]),
                "transition": model.PRIM_TRANSITION_NAMES[c[4]],
            },
            topk,
        )
    if rg is not None:
        report["summary"]["step_read_grad_x_gate_mean"] = float(rg.mean())
        report["summary"]["step_read_grad_x_gate_max"] = float(rg.max())
        report["top"]["step_reads"] = _top_score_items(
            rg,
            lambda c: {
                "address": f"L{c[0]}.B{c[1]}.S{c[2]}.read.{model.STEP_READ_NAMES[c[3]]}",
                "layer": int(c[0]), "block": int(c[1]), "step": int(c[2]),
                "read": model.STEP_READ_NAMES[c[3]],
            },
            topk,
        )
    if lr is not None:
        report["summary"]["layer_route_grad_x_gate_mean"] = float(lr.mean())
        report["summary"]["layer_route_grad_x_gate_max"] = float(lr.max())
        report["top"]["layer_routes"] = _top_score_items(
            lr,
            lambda c: {
                "address": f"L{c[0]}.B{c[1]}.route.{model.LAYER_ROUTE_NAMES[c[2]]}",
                "layer": int(c[0]), "block": int(c[1]), "route": model.LAYER_ROUTE_NAMES[c[2]],
            },
            topk,
        )
    if wg is not None:
        report["summary"]["write_gate_grad_x_gate_mean"] = float(wg.mean())
        report["summary"]["write_gate_grad_x_gate_max"] = float(wg.max())
        report["top"]["step_write_gates"] = _top_score_items(
            wg,
            lambda c: {
                "address": f"L{c[0]}.B{c[1]}.S{c[2]}.write_gate",
                "layer": int(c[0]), "block": int(c[1]), "step": int(c[2]),
            },
            topk,
        )
    if cr is not None:
        report["summary"]["class_read_grad_x_gate_mean"] = float(cr.mean())
        report["summary"]["class_read_grad_x_gate_max"] = float(cr.max())
        def class_name(i: int) -> str:
            return classes[i] if i < len(classes) else f"class_{i}"
        report["top"]["class_reads"] = _top_score_items(
            cr,
            lambda c: {
                "address": f"class.{class_name(c[0])}.read.{aux.slot_names[c[1]] if c[1] < len(aux.slot_names) else 'slot_'+str(c[1])}",
                "class": class_name(c[0]),
                "slot": aux.slot_names[c[1]] if c[1] < len(aux.slot_names) else f"slot_{c[1]}",
            },
            topk,
        )
    return report
'''
if helper not in text:
    if marker not in text:
        raise SystemExit("Train/eval marker not found")
    text = text.replace(marker, helper + marker)

# ---------------------------------------------------------------------------
# Add local variable last_grad_report.
# ---------------------------------------------------------------------------
old = '''    last_aux: Optional[StepProgramAux] = None
    t0 = time.time()
'''
new = '''    last_aux: Optional[StepProgramAux] = None
    last_grad_report: Optional[Dict] = None
    t0 = time.time()
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# Retain grads on selection tensors only on requested batches.
# ---------------------------------------------------------------------------
old = '''            loss = loss + losses["lambda_write_l1_eff"] * losses["write_l1"]
            loss = loss + losses["lambda_slot_collapse_eff"] * losses["slot_collapse"]
        if not torch.isfinite(loss):
'''
new = '''            loss = loss + losses["lambda_write_l1_eff"] * losses["write_l1"]
            loss = loss + losses["lambda_slot_collapse_eff"] * losses["slot_collapse"]
        do_grad_report = bool(args.grad_analytics_every and args.grad_analytics_every > 0 and step % args.grad_analytics_every == 0)
        if do_grad_report:
            for t in (aux.primitive_gates, aux.primitive_transition_gates, aux.step_read_gates, aux.layer_route_gates, aux.step_write_gates, aux.class_read):
                if t is not None and t.requires_grad:
                    t.retain_grad()
        if not torch.isfinite(loss):
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# Capture grad analytics after backward and before optimizer step.
# ---------------------------------------------------------------------------
old = '''        scaler.scale(loss).backward()
        if args.grad_clip > 0:
'''
new = '''        grad_scale_value = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else 1.0
        scaler.scale(loss).backward()
        if do_grad_report:
            last_grad_report = make_grad_report(model, aux, classes, grad_scale=grad_scale_value, topk=args.grad_analytics_topk)
        if args.grad_clip > 0:
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# Return last_grad_report.
# ---------------------------------------------------------------------------
old = '''        "read_temp": read_temp,
        "last_aux": last_aux,
    }
'''
new = '''        "read_temp": read_temp,
        "last_aux": last_aux,
        "last_grad_report": last_grad_report,
    }
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# Include grad report in analysis explicitly.
# ---------------------------------------------------------------------------
old = '''            "program_report": report,
            "notes": {
'''
new = '''            "program_report": report,
            "gradient_report": train.get("last_grad_report"),
            "notes": {
'''
if old in text and new not in text:
    text = text.replace(old, new)

# ---------------------------------------------------------------------------
# Add CLI arguments.
# ---------------------------------------------------------------------------
old = '''    p.add_argument("--lambda-slot-collapse", type=float, default=0.004)
    p.add_argument("--out-dir", type=str, default="./runs/step_program_v1")
'''
new = '''    p.add_argument("--lambda-slot-collapse", type=float, default=0.004)
    p.add_argument("--grad-analytics-every", type=int, default=0, help="log compressed per-address grad_x_gate report every N train batches; 0 disables")
    p.add_argument("--grad-analytics-topk", type=int, default=30, help="number of top addresses to keep per grad analytics group")
    p.add_argument("--out-dir", type=str, default="./runs/step_program_v1")
'''
if old in text and new not in text:
    text = text.replace(old, new)

path.write_text(text, encoding="utf-8")
print(f"patched {path}")
