#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.matrix_program.parallel_core import ParallelSlotConfig, ParallelSlotCore  # noqa: E402


def run(args):
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    cfg = ParallelSlotConfig(
        dim=args.dim,
        layers=args.layers,
        blocks=args.blocks,
        memory_cells=args.memory_cells,
        refine_iters=args.refine_iters,
        topk=args.topk,
        mode=args.mode,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    core = ParallelSlotCore(cfg).to(device)
    head = nn.Linear(args.dim, args.classes).to(device)
    opt = torch.optim.AdamW(list(core.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=0.01)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(1, args.steps + 1):
        x = torch.randn(args.batch, args.tokens, args.dim, device=device)
        # Synthetic target: depends on global mean + local first/last contrast.
        signal = x.mean(dim=(1, 2)) + 0.3 * (x[:, 0].mean(dim=1) - x[:, -1].mean(dim=1))
        y = ((signal * 7).long().abs() % args.classes).to(device)
        h, aux = core(x, warmup_all=step <= args.warmup_steps)
        logits = head(core.readout(h))
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(core.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        wg = aux.get("write_gate")
        ow = aux.get("op_weights")
        row = {
            "step": step,
            "loss": float(loss.item()),
            "acc": float(acc),
            "write_gate_mean": float(wg.mean().item()) if wg is not None else None,
            "op_entropy": float((-(ow.clamp_min(1e-8) * ow.clamp_min(1e-8).log()).sum(dim=-1).mean()).item()) if ow is not None else None,
        }
        rows.append(row)
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / "probe_report.json").write_text(json.dumps({"args": vars(args), "rows": rows[-20:]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", out / "probe_report.json")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sequential_exact", "parallel_refine", "hybrid_groups"], default="parallel_refine")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--memory-cells", type=int, default=6)
    p.add_argument("--refine-iters", type=int, default=2)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out-dir", default="./runs/v14_parallel_slots_probe")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
