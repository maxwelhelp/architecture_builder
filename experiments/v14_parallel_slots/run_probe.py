#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.matrix_program.gradient_signals import slot_gradient_health  # noqa: E402
from src.matrix_program.parallel_core import ParallelSlotConfig, ParallelSlotCore  # noqa: E402


def sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_profiler(args, out: Path):
    if not args.torch_profiler:
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile_dir = out / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=max(0, args.profile_wait), warmup=max(0, args.profile_warmup), active=max(1, args.profile_active), repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
    )


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
    prof = make_profiler(args, out)
    if prof is not None:
        prof.__enter__()
    try:
        for step in range(1, args.steps + 1):
            t0 = time.perf_counter()
            x = torch.randn(args.batch, args.tokens, args.dim, device=device)
            # Synthetic target: depends on global mean + local first/last contrast.
            signal = x.mean(dim=(1, 2)) + 0.3 * (x[:, 0].mean(dim=1) - x[:, -1].mean(dim=1))
            y = ((signal * 7).long().abs() % args.classes).to(device)
            sync_if_cuda(device); t_data = time.perf_counter()

            h, aux = core(x, warmup_all=step <= args.warmup_steps)
            if args.grad_health:
                h.retain_grad()
            logits = head(core.readout(h))
            loss = F.cross_entropy(logits, y)
            sync_if_cuda(device); t_fwd = time.perf_counter()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_health = slot_gradient_health(h.grad if args.grad_health else None) if args.grad_health else {}
            nn.utils.clip_grad_norm_(list(core.parameters()) + list(head.parameters()), 1.0)
            sync_if_cuda(device); t_bwd = time.perf_counter()

            opt.step()
            sync_if_cuda(device); t_opt = time.perf_counter()

            if prof is not None:
                prof.step()

            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            wg = aux.get("write_gate")
            ow = aux.get("op_weights")
            mem_alloc_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if str(device).startswith("cuda") and torch.cuda.is_available() else 0.0
            row = {
                "step": step,
                "loss": float(loss.item()),
                "acc": float(acc),
                "write_gate_mean": float(wg.mean().item()) if wg is not None else None,
                "op_entropy": float((-(ow.clamp_min(1e-8) * ow.clamp_min(1e-8).log()).sum(dim=-1).mean()).item()) if ow is not None else None,
                "data_ms": (t_data - t0) * 1000.0,
                "fwd_ms": (t_fwd - t_data) * 1000.0,
                "bwd_ms": (t_bwd - t_fwd) * 1000.0,
                "opt_ms": (t_opt - t_bwd) * 1000.0,
                "step_ms": (t_opt - t0) * 1000.0,
                "max_cuda_mem_mb": float(mem_alloc_mb),
                **grad_health,
            }
            rows.append(row)
            if step % args.log_every == 0 or step == 1:
                print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if prof is not None:
            prof.__exit__(None, None, None)
            sort_key = "cuda_time_total" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu_time_total"
            table = prof.key_averages().table(sort_by=sort_key, row_limit=args.profile_rows)
            profile_txt = out / "profiles" / "profiler_table.txt"
            profile_txt.write_text(table, encoding="utf-8")
            print("profiler_table", profile_txt, flush=True)
    tail = rows[max(0, len(rows) - args.speed_tail):]
    speed_summary = {
        "mean_step_ms": sum(r["step_ms"] for r in tail) / max(1, len(tail)),
        "mean_fwd_ms": sum(r["fwd_ms"] for r in tail) / max(1, len(tail)),
        "mean_bwd_ms": sum(r["bwd_ms"] for r in tail) / max(1, len(tail)),
        "mean_opt_ms": sum(r["opt_ms"] for r in tail) / max(1, len(tail)),
        "max_cuda_mem_mb": max([r["max_cuda_mem_mb"] for r in rows], default=0.0),
    }
    if args.grad_health:
        for k in ["grad_slot_norm_mean", "grad_slot_norm_max", "grad_layer_entropy", "grad_block_entropy", "grad_dead_slot_frac"]:
            speed_summary[f"mean_{k}"] = sum(float(r.get(k, 0.0)) for r in tail) / max(1, len(tail))
    summary = {"args": vars(args), "rows": rows[-20:], "speed_summary": speed_summary}
    (out / "probe_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    p.add_argument("--speed-tail", type=int, default=30)
    p.add_argument("--grad-health", action="store_true")
    p.add_argument("--torch-profiler", action="store_true")
    p.add_argument("--profile-wait", type=int, default=5)
    p.add_argument("--profile-warmup", type=int, default=5)
    p.add_argument("--profile-active", type=int, default=10)
    p.add_argument("--profile-rows", type=int, default=40)
    p.add_argument("--out-dir", default="./runs/v14_parallel_slots_probe")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
