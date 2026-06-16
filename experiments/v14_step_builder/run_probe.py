#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.matrix_program.grow_prune_controller import GrowPruneConfig, decide_step_actions  # noqa: E402
from src.matrix_program.skill_archive import archive_skill_candidates  # noqa: E402
from src.matrix_program.step_builder import StepBuilderConfig, StepBuilderCore  # noqa: E402
from src.matrix_program.step_utility import summarize_step_utility  # noqa: E402


def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def synthetic_batch(batch: int, tokens: int, dim: int, classes: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(batch, tokens, dim, device=device)
    # Harder than trivial mean: combines early/late contrast, local spike, and channel group signs.
    g0 = x[:, : tokens // 3, : dim // 4].mean(dim=(1, 2))
    g1 = x[:, tokens // 3: 2 * tokens // 3, dim // 4: dim // 2].mean(dim=(1, 2))
    g2 = x[:, -tokens // 4:, dim // 2: 3 * dim // 4].amax(dim=1).mean(dim=1)
    g3 = x[:, 1::2, 3 * dim // 4:].mean(dim=(1, 2)) - x[:, ::2, 3 * dim // 4:].mean(dim=(1, 2))
    score = 2.2 * g0 - 1.7 * g1 + 0.6 * g2 + 1.3 * g3
    y = ((score * 5.0).long().abs() % classes).to(device)
    return x, y


def budget_losses(logits: torch.Tensor, aux: Dict[str, torch.Tensor], *, gate_target: float, logit_target: float, lambda_gate: float, lambda_logit: float, lambda_op_entropy: float) -> tuple[torch.Tensor, Dict[str, float]]:
    loss = logits.new_tensor(0.0)
    logs: Dict[str, float] = {}
    wg = aux["write_gate"].float()
    ow = aux["op_weights"].float()
    op_ent = -(ow.clamp_min(1e-8) * ow.clamp_min(1e-8).log()).sum(dim=-1).mean()
    if lambda_gate > 0:
        l = (wg.mean() - gate_target).pow(2)
        loss = loss + float(lambda_gate) * l.to(logits.dtype)
        logs["loss_gate_budget"] = float(l.detach().item())
    if lambda_logit > 0:
        norm = logits.float().pow(2).mean(dim=-1).sqrt().mean()
        l = torch.relu(norm - logit_target).pow(2)
        loss = loss + float(lambda_logit) * l.to(logits.dtype)
        logs["loss_logit_budget"] = float(l.detach().item())
        logs["logit_norm"] = float(norm.detach().item())
    if lambda_op_entropy > 0:
        # Prevent too-early collapse, not force soup forever.
        l = torch.relu(0.35 - op_ent).pow(2)
        loss = loss + float(lambda_op_entropy) * l.to(logits.dtype)
        logs["loss_op_entropy_floor"] = float(l.detach().item())
    logs["op_entropy"] = float(op_ent.detach().item())
    logs["write_gate_mean"] = float(wg.mean().detach().item())
    logs["write_gate_max"] = float(wg.max().detach().item())
    return loss, logs


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = StepBuilderConfig(
        dim=args.dim,
        layers=args.layers,
        blocks=args.blocks,
        steps=args.steps_per_block,
        memory_cells=args.memory_cells,
        classes=args.classes,
        dropout=args.dropout,
        read_temp=args.read_temp,
        op_temp=args.op_temp,
        write_bias=args.write_bias,
        topk_ops=args.topk_ops,
        hard_topk_after=args.hard_topk_after,
    )
    model = StepBuilderCore(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rows: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    best_acc = 0.0
    best_step = 0
    for step in range(1, args.steps + 1):
        t0 = time.perf_counter()
        x, y = synthetic_batch(args.batch, args.tokens, args.dim, args.classes, device)
        sync(device); t_data = time.perf_counter()
        h, aux = model(x, epoch=step)
        logits = model.logits(h)
        ce = F.cross_entropy(logits, y)
        b_loss, b_logs = budget_losses(
            logits, aux,
            gate_target=args.gate_target,
            logit_target=args.logit_target,
            lambda_gate=args.lambda_gate_budget,
            lambda_logit=args.lambda_logit_budget,
            lambda_op_entropy=args.lambda_op_entropy_floor,
        )
        loss = ce + b_loss
        sync(device); t_fwd = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        sync(device); t_bwd = time.perf_counter()
        opt.step()
        sync(device); t_opt = time.perf_counter()

        acc = (logits.argmax(dim=-1) == y).float().mean().item()
        if acc > best_acc:
            best_acc, best_step = acc, step
        util = summarize_step_utility(aux)
        row = {
            "step": step,
            "loss": float(loss.detach().item()),
            "ce": float(ce.detach().item()),
            "budget_loss": float(b_loss.detach().item()),
            "acc": float(acc),
            "best_acc": float(best_acc),
            "best_step": int(best_step),
            "data_ms": (t_data - t0) * 1000,
            "fwd_ms": (t_fwd - t_data) * 1000,
            "bwd_ms": (t_bwd - t_fwd) * 1000,
            "opt_ms": (t_opt - t_bwd) * 1000,
            "step_ms": (t_opt - t0) * 1000,
            "max_cuda_mem_mb": float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if device.startswith("cuda") and torch.cuda.is_available() else 0.0,
            **b_logs,
            "utility_mean": util["utility_mean"],
            "dead_steps": util["dead_steps"],
            "overloaded_steps": util["overloaded_steps"],
            "saturated_steps": util["saturated_steps"],
            "skill_candidates": util["skill_candidates"],
        }
        rows.append(row)
        history.append({"epoch": step, "train_acc": acc, "val_acc": acc})
        if step % args.controller_every == 0:
            decision = decide_step_actions(history, util, GrowPruneConfig(max_actions=8))
            row["controller"] = decision
            if args.archive_skills:
                n = archive_skill_candidates(out / "step_skills.jsonl", util, source=f"step_{step}", limit=8)
                row["archived_skills"] = n
        if step % args.log_every == 0 or step == 1:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    last_util = summarize_step_utility(aux)
    decision = decide_step_actions(history, last_util, GrowPruneConfig(max_actions=12))
    report = {
        "args": vars(args),
        "best_acc": best_acc,
        "best_step": best_step,
        "last": rows[-1] if rows else {},
        "rows_tail": rows[-30:],
        "utility_report": last_util,
        "controller_decision": decision,
        "read_names": StepBuilderCore.READ_NAMES,
        "op_names": StepBuilderCore.OP_NAMES,
    }
    (out / "probe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Compact markdown for quick review.
    lines = ["# v14 StepBuilder probe", "", f"best_acc: `{best_acc:.4f}` @ step `{best_step}`", "", "## Last row", ""]
    for k, v in rows[-1].items():
        if k != "controller":
            lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Controller decision", "", "```json", json.dumps(decision, ensure_ascii=False, indent=2), "```", "", "## Top utility steps", ""]
    top_steps = sorted(last_util.get("per_step", []), key=lambda x: float(x.get("utility", 0.0)), reverse=True)[:16]
    for x in top_steps:
        lines.append(f"- L{x['layer']}B{x['block']}S{x['step']} utility={x['utility']:.4f} gate={x['gate']:.3f} update={x['update_norm']:.3f} status={x['status']} action={x['action']}")
    (out / "step_builder_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("done", out / "probe_report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--steps-per-block", type=int, default=4)
    ap.add_argument("--memory-cells", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--dropout", type=float, default=0.08)
    ap.add_argument("--read-temp", type=float, default=1.10)
    ap.add_argument("--op-temp", type=float, default=1.20)
    ap.add_argument("--write-bias", type=float, default=-0.35)
    ap.add_argument("--topk-ops", type=int, default=3)
    ap.add_argument("--hard-topk-after", type=int, default=120)
    ap.add_argument("--gate-target", type=float, default=0.72)
    ap.add_argument("--logit-target", type=float, default=8.0)
    ap.add_argument("--lambda-gate-budget", type=float, default=0.01)
    ap.add_argument("--lambda-logit-budget", type=float, default=0.001)
    ap.add_argument("--lambda-op-entropy-floor", type=float, default=0.003)
    ap.add_argument("--grad-clip", type=float, default=0.7)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--controller-every", type=int, default=25)
    ap.add_argument("--archive-skills", action="store_true")
    ap.add_argument("--out-dir", default="./runs/v14_step_builder_probe")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
