#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from run_universal_matrix_program_v1 import BASE_OPS, CATS, HFHiddenTask, devtype, losses, seed_all, write_json  # noqa: E402
from run_universal_matrix_program_v2_1_balanced import to_jsonable  # noqa: E402
from run_universal_matrix_program_v2_2_step_planner import _grad_snapshot  # noqa: E402
from run_universal_matrix_program_v2_5_alive_cost import UniversalProgramV25  # noqa: E402
from run_universal_matrix_program_v2_5b_scheduled_cost import ScheduledAliveCostController, _ramp  # noqa: E402


SEMANTIC_OPS = ["evidence_read", "route_read", "memory_read", "memory_write", "aggregate_merge", "delta_route"]
CHEAP_OPS = ["identity", "residual_repair"]


def _norm(v: torch.Tensor) -> torch.Tensor:
    return v / v.sum().clamp_min(1e-8)


def _load_skill_targets(path: str | None, device: torch.device) -> Dict[str, torch.Tensor | int]:
    op = torch.zeros(len(BASE_OPS), device=device)
    cat = torch.zeros(len(CATS), device=device)
    n = 0
    if path:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    w = float(s.get("metrics", {}).get("estimated_value", 0.5) or 0.5)
                    for step in s.get("program", []):
                        for name, val in (step.get("op_mix") or {}).items():
                            if name in BASE_OPS:
                                op[BASE_OPS.index(name)] += w * float(val)
                        for name, val in (step.get("category_mix") or {}).items():
                            if name in CATS:
                                cat[CATS.index(name)] += w * float(val)
                    n += 1
    if op.sum() <= 1e-8:
        # Safe weak default: keep non-trivial matrix program mass without forcing exact skills.
        for name, weight in {
            "evidence_read": 1.2,
            "residual_repair": 1.0,
            "delta_route": 0.8,
            "route_read": 0.7,
            "memory_read": 0.7,
            "memory_write": 0.5,
            "aggregate_merge": 0.5,
            "identity": 0.4,
        }.items():
            op[BASE_OPS.index(name)] += weight
    if cat.sum() <= 1e-8:
        for name, weight in {"extract": 1.0, "repair": 1.0, "memory": 0.8, "aggregate": 0.6, "compare": 0.4}.items():
            cat[CATS.index(name)] += weight
    return {"op_target": _norm(op), "cat_target": _norm(cat), "skill_count": n}


def _active_prior_distributions(model: UniversalProgramV25) -> Dict[str, torch.Tensor]:
    core = model.core
    ops: List[torch.Tensor] = []
    cats: List[torch.Tensor] = []
    dev = core.op_prior.device
    prim_alive = core._alive(core.primitive_alive_logits).to(dev)
    for lid in core.active_layer_ids:
        nb = int(core.active_blocks[lid].item())
        for b in range(nb):
            ns = int(core.active_steps[lid, b].item())
            for s in range(ns):
                op_logits = core.op_prior[lid, b, s, :len(BASE_OPS)] + torch.log(prim_alive[:len(BASE_OPS)].clamp_min(1e-5))
                cat_logits = core.cat_prior[lid, b, s]
                ops.append(torch.softmax(op_logits / core.primitive_temp, -1))
                cats.append(torch.softmax(cat_logits / core.category_temp, -1))
    if not ops:
        z = torch.zeros(1, len(BASE_OPS), device=dev)
        c = torch.zeros(1, len(CATS), device=dev)
        return {"op_probs": z, "cat_probs": c}
    return {"op_probs": torch.stack(ops, 0), "cat_probs": torch.stack(cats, 0)}


def _loss_driven_architecture_loss(model: UniversalProgramV25, targets: Dict[str, torch.Tensor | int], args) -> Dict[str, torch.Tensor]:
    d = _active_prior_distributions(model)
    op_probs = d["op_probs"]
    cat_probs = d["cat_probs"]
    device = op_probs.device
    op_target = targets["op_target"].to(device)  # type: ignore[union-attr]
    cat_target = targets["cat_target"].to(device)  # type: ignore[union-attr]
    avg_op = op_probs.mean(0)
    avg_cat = cat_probs.mean(0)

    semantic_ids = [BASE_OPS.index(x) for x in SEMANTIC_OPS if x in BASE_OPS]
    cheap_ids = [BASE_OPS.index(x) for x in CHEAP_OPS if x in BASE_OPS]
    semantic_mass = avg_op[semantic_ids].sum() if semantic_ids else torch.tensor(0.0, device=device)
    residual_ratio = avg_op[cheap_ids].sum() if cheap_ids else torch.tensor(0.0, device=device)

    semantic_loss = F.relu(torch.tensor(args.min_semantic_mass, device=device) - semantic_mass)
    semantic_loss = semantic_loss + F.relu(residual_ratio - torch.tensor(args.max_residual_only_ratio, device=device))

    # Skill loss is weak: it nudges priors toward known useful motifs without forcing them.
    skill_op_loss = F.mse_loss(avg_op, op_target)
    skill_cat_loss = F.mse_loss(avg_cat, cat_target)
    skill_loss = skill_op_loss + 0.5 * skill_cat_loss

    # Category-op consistency as real differentiable loss on priors.
    cat_op = model.core.cat_op[:len(CATS), :len(BASE_OPS)].to(device, op_probs.dtype)
    desired_op = cat_probs @ cat_op
    desired_op = desired_op / desired_op.sum(-1, keepdim=True).clamp_min(1e-8)
    consistency_loss = F.mse_loss(op_probs, desired_op)

    # Prior-side expected cost, also differentiable through primitive alive and op priors.
    cost_vec = model.core.primitive_cost[:len(BASE_OPS)].to(device, op_probs.dtype)
    prior_cost = (op_probs * cost_vec.view(1, -1)).sum(-1).mean()
    return {
        "semantic_loss": semantic_loss,
        "skill_loss": skill_loss,
        "consistency_loss": consistency_loss,
        "prior_cost": prior_cost,
        "semantic_mass": semantic_mass.detach(),
        "residual_only_ratio": residual_ratio.detach(),
        "skill_op_loss": skill_op_loss.detach(),
        "skill_cat_loss": skill_cat_loss.detach(),
        "consistency_metric": (1.0 - consistency_loss.detach()).clamp_min(0.0),
    }


def run(args) -> None:
    seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    task = HFHiddenTask(args.model_name, args.layer_idx, args.device)
    model = UniversalProgramV25(task.dim, args).to(args.device)
    skill_targets = _load_skill_targets(args.skill_bank if args.use_skill_loss else None, torch.device(args.device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    active = [x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl = ScheduledAliveCostController(args)

    fields = [
        "epoch","score","best_score","mode","action","target_layer","target_block","target_step","reason","growth_detail",
        "step_score","depth_score","width_score","primitive_score","prune_score","repair_category_score","descriptor_repair_score","soft_prune_score",
        "expected_cost","alive_mean","cost_weight","alive_weight","semantic_weight","skill_weight","consistency_weight",
        "semantic_mass","residual_only_ratio","skill_loss","consistency_metric","prior_cost","skill_count","seconds",
    ] + [f"train_{n}_{k}" for n in active for k in ("mse","cos")] + [f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for ep in range(1, args.epochs + 1):
        model.train(); sums: Dict[str, float] = {}; t0 = time.time(); last_grad = None
        cost_w = args.lambda_op_cost * _ramp(ep, args.epochs, args.cost_start_frac, args.cost_end_frac)
        alive_w = args.lambda_alive_sparsity * _ramp(ep, args.epochs, args.alive_start_frac, args.alive_end_frac)
        sem_w = args.lambda_semantic_diversity * _ramp(ep, args.epochs, args.semantic_start_frac, args.semantic_end_frac)
        skill_w = args.lambda_skill_loss * _ramp(ep, args.epochs, args.skill_start_frac, args.skill_end_frac)
        cons_w = args.lambda_category_consistency * _ramp(ep, args.epochs, args.consistency_start_frac, args.consistency_end_frac)
        last_arch = None
        for _ in range(args.steps_per_epoch):
            b = task.sample(args.batch_size, args.seq_len, args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                pred, aux = model(b["x"])
                loss, met = losses(pred, b, active, args.lambda_cos)
                arch = _loss_driven_architecture_loss(model, skill_targets, args)
                last_arch = arch
                cost_loss = aux.get("expected_cost", torch.tensor(0.0, device=loss.device))
                alive_loss = aux.get("alive_mean", torch.tensor(0.0, device=loss.device))
                loss = loss + cost_w * cost_loss + alive_w * alive_loss
                loss = loss + sem_w * arch["semantic_loss"] + skill_w * arch["skill_loss"] + cons_w * arch["consistency_loss"]
                loss = loss + args.lambda_prior_cost * cost_w * arch["prior_cost"]
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            last_grad = _grad_snapshot(model)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            for k, v in met.items(): sums[k] = sums.get(k, 0.0) + v
        train = {k: v / args.steps_per_epoch for k, v in sums.items()} | {"seconds": time.time() - t0}

        model.eval(); sums = {}; last_aux = None
        with torch.no_grad():
            for _ in range(args.val_steps):
                b = task.sample(args.eval_batch_size, args.seq_len, args.device)
                with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                    pred, aux = model(b["x"]); _, met = losses(pred, b, active, args.lambda_cos)
                for k, v in met.items(): sums[k] = sums.get(k, 0.0) + v
                last_aux = aux
        val = {k: v / args.val_steps for k, v in sums.items()}; aux_json = to_jsonable(last_aux or {})
        ev = ctrl.step(model, ep, val, aux_json, active, last_grad)
        expected_cost = float(aux_json.get("expected_cost", 0.0)) if isinstance(aux_json, dict) else 0.0
        alive_mean = float(aux_json.get("alive_mean", 0.0)) if isinstance(aux_json, dict) else 0.0
        scores = ctrl.last_scores or {}
        arch = last_arch or _loss_driven_architecture_loss(model, skill_targets, args)
        row = {
            "epoch": ep, "score": ev.score, "best_score": ev.best_score, "mode": ev.mode, "action": ev.action,
            "target_layer": ev.target_layer, "target_block": ev.target_block, "target_step": getattr(ev, "target_step", -1),
            "reason": ev.reason, "growth_detail": ev.growth_detail,
            "step_score": ev.step_score, "depth_score": ev.depth_score, "width_score": ev.width_score,
            "primitive_score": ev.primitive_score, "prune_score": ev.prune_score,
            "repair_category_score": ev.repair_category_score,
            "descriptor_repair_score": float(scores.get("descriptor_repair", 0.0)),
            "soft_prune_score": float(scores.get("soft_prune", 0.0)),
            "expected_cost": expected_cost, "alive_mean": alive_mean,
            "cost_weight": cost_w, "alive_weight": alive_w, "semantic_weight": sem_w, "skill_weight": skill_w, "consistency_weight": cons_w,
            "semantic_mass": float(arch["semantic_mass"].detach().cpu()),
            "residual_only_ratio": float(arch["residual_only_ratio"].detach().cpu()),
            "skill_loss": float(arch["skill_loss"].detach().cpu()),
            "consistency_metric": float(arch["consistency_metric"].detach().cpu()),
            "prior_cost": float(arch["prior_cost"].detach().cpu()),
            "skill_count": int(skill_targets.get("skill_count", 0)),
            "seconds": train["seconds"],
        }
        for n in active:
            row[f"train_{n}_mse"] = train.get(f"{n}_mse", ""); row[f"train_{n}_cos"] = train.get(f"{n}_cos", "")
            row[f"val_{n}_mse"] = val.get(f"{n}_mse", ""); row[f"val_{n}_cos"] = val.get(f"{n}_cos", "")
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        report = {
            "epoch": ep, "train": train, "val": val, "controller": asdict(ev), "action_history": ctrl.history,
            "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending), "grad_health_last_batch": last_grad,
            "matrix_program": aux_json, "arch_state": model.export_arch_state(),
            "scheduled_weights": {"cost": cost_w, "alive": alive_w, "semantic": sem_w, "skill": skill_w, "consistency": cons_w},
            "loss_driven_debug": {k: float(v.detach().cpu()) for k, v in arch.items() if torch.is_tensor(v)},
            "skill_bank": {"path": args.skill_bank, "count": int(skill_targets.get("skill_count", 0))},
            "args": vars(args),
        }
        write_json(out / f"analysis_epoch_{ep:03d}.json", report)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": asdict(ev), "scores": scores, "loss_debug": report["loss_driven_debug"], "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending)}, ensure_ascii=False) + "\n")
        print(
            f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} "
            f"target=L{ev.target_layer}B{ev.target_block} scores(step={ev.step_score:.3f},depth={ev.depth_score:.3f},width={ev.width_score:.3f},"
            f"softprune={float(scores.get('soft_prune',0.0)):.3f},desc={float(scores.get('descriptor_repair',0.0)):.3f},"
            f"cost={expected_cost:.3f},alive={alive_mean:.3f},sem={row['semantic_mass']:.3f},resid={row['residual_only_ratio']:.3f},"
            f"skill={row['skill_loss']:.4f},cons={row['consistency_metric']:.3f}) {ev.growth_detail} "
            + " ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]),
            flush=True,
        )
    write_json(out / "final_report.json", {"architecture": "universal_matrix_program_v2_6_loss_driven_skills", "final_arch_state": model.export_arch_state(), "action_history": ctrl.history, "args": vars(args)})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="google/bert_uncased_L-2_H-128_A-2"); p.add_argument("--layer-idx", type=int, default=0)
    p.add_argument("--device", default="cuda"); p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=64); p.add_argument("--targets", default="block,delta,norm")
    p.add_argument("--evidence-cells", type=int, default=40); p.add_argument("--task-cells", type=int, default=12); p.add_argument("--memory-cells", type=int, default=6); p.add_argument("--global-cells", type=int, default=3)
    p.add_argument("--init-layers", type=int, default=4); p.add_argument("--max-layers", type=int, default=8)
    p.add_argument("--init-blocks-per-layer", type=int, default=4); p.add_argument("--max-blocks-per-layer", type=int, default=8)
    p.add_argument("--init-steps-per-block", type=int, default=1); p.add_argument("--max-steps-per-block", type=int, default=4); p.add_argument("--max-primitives", type=int, default=24)
    p.add_argument("--category-temp", type=float, default=1.4); p.add_argument("--primitive-temp", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--eval-batch-size", type=int, default=128); p.add_argument("--steps-per-epoch", type=int, default=120); p.add_argument("--val-steps", type=int, default=20); p.add_argument("--epochs", type=int, default=45)
    p.add_argument("--lr", type=float, default=7e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=0.7); p.add_argument("--lambda-cos", type=float, default=0.25)
    p.add_argument("--controller-min-delta", type=float, default=0.0015); p.add_argument("--controller-patience", type=int, default=2); p.add_argument("--rollback-patience", type=int, default=1); p.add_argument("--pending-epochs", type=int, default=2); p.add_argument("--weak-action-reward", type=float, default=0.00025)
    p.add_argument("--cleanup-start-frac", type=float, default=0.62); p.add_argument("--late-growth-start-frac", type=float, default=0.70); p.add_argument("--late-growth-penalty", type=float, default=0.45)
    p.add_argument("--descriptor-action-boost", type=float, default=0.15); p.add_argument("--soft-prune-strength", type=float, default=0.30)
    p.add_argument("--alive-init-logit", type=float, default=2.2); p.add_argument("--alive-floor", type=float, default=0.05)
    p.add_argument("--lambda-op-cost", type=float, default=0.0012); p.add_argument("--lambda-alive-sparsity", type=float, default=0.00025); p.add_argument("--lambda-prior-cost", type=float, default=0.25)
    p.add_argument("--cost-start-frac", type=float, default=0.55); p.add_argument("--cost-end-frac", type=float, default=0.90)
    p.add_argument("--alive-start-frac", type=float, default=0.60); p.add_argument("--alive-end-frac", type=float, default=0.95)
    p.add_argument("--lambda-semantic-diversity", type=float, default=0.012); p.add_argument("--semantic-start-frac", type=float, default=0.15); p.add_argument("--semantic-end-frac", type=float, default=0.55)
    p.add_argument("--min-semantic-mass", type=float, default=0.24); p.add_argument("--max-residual-only-ratio", type=float, default=0.76)
    p.add_argument("--use-skill-loss", action="store_true"); p.add_argument("--skill-bank", default="./skills/skill_bank.jsonl")
    p.add_argument("--lambda-skill-loss", type=float, default=0.006); p.add_argument("--skill-start-frac", type=float, default=0.05); p.add_argument("--skill-end-frac", type=float, default=0.45)
    p.add_argument("--lambda-category-consistency", type=float, default=0.004); p.add_argument("--consistency-start-frac", type=float, default=0.10); p.add_argument("--consistency-end-frac", type=float, default=0.70)
    p.add_argument("--candidate-primitive-cost", type=float, default=0.85)
    p.add_argument("--out-dir", default="./runs/universal_matrix_program_v2_6_loss_driven_skills_hf_45ep")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
