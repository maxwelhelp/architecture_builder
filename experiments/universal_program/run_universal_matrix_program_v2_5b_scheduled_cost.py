#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, sys, time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from run_universal_matrix_program_v1 import BASE_OPS, HFHiddenTask, devtype, losses, seed_all, write_json  # noqa: E402
from run_universal_matrix_program_v2_1_balanced import to_jsonable  # noqa: E402
from run_universal_matrix_program_v2_2_step_planner import _grad_snapshot  # noqa: E402
from run_universal_matrix_program_v2_5_alive_cost import AliveCostController, UniversalProgramV25  # noqa: E402


def _ramp(epoch: int, total: int, start_frac: float, end_frac: float) -> float:
    if total <= 1:
        return 1.0
    x = epoch / float(total)
    if x <= start_frac:
        return 0.0
    if x >= end_frac:
        return 1.0
    return (x - start_frac) / max(1e-8, end_frac - start_frac)


def _semantic_loss(aux: Dict[str, Any], device: torch.device, args) -> Dict[str, torch.Tensor]:
    """Prevent cheap collapse into identity/residual-only programs.

    This is a soft regularizer, not a hard rule. It keeps a minimum amount of
    evidence/route/memory/aggregate/delta operators alive while still allowing the
    cost term to optimize away useless duplicates.
    """
    opmix = aux.get("operator_mix_by_layer")
    if not torch.is_tensor(opmix) or opmix.numel() == 0:
        z = torch.tensor(0.0, device=device)
        return {"semantic_loss": z, "semantic_mass": z, "residual_only_ratio": z}
    opmix = opmix.float().to(device)
    names = list(BASE_OPS)
    def idx(name: str) -> int:
        return names.index(name)
    semantic_names = ["evidence_read", "route_read", "memory_read", "memory_write", "aggregate_merge", "delta_route"]
    semantic_ids = [idx(n) for n in semantic_names if n in names]
    cheap_ids = [idx(n) for n in ["identity", "residual_repair"] if n in names]
    semantic_mass = opmix[:, semantic_ids].sum(-1).mean() if semantic_ids else torch.tensor(0.0, device=device)
    residual_ratio = opmix[:, cheap_ids].sum(-1).mean() if cheap_ids else torch.tensor(0.0, device=device)
    loss = F.relu(torch.tensor(args.min_semantic_mass, device=device) - semantic_mass)
    loss = loss + F.relu(residual_ratio - torch.tensor(args.max_residual_only_ratio, device=device))
    return {"semantic_loss": loss, "semantic_mass": semantic_mass.detach(), "residual_only_ratio": residual_ratio.detach()}


class ScheduledAliveCostController(AliveCostController):
    """v2.5b controller.

    Changes after v2.5:
    - late growth is penalized earlier/stronger;
    - cleanup actions get a bigger chance late;
    - structural growth is suppressed when score is still improving normally;
    - descriptor/soft-prune are preferred when the program already has enough blocks/layers.
    """

    def _action_scores(self, model: UniversalProgramV25, val: Dict[str, float], feat: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, float]:
        scores = super()._action_scores(model, val, feat, aux)
        frac = self.epoch_frac
        active_layers = len(aux.get("active_layer_ids", [])) if isinstance(aux, dict) else 0
        layer = int(feat.get("layer", -1))
        # Stronger late guard: no pointless ep40+ width/depth.
        if frac >= 0.70:
            scores["add_width"] -= 0.45
            scores["add_depth"] -= 0.45
            scores["add_step"] -= 0.12
            scores["descriptor_repair"] = scores.get("descriptor_repair", 0.0) + 0.14
            scores["soft_prune"] = scores.get("soft_prune", 0.0) + 0.10
            scores["repair_category"] = scores.get("repair_category", 0.0) + 0.08
        if frac >= 0.82:
            scores["add_width"] = min(scores.get("add_width", -999.0), -0.20)
            scores["add_depth"] = min(scores.get("add_depth", -999.0), -0.20)
        # If architecture already has enough stages, prefer cleanup over more stages.
        if active_layers >= 6:
            scores["add_depth"] -= 0.25
            scores["descriptor_repair"] = scores.get("descriptor_repair", 0.0) + 0.05
        # Avoid very late growth in early layer 0 unless it is clearly better.
        if frac >= 0.60 and layer == 0:
            scores["add_width"] -= 0.20
        return scores


def run(args) -> None:
    seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    task = HFHiddenTask(args.model_name, args.layer_idx, args.device)
    model = UniversalProgramV25(task.dim, args).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    active = [x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl = ScheduledAliveCostController(args)

    fields = [
        "epoch","score","best_score","mode","action","target_layer","target_block","target_step","reason","growth_detail",
        "step_score","depth_score","width_score","primitive_score","prune_score","repair_category_score","descriptor_repair_score","soft_prune_score",
        "category_consistency","expected_cost","alive_mean","cost_weight","alive_weight","semantic_weight","semantic_mass","residual_only_ratio","seconds",
    ] + [f"train_{n}_{k}" for n in active for k in ("mse","cos")] + [f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for ep in range(1, args.epochs + 1):
        model.train(); sums: Dict[str, float] = {}; t0 = time.time(); last_grad = None
        cost_w = args.lambda_op_cost * _ramp(ep, args.epochs, args.cost_start_frac, args.cost_end_frac)
        alive_w = args.lambda_alive_sparsity * _ramp(ep, args.epochs, args.alive_start_frac, args.alive_end_frac)
        sem_w = args.lambda_semantic_diversity * _ramp(ep, args.epochs, args.semantic_start_frac, args.semantic_end_frac)
        last_sem = {"semantic_mass": torch.tensor(0.0), "residual_only_ratio": torch.tensor(0.0), "semantic_loss": torch.tensor(0.0)}

        for _ in range(args.steps_per_epoch):
            b = task.sample(args.batch_size, args.seq_len, args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                pred, aux = model(b["x"])
                loss, met = losses(pred, b, active, args.lambda_cos)
                cost_loss = aux.get("expected_cost", torch.tensor(0.0, device=loss.device))
                alive_loss = aux.get("alive_mean", torch.tensor(0.0, device=loss.device))
                sem = _semantic_loss(aux, loss.device, args)
                last_sem = sem
                loss = loss + cost_w * cost_loss + alive_w * alive_loss + sem_w * sem["semantic_loss"]
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
        sem_mass = float(last_sem["semantic_mass"].detach().cpu())
        residual_ratio = float(last_sem["residual_only_ratio"].detach().cpu())
        row = {
            "epoch": ep, "score": ev.score, "best_score": ev.best_score, "mode": ev.mode, "action": ev.action,
            "target_layer": ev.target_layer, "target_block": ev.target_block, "target_step": getattr(ev, "target_step", -1),
            "reason": ev.reason, "growth_detail": ev.growth_detail,
            "step_score": ev.step_score, "depth_score": ev.depth_score, "width_score": ev.width_score,
            "primitive_score": ev.primitive_score, "prune_score": ev.prune_score,
            "repair_category_score": ev.repair_category_score,
            "descriptor_repair_score": float(scores.get("descriptor_repair", 0.0)),
            "soft_prune_score": float(scores.get("soft_prune", 0.0)),
            "category_consistency": getattr(ev, "category_consistency", 0.0),
            "expected_cost": expected_cost, "alive_mean": alive_mean,
            "cost_weight": cost_w, "alive_weight": alive_w, "semantic_weight": sem_w,
            "semantic_mass": sem_mass, "residual_only_ratio": residual_ratio,
            "seconds": train["seconds"],
        }
        for n in active:
            row[f"train_{n}_mse"] = train.get(f"{n}_mse", ""); row[f"train_{n}_cos"] = train.get(f"{n}_cos", "")
            row[f"val_{n}_mse"] = val.get(f"{n}_mse", ""); row[f"val_{n}_cos"] = val.get(f"{n}_cos", "")
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        report = {
            "epoch": ep, "train": train, "val": val, "controller": ev.__dict__, "action_history": ctrl.history,
            "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending), "grad_health_last_batch": last_grad,
            "matrix_program": aux_json, "arch_state": model.export_arch_state(),
            "scheduled_weights": {"cost": cost_w, "alive": alive_w, "semantic": sem_w},
            "semantic_debug": {"semantic_mass": sem_mass, "residual_only_ratio": residual_ratio},
            "args": vars(args),
        }
        write_json(out / f"analysis_epoch_{ep:03d}.json", report)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": ev.__dict__, "scores": scores, "semantic": report["semantic_debug"], "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending)}, ensure_ascii=False) + "\n")
        print(
            f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} "
            f"target=L{ev.target_layer}B{ev.target_block} scores(step={ev.step_score:.3f},depth={ev.depth_score:.3f},width={ev.width_score:.3f},"
            f"softprune={float(scores.get('soft_prune',0.0)):.3f},repair={ev.repair_category_score:.3f},desc={float(scores.get('descriptor_repair',0.0)):.3f},"
            f"cost={expected_cost:.3f},alive={alive_mean:.3f},sem={sem_mass:.3f},resid={residual_ratio:.3f},cw={cost_w:.4g}) {ev.growth_detail} "
            + " ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]),
            flush=True,
        )
    write_json(out / "final_report.json", {"architecture": "universal_matrix_program_v2_5b_scheduled_cost", "final_arch_state": model.export_arch_state(), "action_history": ctrl.history, "args": vars(args)})


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
    p.add_argument("--lambda-op-cost", type=float, default=0.0015); p.add_argument("--lambda-alive-sparsity", type=float, default=0.0003)
    p.add_argument("--cost-start-frac", type=float, default=0.50); p.add_argument("--cost-end-frac", type=float, default=0.85)
    p.add_argument("--alive-start-frac", type=float, default=0.55); p.add_argument("--alive-end-frac", type=float, default=0.90)
    p.add_argument("--lambda-semantic-diversity", type=float, default=0.010); p.add_argument("--semantic-start-frac", type=float, default=0.25); p.add_argument("--semantic-end-frac", type=float, default=0.70)
    p.add_argument("--min-semantic-mass", type=float, default=0.22); p.add_argument("--max-residual-only-ratio", type=float, default=0.78)
    p.add_argument("--candidate-primitive-cost", type=float, default=0.85)
    p.add_argument("--out-dir", default="./runs/universal_matrix_program_v2_5b_scheduled_cost_hf_45ep")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
