#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from run_universal_matrix_program_v1 import BASE_OPS, CATS, HFHiddenTask, devtype, losses, seed_all, write_json  # noqa: E402
from run_universal_matrix_program_v2_1_balanced import UniversalProgramV21, to_jsonable  # noqa: E402
from run_universal_matrix_program_v2_2_step_planner import _grad_snapshot  # noqa: E402
from run_universal_matrix_program_v2_3_cleanup import CleanupPlannerController  # noqa: E402


PRIMITIVE_DESC: Dict[str, Dict[str, float]] = {
    "identity": {"preserve": 1.0, "early": 0.5, "late": 0.2},
    "evidence_read": {"extract": 1.0, "read": 1.0, "early": 1.0},
    "task_read": {"task": 1.0, "read": 1.0, "global": 0.6},
    "route_read": {"route": 1.0, "read": 1.0, "middle": 0.9},
    "delta_route": {"delta": 1.0, "route": 0.7, "repair": 0.4, "middle": 0.8, "late": 0.6},
    "compare_mul": {"compare": 1.0, "relation": 0.8, "middle": 0.7},
    "memory_read": {"memory": 1.0, "read": 1.0, "middle": 0.8, "late": 0.5},
    "memory_write": {"memory": 1.0, "write": 1.0, "middle": 0.6, "late": 0.8},
    "global_summary": {"global": 1.0, "aggregate": 0.8, "late": 0.7},
    "suppress_global": {"suppress": 1.0, "global": 0.7, "repair": 0.5},
    "normalize": {"normalize": 1.0, "repair": 0.7, "late": 0.6},
    "energy_count": {"stat": 1.0, "extract": 0.6, "global": 0.4},
    "residual_repair": {"repair": 1.0, "delta": 0.5, "late": 0.6},
    "aggregate_merge": {"aggregate": 1.0, "memory": 0.4, "late": 0.8},
}

CAT_DESC = {
    "extract": ["extract", "read", "early"],
    "compare": ["compare", "relation", "route"],
    "memory": ["memory", "read", "write"],
    "repair": ["repair", "delta", "normalize"],
    "aggregate": ["aggregate", "global", "late"],
    "suppress": ["suppress", "repair"],
    "transform": ["preserve", "route", "delta"],
}


def _need_vector(val: Dict[str, float], feat: Dict[str, Any]) -> Dict[str, float]:
    delta_gap = max(0.0, float(val.get("block_cos", 0.0)) - float(val.get("delta_cos", 0.0)))
    norm_pain = float(val.get("norm_mse", 0.0))
    pressure = float(feat.get("pressure", 0.0))
    entropy = float(feat.get("op_entropy", 0.0))
    consistency = float(feat.get("category_consistency", 0.0))
    step_count = int(feat.get("step_count", 1))
    return {
        "delta": 0.8 * delta_gap + 0.5 * pressure,
        "repair": 0.6 * delta_gap + 0.3 * norm_pain + 0.4 * max(0.0, 0.45 - consistency),
        "memory": 0.3 * pressure + 0.2 * max(0, step_count - 1),
        "route": 0.4 * pressure + 0.2 * entropy,
        "extract": 0.2 * max(0.0, 0.45 - consistency),
        "aggregate": 0.3 * pressure + 0.2 * norm_pain,
        "normalize": 0.7 * norm_pain,
        "read": 0.2 * pressure,
        "write": 0.15 * max(0, step_count - 1),
    }


def _primitive_scores_from_need(need: Dict[str, float]) -> torch.Tensor:
    scores = []
    for op in BASE_OPS:
        desc = PRIMITIVE_DESC.get(op, {})
        scores.append(sum(float(need.get(k, 0.0)) * float(v) for k, v in desc.items()))
    t = torch.tensor(scores, dtype=torch.float32)
    if float(t.abs().sum()) <= 1e-8:
        return torch.zeros(len(BASE_OPS), dtype=torch.float32)
    return t / t.abs().max().clamp_min(1e-8)


def _category_scores_from_need(need: Dict[str, float]) -> torch.Tensor:
    scores = []
    for cat in CATS:
        scores.append(sum(float(need.get(k, 0.0)) for k in CAT_DESC.get(cat, [])))
    t = torch.tensor(scores, dtype=torch.float32)
    if float(t.abs().sum()) <= 1e-8:
        return torch.zeros(len(CATS), dtype=torch.float32)
    return t / t.abs().max().clamp_min(1e-8)


@dataclass
class DescriptorEvent:
    epoch: int
    mode: str
    action: str
    score: float
    best_score: float
    reward: float
    accepted: bool
    target_layer: int
    target_block: int
    target_step: int
    reason: str
    growth_detail: str
    step_score: float
    depth_score: float
    width_score: float
    primitive_score: float
    prune_score: float
    repair_category_score: float
    descriptor_repair_score: float
    category_consistency: float
    planner_attention_top: List[Dict[str, Any]]


class DescriptorAliveController(CleanupPlannerController):
    def _action_scores(self, model: UniversalProgramV21, val: Dict[str, float], feat: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, float]:
        scores = super()._action_scores(model, val, feat, aux)
        need = _need_vector(val, feat)
        desc_mag = sum(abs(x) for x in need.values())
        consistency = float(feat.get("category_consistency", 0.0))
        pressure = float(feat.get("pressure", 0.0))
        grad = float(feat.get("grad", 0.0))
        layer = int(feat.get("layer", 0)); block = int(feat.get("block", 0)); step = int(feat.get("step_count", 1))
        scores["descriptor_repair"] = 0.22 * min(1.0, desc_mag) + 0.18 * max(0.0, 0.55 - consistency) + 0.06 * pressure + 0.06 * grad - self._cool("descriptor_repair", layer, block, step)
        # If descriptor says old primitives have a strong match, do not add a new primitive too early.
        prim_scores = _primitive_scores_from_need(need)
        if float(prim_scores.max()) > 0.5:
            scores["add_primitive"] -= 0.10
        return scores

    def _descriptor_repair(self, model: UniversalProgramV21, feat: Dict[str, Any], val: Dict[str, float]) -> Tuple[bool, str]:
        layer = int(feat.get("layer", 0)); block = int(feat.get("block", 0))
        steps = int(model.core.active_steps[layer, block].item())
        need = _need_vector(val, feat)
        op_score = _primitive_scores_from_need(need).to(model.core.op_prior.device)
        cat_score = _category_scores_from_need(need).to(model.core.cat_prior.device)
        with torch.no_grad():
            for s in range(steps):
                model.core.op_prior[layer, block, s, :len(BASE_OPS)] += 0.06 * op_score
                model.core.cat_prior[layer, block, s, :] += 0.05 * cat_score
        top_ops = torch.topk(op_score, k=min(3, op_score.numel())).indices.tolist()
        names = [BASE_OPS[i] for i in top_ops]
        return True, f"descriptor_repair_L{layer}B{block}_steps_{steps}_ops_{','.join(names)}"

    def _do_action(self, model: UniversalProgramV21, action: str, feat: Dict[str, Any], aux: Dict[str, Any]) -> Tuple[bool, str]:
        if action == "descriptor_repair":
            return self._descriptor_repair(model, feat, self._last_val)
        return super()._do_action(model, action, feat, aux)

    def step(self, model: UniversalProgramV21, epoch: int, val: Dict[str, float], aux: Dict[str, Any], active: List[str], grad_health: Any) -> DescriptorEvent:
        self._last_val = val
        base = super().step(model, epoch, val, aux, active, grad_health)
        scores = self.last_scores or {}
        return DescriptorEvent(
            epoch=base.epoch, mode=base.mode, action=base.action, score=base.score,
            best_score=base.best_score, reward=base.reward, accepted=base.accepted,
            target_layer=base.target_layer, target_block=base.target_block, target_step=base.target_step,
            reason=base.reason, growth_detail=base.growth_detail,
            step_score=base.step_score, depth_score=base.depth_score, width_score=base.width_score,
            primitive_score=base.primitive_score, prune_score=base.prune_score,
            repair_category_score=base.repair_category_score,
            descriptor_repair_score=float(scores.get("descriptor_repair", 0.0)),
            category_consistency=base.category_consistency,
            planner_attention_top=base.planner_attention_top,
        )


def _avg_alive_proxy(model: UniversalProgramV21) -> Dict[str, float]:
    # True alive-gates are planned for a full core rewrite. For v2.4 we expose a soft
    # proxy from gates/priors so the report can track cleanup pressure without changing
    # the stable v2.1 forward graph.
    with torch.no_grad():
        op_energy = model.core.op_prior.abs().mean().item()
        cat_energy = model.core.cat_prior.abs().mean().item()
    return {"op_prior_energy": float(op_energy), "cat_prior_energy": float(cat_energy)}


def run(args) -> None:
    seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    task = HFHiddenTask(args.model_name, args.layer_idx, args.device)
    model = UniversalProgramV21(task.dim, args).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    active = [x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl = DescriptorAliveController(args)
    fields = ["epoch","score","best_score","mode","action","target_layer","target_block","target_step","reason","growth_detail","step_score","depth_score","width_score","primitive_score","prune_score","repair_category_score","descriptor_repair_score","category_consistency","seconds"] + [f"train_{n}_{k}" for n in active for k in ("mse","cos")] + [f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    for ep in range(1, args.epochs + 1):
        model.train(); sums: Dict[str, float] = {}; t0 = time.time(); last_grad = None
        for _ in range(args.steps_per_epoch):
            b = task.sample(args.batch_size, args.seq_len, args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                pred, _ = model(b["x"]); loss, met = losses(pred, b, active, args.lambda_cos)
            scaler.scale(loss).backward(); scaler.unscale_(opt); last_grad = _grad_snapshot(model)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); scaler.step(opt); scaler.update()
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
        val = {k: v / args.val_steps for k, v in sums.items()}; aux = to_jsonable(last_aux or {})
        ev = ctrl.step(model, ep, val, aux, active, last_grad)
        row = {"epoch": ep, "score": ev.score, "best_score": ev.best_score, "mode": ev.mode, "action": ev.action, "target_layer": ev.target_layer, "target_block": ev.target_block, "target_step": ev.target_step, "reason": ev.reason, "growth_detail": ev.growth_detail, "step_score": ev.step_score, "depth_score": ev.depth_score, "width_score": ev.width_score, "primitive_score": ev.primitive_score, "prune_score": ev.prune_score, "repair_category_score": ev.repair_category_score, "descriptor_repair_score": ev.descriptor_repair_score, "category_consistency": ev.category_consistency, "seconds": train["seconds"]}
        for n in active:
            row[f"train_{n}_mse"] = train.get(f"{n}_mse", ""); row[f"train_{n}_cos"] = train.get(f"{n}_cos", "")
            row[f"val_{n}_mse"] = val.get(f"{n}_mse", ""); row[f"val_{n}_cos"] = val.get(f"{n}_cos", "")
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        report = {"epoch": ep, "train": train, "val": val, "controller": asdict(ev), "action_history": ctrl.history, "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending), "grad_health_last_batch": last_grad, "alive_proxy": _avg_alive_proxy(model), "primitive_descriptors": PRIMITIVE_DESC, "matrix_program": aux, "arch_state": model.export_arch_state(), "args": vars(args)}
        write_json(out / f"analysis_epoch_{ep:03d}.json", report)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": asdict(ev), "history_tail": ctrl.history[-3:], "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending)}, ensure_ascii=False) + "\n")
        print(f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} target=L{ev.target_layer}B{ev.target_block}S{ev.target_step} scores(step={ev.step_score:.3f},depth={ev.depth_score:.3f},width={ev.width_score:.3f},prim={ev.primitive_score:.3f},prune={ev.prune_score:.3f},repair={ev.repair_category_score:.3f},desc={ev.descriptor_repair_score:.3f},cons={ev.category_consistency:.3f}) {ev.growth_detail} " + " ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]), flush=True)
    write_json(out / "final_report.json", {"architecture": "universal_matrix_program_v2_4_descriptor_alive", "final_arch_state": model.export_arch_state(), "action_history": ctrl.history, "args": vars(args)})


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
    p.add_argument("--out-dir", default="./runs/universal_matrix_program_v2_4_descriptor_alive_hf_45ep")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
