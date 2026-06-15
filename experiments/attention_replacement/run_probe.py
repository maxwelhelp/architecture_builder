#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, random, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from probe_core import (
    CATS, OPS, ROLES,
    FrozenTeacherMHA,
    MatrixAttentionReplacementStudent,
    StructuredSequenceGenerator,
)


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def devtype(device: str) -> str:
    return device.split(":", 1)[0]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class ControllerEvent:
    epoch: int
    mode: str
    action: str
    score: float
    best_score: float
    reward: float
    accepted: bool
    category_temp: float
    primitive_temp: float
    reason: str


class AdaptiveController:
    def __init__(self, patience: int = 2, rollback: int = 2, min_delta: float = 0.002) -> None:
        self.patience, self.rollback, self.min_delta = patience, rollback, min_delta
        self.best_score, self.best_state, self.prev_score = -1e9, None, None
        self.bad, self.plateau, self.i = 0, 0, 0
        self.actions = [
            "explore_categories", "boost_l2_memory", "boost_l2_repair",
            "boost_l1_compare", "boost_aggregate_memory", "sharpen_exploit", "reduce_priors",
        ]

    def score(self, mse: float, cos: float, aux: Dict[str, Any]) -> float:
        route = float(aux.get("route_entropy_mean", 0.0))
        token = float(aux.get("token_block_entropy", 0.0))
        return -mse + 0.20 * cos + 0.01 * route + 0.005 * token

    def step(self, model: MatrixAttentionReplacementStudent, epoch: int, mse: float, cos: float, aux: Dict[str, Any]) -> ControllerEvent:
        score = self.score(mse, cos, aux)
        reward = 0.0 if self.prev_score is None else score - self.prev_score
        accepted = score > self.best_score + self.min_delta
        mode, action, reason = "WAIT", "noop", "pending effect buffer"
        if accepted:
            self.best_score, self.best_state = score, model.export_arch_state()
            self.bad, self.plateau = 0, 0
            mode, action, reason = "EXPLOIT", "sharpen_exploit" if epoch > 3 else "noop", "new best score"
        else:
            if self.prev_score is not None and score < self.prev_score - self.min_delta:
                self.bad += 1
            else:
                self.plateau += 1
            if self.bad >= self.rollback and self.best_state is not None:
                model.load_arch_state(self.best_state)
                action = "reduce_priors"; model.apply_action(action, 0.04)
                mode, reason, self.bad = "ROLLBACK", "score dropped; restored best arch state", 0
            elif self.plateau >= self.patience:
                action = self.actions[self.i % len(self.actions)]; self.i += 1
                model.apply_action(action, 0.06)
                mode, reason, self.plateau = "EXPLORE", "plateau; trying soft architecture action", 0
        if action == "sharpen_exploit":
            model.apply_action(action, 0.03)
        self.prev_score = score
        return ControllerEvent(epoch, mode, action, score, self.best_score, reward, accepted, model.category_temp, model.primitive_temp, reason)


def summarize(aux_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    role = torch.stack([a["role_mix_by_layer"] for a in aux_list]).mean(0)
    cat = torch.stack([a["category_mix_by_layer"] for a in aux_list]).mean(0)
    op = torch.stack([a["operator_mix_by_layer"] for a in aux_list]).mean(0)
    route = torch.stack([a["route_entropy_by_layer"] for a in aux_list]).mean(0)
    token = torch.stack([a["token_block_entropy"] for a in aux_list]).mean()
    return {
        "role_mix_by_layer": [{"layer": i, "roles": {ROLES[j]: float(role[i, j]) for j in range(len(ROLES))}} for i in range(role.shape[0])],
        "category_mix_by_layer": [{"layer": i, "categories": {CATS[j]: float(cat[i, j]) for j in range(len(CATS))}} for i in range(cat.shape[0])],
        "operator_mix_by_layer": [{"layer": i, "ops": {OPS[j]: float(op[i, j]) for j in range(len(OPS))}} for i in range(op.shape[0])],
        "route_entropy_by_layer": [float(x) for x in route],
        "route_entropy_mean": float(route.mean()),
        "token_block_entropy": float(token),
    }


def train_epoch(model, teacher, gen, opt, scaler, args, amp_dtype):
    model.train(); loss_sum = mse_sum = cos_sum = 0.0; t0 = time.time()
    for _ in range(args.steps_per_epoch):
        x = gen.batch(args.batch_size); y = teacher(x)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
            pred, _ = model(x, False)
            mse = F.mse_loss(pred.float(), y.float())
            cos = F.cosine_similarity(pred.float().flatten(1), y.float().flatten(1), -1).mean()
            loss = mse + args.lambda_cos * (1 - cos)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); scaler.step(opt); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        loss_sum += float(loss.detach()); mse_sum += float(mse.detach()); cos_sum += float(cos.detach())
    n = args.steps_per_epoch
    return {"loss": loss_sum / n, "mse": mse_sum / n, "cos": cos_sum / n, "seconds": time.time() - t0}


@torch.no_grad()
def evaluate(model, teacher, gen, args, amp_dtype):
    model.eval(); mse_sum = cos_sum = 0.0; aux_list = []
    for _ in range(args.val_steps):
        x = gen.batch(args.eval_batch_size); y = teacher(x)
        with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
            pred, aux = model(x, True)
            mse = F.mse_loss(pred.float(), y.float())
            cos = F.cosine_similarity(pred.float().flatten(1), y.float().flatten(1), -1).mean()
        mse_sum += float(mse); cos_sum += float(cos); aux_list.append(aux)
    return {"mse": mse_sum / args.val_steps, "cos": cos_sum / args.val_steps}, summarize(aux_list)


def run(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    gen = StructuredSequenceGenerator(args.seq_len, args.dim, args.device, args.seed)
    teacher = FrozenTeacherMHA(args.dim, args.teacher_heads, args.seed + 7).to(device)
    model = MatrixAttentionReplacementStudent(
        args.dim, args.evidence_cells, args.task_cells, args.num_layers, args.blocks_per_layer,
        args.memory_cells, args.global_cells, args.category_temp, args.primitive_temp,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and args.device.startswith("cuda")) if args.amp != "off" else None
    ctrl = AdaptiveController(args.controller_patience, args.rollback_patience, args.controller_min_delta)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_mse", "train_cos", "val_mse", "val_cos", "score", "best_score", "mode", "action", "reward", "category_temp", "primitive_temp", "seconds"])
    best = {"score": -1e9, "epoch": 0, "val_mse": None, "val_cos": None}
    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, teacher, gen, opt, scaler, args, amp_dtype)
        val, aux = evaluate(model, teacher, gen, args, amp_dtype)
        ev = ctrl.step(model, ep, val["mse"], val["cos"], aux) if args.controller else ControllerEvent(ep, "FIXED", "noop", -val["mse"] + 0.2 * val["cos"], best["score"], 0, False, model.category_temp, model.primitive_temp, "off")
        if ev.score > best["score"]:
            best = {"score": ev.score, "epoch": ep, "val_mse": val["mse"], "val_cos": val["cos"]}
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, tr["loss"], tr["mse"], tr["cos"], val["mse"], val["cos"], ev.score, best["score"], ev.mode, ev.action, ev.reward, ev.category_temp, ev.primitive_temp, tr["seconds"]])
        analysis = {"epoch": ep, "train": tr, "val": val, "controller": asdict(ev), "best": best, "matrix_program": aux, "arch_state": model.export_arch_state()}
        write_json(out / f"analysis_epoch_{ep:03d}.json", analysis)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        print(f"epoch {ep:03d}/{args.epochs} train_mse={tr['mse']:.5f} train_cos={tr['cos']:.4f} val_mse={val['mse']:.5f} val_cos={val['cos']:.4f} score={ev.score:.5f} best={best['score']:.5f}@{best['epoch']} mode={ev.mode} action={ev.action} temp=({ev.category_temp:.2f},{ev.primitive_temp:.2f})", flush=True)
    write_json(out / "final_report.json", {"best": best, "args": vars(args), "architecture": "matrix_attention_replacement_probe_v1"})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda"); p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=128); p.add_argument("--seq-len", type=int, default=64); p.add_argument("--teacher-heads", type=int, default=4)
    p.add_argument("--evidence-cells", type=int, default=40); p.add_argument("--task-cells", type=int, default=12); p.add_argument("--num-layers", type=int, default=4); p.add_argument("--blocks-per-layer", type=int, default=4); p.add_argument("--memory-cells", type=int, default=6); p.add_argument("--global-cells", type=int, default=3)
    p.add_argument("--category-temp", type=float, default=1.4); p.add_argument("--primitive-temp", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--eval-batch-size", type=int, default=256); p.add_argument("--steps-per-epoch", type=int, default=120); p.add_argument("--val-steps", type=int, default=20); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=7e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=0.7); p.add_argument("--lambda-cos", type=float, default=0.25)
    p.add_argument("--controller", action="store_true", default=True); p.add_argument("--no-controller", dest="controller", action="store_false"); p.add_argument("--controller-patience", type=int, default=2); p.add_argument("--rollback-patience", type=int, default=2); p.add_argument("--controller-min-delta", type=float, default=0.002)
    p.add_argument("--out-dir", default="./runs/attention_replacement_probe_v1")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
