#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, random, sys, time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "attention_replacement"))
from probe_core import CATS, OPS, ROLES, MatrixAttentionReplacementStudent, StructuredSequenceGenerator, entropy  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def devtype(device: str) -> str:
    return device.split(":", 1)[0]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class FrozenTeacherBlock(nn.Module):
    def __init__(self, dim: int, heads: int, seed: int) -> None:
        super().__init__(); torch.manual_seed(seed)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward_targets(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        a, _ = self.attn(x, x, x, need_weights=False)
        attention = self.norm1(x + a)
        ffn = self.norm2(x + self.ffn(x))
        block = self.norm2(attention + self.ffn(attention))
        return {"attention": attention, "ffn": ffn, "block": block}


class MultiHeadMatrixDropIn(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.backend = MatrixAttentionReplacementStudent(
            dim=args.dim,
            evidence_cells=args.evidence_cells,
            task_cells=args.task_cells,
            layers=args.num_layers,
            blocks=args.blocks_per_layer,
            memory_cells=args.memory_cells,
            global_cells=args.global_cells,
            category_temp=args.category_temp,
            primitive_temp=args.primitive_temp,
        )
        self.heads = nn.ModuleDict({
            "attention": nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, args.dim)),
            "ffn": nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, args.dim)),
            "block": nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, args.dim)),
        })

    @property
    def category_temp(self): return self.backend.category_temp
    @property
    def primitive_temp(self): return self.backend.primitive_temp
    def set_knobs(self, *a, **kw): return self.backend.set_knobs(*a, **kw)
    def export_arch_state(self): return self.backend.export_arch_state()
    def load_arch_state(self, s): return self.backend.load_arch_state(s)
    def apply_action(self, action: str, strength: float = 0.06): return self.backend.apply_action(action, strength)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        h, aux = self.backend(x, return_aux=return_aux)
        out = {name: head(h) for name, head in self.heads.items()}
        return out, aux


class AdaptiveController:
    def __init__(self, patience=2, rollback=2, min_delta=0.002):
        self.patience, self.rollback, self.min_delta = patience, rollback, min_delta
        self.best_score, self.best_state, self.prev_score = -1e9, None, None
        self.bad, self.plateau, self.i = 0, 0, 0
        self.actions = ["explore_categories", "boost_l2_memory", "boost_l2_repair", "boost_l1_compare", "boost_aggregate_memory", "sharpen_exploit", "reduce_priors"]

    def step(self, model, epoch: int, score: float) -> Dict[str, Any]:
        reward = 0.0 if self.prev_score is None else score - self.prev_score
        accepted = score > self.best_score + self.min_delta
        mode, action, reason = "WAIT", "noop", "pending effect buffer"
        if accepted:
            self.best_score, self.best_state = score, model.export_arch_state()
            self.bad, self.plateau = 0, 0
            mode, action, reason = "EXPLOIT", "sharpen_exploit" if epoch > 3 else "noop", "new best"
        else:
            if self.prev_score is not None and score < self.prev_score - self.min_delta: self.bad += 1
            else: self.plateau += 1
            if self.bad >= self.rollback and self.best_state is not None:
                model.load_arch_state(self.best_state); action = "reduce_priors"; model.apply_action(action, 0.04)
                mode, reason, self.bad = "ROLLBACK", "score dropped; rollback best arch", 0
            elif self.plateau >= self.patience:
                action = self.actions[self.i % len(self.actions)]; self.i += 1; model.apply_action(action, 0.06)
                mode, reason, self.plateau = "EXPLORE", "plateau; soft architecture action", 0
        if action == "sharpen_exploit": model.apply_action(action, 0.03)
        self.prev_score = score
        return {"epoch": epoch, "mode": mode, "action": action, "score": score, "best_score": self.best_score, "reward": reward, "accepted": accepted, "category_temp": model.category_temp, "primitive_temp": model.primitive_temp, "reason": reason}


def summarize_aux(aux_list: List[Dict[str, Any]]) -> Dict[str, Any]:
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


def compute_losses(pred, targets, active, weights, lambda_cos):
    loss = torch.zeros((), device=next(iter(pred.values())).device)
    metrics = {}
    for name in active:
        mse = F.mse_loss(pred[name].float(), targets[name].float())
        cos = F.cosine_similarity(pred[name].float().flatten(1), targets[name].float().flatten(1), -1).mean()
        loss = loss + weights.get(name, 1.0) * (mse + lambda_cos * (1 - cos))
        metrics[f"{name}_mse"] = float(mse.detach()); metrics[f"{name}_cos"] = float(cos.detach())
    metrics["loss"] = float(loss.detach())
    return loss, metrics


def train_epoch(model, teacher, gen, opt, scaler, args, amp_dtype, active, weights):
    model.train(); sums: Dict[str, float] = {}; t0 = time.time()
    for _ in range(args.steps_per_epoch):
        x = gen.batch(args.batch_size); targets = teacher.forward_targets(x)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
            pred, _ = model(x, False); loss, met = compute_losses(pred, targets, active, weights, args.lambda_cos)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); scaler.step(opt); scaler.update()
        else:
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt.step()
        for k, v in met.items(): sums[k] = sums.get(k, 0.0) + v
    return {k: v / args.steps_per_epoch for k, v in sums.items()} | {"seconds": time.time() - t0}


@torch.no_grad()
def evaluate(model, teacher, gen, args, amp_dtype, active, weights):
    model.eval(); sums: Dict[str, float] = {}; aux_list = []
    for _ in range(args.val_steps):
        x = gen.batch(args.eval_batch_size); targets = teacher.forward_targets(x)
        with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
            pred, aux = model(x, True); _, met = compute_losses(pred, targets, active, weights, args.lambda_cos)
        for k, v in met.items(): sums[k] = sums.get(k, 0.0) + v
        aux_list.append(aux)
    return {k: v / args.val_steps for k, v in sums.items()}, summarize_aux(aux_list)


def score_from_val(val: Dict[str, float], aux: Dict[str, Any], active: List[str]) -> float:
    score = 0.0
    for name in active:
        score += -val[f"{name}_mse"] + 0.20 * val[f"{name}_cos"]
    score /= max(1, len(active))
    score += 0.005 * aux.get("route_entropy_mean", 0.0) + 0.002 * aux.get("token_block_entropy", 0.0)
    return float(score)


def run(args):
    set_seed(args.seed); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    gen = StructuredSequenceGenerator(args.seq_len, args.dim, args.device, args.seed)
    teacher = FrozenTeacherBlock(args.dim, args.teacher_heads, args.seed + 17).to(args.device)
    model = MultiHeadMatrixDropIn(args).to(args.device)
    active = [x.strip() for x in args.targets.split(",") if x.strip()]
    weights = {"attention": args.weight_attention, "ffn": args.weight_ffn, "block": args.weight_block}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp == "fp16" and args.device.startswith("cuda")) if args.amp != "off" else None
    ctrl = AdaptiveController(args.controller_patience, args.rollback_patience, args.controller_min_delta)
    fields = ["epoch", "score", "best_score", "mode", "action", "reward", "category_temp", "primitive_temp", "seconds"]
    for name in ["attention", "ffn", "block"]: fields += [f"train_{name}_mse", f"train_{name}_cos", f"val_{name}_mse", f"val_{name}_cos"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=fields).writeheader()
    best = {"score": -1e9, "epoch": 0}
    for ep in range(1, args.epochs + 1):
        train = train_epoch(model, teacher, gen, opt, scaler, args, amp_dtype, active, weights)
        val, aux = evaluate(model, teacher, gen, args, amp_dtype, active, weights)
        score = score_from_val(val, aux, active)
        event = ctrl.step(model, ep, score) if args.controller else {"epoch": ep, "mode": "FIXED", "action": "noop", "score": score, "best_score": best["score"], "reward": 0.0, "category_temp": model.category_temp, "primitive_temp": model.primitive_temp, "accepted": False}
        if score > best["score"]: best = {"score": score, "epoch": ep, "val": val}
        row = {"epoch": ep, "score": score, "best_score": best["score"], "mode": event["mode"], "action": event["action"], "reward": event["reward"], "category_temp": event["category_temp"], "primitive_temp": event["primitive_temp"], "seconds": train["seconds"]}
        for name in ["attention", "ffn", "block"]:
            row[f"train_{name}_mse"] = train.get(f"{name}_mse", ""); row[f"train_{name}_cos"] = train.get(f"{name}_cos", "")
            row[f"val_{name}_mse"] = val.get(f"{name}_mse", ""); row[f"val_{name}_cos"] = val.get(f"{name}_cos", "")
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f: csv.DictWriter(f, fieldnames=fields).writerow(row)
        analysis = {"epoch": ep, "train": train, "val": val, "score": score, "best": best, "controller": event, "matrix_program": aux, "arch_state": model.export_arch_state(), "active_targets": active}
        write_json(out / f"analysis_epoch_{ep:03d}.json", analysis)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f: f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"ep {ep:03d}/{args.epochs} score={score:.5f} best={best['score']:.5f}@{best['epoch']} mode={event['mode']} action={event['action']} val=" + " ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]), flush=True)
    write_json(out / "final_report.json", {"best": best, "args": vars(args), "architecture": "multihead_matrix_dropin_probe_v1"})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda"); p.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=128); p.add_argument("--seq-len", type=int, default=64); p.add_argument("--teacher-heads", type=int, default=4)
    p.add_argument("--evidence-cells", type=int, default=40); p.add_argument("--task-cells", type=int, default=12); p.add_argument("--num-layers", type=int, default=4); p.add_argument("--blocks-per-layer", type=int, default=4); p.add_argument("--memory-cells", type=int, default=6); p.add_argument("--global-cells", type=int, default=3)
    p.add_argument("--category-temp", type=float, default=1.4); p.add_argument("--primitive-temp", type=float, default=1.2)
    p.add_argument("--targets", default="attention,ffn,block"); p.add_argument("--weight-attention", type=float, default=1.0); p.add_argument("--weight-ffn", type=float, default=1.0); p.add_argument("--weight-block", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--eval-batch-size", type=int, default=256); p.add_argument("--steps-per-epoch", type=int, default=120); p.add_argument("--val-steps", type=int, default=20); p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=7e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=0.7); p.add_argument("--lambda-cos", type=float, default=0.25)
    p.add_argument("--controller", action="store_true", default=True); p.add_argument("--no-controller", dest="controller", action="store_false"); p.add_argument("--controller-patience", type=int, default=2); p.add_argument("--rollback-patience", type=int, default=2); p.add_argument("--controller-min-delta", type=float, default=0.002)
    p.add_argument("--out-dir", default="./runs/multihead_dropin_probe_v1_40ep")
    return p


if __name__ == "__main__": run(parser().parse_args())
