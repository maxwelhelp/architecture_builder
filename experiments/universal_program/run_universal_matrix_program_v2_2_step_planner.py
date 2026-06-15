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
from run_universal_matrix_program_v1 import (  # noqa: E402
    BASE_OPS, HFHiddenTask, devtype, losses, seed_all, write_json,
)
from run_universal_matrix_program_v2_1_balanced import (  # noqa: E402
    UniversalProgramV21, to_jsonable,
)


def _entropy(vals: List[float]) -> float:
    if not vals:
        return 0.0
    t = torch.tensor(vals, dtype=torch.float32).clamp_min(1e-8)
    t = t / t.sum().clamp_min(1e-8)
    return float((-(t * t.log()).sum() / math.log(max(2, t.numel()))).item())


def _safe_get(xs, idx, default=0.0):
    try:
        return xs[idx]
    except Exception:
        return default


@dataclass
class PlannerEvent:
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
    planner_attention_top: List[Dict[str, Any]]


class StepPlannerController:
    """Controller with matrix-like planner attention over all layer/block/step slots.

    Growth is still discrete between epochs, but the selection policy uses a dense
    attention-style scoring pass over all current program elements.
    """

    def __init__(self, args) -> None:
        self.min_delta = args.controller_min_delta
        self.patience = args.controller_patience
        self.rollback = args.rollback_patience
        self.pending_epochs = args.pending_epochs
        self.weak_reward = args.weak_action_reward
        self.prev: float | None = None
        self.absolute_best = -1e9
        self.absolute_state = None
        self.significant_best = -1e9
        self.bad = 0
        self.flat = 0
        self.cooldown: Dict[str, int] = {}
        self.pending: Dict[str, Any] | None = None
        self.history: List[Dict[str, Any]] = []
        self.last_scores: Dict[str, float] = {"add_step": 0.0, "add_depth": 0.0, "add_width": 0.0, "add_primitive": 0.0, "prune_step": 0.0}
        self.last_top: List[Dict[str, Any]] = []

    def score(self, val: Dict[str, float], aux: Dict[str, Any], active: List[str]) -> float:
        s = sum([-val[f"{n}_mse"] + 0.2 * val[f"{n}_cos"] for n in active]) / max(1, len(active))
        routes = aux.get("route_entropy_by_layer", []) or [0.0]
        s += 0.002 * float(torch.tensor(routes).float().mean())
        return float(s)

    def _cool_key(self, action: str, layer: int, block: int, step: int = -1) -> str:
        return f"{action}:L{layer}:B{block}:S{step}"

    def _cool(self, action: str, layer: int, block: int, step: int = -1) -> float:
        return 1.0 if self.cooldown.get(self._cool_key(action, layer, block, step), 0) > 0 else 0.0

    def _tick(self) -> None:
        for k in list(self.cooldown.keys()):
            self.cooldown[k] = max(0, self.cooldown[k] - 1)
            if self.cooldown[k] == 0:
                del self.cooldown[k]

    def _head_pressure(self, val: Dict[str, float], aux: Dict[str, Any]) -> List[float]:
        block_map = aux.get("block_index_map", [])
        reads = aux.get("head_block_read", {})
        delta_read = reads.get("delta", [0.0] * len(block_map))
        norm_read = reads.get("norm", [0.0] * len(block_map))
        delta_pain = float(val.get("delta_mse", 0.0)) + max(0.0, float(val.get("block_cos", 0.0)) - float(val.get("delta_cos", 0.0)))
        norm_pain = float(val.get("norm_mse", 0.0))
        out = []
        for i in range(len(block_map)):
            out.append(delta_pain * float(_safe_get(delta_read, i, 0.0)) + 0.5 * norm_pain * float(_safe_get(norm_read, i, 0.0)))
        return out

    def _grad_health(self, grad_health: Any, layer: int, block: int, step: int) -> float:
        try:
            return float(grad_health[layer][block][step])
        except Exception:
            return 0.0

    def _block_step_features(self, model: UniversalProgramV21, val: Dict[str, float], aux: Dict[str, Any], grad_health: Any) -> List[Dict[str, Any]]:
        block_map = aux.get("block_index_map", [])
        pressures = self._head_pressure(val, aux)
        op_steps = aux.get("operator_mix_by_layer_block_step", [])
        gate_steps = aux.get("gate_by_layer_block_step", [])
        route_steps = aux.get("route_entropy_by_layer_block_step", [])
        block_cos = aux.get("block_cosine_by_layer", []) or []
        feats = []
        for global_idx, bm in enumerate(block_map):
            layer = int(bm["layer_id"])
            block = int(bm["block_id"])
            order = int(bm["order_pos"])
            step_count = int(bm.get("active_steps", 1))
            pressure = float(_safe_get(pressures, global_idx, 0.0))
            block_cos_i = float(_safe_get(block_cos, order, 0.0))
            b_ops = _safe_get(_safe_get(op_steps, order, []), block, []) or []
            b_gates = _safe_get(_safe_get(gate_steps, order, []), block, []) or []
            b_routes = _safe_get(_safe_get(route_steps, order, []), block, []) or []
            op_entropy = sum(_entropy(step_ops[: len(BASE_OPS)]) for step_ops in b_ops) / max(1, len(b_ops))
            gate_mean = sum(float(x) for x in b_gates) / max(1, len(b_gates))
            route_mean = sum(float(x) for x in b_routes) / max(1, len(b_routes))
            grad_mean = sum(self._grad_health(grad_health, layer, block, s) for s in range(step_count)) / max(1, step_count)
            # Planner attention logit: this is the controller's "attention" over all existing program elements.
            planner_logit = 3.0 * pressure + 0.55 * op_entropy + 0.25 * route_mean + 0.20 * grad_mean + 0.15 * block_cos_i - 0.08 * max(0, step_count - 1)
            feats.append({
                "global_index": global_idx,
                "layer": layer,
                "block": block,
                "order": order,
                "step_count": step_count,
                "pressure": pressure,
                "op_entropy": op_entropy,
                "gate": gate_mean,
                "route_entropy": route_mean,
                "grad": grad_mean,
                "block_cos": block_cos_i,
                "planner_logit": float(planner_logit),
            })
        if feats:
            logits = torch.tensor([f["planner_logit"] for f in feats], dtype=torch.float32)
            attn = torch.softmax(logits, 0).tolist()
            for f, a in zip(feats, attn):
                f["planner_attention"] = float(a)
        return feats

    def _action_scores(self, model: UniversalProgramV21, val: Dict[str, float], aux: Dict[str, Any], feat: Dict[str, Any]) -> Dict[str, float]:
        layer = int(feat["layer"]); block = int(feat["block"]); step_count = int(feat["step_count"])
        active_layers = len(aux.get("active_layer_ids", []))
        active_blocks = int(model.core.active_blocks[layer].item())
        active_prims = int(model.core.primitive_active.sum().item())
        delta_gap = max(0.0, float(val.get("block_cos", 0.0)) - float(val.get("delta_cos", 0.0)))
        pressure = float(feat["pressure"])
        entropy = float(feat["op_entropy"])
        grad = float(feat["grad"])
        route = float(feat["route_entropy"])
        gate = float(feat["gate"])
        block_cos = float(feat["block_cos"])
        attn = float(feat.get("planner_attention", 0.0))
        low_entropy = max(0.0, 0.45 - entropy)
        scores = {
            "add_step": 0.45 * attn + 0.35 * entropy + 0.20 * pressure + 0.15 * grad + 0.08 * route - 0.22 * max(0, step_count - 1) - self._cool("add_step", layer, block, step_count),
            "add_depth": 0.55 * delta_gap + 0.30 * pressure + 0.18 * route + 0.15 * grad - 0.06 * max(0, active_layers - 4) - self._cool("add_depth", layer, block),
            "add_width": 0.40 * block_cos + 0.35 * pressure + 0.25 * attn - 0.09 * max(0, active_blocks - 4) - self._cool("add_width", layer, block),
            "add_primitive": 0.35 * low_entropy + 0.20 * grad + 0.10 * delta_gap - 0.12 * max(0, active_prims - len(BASE_OPS)) - self._cool("add_primitive", layer, block),
            "prune_step": 0.18 * max(0, step_count - 1) + 0.12 * max(0.0, 0.25 - gate) - 0.35 * pressure - 0.20 * grad - self._cool("prune_step", layer, block),
        }
        if step_count >= model.core.max_steps:
            scores["add_step"] = -999.0
        if active_layers >= model.core.max_layers:
            scores["add_depth"] = -999.0
        if active_blocks >= model.core.max_blocks:
            scores["add_width"] = -999.0
        if active_prims >= model.core.max_primitives:
            scores["add_primitive"] = -999.0
        # Primitive only after structural options are weak: new operations are expensive hypotheses.
        if max(scores["add_step"], scores["add_depth"], scores["add_width"]) > 0.12:
            scores["add_primitive"] -= 0.20
        # Do not prune if we are still strongly improving or the block has real pressure.
        if pressure > 0.01 or step_count <= 1:
            scores["prune_step"] = -999.0
        return scores

    def _choose(self, model: UniversalProgramV21, val: Dict[str, float], aux: Dict[str, Any], grad_health: Any) -> Tuple[str, Dict[str, Any], Dict[str, float], List[Dict[str, Any]]]:
        feats = self._block_step_features(model, val, aux, grad_health)
        if not feats:
            return "soft_explore", {"layer": 0, "block": 0, "order": 0, "step_count": 1}, self.last_scores, []
        ranked = sorted(feats, key=lambda f: f.get("planner_attention", 0.0), reverse=True)
        best_action = "soft_explore"; best_feat = ranked[0]; best_scores = {"add_step": -999.0, "add_depth": -999.0, "add_width": -999.0, "add_primitive": -999.0, "prune_step": -999.0}
        best_val = -999.0
        # Evaluate top planner-attended program elements, not just the best one.
        for feat in ranked[: min(8, len(ranked))]:
            scores = self._action_scores(model, val, aux, feat)
            action = max(scores, key=scores.get)
            if scores[action] > best_val:
                best_val = scores[action]; best_action = action; best_feat = feat; best_scores = scores
        top = [{k: f[k] for k in ["layer", "block", "step_count", "pressure", "op_entropy", "grad", "gate", "route_entropy", "planner_attention"] if k in f} for f in ranked[:8]]
        if best_val < -0.05:
            best_action = "soft_explore"
        return best_action, best_feat, best_scores, top

    def _do_action(self, model: UniversalProgramV21, action: str, feat: Dict[str, Any], aux: Dict[str, Any]) -> Tuple[bool, str]:
        layer = int(feat.get("layer", 0)); block = int(feat.get("block", 0)); order = int(feat.get("order", 0))
        if action == "add_step":
            return model.core.add_step(layer, block)
        if action == "add_depth":
            return model.core.add_depth(before_order_pos=order + 1)
        if action == "add_width":
            return model.core.add_width(layer)
        if action == "prune_step" and hasattr(model.core, "prune_step"):
            return model.core.prune_step(layer, block)
        if action == "add_primitive":
            rows = aux.get("operator_mix_by_layer", [])
            hint = BASE_OPS.index("delta_route")
            if order < len(rows):
                hint = int(torch.argmax(torch.tensor(rows[order][: len(BASE_OPS)]).float()).item())
            return model.core.add_primitive(layer, hint)
        model.core.soft_action("explore", layer)
        return True, "soft_explore"

    def step(self, model: UniversalProgramV21, epoch: int, val: Dict[str, float], aux: Dict[str, Any], active: List[str], grad_health: Any) -> PlannerEvent:
        self._tick()
        score = self.score(val, aux, active)
        reward = 0.0 if self.prev is None else score - self.prev
        accepted = False
        mode = "WAIT"; action = "noop"; reason = "normal"; detail = "none"
        target_layer = -1; target_block = -1; target_step = -1
        scores = self.last_scores
        top = self.last_top

        if score > self.absolute_best:
            self.absolute_best = score
            self.absolute_state = model.export_arch_state()

        # Evaluate a pending action after a short buffer.
        if self.pending is not None and epoch - int(self.pending["epoch"]) >= self.pending_epochs:
            pr = score - float(self.pending["score_before"])
            rec = dict(self.pending)
            rec.update({"epoch_eval": epoch, "score_after": score, "reward": pr})
            self.history.append(rec)
            layer = int(self.pending["layer"]); block = int(self.pending["block"]); step = int(self.pending.get("step", -1)); act = str(self.pending["action"])
            key = self._cool_key(act, layer, block, step)
            if pr < -self.min_delta and self.pending.get("state_before") is not None:
                model.load_arch_state(self.pending["state_before"])
                self.cooldown[key] = 3
                mode = "ROLLBACK"; action = "rollback_rejected_growth"; reason = "pending_action_hurt"; detail = f"rejected_{act}_reward_{pr:.5g}"
                self.pending = None; self.prev = score
                return PlannerEvent(epoch, mode, action, score, self.absolute_best, reward, False, -1, -1, -1, reason, detail, **self._scores_for_event(scores), planner_attention_top=top)
            if pr < self.weak_reward:
                self.cooldown[key] = 2
            self.pending = None

        if score > self.significant_best + self.min_delta:
            self.significant_best = score
            accepted = True
            self.flat = 0; self.bad = 0
            mode = "EXPLOIT"; action = "sharpen"; reason = "significant_best"
            model.core.soft_action("sharpen")
        else:
            if self.prev is not None and score < self.prev - self.min_delta:
                self.bad += 1
            else:
                self.flat += 1
            if self.bad >= self.rollback and self.absolute_state is not None:
                model.load_arch_state(self.absolute_state)
                model.core.soft_action("reduce_priors")
                mode = "ROLLBACK"; action = "rollback_absolute_best"; reason = "score_drop"; self.bad = 0
            elif self.flat >= self.patience:
                action, feat, scores, top = self._choose(model, val, aux, grad_health)
                self.last_scores = scores; self.last_top = top
                state_before = model.export_arch_state()
                ok, detail = self._do_action(model, action, feat, aux)
                target_layer = int(feat.get("layer", -1)); target_block = int(feat.get("block", -1)); target_step = int(feat.get("step_count", -1))
                if not ok:
                    action = "soft_explore"; model.core.soft_action("explore", target_layer); detail = "fallback_soft_explore"
                if action.startswith("add") or action.startswith("prune"):
                    self.pending = {"epoch": epoch, "action": action, "layer": target_layer, "block": target_block, "step": target_step, "score_before": score, "state_before": state_before, "detail": detail, "scores": scores}
                mode = "GROW" if action.startswith("add") or action.startswith("prune") else "EXPLORE"
                reason = "plateau_step_planner_attention"
                self.flat = 0
        self.prev = score
        return PlannerEvent(epoch, mode, action, score, self.absolute_best, reward, accepted, target_layer, target_block, target_step, reason, detail, **self._scores_for_event(scores), planner_attention_top=top)

    def _scores_for_event(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {
            "step_score": float(scores.get("add_step", 0.0)),
            "depth_score": float(scores.get("add_depth", 0.0)),
            "width_score": float(scores.get("add_width", 0.0)),
            "primitive_score": float(scores.get("add_primitive", 0.0)),
            "prune_score": float(scores.get("prune_step", 0.0)),
        }


def _grad_snapshot(model: UniversalProgramV21) -> Any:
    with torch.no_grad():
        g1 = model.core.cat_logits.grad
        g2 = model.core.role_logits.grad
        if g1 is None and g2 is None:
            return None
        out = 0.0
        if g1 is not None:
            out = out + g1.detach().float().pow(2).sum(-1).sqrt()
        if g2 is not None:
            out = out + g2.detach().float().pow(2).sum(-1).sqrt()
        return out.cpu().tolist()


def run(args) -> None:
    seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    task = HFHiddenTask(args.model_name, args.layer_idx, args.device)
    model = UniversalProgramV21(task.dim, args).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    active = [x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl = StepPlannerController(args)
    fields = ["epoch","score","best_score","mode","action","target_layer","target_block","target_step","reason","growth_detail","step_score","depth_score","width_score","primitive_score","prune_score","seconds"] + [f"train_{n}_{k}" for n in active for k in ("mse","cos")] + [f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    for ep in range(1, args.epochs + 1):
        model.train(); sums: Dict[str, float] = {}; t0 = time.time(); last_grad = None
        for _ in range(args.steps_per_epoch):
            b = task.sample(args.batch_size, args.seq_len, args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                pred, _ = model(b["x"])
                loss, met = losses(pred, b, active, args.lambda_cos)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            last_grad = _grad_snapshot(model)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            for k, v in met.items():
                sums[k] = sums.get(k, 0.0) + v
        train = {k: v / args.steps_per_epoch for k, v in sums.items()} | {"seconds": time.time() - t0}
        model.eval(); sums = {}; last_aux = None
        with torch.no_grad():
            for _ in range(args.val_steps):
                b = task.sample(args.eval_batch_size, args.seq_len, args.device)
                with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                    pred, aux = model(b["x"])
                    _, met = losses(pred, b, active, args.lambda_cos)
                for k, v in met.items():
                    sums[k] = sums.get(k, 0.0) + v
                last_aux = aux
        val = {k: v / args.val_steps for k, v in sums.items()}
        aux = to_jsonable(last_aux or {})
        ev = ctrl.step(model, ep, val, aux, active, last_grad)
        row = {"epoch": ep, "score": ev.score, "best_score": ev.best_score, "mode": ev.mode, "action": ev.action, "target_layer": ev.target_layer, "target_block": ev.target_block, "target_step": ev.target_step, "reason": ev.reason, "growth_detail": ev.growth_detail, "step_score": ev.step_score, "depth_score": ev.depth_score, "width_score": ev.width_score, "primitive_score": ev.primitive_score, "prune_score": ev.prune_score, "seconds": train["seconds"]}
        for n in active:
            row[f"train_{n}_mse"] = train.get(f"{n}_mse", ""); row[f"train_{n}_cos"] = train.get(f"{n}_cos", "")
            row[f"val_{n}_mse"] = val.get(f"{n}_mse", ""); row[f"val_{n}_cos"] = val.get(f"{n}_cos", "")
        with (out / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        report = {"epoch": ep, "train": train, "val": val, "controller": asdict(ev), "action_history": ctrl.history, "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending), "grad_health_last_batch": last_grad, "matrix_program": aux, "arch_state": model.export_arch_state(), "args": vars(args)}
        write_json(out / f"analysis_epoch_{ep:03d}.json", report)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": asdict(ev), "history_tail": ctrl.history[-3:], "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending)}, ensure_ascii=False) + "\n")
        print(f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} target=L{ev.target_layer}B{ev.target_block}S{ev.target_step} scores(step={ev.step_score:.3f},depth={ev.depth_score:.3f},width={ev.width_score:.3f},prim={ev.primitive_score:.3f},prune={ev.prune_score:.3f}) {ev.growth_detail} " + " ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]), flush=True)
    write_json(out / "final_report.json", {"architecture": "universal_matrix_program_v2_2_step_planner", "final_arch_state": model.export_arch_state(), "action_history": ctrl.history, "args": vars(args)})


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
    p.add_argument("--out-dir", default="./runs/universal_matrix_program_v2_2_step_planner_hf_45ep")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
