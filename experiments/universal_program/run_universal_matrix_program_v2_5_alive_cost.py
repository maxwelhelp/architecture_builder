#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from run_universal_matrix_program_v1 import (  # noqa: E402
    BASE_OPS, CATS, HEAD_OPS, ROLES,
    HFHiddenTask, devtype, ent, losses, seed_all, write_json,
)
from run_universal_matrix_program_v2_1_balanced import TraceMatrixHead, TraceStepCore, to_jsonable  # noqa: E402
from run_universal_matrix_program_v2_3_cleanup import CleanupPlannerController, CleanupEvent  # noqa: E402
from run_universal_matrix_program_v2_4_descriptor_alive import (  # noqa: E402
    PRIMITIVE_DESC, _need_vector, _primitive_scores_from_need, _category_scores_from_need,
)
from run_universal_matrix_program_v2_2_step_planner import _grad_snapshot  # noqa: E402


# Rough relative compute/complexity costs. These are not wall-clock exact; they are a
# weak differentiable pressure toward simpler programs.
PRIMITIVE_COST = {
    "identity": 0.05,
    "evidence_read": 0.35,
    "task_read": 0.25,
    "route_read": 0.55,
    "delta_route": 0.45,
    "compare_mul": 0.50,
    "memory_read": 0.65,
    "memory_write": 0.70,
    "global_summary": 0.30,
    "suppress_global": 0.35,
    "normalize": 0.25,
    "energy_count": 0.25,
    "residual_repair": 0.20,
    "aggregate_merge": 0.45,
}


class AliveCostCore(TraceStepCore):
    """Trace core with differentiable alive gates and primitive cost logging.

    Discrete add_depth/add_width/add_step still activates slots between epochs, but
    every active block/step/primitive now has a soft alive gate in the forward path.
    This makes pruning/cleanup differentiable instead of only controller-discrete.
    """

    def __init__(self, dim: int, args) -> None:
        super().__init__(dim, args)
        # Start near 1.0 alive, not exactly 1.0, so gradients can move it.
        init_alive = float(args.alive_init_logit)
        self.step_alive_logits = nn.Parameter(torch.full((self.max_layers, self.max_blocks, self.max_steps), init_alive))
        self.block_alive_logits = nn.Parameter(torch.full((self.max_layers, self.max_blocks), init_alive))
        self.primitive_alive_logits = nn.Parameter(torch.full((self.max_primitives,), init_alive))
        cost = torch.ones(self.max_primitives) * 0.60
        for i, name in enumerate(self.primitive_names[:len(BASE_OPS)]):
            cost[i] = float(PRIMITIVE_COST.get(name, 0.60))
        if self.max_primitives > len(BASE_OPS):
            cost[len(BASE_OPS):] = float(args.candidate_primitive_cost)
        self.register_buffer("primitive_cost", cost)
        self.alive_floor = float(args.alive_floor)

    def _alive(self, logits: torch.Tensor) -> torch.Tensor:
        return self.alive_floor + (1.0 - self.alive_floor) * torch.sigmoid(logits)

    def export_arch_state(self) -> Dict[str, Any]:
        s = super().export_arch_state()
        s.update({
            "step_alive_logits": self.step_alive_logits.detach().cpu().tolist(),
            "block_alive_logits": self.block_alive_logits.detach().cpu().tolist(),
            "primitive_alive_logits": self.primitive_alive_logits.detach().cpu().tolist(),
        })
        return s

    def load_arch_state(self, s: Dict[str, Any]) -> None:
        super().load_arch_state(s)
        with torch.no_grad():
            if "step_alive_logits" in s:
                self.step_alive_logits.copy_(torch.tensor(s["step_alive_logits"], device=self.step_alive_logits.device))
            if "block_alive_logits" in s:
                self.block_alive_logits.copy_(torch.tensor(s["block_alive_logits"], device=self.block_alive_logits.device))
            if "primitive_alive_logits" in s:
                self.primitive_alive_logits.copy_(torch.tensor(s["primitive_alive_logits"], device=self.primitive_alive_logits.device))

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:  # noqa: C901
        B, N, D = x.shape
        ev = self.evidence(x) + self.space_emb[1][None, None]
        task = self.task[None].expand(B, -1, -1) + self.space_emb[2][None, None]
        mem = self.memory0[None].expand(B, -1, -1) + self.space_emb[4][None, None]
        glob = self.global0[None].expand(B, -1, -1) + self.space_emb[5][None, None]
        base = ev.mean(1)
        prev = base[:, None, :]
        blocks_all: List[torch.Tensor] = []
        role_logs=[]; cat_logs=[]; op_logs=[]; route_logs=[]; gate_logs=[]; block_cos_logs=[]; step_logs=[]
        per_step_ops=[]; per_step_cats=[]; per_step_gates=[]; per_step_routes=[]; per_step_alive=[]; block_alive_logs=[]
        block_index_map=[]; cost_terms=[]; alive_terms=[]; route_entropy_terms=[]

        primitive_alive = self._alive(self.primitive_alive_logits).to(x.device, x.dtype)
        primitive_cost = self.primitive_cost.to(x.device, x.dtype)
        primitive_active = self.primitive_active.to(x.device).view(1, 1, -1)

        for order_pos, lid in enumerate(self.active_layer_ids):
            nb = int(self.active_blocks[lid].item())
            if nb <= 0:
                continue
            h0 = prev[:, :nb] if prev.shape[1] >= nb else base[:, None].expand(B, nb, D)
            b_alive = self._alive(self.block_alive_logits[lid, :nb]).to(x.device, x.dtype)
            h = self.norm(h0 + self.block_addr[lid, :nb][None] + self.layer_pos_emb[lid][None, None] + self.block_pos_emb[:nb][None])
            max_active_steps = int(self.active_steps[lid, :nb].max().item())
            layer_role=[]; layer_cat=[]; layer_op=[]; layer_route=[]; layer_gate=[]
            layer_step_ops=[[] for _ in range(nb)]
            layer_step_cats=[[] for _ in range(nb)]
            layer_step_gates=[[] for _ in range(nb)]
            layer_step_routes=[[] for _ in range(nb)]
            layer_step_alive=[[] for _ in range(nb)]

            for step in range(max_active_steps):
                active_mask = (self.active_steps[lid, :nb] > step).to(x.device).view(1, nb, 1)
                if not bool(active_mask.any()):
                    continue
                hs = h + self.step_emb[step][None, None]
                operands, route, evc, tc, ra = self._ops(hs, ev, task, mem, glob, prev)
                role = torch.softmax(self.role_logits[lid, :nb, step].float(), -1).to(x.dtype)
                cat = torch.softmax(((self.cat_logits[lid, :nb, step] + self.cat_prior[lid, :nb, step]) / self.category_temp).float(), -1).to(x.dtype)
                cat_prior = cat @ self.cat_op.to(x.device, x.dtype)
                logits = (hs @ self.op_addr.t()) / math.sqrt(D) + self.op_bias(torch.cat([hs, route, evc, tc], -1)) + self.op_prior[lid, :nb, step][None]
                # Differentiable primitive alive: inactive hard-mask stays hard, but active primitives can be softly reduced.
                logits = logits + torch.log(primitive_alive.view(1, 1, -1).clamp_min(1e-5))
                logits = logits.masked_fill(~primitive_active, -1e4)
                prim = torch.softmax((logits / self.primitive_temp).float(), -1).to(x.dtype)
                op = prim * cat_prior.clamp_min(1e-6)
                op = op / op.sum(-1, keepdim=True).clamp_min(1e-6)
                upd = (op[..., None] * operands).sum(2)
                g = torch.sigmoid(self.gate(torch.cat([hs, upd, route, tc], -1))).squeeze(-1)
                s_alive = self._alive(self.step_alive_logits[lid, :nb, step]).to(x.device, x.dtype)
                alive = (s_alive * b_alive).view(1, nb)
                update_scale = alive * (0.10 + 0.90 * g)
                h_new = self.norm(h + update_scale.unsqueeze(-1) * upd)
                h = torch.where(active_mask.bool(), h_new, h)

                # Differentiable efficiency losses/logs.
                expected_cost = (op.float() * primitive_cost.float().view(1, 1, -1)).sum(-1) * alive.float()
                cost_terms.append(expected_cost.mean())
                alive_terms.append(alive.float().mean())
                route_entropy_terms.append(ent(ra.detach().float(), -1).mean())

                op_mean = op.detach().mean(0).cpu()
                cat_cpu = cat.detach().cpu()
                gate_cpu = g.detach().mean(0).cpu()
                route_cpu = ent(ra.detach().float(), -1).mean(0).cpu()
                alive_cpu = alive.detach().mean(0).cpu()
                for b in range(nb):
                    if int(self.active_steps[lid, b].item()) > step:
                        layer_step_ops[b].append(op_mean[b])
                        layer_step_cats[b].append(cat_cpu[b])
                        layer_step_gates[b].append(gate_cpu[b])
                        layer_step_routes[b].append(route_cpu[b])
                        layer_step_alive[b].append(alive_cpu[b])
                layer_role.append(role.detach().mean(0).cpu())
                layer_cat.append(cat.detach().mean(0).cpu())
                layer_op.append(op.detach().mean((0, 1)).cpu())
                layer_route.append(ent(ra.detach().float(), -1).mean().cpu())
                layer_gate.append(g.detach().mean(0).cpu())

            # A soft block alive gate also controls how much the layer writes forward.
            h = h * b_alive.view(1, nb, 1) + h0[:, :nb] * (1.0 - b_alive.view(1, nb, 1)) if h0.shape[1] >= nb else h
            blocks_all.append(h)
            for b in range(nb):
                block_index_map.append({
                    "global_index": len(block_index_map),
                    "layer_id": int(lid),
                    "order_pos": int(order_pos),
                    "block_id": int(b),
                    "active_steps": int(self.active_steps[lid, b].item()),
                    "block_alive": float(b_alive.detach().cpu()[b]),
                })
            prev = h
            glob = glob + 0.05 * h.mean(1, keepdim=True).expand_as(glob)
            mem = mem + 0.03 * h.mean(1, keepdim=True).expand_as(mem)
            with torch.no_grad():
                flat = F.normalize(h.mean(0), dim=-1)
                cos = flat @ flat.T
                mask2 = ~torch.eye(nb, dtype=torch.bool, device=cos.device)
                bc = cos[mask2].abs().mean() if nb > 1 else torch.tensor(0.0, device=cos.device)

            role_logs.append(torch.stack(layer_role).mean(0) if layer_role else torch.zeros(len(ROLES)))
            cat_logs.append(torch.stack(layer_cat).mean(0) if layer_cat else torch.zeros(len(CATS)))
            op_logs.append(torch.stack(layer_op).mean(0) if layer_op else torch.zeros(self.max_primitives))
            route_logs.append(torch.stack(layer_route).mean(0) if layer_route else torch.tensor(0.0))
            gate_logs.append(torch.stack(layer_gate).mean(0) if layer_gate else torch.zeros(nb))
            block_cos_logs.append(bc.cpu())
            block_alive_logs.append([float(v) for v in b_alive.detach().cpu().tolist()])
            step_logs.append([int(x) for x in self.active_steps[lid, :nb].cpu().tolist()])
            per_step_ops.append([[v.tolist() for v in block_steps] for block_steps in layer_step_ops])
            per_step_cats.append([[v.tolist() for v in block_steps] for block_steps in layer_step_cats])
            per_step_gates.append([[float(v) for v in block_steps] for block_steps in layer_step_gates])
            per_step_routes.append([[float(v) for v in block_steps] for block_steps in layer_step_routes])
            per_step_alive.append([[float(v) for v in block_steps] for block_steps in layer_step_alive])

        blocks = torch.cat(blocks_all, 1) if blocks_all else prev
        expected_cost = torch.stack(cost_terms).mean() if cost_terms else x.new_tensor(0.0)
        alive_mean = torch.stack(alive_terms).mean() if alive_terms else x.new_tensor(0.0)
        aux = {
            "active_layer_ids": list(self.active_layer_ids),
            "active_blocks": {str(i): int(self.active_blocks[i]) for i in self.active_layer_ids},
            "active_steps": {str(i): [int(x) for x in self.active_steps[i, :int(self.active_blocks[i])].cpu().tolist()] for i in self.active_layer_ids},
            "active_primitives": [self.primitive_names[i] for i, v in enumerate(self.primitive_active.tolist()) if v],
            "primitive_names": self.primitive_names,
            "block_index_map": block_index_map,
            "role_mix_by_layer": torch.stack(role_logs) if role_logs else torch.zeros(0, len(ROLES)),
            "category_mix_by_layer": torch.stack(cat_logs) if cat_logs else torch.zeros(0, len(CATS)),
            "operator_mix_by_layer": torch.stack(op_logs) if op_logs else torch.zeros(0, self.max_primitives),
            "route_entropy_by_layer": torch.stack(route_logs) if route_logs else torch.zeros(0),
            "gate_by_layer_block": [x.tolist() for x in gate_logs],
            "block_cosine_by_layer": [float(x) for x in block_cos_logs],
            "step_count_by_layer": step_logs,
            "operator_mix_by_layer_block_step": per_step_ops,
            "category_mix_by_layer_block_step": per_step_cats,
            "gate_by_layer_block_step": per_step_gates,
            "route_entropy_by_layer_block_step": per_step_routes,
            "step_alive_by_layer_block_step": per_step_alive,
            "block_alive_by_layer": block_alive_logs,
            "primitive_alive": [float(x) for x in primitive_alive.detach().cpu().tolist()],
            "primitive_cost": [float(x) for x in self.primitive_cost.detach().cpu().tolist()],
            "expected_cost": expected_cost,
            "alive_mean": alive_mean,
            "attention_debug": {
                "route_entropy_mean": float(torch.stack(route_entropy_terms).mean().detach().cpu()) if route_entropy_terms else 0.0,
                "cost_mean": float(expected_cost.detach().cpu()),
                "alive_mean": float(alive_mean.detach().cpu()),
            },
        }
        return {"blocks": blocks, "evidence": ev, "task": task, "memory": mem, "global": glob, "aux": aux}


class UniversalProgramV25(nn.Module):
    def __init__(self, dim: int, args) -> None:
        super().__init__()
        self.core = AliveCostCore(dim, args)
        self.delta_head = TraceMatrixHead(dim)
        self.norm_head = TraceMatrixHead(dim)

    def export_arch_state(self):
        return self.core.export_arch_state()

    def load_arch_state(self, s):
        return self.core.load_arch_state(s)

    def forward(self, x):
        core = self.core(x)
        delta, dop, dread = self.delta_head(x, core)
        block = x + delta
        norm, nop, nread = self.norm_head(block, core)
        core["aux"]["head_operator_mix"] = {
            "delta": {HEAD_OPS[i]: float(v) for i, v in enumerate(dop)},
            "norm": {HEAD_OPS[i]: float(v) for i, v in enumerate(nop)},
        }
        core["aux"]["head_block_read"] = {
            "delta": [float(v) for v in dread],
            "norm": [float(v) for v in nread],
        }
        return {"block": block, "delta": delta, "norm": norm}, core["aux"]


class AliveCostController(CleanupPlannerController):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.max_epochs = int(args.epochs)
        self.cleanup_start = float(args.cleanup_start_frac)
        self.late_growth_start = float(args.late_growth_start_frac)
        self.late_growth_penalty = float(args.late_growth_penalty)
        self.soft_prune_strength = float(args.soft_prune_strength)
        self.descriptor_boost = float(args.descriptor_action_boost)
        self.epoch_frac = 0.0

    def _action_scores(self, model: UniversalProgramV25, val: Dict[str, float], feat: Dict[str, Any], aux: Dict[str, Any]) -> Dict[str, float]:
        scores = super()._action_scores(model, val, feat, aux)
        need = _need_vector(val, feat)
        desc_mag = min(1.0, sum(abs(x) for x in need.values()))
        consistency = float(feat.get("category_consistency", 0.0))
        pressure = float(feat.get("pressure", 0.0))
        grad = float(feat.get("grad", 0.0))
        layer = int(feat.get("layer", 0)); block = int(feat.get("block", 0)); step_count = int(feat.get("step_count", 1))
        scores["descriptor_repair"] = 0.20 * desc_mag + 0.22 * max(0.0, 0.55 - consistency) + 0.05 * pressure + 0.05 * grad - self._cool("descriptor_repair", layer, block, step_count)
        # Soft prune is allowed even for single-step blocks: it lowers alive gates, not hard deletes.
        gate = float(feat.get("gate", 0.0))
        block_cos = float(feat.get("block_cos", 0.0))
        scores["soft_prune"] = 0.10 * max(0.0, 0.50 - gate) + 0.12 * max(0.0, block_cos - 0.75) + 0.10 * max(0.0, 0.35 - consistency) - 0.10 * pressure - 0.08 * grad - self._cool("soft_prune", layer, block, step_count)
        # Cost pressure: when cost is high, prefer cleanup over adding new structure.
        cost = float(aux.get("attention_debug", {}).get("cost_mean", 0.0))
        if cost > 0.45:
            scores["soft_prune"] += 0.06 * (cost - 0.45)
            scores["descriptor_repair"] += 0.04 * (cost - 0.45)
        if self.epoch_frac >= self.cleanup_start:
            scores["descriptor_repair"] += self.descriptor_boost
            scores["repair_category"] += 0.05
            scores["soft_prune"] += 0.08
        if self.epoch_frac >= self.late_growth_start:
            scores["add_width"] -= self.late_growth_penalty
            scores["add_depth"] -= self.late_growth_penalty
            scores["add_step"] -= 0.35 * self.late_growth_penalty
        # If descriptor says old primitives match the need, avoid growing new primitive.
        prim_scores = _primitive_scores_from_need(need)
        if float(prim_scores.max()) > 0.5:
            scores["add_primitive"] -= 0.10
        return scores

    def _descriptor_repair(self, model: UniversalProgramV25, feat: Dict[str, Any], val: Dict[str, float]) -> Tuple[bool, str]:
        layer = int(feat.get("layer", 0)); block = int(feat.get("block", 0))
        steps = int(model.core.active_steps[layer, block].item())
        need = _need_vector(val, feat)
        op_score = _primitive_scores_from_need(need).to(model.core.op_prior.device)
        cat_score = _category_scores_from_need(need).to(model.core.cat_prior.device)
        with torch.no_grad():
            for s in range(max(1, steps)):
                model.core.op_prior[layer, block, s, :len(BASE_OPS)] += 0.08 * op_score
                model.core.cat_prior[layer, block, s, :] += 0.06 * cat_score
        names = [BASE_OPS[i] for i in torch.topk(op_score, k=min(3, op_score.numel())).indices.tolist()]
        return True, f"descriptor_repair_L{layer}B{block}_steps_{steps}_ops_{','.join(names)}"

    def _soft_prune(self, model: UniversalProgramV25, feat: Dict[str, Any]) -> Tuple[bool, str]:
        layer = int(feat.get("layer", 0)); block = int(feat.get("block", 0))
        steps = int(model.core.active_steps[layer, block].item())
        if steps <= 0:
            return False, "no_active_steps_to_soft_prune"
        target_step = max(0, steps - 1)
        with torch.no_grad():
            # Prefer pruning extra step, otherwise lower block alive a little.
            if steps > 1:
                model.core.step_alive_logits[layer, block, target_step] -= self.soft_prune_strength
                return True, f"soft_pruned_step_alive_L{layer}B{block}S{target_step}"
            model.core.block_alive_logits[layer, block] -= 0.5 * self.soft_prune_strength
            return True, f"soft_pruned_block_alive_L{layer}B{block}"

    def _do_action(self, model: UniversalProgramV25, action: str, feat: Dict[str, Any], aux: Dict[str, Any]) -> Tuple[bool, str]:
        if action == "descriptor_repair":
            return self._descriptor_repair(model, feat, self._last_val)
        if action == "soft_prune":
            return self._soft_prune(model, feat)
        return super()._do_action(model, action, feat, aux)

    def step(self, model: UniversalProgramV25, epoch: int, val: Dict[str, float], aux: Dict[str, Any], active: List[str], grad_health: Any) -> CleanupEvent:
        self._last_val = val
        self.epoch_frac = float(epoch) / max(1, self.max_epochs)
        return super().step(model, epoch, val, aux, active, grad_health)


def run(args) -> None:
    seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    task = HFHiddenTask(args.model_name, args.layer_idx, args.device)
    model = UniversalProgramV25(task.dim, args).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and args.device.startswith("cuda"))
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    active = [x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl = AliveCostController(args)
    fields = [
        "epoch","score","best_score","mode","action","target_layer","target_block","target_step","reason","growth_detail",
        "step_score","depth_score","width_score","primitive_score","prune_score","repair_category_score","descriptor_repair_score","soft_prune_score","category_consistency","expected_cost","alive_mean","seconds",
    ] + [f"train_{n}_{k}" for n in active for k in ("mse","cos")] + [f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for ep in range(1, args.epochs + 1):
        model.train(); sums: Dict[str, float] = {}; t0 = time.time(); last_grad = None; last_aux_train = None
        for _ in range(args.steps_per_epoch):
            b = task.sample(args.batch_size, args.seq_len, args.device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device), dtype=amp_dtype, enabled=args.amp != "off" and args.device.startswith("cuda")):
                pred, aux = model(b["x"])
                loss, met = losses(pred, b, active, args.lambda_cos)
                cost_loss = aux.get("expected_cost", torch.tensor(0.0, device=loss.device))
                alive_loss = aux.get("alive_mean", torch.tensor(0.0, device=loss.device))
                loss = loss + args.lambda_op_cost * cost_loss + args.lambda_alive_sparsity * alive_loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            last_grad = _grad_snapshot(model)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            last_aux_train = aux
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
        val = {k: v / args.val_steps for k, v in sums.items()}
        aux_json = to_jsonable(last_aux or {})
        ev = ctrl.step(model, ep, val, aux_json, active, last_grad)
        expected_cost = float(aux_json.get("expected_cost", 0.0)) if isinstance(aux_json, dict) else 0.0
        alive_mean = float(aux_json.get("alive_mean", 0.0)) if isinstance(aux_json, dict) else 0.0
        scores = ctrl.last_scores or {}
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
            "expected_cost": expected_cost,
            "alive_mean": alive_mean,
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
            "matrix_program": aux_json, "arch_state": model.export_arch_state(), "primitive_cost": PRIMITIVE_COST,
            "args": vars(args),
        }
        write_json(out / f"analysis_epoch_{ep:03d}.json", report)
        with (out / "controller_state.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": asdict(ev), "scores": scores, "history_tail": ctrl.history[-3:], "cooldown": ctrl.cooldown, "pending": to_jsonable(ctrl.pending)}, ensure_ascii=False) + "\n")
        print(
            f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} "
            f"target=L{ev.target_layer}B{ev.target_block} scores(step={ev.step_score:.3f},depth={ev.depth_score:.3f},width={ev.width_score:.3f},"
            f"prim={ev.primitive_score:.3f},prune={ev.prune_score:.3f},softprune={float(scores.get('soft_prune',0.0)):.3f},"
            f"repair={ev.repair_category_score:.3f},desc={float(scores.get('descriptor_repair',0.0)):.3f},cost={expected_cost:.3f},alive={alive_mean:.3f}) {ev.growth_detail} "
            + " ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]),
            flush=True,
        )
    write_json(out / "final_report.json", {"architecture": "universal_matrix_program_v2_5_alive_cost", "final_arch_state": model.export_arch_state(), "action_history": ctrl.history, "args": vars(args)})


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
    p.add_argument("--cleanup-start-frac", type=float, default=0.62); p.add_argument("--late-growth-start-frac", type=float, default=0.78); p.add_argument("--late-growth-penalty", type=float, default=0.25)
    p.add_argument("--descriptor-action-boost", type=float, default=0.08); p.add_argument("--soft-prune-strength", type=float, default=0.45)
    p.add_argument("--alive-init-logit", type=float, default=2.2); p.add_argument("--alive-floor", type=float, default=0.05)
    p.add_argument("--lambda-op-cost", type=float, default=0.004); p.add_argument("--lambda-alive-sparsity", type=float, default=0.001)
    p.add_argument("--candidate-primitive-cost", type=float, default=0.85)
    p.add_argument("--out-dir", default="./runs/universal_matrix_program_v2_5_alive_cost_hf_45ep")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
