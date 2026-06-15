#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse task/io helpers and constants from v1, but replace the core with
# step-slots: layer -> block -> step -> operator_mix.
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from run_universal_matrix_program_v1 import (  # noqa: E402
    BASE_OPS, CATS, HEAD_OPS, ROLES,
    HFHiddenTask, MatrixHead, UniversalEvidence,
    devtype, ent, losses, seed_all, write_json,
)


def _mean_or_zero(xs: List[torch.Tensor], shape: Tuple[int, ...]) -> torch.Tensor:
    if not xs:
        return torch.zeros(*shape)
    return torch.stack(xs).mean(0)


class StepMatrixCore(nn.Module):
    def __init__(self, dim: int, args) -> None:
        super().__init__()
        self.dim = dim
        self.max_layers = args.max_layers
        self.max_blocks = args.max_blocks_per_layer
        self.max_steps = args.max_steps_per_block
        self.max_primitives = args.max_primitives
        self.base_ops = list(BASE_OPS)
        self.primitive_names = self.base_ops + [f"candidate_{i}" for i in range(self.max_primitives - len(self.base_ops))]
        self.active_layer_ids: List[int] = list(range(args.init_layers))
        self.unused_layer_ids: List[int] = list(range(args.init_layers, self.max_layers))
        self.register_buffer("active_blocks", torch.zeros(self.max_layers, dtype=torch.long))
        self.active_blocks[:args.init_layers] = args.init_blocks_per_layer
        self.register_buffer("active_steps", torch.zeros(self.max_layers, self.max_blocks, dtype=torch.long))
        self.active_steps[:args.init_layers, :args.init_blocks_per_layer] = args.init_steps_per_block
        self.register_buffer("primitive_active", torch.zeros(self.max_primitives, dtype=torch.bool))
        self.primitive_active[:len(self.base_ops)] = True
        self.category_temp = float(args.category_temp)
        self.primitive_temp = float(args.primitive_temp)

        self.evidence = UniversalEvidence(dim, args.evidence_cells)
        self.task = nn.Parameter(torch.randn(args.task_cells, dim) * 0.02)
        self.memory0 = nn.Parameter(torch.randn(args.memory_cells, dim) * 0.02)
        self.global0 = nn.Parameter(torch.randn(args.global_cells, dim) * 0.02)
        self.block_addr = nn.Parameter(torch.randn(self.max_layers, self.max_blocks, dim) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(self.max_steps, dim) * 0.02)
        self.space_emb = nn.Parameter(torch.randn(6, dim) * 0.02)
        self.layer_pos_emb = nn.Parameter(torch.randn(self.max_layers, dim) * 0.02)
        self.block_pos_emb = nn.Parameter(torch.randn(self.max_blocks, dim) * 0.02)

        self.qr = nn.Linear(dim, dim, bias=False); self.kr = nn.Linear(dim, dim, bias=False); self.vr = nn.Linear(dim, dim, bias=False)
        self.qe = nn.Linear(dim, dim, bias=False); self.ke = nn.Linear(dim, dim, bias=False); self.ve = nn.Linear(dim, dim, bias=False)
        self.qt = nn.Linear(dim, dim, bias=False); self.kt = nn.Linear(dim, dim, bias=False); self.vt = nn.Linear(dim, dim, bias=False)
        self.qm = nn.Linear(dim, dim, bias=False); self.km = nn.Linear(dim, dim, bias=False); self.vm = nn.Linear(dim, dim, bias=False)
        self.role_logits = nn.Parameter(torch.zeros(self.max_layers, self.max_blocks, self.max_steps, len(ROLES)))
        self.cat_logits = nn.Parameter(torch.zeros(self.max_layers, self.max_blocks, self.max_steps, len(CATS)))
        self.op_addr = nn.Parameter(torch.randn(self.max_primitives, dim) * 0.02)
        self.op_bias = nn.Linear(dim * 4, self.max_primitives)
        self.gate = nn.Linear(dim * 4, 1)
        self.norm = nn.LayerNorm(dim)
        self.candidate_mix = nn.Parameter(torch.randn(max(1, self.max_primitives - len(self.base_ops)), len(self.base_ops)) * 0.02)
        self.register_buffer("cat_prior", torch.zeros(self.max_layers, self.max_blocks, self.max_steps, len(CATS)))
        self.register_buffer("op_prior", torch.zeros(self.max_layers, self.max_blocks, self.max_steps, self.max_primitives))

        cat_op = torch.zeros(len(CATS), self.max_primitives)
        def add(cat: str, ops: Tuple[str, ...]) -> None:
            for op in ops:
                cat_op[CATS.index(cat), self.primitive_names.index(op)] = 1.0
        add("extract", ("evidence_read", "delta_route", "energy_count", "identity"))
        add("compare", ("compare_mul", "delta_route", "route_read", "normalize", "residual_repair"))
        add("memory", ("memory_read", "memory_write", "aggregate_merge", "identity"))
        add("repair", ("residual_repair", "suppress_global", "normalize", "delta_route", "compare_mul"))
        add("aggregate", ("global_summary", "aggregate_merge", "energy_count", "memory_write"))
        add("suppress", ("suppress_global", "normalize", "delta_route"))
        add("transform", ("identity", "route_read", "evidence_read", "task_read", "aggregate_merge"))
        if self.max_primitives > len(self.base_ops):
            cat_op[:, len(self.base_ops):] = 1.0 / len(CATS)
        self.register_buffer("cat_op", cat_op / cat_op.sum(-1, keepdim=True).clamp_min(1.0))

        # Weak starting priors only. Specialization is learned/controlled by health-map.
        with torch.no_grad():
            if args.init_layers > 0:
                self.cat_logits[0, :, :, CATS.index("extract")] = 0.8
                self.role_logits[0, :, :, ROLES.index("extract")] = 0.8
            if args.init_layers > 1:
                self.cat_logits[1, :, :, CATS.index("compare")] = 0.4
                self.cat_logits[1, :, :, CATS.index("repair")] = 0.4
            if args.init_layers > 2:
                self.cat_logits[2, :, :, CATS.index("memory")] = 0.4
                self.cat_logits[2, :, :, CATS.index("repair")] = 0.4
            if args.init_layers > 3:
                self.cat_logits[3, :, :, CATS.index("aggregate")] = 0.6

    def set_knobs(self, category_temp=None, primitive_temp=None) -> None:
        if category_temp is not None:
            self.category_temp = float(max(0.5, min(2.5, category_temp)))
        if primitive_temp is not None:
            self.primitive_temp = float(max(0.5, min(2.5, primitive_temp)))

    def export_arch_state(self) -> Dict[str, Any]:
        return {
            "active_layer_ids": list(self.active_layer_ids),
            "unused_layer_ids": list(self.unused_layer_ids),
            "active_blocks": self.active_blocks.cpu().tolist(),
            "active_steps": self.active_steps.cpu().tolist(),
            "primitive_active": self.primitive_active.cpu().tolist(),
            "category_temp": self.category_temp,
            "primitive_temp": self.primitive_temp,
            "cat_prior": self.cat_prior.cpu().tolist(),
            "op_prior": self.op_prior.cpu().tolist(),
        }

    def load_arch_state(self, s: Dict[str, Any]) -> None:
        self.active_layer_ids = [int(x) for x in s["active_layer_ids"]]
        self.unused_layer_ids = [int(x) for x in s["unused_layer_ids"]]
        self.category_temp = float(s["category_temp"])
        self.primitive_temp = float(s["primitive_temp"])
        with torch.no_grad():
            self.active_blocks.copy_(torch.tensor(s["active_blocks"], device=self.active_blocks.device, dtype=torch.long))
            self.active_steps.copy_(torch.tensor(s["active_steps"], device=self.active_steps.device, dtype=torch.long))
            self.primitive_active.copy_(torch.tensor(s["primitive_active"], device=self.primitive_active.device, dtype=torch.bool))
            self.cat_prior.copy_(torch.tensor(s["cat_prior"], device=self.cat_prior.device))
            self.op_prior.copy_(torch.tensor(s["op_prior"], device=self.op_prior.device))

    def attn(self, q, k, v):
        a = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1]), -1).to(q.dtype)
        return a @ v, a

    def _candidate_outputs(self, base: torch.Tensor) -> torch.Tensor:
        if self.max_primitives <= len(self.base_ops):
            return base.new_zeros(base.shape[0], base.shape[1], 0, base.shape[-1])
        mix = torch.softmax(self.candidate_mix.float(), -1).to(base.dtype)
        return torch.einsum("co,bnod->bncd", mix, base)

    def _ops(self, h, ev, task, mem, glob, prev):
        B, nb, D = h.shape
        bank = torch.cat([ev.mean(1, keepdim=True), glob, prev], 1)
        route, ra = self.attn(self.qr(h), self.kr(bank), self.vr(bank))
        evc, _ = self.attn(self.qe(h), self.ke(ev), self.ve(ev))
        tc, _ = self.attn(self.qt(h), self.kt(task), self.vt(task))
        mc, _ = self.attn(self.qm(h), self.km(mem), self.vm(mem))
        gc = glob.mean(1, keepdim=True).expand(B, nb, D)
        energy = ev.pow(2).mean(1, keepdim=True).expand(B, nb, D)
        # Data-derived matrix candidates are present as regular operands; candidate primitives can later mix them.
        base_ops = torch.stack([
            h, evc, tc, route, h - route, h * torch.tanh(route), mc, h + 0.5 * mc,
            gc, h - gc, F.layer_norm(h, (D,)), energy, h + (evc - route), (h + route + evc + tc) / 4,
        ], 2)
        operands = torch.cat([base_ops, self._candidate_outputs(base_ops)], 2)
        return operands, route, evc, tc, ra

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        B, N, D = x.shape
        ev = self.evidence(x) + self.space_emb[1][None, None]
        task = self.task[None].expand(B, -1, -1) + self.space_emb[2][None, None]
        mem = self.memory0[None].expand(B, -1, -1) + self.space_emb[4][None, None]
        glob = self.global0[None].expand(B, -1, -1) + self.space_emb[5][None, None]
        base = ev.mean(1)
        prev = base[:, None, :]
        blocks_all = []
        role_logs=[]; cat_logs=[]; op_logs=[]; route_logs=[]; gate_logs=[]; block_cos_logs=[]; step_logs=[]
        block_index_map=[]
        for order_pos, lid in enumerate(self.active_layer_ids):
            nb = int(self.active_blocks[lid].item())
            if nb <= 0: continue
            h0 = prev[:, :nb] if prev.shape[1] >= nb else base[:, None].expand(B, nb, D)
            h = self.norm(h0 + self.block_addr[lid, :nb][None] + self.layer_pos_emb[lid][None, None] + self.block_pos_emb[:nb][None])
            max_active_steps = int(self.active_steps[lid, :nb].max().item())
            layer_role=[]; layer_cat=[]; layer_op=[]; layer_route=[]; layer_gate=[]
            for step in range(max_active_steps):
                mask = (self.active_steps[lid, :nb] > step).to(x.device).view(1, nb, 1)
                if not bool(mask.any()):
                    continue
                hs = h + self.step_emb[step][None, None]
                operands, route, evc, tc, ra = self._ops(hs, ev, task, mem, glob, prev)
                role = torch.softmax(self.role_logits[lid, :nb, step].float(), -1).to(x.dtype)
                cat = torch.softmax(((self.cat_logits[lid, :nb, step] + self.cat_prior[lid, :nb, step]) / self.category_temp).float(), -1).to(x.dtype)
                cat_prior = cat @ self.cat_op.to(x.device, x.dtype)
                logits = (hs @ self.op_addr.t()) / math.sqrt(D) + self.op_bias(torch.cat([hs, route, evc, tc], -1)) + self.op_prior[lid, :nb, step][None]
                active = self.primitive_active.to(x.device).view(1, 1, -1)
                logits = logits.masked_fill(~active, -1e4)
                prim = torch.softmax((logits / self.primitive_temp).float(), -1).to(x.dtype)
                op = prim * cat_prior.clamp_min(1e-6)
                op = op / op.sum(-1, keepdim=True).clamp_min(1e-6)
                upd = (op[..., None] * operands).sum(2)
                g = torch.sigmoid(self.gate(torch.cat([hs, upd, route, tc], -1))).squeeze(-1)
                h_new = self.norm(h + (0.10 + 0.90 * g).unsqueeze(-1) * upd)
                h = torch.where(mask.bool(), h_new, h)
                layer_role.append(role.detach().mean(0).cpu()); layer_cat.append(cat.detach().mean(0).cpu()); layer_op.append(op.detach().mean((0,1)).cpu()); layer_route.append(ent(ra.detach().float(), -1).mean().cpu()); layer_gate.append(g.detach().mean(0).cpu())
            blocks_all.append(h)
            for b in range(nb): block_index_map.append({"global_index": len(block_index_map), "layer_id": int(lid), "order_pos": int(order_pos), "block_id": int(b), "active_steps": int(self.active_steps[lid,b])})
            prev = h
            glob = glob + 0.05 * h.mean(1, keepdim=True).expand_as(glob)
            mem = mem + 0.03 * h.mean(1, keepdim=True).expand_as(mem)
            with torch.no_grad():
                flat = F.normalize(h.mean(0), dim=-1); cos = flat @ flat.T
                mask2 = ~torch.eye(nb, dtype=torch.bool, device=cos.device)
                bc = cos[mask2].abs().mean() if nb > 1 else torch.tensor(0.0, device=cos.device)
            role_logs.append(_mean_or_zero(layer_role, (len(ROLES),)))
            cat_logs.append(_mean_or_zero(layer_cat, (len(CATS),)))
            op_logs.append(_mean_or_zero(layer_op, (self.max_primitives,)))
            route_logs.append(_mean_or_zero(layer_route, ()))
            gate_logs.append(_mean_or_zero(layer_gate, (nb,)))
            block_cos_logs.append(bc.cpu())
            step_logs.append([int(x) for x in self.active_steps[lid, :nb].cpu().tolist()])
        blocks = torch.cat(blocks_all, 1) if blocks_all else prev
        aux = {
            "active_layer_ids": list(self.active_layer_ids),
            "active_blocks": {str(i): int(self.active_blocks[i]) for i in self.active_layer_ids},
            "active_steps": {str(i): [int(x) for x in self.active_steps[i, :int(self.active_blocks[i])].cpu().tolist()] for i in self.active_layer_ids},
            "active_primitives": [self.primitive_names[i] for i,v in enumerate(self.primitive_active.tolist()) if v],
            "block_index_map": block_index_map,
            "role_mix_by_layer": torch.stack(role_logs) if role_logs else torch.zeros(0, len(ROLES)),
            "category_mix_by_layer": torch.stack(cat_logs) if cat_logs else torch.zeros(0, len(CATS)),
            "operator_mix_by_layer": torch.stack(op_logs) if op_logs else torch.zeros(0, self.max_primitives),
            "route_entropy_by_layer": torch.stack(route_logs) if route_logs else torch.zeros(0),
            "gate_by_layer_block": [x.tolist() for x in gate_logs],
            "block_cosine_by_layer": [float(x) for x in block_cos_logs],
            "step_count_by_layer": step_logs,
        }
        return {"blocks": blocks, "evidence": ev, "task": task, "memory": mem, "global": glob, "aux": aux}

    def soft_action(self, action: str, layer_id: int | None = None) -> None:
        with torch.no_grad():
            if action == "explore":
                self.set_knobs(self.category_temp + 0.12, self.primitive_temp + 0.08)
                self.cat_prior.mul_(0.97); self.op_prior.mul_(0.98)
            elif action == "sharpen":
                self.set_knobs(self.category_temp - 0.08, self.primitive_temp - 0.06)
            elif layer_id is not None:
                nb = int(self.active_blocks[layer_id])
                if "memory" in action: self.cat_prior[layer_id, :nb, :, CATS.index("memory")] += 0.04
                if "repair" in action: self.cat_prior[layer_id, :nb, :, CATS.index("repair")] += 0.04
                if "compare" in action: self.cat_prior[layer_id, :nb, :, CATS.index("compare")] += 0.04
            elif action == "reduce_priors":
                self.cat_prior.mul_(0.85); self.op_prior.mul_(0.85)

    def add_step(self, layer_id: int, block_id: int) -> Tuple[bool, str]:
        cur = int(self.active_steps[layer_id, block_id].item())
        if cur >= self.max_steps:
            return False, "steps_full"
        with torch.no_grad():
            self.active_steps[layer_id, block_id] = cur + 1
            ref = max(0, cur - 1)
            self.cat_logits[layer_id, block_id, cur].copy_(self.cat_logits[layer_id, block_id, ref])
            self.role_logits[layer_id, block_id, cur].copy_(self.role_logits[layer_id, block_id, ref])
            self.cat_prior[layer_id, block_id, cur].copy_(self.cat_prior[layer_id, block_id, ref] * 0.5)
            self.op_prior[layer_id, block_id, cur].copy_(self.op_prior[layer_id, block_id, ref] * 0.5)
        return True, f"activated_step_{cur}_in_L{layer_id}B{block_id}"

    def add_depth(self, before_order_pos: int) -> Tuple[bool, str]:
        if not self.unused_layer_ids:
            return False, "no_unused_layer"
        new = self.unused_layer_ids.pop(0)
        before_order_pos = max(0, min(before_order_pos, len(self.active_layer_ids)))
        ref = self.active_layer_ids[min(before_order_pos, len(self.active_layer_ids)-1)] if self.active_layer_ids else 0
        with torch.no_grad():
            self.active_blocks[new] = max(1, int(self.active_blocks[ref]))
            self.active_steps[new, :int(self.active_blocks[new])] = torch.clamp(self.active_steps[ref, :int(self.active_blocks[new])], min=1)
            self.cat_logits[new].copy_(self.cat_logits[ref]); self.role_logits[new].copy_(self.role_logits[ref]); self.cat_prior[new].copy_(self.cat_prior[ref] * 0.5); self.op_prior[new].copy_(self.op_prior[ref] * 0.5)
        self.active_layer_ids.insert(before_order_pos, new)
        return True, f"activated_layer_{new}_at_order_{before_order_pos}_ref_{ref}"

    def add_width(self, layer_id: int) -> Tuple[bool, str]:
        nb = int(self.active_blocks[layer_id])
        if nb >= self.max_blocks:
            return False, "width_full"
        with torch.no_grad():
            self.active_blocks[layer_id] = nb + 1
            ref = max(0, nb - 1)
            self.active_steps[layer_id, nb] = max(1, int(self.active_steps[layer_id, ref]))
            self.cat_logits[layer_id, nb].copy_(self.cat_logits[layer_id, ref]); self.role_logits[layer_id, nb].copy_(self.role_logits[layer_id, ref]); self.cat_prior[layer_id, nb].copy_(self.cat_prior[layer_id, ref] * 0.5); self.op_prior[layer_id, nb].copy_(self.op_prior[layer_id, ref] * 0.5)
        return True, f"activated_block_{nb}_in_layer_{layer_id}"

    def add_primitive(self, layer_id: int, op_hint: int) -> Tuple[bool, str]:
        active = self.primitive_active.tolist()
        for i in range(len(self.base_ops), self.max_primitives):
            if not active[i]:
                with torch.no_grad():
                    self.primitive_active[i] = True
                    c = i - len(self.base_ops)
                    self.candidate_mix[c].fill_(-3.0)
                    self.candidate_mix[c, max(0, min(op_hint, len(self.base_ops)-1))] = 3.0
                    nb = int(self.active_blocks[layer_id])
                    self.op_prior[layer_id, :nb, :, i] += 0.08
                return True, f"activated_candidate_primitive_{i}_from_{self.primitive_names[op_hint]}_L{layer_id}"
        return False, "primitive_slots_full"


class UniversalProgramV2(nn.Module):
    def __init__(self, dim: int, args) -> None:
        super().__init__()
        self.core = StepMatrixCore(dim, args)
        self.delta_head = MatrixHead(dim)
        self.norm_head = MatrixHead(dim)

    def export_arch_state(self): return self.core.export_arch_state()
    def load_arch_state(self, s): return self.core.load_arch_state(s)

    def forward(self, x):
        core = self.core(x)
        delta, dop = self.delta_head(x, core)
        block = x + delta
        norm, nop = self.norm_head(block, core)
        core["aux"]["head_operator_mix"] = {
            "delta": {HEAD_OPS[i]: float(v) for i, v in enumerate(dop)},
            "norm": {HEAD_OPS[i]: float(v) for i, v in enumerate(nop)},
        }
        return {"block": block, "delta": delta, "norm": norm}, core["aux"]


@dataclass
class Event:
    epoch:int; mode:str; action:str; score:float; best_score:float; reward:float; accepted:bool; target_layer:int; target_block:int; reason:str; growth_detail:str


class StepGrowthController:
    def __init__(self, args) -> None:
        self.min_delta=args.controller_min_delta; self.patience=args.controller_patience; self.rollback=args.rollback_patience
        self.prev=None; self.absolute_best=-1e9; self.absolute_state=None; self.significant_best=-1e9; self.bad=0; self.flat=0

    def score(self, val: Dict[str,float], aux: Dict[str,Any], active: List[str]) -> float:
        s=sum([-val[f"{n}_mse"]+0.2*val[f"{n}_cos"] for n in active])/max(1,len(active))
        routes=aux.get("route_entropy_by_layer", []) or [0.0]
        s += 0.002 * float(torch.tensor(routes).float().mean())
        return float(s)

    def target_from_health(self, val: Dict[str,float], aux: Dict[str,Any]) -> Tuple[int,int,int,str]:
        ids=aux.get("active_layer_ids", [])
        if not ids: return 0,0,0,"fallback"
        op_rows=aux.get("operator_mix_by_layer", [])
        block_cos=aux.get("block_cosine_by_layer", []) or [0.0 for _ in ids]
        best=(-1e9, ids[-1], 0, len(ids)-1, "fallback")
        for pos,lid in enumerate(ids):
            row=op_rows[pos] if pos < len(op_rows) else []
            get=lambda name: float(row[BASE_OPS.index(name)]) if len(row)>BASE_OPS.index(name) else 0.0
            # Head-loss pressure: delta gap makes late delta/repair layers more suspicious.
            delta_gap=max(0.0, val.get("block_cos",0.0)-val.get("delta_cos",0.0))
            pressure = delta_gap * (get("delta_route") + get("residual_repair") + 0.5*get("memory_read")) + 0.15*float(block_cos[pos] if pos < len(block_cos) else 0.0)
            if pressure > best[0]:
                best=(pressure,int(lid),0,pos,"head_loss_weighted_operator_pressure")
        return best[1], best[2], best[3], best[4]

    def op_hint(self, aux: Dict[str,Any], order_pos:int) -> int:
        rows=aux.get("operator_mix_by_layer", [])
        if order_pos < len(rows):
            row=torch.tensor(rows[order_pos][:len(BASE_OPS)]).float()
            return int(torch.argmax(row).item())
        return BASE_OPS.index("delta_route")

    def step(self, model: UniversalProgramV2, epoch:int, val:Dict[str,float], aux:Dict[str,Any], active:List[str]) -> Event:
        score=self.score(val,aux,active); reward=0.0 if self.prev is None else score-self.prev
        accepted=False; mode="WAIT"; action="noop"; reason="normal"; detail="none"; tl=-1; tb=-1
        if score>self.absolute_best:
            self.absolute_best=score; self.absolute_state=model.export_arch_state()
        if score>self.significant_best+self.min_delta:
            self.significant_best=score; accepted=True; self.flat=0; self.bad=0; mode="EXPLOIT"; action="sharpen"; reason="significant_best"; model.core.soft_action("sharpen")
        else:
            if self.prev is not None and score<self.prev-self.min_delta: self.bad+=1
            else: self.flat+=1
            if self.bad>=self.rollback and self.absolute_state is not None:
                model.load_arch_state(self.absolute_state); model.core.soft_action("reduce_priors"); mode="ROLLBACK"; action="rollback_absolute_best"; reason="score_drop"; self.bad=0
            elif self.flat>=self.patience:
                tl,tb,pos,why=self.target_from_health(val,aux); reason="plateau_"+why
                delta_gap=val.get("block_cos",0.0)-val.get("delta_cos",0.0)
                block_cos=max(aux.get("block_cosine_by_layer", [0.0]) or [0.0])
                # Priority: step first, because it adds ordered operation sequence without global topology churn.
                ok,detail=model.core.add_step(tl,tb); action="add_step" if ok else ""
                if not ok and delta_gap>0.04 and model.core.unused_layer_ids:
                    ok,detail=model.core.add_depth(before_order_pos=pos+1); action="add_depth" if ok else ""
                if not ok and block_cos>0.35:
                    ok,detail=model.core.add_width(tl); action="add_width" if ok else ""
                if not ok:
                    hint=self.op_hint(aux,pos); ok,detail=model.core.add_primitive(tl,hint); action="add_primitive" if ok else "soft_explore"
                if action=="soft_explore": model.core.soft_action("explore",tl); detail="soft_explore_no_slots_or_unclear_bottleneck"
                mode="GROW" if action.startswith("add") else "EXPLORE"; self.flat=0
        self.prev=score
        return Event(epoch,mode,action,score,self.absolute_best,reward,accepted,tl,tb,reason,detail)


def tensor_to_jsonable_aux(aux:Dict[str,Any]) -> Dict[str,Any]:
    out={}
    for k,v in aux.items():
        if isinstance(v, torch.Tensor): out[k]=v.tolist()
        else: out[k]=v
    return out


def run(args):
    seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    task=HFHiddenTask(args.model_name,args.layer_idx,args.device)
    model=UniversalProgramV2(task.dim,args).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler("cuda",enabled=args.amp=="fp16" and args.device.startswith("cuda"))
    amp_dtype=torch.bfloat16 if args.amp=="bf16" else torch.float16
    active=[x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl=StepGrowthController(args)
    fields=["epoch","score","best_score","mode","action","target_layer","target_block","reason","growth_detail","seconds"]+[f"train_{n}_{k}" for n in active for k in ("mse","cos")]+[f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    for ep in range(1,args.epochs+1):
        model.train(); sums={}; t0=time.time()
        for _ in range(args.steps_per_epoch):
            b=task.sample(args.batch_size,args.seq_len,args.device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device),dtype=amp_dtype,enabled=args.amp!="off" and args.device.startswith("cuda")):
                pred,_=model(b["x"]); loss,met=losses(pred,b,active,args.lambda_cos)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); scaler.step(opt); scaler.update()
            for k,v in met.items(): sums[k]=sums.get(k,0.0)+v
        train={k:v/args.steps_per_epoch for k,v in sums.items()}|{"seconds":time.time()-t0}
        model.eval(); sums={}; last_aux=None
        with torch.no_grad():
            for _ in range(args.val_steps):
                b=task.sample(args.eval_batch_size,args.seq_len,args.device)
                with torch.autocast(device_type=devtype(args.device),dtype=amp_dtype,enabled=args.amp!="off" and args.device.startswith("cuda")):
                    pred,aux=model(b["x"]); _,met=losses(pred,b,active,args.lambda_cos)
                for k,v in met.items(): sums[k]=sums.get(k,0.0)+v
                last_aux=aux
        val={k:v/args.val_steps for k,v in sums.items()}; aux=tensor_to_jsonable_aux(last_aux or {})
        ev=ctrl.step(model,ep,val,aux,active)
        row={"epoch":ep,"score":ev.score,"best_score":ev.best_score,"mode":ev.mode,"action":ev.action,"target_layer":ev.target_layer,"target_block":ev.target_block,"reason":ev.reason,"growth_detail":ev.growth_detail,"seconds":train["seconds"]}
        for n in active:
            row[f"train_{n}_mse"]=train.get(f"{n}_mse",""); row[f"train_{n}_cos"]=train.get(f"{n}_cos",""); row[f"val_{n}_mse"]=val.get(f"{n}_mse",""); row[f"val_{n}_cos"]=val.get(f"{n}_cos","")
        with (out/"metrics.csv").open("a",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
        report={"epoch":ep,"train":train,"val":val,"controller":asdict(ev),"matrix_program":aux,"arch_state":model.export_arch_state(),"args":vars(args)}; write_json(out/f"analysis_epoch_{ep:03d}.json",report)
        with (out/"controller_state.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(asdict(ev),ensure_ascii=False)+"\n")
        print(f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} target=L{ev.target_layer}B{ev.target_block} {ev.growth_detail} "+" ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]),flush=True)
    write_json(out/"final_report.json",{"architecture":"universal_matrix_program_v2_steps","final_arch_state":model.export_arch_state(),"args":vars(args)})


def parser():
    p=argparse.ArgumentParser(); p.add_argument("--model-name",default="google/bert_uncased_L-2_H-128_A-2"); p.add_argument("--layer-idx",type=int,default=0); p.add_argument("--device",default="cuda"); p.add_argument("--amp",choices=["bf16","fp16","off"],default="bf16"); p.add_argument("--seed",type=int,default=42); p.add_argument("--seq-len",type=int,default=64); p.add_argument("--targets",default="block,delta,norm"); p.add_argument("--evidence-cells",type=int,default=40); p.add_argument("--task-cells",type=int,default=12); p.add_argument("--memory-cells",type=int,default=6); p.add_argument("--global-cells",type=int,default=3); p.add_argument("--init-layers",type=int,default=4); p.add_argument("--max-layers",type=int,default=8); p.add_argument("--init-blocks-per-layer",type=int,default=4); p.add_argument("--max-blocks-per-layer",type=int,default=8); p.add_argument("--init-steps-per-block",type=int,default=1); p.add_argument("--max-steps-per-block",type=int,default=4); p.add_argument("--max-primitives",type=int,default=24); p.add_argument("--category-temp",type=float,default=1.4); p.add_argument("--primitive-temp",type=float,default=1.2); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--steps-per-epoch",type=int,default=120); p.add_argument("--val-steps",type=int,default=20); p.add_argument("--epochs",type=int,default=45); p.add_argument("--lr",type=float,default=7e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-clip",type=float,default=0.7); p.add_argument("--lambda-cos",type=float,default=0.25); p.add_argument("--controller-min-delta",type=float,default=0.0015); p.add_argument("--controller-patience",type=int,default=2); p.add_argument("--rollback-patience",type=int,default=1); p.add_argument("--out-dir",default="./runs/universal_matrix_program_v2_steps_hf_45ep"); return p
if __name__=="__main__": run(parser().parse_args())
