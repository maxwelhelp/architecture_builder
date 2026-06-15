#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
from run_universal_matrix_program_v1 import (  # noqa: E402
    BASE_OPS, CATS, HEAD_OPS, ROLES,
    HFHiddenTask, UniversalEvidence,
    devtype, ent, losses, seed_all, write_json,
)
from run_universal_matrix_program_v2_steps import StepMatrixCore  # noqa: E402


def to_jsonable(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def entropy_list(vals: torch.Tensor) -> float:
    p = vals.float().clamp_min(1e-8)
    p = p / p.sum().clamp_min(1e-8)
    return float((-(p * p.log()).sum() / math.log(max(2, p.numel()))).detach().cpu())


class TraceStepCore(StepMatrixCore):
    """V2 core with per layer/block/step logs.

    Growth slots still come from StepMatrixCore:
    active_layer_ids, active_blocks, active_steps, primitive_active.
    This class only improves observability.
    """

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
        per_step_ops=[]; per_step_cats=[]; per_step_gates=[]; per_step_routes=[]; block_index_map=[]

        for order_pos, lid in enumerate(self.active_layer_ids):
            nb = int(self.active_blocks[lid].item())
            if nb <= 0:
                continue
            h0 = prev[:, :nb] if prev.shape[1] >= nb else base[:, None].expand(B, nb, D)
            h = self.norm(h0 + self.block_addr[lid, :nb][None] + self.layer_pos_emb[lid][None, None] + self.block_pos_emb[:nb][None])
            max_active_steps = int(self.active_steps[lid, :nb].max().item())
            layer_role=[]; layer_cat=[]; layer_op=[]; layer_route=[]; layer_gate=[]
            layer_step_ops=[[] for _ in range(nb)]
            layer_step_cats=[[] for _ in range(nb)]
            layer_step_gates=[[] for _ in range(nb)]
            layer_step_routes=[[] for _ in range(nb)]

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
                primitive_active = self.primitive_active.to(x.device).view(1, 1, -1)
                logits = logits.masked_fill(~primitive_active, -1e4)
                prim = torch.softmax((logits / self.primitive_temp).float(), -1).to(x.dtype)
                op = prim * cat_prior.clamp_min(1e-6)
                op = op / op.sum(-1, keepdim=True).clamp_min(1e-6)
                upd = (op[..., None] * operands).sum(2)
                g = torch.sigmoid(self.gate(torch.cat([hs, upd, route, tc], -1))).squeeze(-1)
                h_new = self.norm(h + (0.10 + 0.90 * g).unsqueeze(-1) * upd)
                h = torch.where(active_mask.bool(), h_new, h)

                op_mean = op.detach().mean(0).cpu()              # [nb,P]
                cat_cpu = cat.detach().cpu()                     # [nb,C]
                gate_cpu = g.detach().mean(0).cpu()              # [nb]
                route_cpu = ent(ra.detach().float(), -1).mean(0).cpu()  # [nb]
                for b in range(nb):
                    if int(self.active_steps[lid, b].item()) > step:
                        layer_step_ops[b].append(op_mean[b])
                        layer_step_cats[b].append(cat_cpu[b])
                        layer_step_gates[b].append(gate_cpu[b])
                        layer_step_routes[b].append(route_cpu[b])
                layer_role.append(role.detach().mean(0).cpu())
                layer_cat.append(cat.detach().mean(0).cpu())
                layer_op.append(op.detach().mean((0, 1)).cpu())
                layer_route.append(ent(ra.detach().float(), -1).mean().cpu())
                layer_gate.append(g.detach().mean(0).cpu())

            blocks_all.append(h)
            for b in range(nb):
                block_index_map.append({
                    "global_index": len(block_index_map),
                    "layer_id": int(lid),
                    "order_pos": int(order_pos),
                    "block_id": int(b),
                    "active_steps": int(self.active_steps[lid, b].item()),
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
            step_logs.append([int(x) for x in self.active_steps[lid, :nb].cpu().tolist()])
            per_step_ops.append([[v.tolist() for v in block_steps] for block_steps in layer_step_ops])
            per_step_cats.append([[v.tolist() for v in block_steps] for block_steps in layer_step_cats])
            per_step_gates.append([[float(v) for v in block_steps] for block_steps in layer_step_gates])
            per_step_routes.append([[float(v) for v in block_steps] for block_steps in layer_step_routes])

        blocks = torch.cat(blocks_all, 1) if blocks_all else prev
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
        }
        return {"blocks": blocks, "evidence": ev, "task": task, "memory": mem, "global": glob, "aux": aux}

    def prune_step(self, layer_id: int, block_id: int) -> Tuple[bool, str]:
        cur = int(self.active_steps[layer_id, block_id].item())
        if cur <= 1:
            return False, "cannot_prune_last_step"
        with torch.no_grad():
            self.active_steps[layer_id, block_id] = cur - 1
        return True, f"pruned_step_{cur-1}_from_L{layer_id}B{block_id}"

    def prune_width(self, layer_id: int) -> Tuple[bool, str]:
        nb = int(self.active_blocks[layer_id].item())
        if nb <= 1:
            return False, "cannot_prune_last_block"
        with torch.no_grad():
            self.active_blocks[layer_id] = nb - 1
        return True, f"pruned_block_{nb-1}_from_L{layer_id}"


class TraceMatrixHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q=nn.Linear(dim,dim,bias=False); self.kb=nn.Linear(dim,dim,bias=False); self.vb=nn.Linear(dim,dim,bias=False)
        self.ke=nn.Linear(dim,dim,bias=False); self.ve=nn.Linear(dim,dim,bias=False)
        self.km=nn.Linear(dim,dim,bias=False); self.vm=nn.Linear(dim,dim,bias=False)
        self.op=nn.Parameter(torch.randn(len(HEAD_OPS),dim)*0.02)
        self.gate=nn.Linear(dim*3,1); self.norm=nn.LayerNorm(dim)

    def read(self,q,k,v):
        a=torch.softmax((q@k.transpose(-1,-2)).float()/math.sqrt(q.shape[-1]),-1).to(q.dtype)
        return a@v,a

    def forward(self,x,core):
        q=self.q(x)
        bc,ba=self.read(q,self.kb(core["blocks"]),self.vb(core["blocks"]))
        ec,_=self.read(q,self.ke(core["evidence"]),self.ve(core["evidence"]))
        mc,_=self.read(q,self.km(core["memory"]),self.vm(core["memory"]))
        gc=core["global"].mean(1,keepdim=True).expand_as(x)
        cand=torch.stack([x,bc,ec,mc,gc,x+bc,bc-x,x*torch.tanh(bc),F.layer_norm(bc,(x.shape[-1],))],2)
        w=torch.softmax(((q@self.op.t())/math.sqrt(x.shape[-1])).float(),-1).to(x.dtype)
        upd=(w[...,None]*cand).sum(2)
        g=torch.sigmoid(self.gate(torch.cat([x,upd,bc],-1)))
        out=self.norm(x+g*upd)
        return out, w.detach().mean((0,1)).cpu(), ba.detach().float().mean((0,1)).cpu()


class UniversalProgramV21(nn.Module):
    def __init__(self, dim: int, args) -> None:
        super().__init__()
        self.core = TraceStepCore(dim, args)
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


@dataclass
class GrowthEvent:
    epoch:int; mode:str; action:str; score:float; best_score:float; reward:float; accepted:bool
    target_layer:int; target_block:int; reason:str; growth_detail:str
    step_score:float; depth_score:float; width_score:float; primitive_score:float


class BalancedGrowthController:
    def __init__(self, args) -> None:
        self.min_delta=args.controller_min_delta
        self.patience=args.controller_patience
        self.rollback=args.rollback_patience
        self.pending_epochs=args.pending_epochs
        self.weak_reward=args.weak_action_reward
        self.prev=None
        self.absolute_best=-1e9
        self.absolute_state=None
        self.significant_best=-1e9
        self.bad=0
        self.flat=0
        self.cooldown: Dict[str,int] = {}
        self.pending: Dict[str,Any] | None = None
        self.history: List[Dict[str,Any]] = []
        self.last_action_scores: Dict[str,float] = {}

    def _cool_key(self, action: str, layer: int, block: int) -> str:
        return f"{action}:L{layer}:B{block}"

    def _cool(self, key: str) -> float:
        return 1.0 if self.cooldown.get(key, 0) > 0 else 0.0

    def _tick(self) -> None:
        for k in list(self.cooldown.keys()):
            self.cooldown[k] = max(0, self.cooldown[k] - 1)
            if self.cooldown[k] == 0:
                del self.cooldown[k]

    def score(self, val: Dict[str,float], aux: Dict[str,Any], active: List[str]) -> float:
        s=sum([-val[f"{n}_mse"]+0.2*val[f"{n}_cos"] for n in active])/max(1,len(active))
        routes=aux.get("route_entropy_by_layer", []) or [0.0]
        s += 0.002 * float(torch.tensor(routes).float().mean())
        return float(s)

    def _target_from_head_reads(self, val: Dict[str,float], aux: Dict[str,Any]) -> Tuple[int,int,int,int,str,float]:
        block_map = aux.get("block_index_map", [])
        reads = aux.get("head_block_read", {})
        if not block_map or not reads:
            ids=aux.get("active_layer_ids", [0])
            return int(ids[min(1, len(ids)-1)]), 0, min(1, len(ids)-1), 0, "fallback_no_head_reads", 0.0
        delta_read = reads.get("delta", [0.0]*len(block_map))
        norm_read = reads.get("norm", [0.0]*len(block_map))
        delta_err = float(val.get("delta_mse", 0.0)) + max(0.0, float(val.get("block_cos",0.0))-float(val.get("delta_cos",0.0)))
        norm_err = float(val.get("norm_mse", 0.0))
        pressures=[]
        for i, bm in enumerate(block_map):
            p = delta_err * float(delta_read[i] if i < len(delta_read) else 0.0) + 0.5 * norm_err * float(norm_read[i] if i < len(norm_read) else 0.0)
            pressures.append((p, bm))
        p, bm = max(pressures, key=lambda x: x[0])
        return int(bm["layer_id"]), int(bm["block_id"]), int(bm["order_pos"]), int(bm["global_index"]), "head_loss_read_pressure", float(p)

    def _action_scores(self, model: UniversalProgramV21, val: Dict[str,float], aux: Dict[str,Any], layer:int, block:int, order_pos:int, pressure:float) -> Dict[str,float]:
        ops = aux.get("operator_mix_by_layer", [])
        op_row = torch.tensor(ops[order_pos]).float() if order_pos < len(ops) else torch.ones(model.core.max_primitives) / model.core.max_primitives
        op_entropy = entropy_list(op_row[:len(BASE_OPS)])
        active_layers = len(aux.get("active_layer_ids", []))
        active_blocks = int(model.core.active_blocks[layer].item())
        step_count = int(model.core.active_steps[layer, block].item())
        block_cos = float((aux.get("block_cosine_by_layer", []) or [0.0])[min(order_pos, len(aux.get("block_cosine_by_layer", [0.0]))-1)])
        delta_gap = max(0.0, float(val.get("block_cos",0.0))-float(val.get("delta_cos",0.0)))
        route = float((aux.get("route_entropy_by_layer", []) or [0.0])[min(order_pos, len(aux.get("route_entropy_by_layer", [0.0]))-1)])
        low_op_entropy = max(0.0, 0.45 - op_entropy)
        multi_op_pressure = op_entropy * min(1.0, pressure * 20.0)
        step_key=self._cool_key("add_step",layer,block)
        depth_key=self._cool_key("add_depth",layer,block)
        width_key=self._cool_key("add_width",layer,block)
        primitive_key=self._cool_key("add_primitive",layer,block)
        scores = {
            "add_step": 0.55*multi_op_pressure + 0.12*route + 0.10*delta_gap - 0.18*max(0, step_count-1) - self._cool(step_key),
            "add_depth": 0.70*delta_gap + 0.20*pressure + 0.15*route - 0.055*max(0, active_layers-4) - self._cool(depth_key),
            "add_width": 0.45*block_cos + 0.35*pressure - 0.08*max(0, active_blocks-4) - self._cool(width_key),
            "add_primitive": 0.25*low_op_entropy + 0.10*delta_gap - 0.15*(len([x for x in model.core.primitive_active.tolist() if x])-len(BASE_OPS)) - self._cool(primitive_key),
        }
        # Do not add primitive while basic structural slots are clearly available and promising.
        if scores["add_step"] > 0.05 or scores["add_depth"] > 0.05 or scores["add_width"] > 0.05:
            scores["add_primitive"] -= 0.15
        if step_count >= model.core.max_steps:
            scores["add_step"] = -999.0
        if active_layers >= model.core.max_layers:
            scores["add_depth"] = -999.0
        if active_blocks >= model.core.max_blocks:
            scores["add_width"] = -999.0
        if int(model.core.primitive_active.sum().item()) >= model.core.max_primitives:
            scores["add_primitive"] = -999.0
        return scores

    def _apply_action(self, model: UniversalProgramV21, action:str, layer:int, block:int, order_pos:int, aux:Dict[str,Any]) -> Tuple[str,bool]:
        if action == "add_step":
            return model.core.add_step(layer, block)[1], model.core.add_step(layer, block)[0]  # unreachable double call guard below
        return "none", False

    def _do_action(self, model: UniversalProgramV21, action:str, layer:int, block:int, order_pos:int, aux:Dict[str,Any]) -> Tuple[bool,str]:
        if action == "add_step":
            return model.core.add_step(layer, block)
        if action == "add_depth":
            return model.core.add_depth(before_order_pos=order_pos+1)
        if action == "add_width":
            return model.core.add_width(layer)
        if action == "add_primitive":
            rows=aux.get("operator_mix_by_layer", [])
            hint=BASE_OPS.index("delta_route")
            if order_pos < len(rows):
                hint=int(torch.argmax(torch.tensor(rows[order_pos][:len(BASE_OPS)]).float()).item())
            return model.core.add_primitive(layer, hint)
        model.core.soft_action("explore", layer)
        return True, "soft_explore"

    def step(self, model: UniversalProgramV21, epoch:int, val:Dict[str,float], aux:Dict[str,Any], active:List[str]) -> GrowthEvent:
        self._tick()
        score=self.score(val,aux,active)
        reward=0.0 if self.prev is None else score-self.prev
        mode="WAIT"; action="noop"; reason="normal"; detail="none"; target_layer=-1; target_block=-1; accepted=False

        if score > self.absolute_best:
            self.absolute_best = score
            self.absolute_state = model.export_arch_state()

        # Evaluate pending growth after a small buffer.
        if self.pending is not None and epoch - int(self.pending["epoch"]) >= self.pending_epochs:
            pr = score - float(self.pending["score_before"])
            rec = dict(self.pending)
            rec.update({"epoch_eval": epoch, "score_after": score, "reward": pr})
            self.history.append(rec)
            key = self._cool_key(str(self.pending["action"]), int(self.pending["layer"]), int(self.pending["block"]))
            if pr < -self.min_delta and self.pending.get("state_before") is not None:
                model.load_arch_state(self.pending["state_before"])
                self.cooldown[key] = 3
                mode="ROLLBACK"; action="rollback_rejected_growth"; reason="pending_action_hurt"; detail=f"rejected_{self.pending['action']}_reward_{pr:.5g}"
                self.pending=None; self.prev=score
                return GrowthEvent(epoch,mode,action,score,self.absolute_best,reward,False,-1,-1,reason,detail,**self._score_defaults())
            if pr < self.weak_reward:
                self.cooldown[key] = 2
            self.pending=None

        if score > self.significant_best + self.min_delta:
            self.significant_best=score; accepted=True; self.flat=0; self.bad=0
            mode="EXPLOIT"; action="sharpen"; reason="significant_best"; model.core.soft_action("sharpen")
        else:
            if self.prev is not None and score < self.prev - self.min_delta:
                self.bad += 1
            else:
                self.flat += 1
            if self.bad >= self.rollback and self.absolute_state is not None:
                model.load_arch_state(self.absolute_state)
                model.core.soft_action("reduce_priors")
                mode="ROLLBACK"; action="rollback_absolute_best"; reason="score_drop"; self.bad=0
            elif self.flat >= self.patience:
                target_layer,target_block,order_pos,global_idx,why,pressure=self._target_from_head_reads(val,aux)
                scores=self._action_scores(model,val,aux,target_layer,target_block,order_pos,pressure)
                self.last_action_scores=scores
                action=max(scores, key=scores.get)
                if scores[action] < -0.05:
                    action="soft_explore"
                state_before=model.export_arch_state()
                ok,detail=self._do_action(model,action,target_layer,target_block,order_pos,aux)
                if not ok:
                    action="soft_explore"; model.core.soft_action("explore", target_layer); detail="fallback_soft_explore"
                if action.startswith("add"):
                    self.pending={"epoch":epoch,"action":action,"layer":target_layer,"block":target_block,"score_before":score,"state_before":state_before,"detail":detail}
                mode="GROW" if action.startswith("add") else "EXPLORE"
                reason="plateau_"+why
                target_block=int(target_block); self.flat=0
        self.prev=score
        return GrowthEvent(epoch,mode,action,score,self.absolute_best,reward,accepted,int(target_layer),int(target_block),reason,detail,**self.last_action_scores_or_default())

    def _score_defaults(self):
        return {"step_score":0.0,"depth_score":0.0,"width_score":0.0,"primitive_score":0.0}

    def last_action_scores_or_default(self):
        s=self.last_action_scores or {}
        return {"step_score":float(s.get("add_step",0.0)),"depth_score":float(s.get("add_depth",0.0)),"width_score":float(s.get("add_width",0.0)),"primitive_score":float(s.get("add_primitive",0.0))}


def run(args):
    seed_all(args.seed)
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    task=HFHiddenTask(args.model_name,args.layer_idx,args.device)
    model=UniversalProgramV21(task.dim,args).to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9,0.95))
    scaler=torch.amp.GradScaler("cuda",enabled=args.amp=="fp16" and args.device.startswith("cuda"))
    amp_dtype=torch.bfloat16 if args.amp=="bf16" else torch.float16
    active=[x.strip() for x in args.targets.split(",") if x.strip()]
    ctrl=BalancedGrowthController(args)
    fields=["epoch","score","best_score","mode","action","target_layer","target_block","reason","growth_detail","step_score","depth_score","width_score","primitive_score","seconds"]+[f"train_{n}_{k}" for n in active for k in ("mse","cos")]+[f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        csv.DictWriter(f,fieldnames=fields).writeheader()
    for ep in range(1,args.epochs+1):
        model.train(); sums={}; t0=time.time()
        for _ in range(args.steps_per_epoch):
            b=task.sample(args.batch_size,args.seq_len,args.device)
            opt.zero_grad(set_to_none=True)
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
        val={k:v/args.val_steps for k,v in sums.items()}
        aux=to_jsonable(last_aux or {})
        ev=ctrl.step(model,ep,val,aux,active)
        row={"epoch":ep,"score":ev.score,"best_score":ev.best_score,"mode":ev.mode,"action":ev.action,"target_layer":ev.target_layer,"target_block":ev.target_block,"reason":ev.reason,"growth_detail":ev.growth_detail,"step_score":ev.step_score,"depth_score":ev.depth_score,"width_score":ev.width_score,"primitive_score":ev.primitive_score,"seconds":train["seconds"]}
        for n in active:
            row[f"train_{n}_mse"]=train.get(f"{n}_mse",""); row[f"train_{n}_cos"]=train.get(f"{n}_cos",""); row[f"val_{n}_mse"]=val.get(f"{n}_mse",""); row[f"val_{n}_cos"]=val.get(f"{n}_cos","")
        with (out/"metrics.csv").open("a",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=fields).writerow(row)
        report={"epoch":ep,"train":train,"val":val,"controller":asdict(ev),"action_history":ctrl.history,"cooldown":ctrl.cooldown,"pending":to_jsonable(ctrl.pending),"matrix_program":aux,"arch_state":model.export_arch_state(),"args":vars(args)}
        write_json(out/f"analysis_epoch_{ep:03d}.json",report)
        with (out/"controller_state.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps({"event":asdict(ev),"history_tail":ctrl.history[-3:],"cooldown":ctrl.cooldown},ensure_ascii=False)+"\n")
        print(f"ep {ep:03d}/{args.epochs} score={ev.score:.5f} best={ev.best_score:.5f} mode={ev.mode} action={ev.action} target=L{ev.target_layer}B{ev.target_block} scores(step={ev.step_score:.3f},depth={ev.depth_score:.3f},width={ev.width_score:.3f},prim={ev.primitive_score:.3f}) {ev.growth_detail} "+" ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]),flush=True)
    write_json(out/"final_report.json",{"architecture":"universal_matrix_program_v2_1_balanced","final_arch_state":model.export_arch_state(),"action_history":ctrl.history,"args":vars(args)})


def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--model-name",default="google/bert_uncased_L-2_H-128_A-2"); p.add_argument("--layer-idx",type=int,default=0)
    p.add_argument("--device",default="cuda"); p.add_argument("--amp",choices=["bf16","fp16","off"],default="bf16"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--seq-len",type=int,default=64); p.add_argument("--targets",default="block,delta,norm")
    p.add_argument("--evidence-cells",type=int,default=40); p.add_argument("--task-cells",type=int,default=12); p.add_argument("--memory-cells",type=int,default=6); p.add_argument("--global-cells",type=int,default=3)
    p.add_argument("--init-layers",type=int,default=4); p.add_argument("--max-layers",type=int,default=8)
    p.add_argument("--init-blocks-per-layer",type=int,default=4); p.add_argument("--max-blocks-per-layer",type=int,default=8)
    p.add_argument("--init-steps-per-block",type=int,default=1); p.add_argument("--max-steps-per-block",type=int,default=4); p.add_argument("--max-primitives",type=int,default=24)
    p.add_argument("--category-temp",type=float,default=1.4); p.add_argument("--primitive-temp",type=float,default=1.2)
    p.add_argument("--batch-size",type=int,default=64); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--steps-per-epoch",type=int,default=120); p.add_argument("--val-steps",type=int,default=20); p.add_argument("--epochs",type=int,default=45)
    p.add_argument("--lr",type=float,default=7e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-clip",type=float,default=0.7); p.add_argument("--lambda-cos",type=float,default=0.25)
    p.add_argument("--controller-min-delta",type=float,default=0.0015); p.add_argument("--controller-patience",type=int,default=2); p.add_argument("--rollback-patience",type=int,default=1); p.add_argument("--pending-epochs",type=int,default=2); p.add_argument("--weak-action-reward",type=float,default=0.00025)
    p.add_argument("--out-dir",default="./runs/universal_matrix_program_v2_1_balanced_hf_45ep")
    return p


if __name__=="__main__":
    run(parser().parse_args())
