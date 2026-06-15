#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json, math, random, sys, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "attention_replacement"))
from probe_core import CATS, OPS, ROLES, entropy  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def devtype(device: str) -> str:
    return device.split(":", 1)[0]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


class HFHiddenTeacher(nn.Module):
    def __init__(self, model_name: str, layer_idx: int, device: str) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except Exception as exc:
            raise RuntimeError("Install transformers: pip install transformers") from exc
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.config = self.model.config
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(getattr(self.config, "hidden_size", getattr(self.config, "n_embd", 0)))
        self.vocab_size = int(getattr(self.config, "vocab_size", 30522))
        self.max_pos = int(getattr(self.config, "max_position_embeddings", 512))

    @torch.no_grad()
    def batch(self, batch_size: int, seq_len: int, device: str) -> Dict[str, torch.Tensor]:
        hi = max(16, min(self.vocab_size, 30000))
        input_ids = torch.randint(0, hi, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if self.layer_idx + 1 >= len(hs):
            raise RuntimeError(f"layer_idx={self.layer_idx} invalid for hidden_states len={len(hs)}")
        x = hs[self.layer_idx].detach()
        y = hs[self.layer_idx + 1].detach()
        return {"x": x, "block": y, "delta": y - x, "norm": F.layer_norm(y, (y.shape[-1],))}


class SequenceEvidence(nn.Module):
    def __init__(self, dim: int, cells: int) -> None:
        super().__init__(); self.cells = cells
        self.type = nn.Parameter(torch.randn(cells, dim) * 0.02)
        self.proj = nn.Linear(dim, dim, bias=False); self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape; feats: List[torch.Tensor] = []
        local = min(24, max(8, self.cells - 8))
        for i in range(local):
            a = int(i * N / local); b = max(a + 1, int((i + 1) * N / local))
            feats.append(x[:, a:b].mean(1))
        q = max(1, N // 4)
        feats += [x.mean(1), x.std(1), x.amax(1), x[:, :q].mean(1), x[:, -q:].mean(1), x[:, 1:].mean(1) - x[:, :-1].mean(1), x[:, -q:].mean(1) - x[:, :q].mean(1)]
        orig = len(feats)
        while len(feats) < self.cells: feats.append(feats[len(feats) % orig])
        ev = torch.stack(feats[:self.cells], 1)
        return self.norm(self.proj(ev) + self.type.view(1, self.cells, D))


class MatrixCore(nn.Module):
    def __init__(self, dim: int, evidence_cells: int, task_cells: int, layers: int, blocks: int, memory_cells: int, global_cells: int, cat_temp: float, prim_temp: float) -> None:
        super().__init__(); self.dim, self.layers, self.blocks = dim, layers, blocks
        self.category_temp, self.primitive_temp = float(cat_temp), float(prim_temp)
        self.evidence = SequenceEvidence(dim, evidence_cells)
        self.task = nn.Parameter(torch.randn(task_cells, dim) * 0.02)
        self.addr = nn.Parameter(torch.randn(layers, blocks, dim) * 0.02)
        self.mem0 = nn.Parameter(torch.randn(memory_cells, dim) * 0.02); self.glob0 = nn.Parameter(torch.randn(global_cells, dim) * 0.02)
        self.qr = nn.Linear(dim, dim, bias=False); self.kr = nn.Linear(dim, dim, bias=False); self.vr = nn.Linear(dim, dim, bias=False)
        self.qe = nn.Linear(dim, dim, bias=False); self.ke = nn.Linear(dim, dim, bias=False); self.ve = nn.Linear(dim, dim, bias=False)
        self.qt = nn.Linear(dim, dim, bias=False); self.kt = nn.Linear(dim, dim, bias=False); self.vt = nn.Linear(dim, dim, bias=False)
        self.qm = nn.Linear(dim, dim, bias=False); self.km = nn.Linear(dim, dim, bias=False); self.vm = nn.Linear(dim, dim, bias=False)
        self.role_logits = nn.Parameter(torch.zeros(layers, len(ROLES))); self.cat_logits = nn.Parameter(torch.zeros(layers, blocks, len(CATS)))
        self.op_addr = nn.Parameter(torch.randn(len(OPS), dim) * 0.02); self.op_bias = nn.Linear(dim * 4, len(OPS)); self.gate = nn.Linear(dim * 4, 1); self.norm = nn.LayerNorm(dim)
        self.register_buffer("role_prior", torch.zeros(layers, len(ROLES))); self.register_buffer("cat_prior", torch.zeros(layers, blocks, len(CATS))); self.register_buffer("op_prior", torch.zeros(layers, blocks, len(OPS)))
        cat_op = torch.zeros(len(CATS), len(OPS))
        def add(c, names):
            for n in names: cat_op[CATS.index(c), OPS.index(n)] = 1.0
        add("extract", ("evidence_read", "delta_route", "energy_count", "identity")); add("compare", ("compare_mul", "delta_route", "route_read", "normalize", "residual_repair")); add("memory", ("memory_read", "memory_write", "aggregate_merge", "identity")); add("repair", ("residual_repair", "suppress_global", "normalize", "delta_route", "compare_mul")); add("aggregate", ("global_summary", "aggregate_merge", "energy_count", "memory_write")); add("suppress", ("suppress_global", "normalize", "delta_route")); add("transform", ("identity", "route_read", "evidence_read", "task_read", "aggregate_merge"))
        self.register_buffer("cat_op", cat_op / cat_op.sum(-1, keepdim=True).clamp_min(1.0))
        with torch.no_grad():
            self.role_logits[0, ROLES.index("extract")] = 2.0; self.role_logits[1, ROLES.index("contrast")] = 1.1; self.role_logits[1, ROLES.index("repair")] = 0.8; self.role_logits[2, ROLES.index("memory")] = 1.0; self.role_logits[2, ROLES.index("repair")] = 1.0; self.role_logits[-1, ROLES.index("aggregate")] = 1.8
            self.cat_logits[0, :, CATS.index("extract")] = 1.8; self.cat_logits[1, :, CATS.index("compare")] = 0.9; self.cat_logits[1, :, CATS.index("repair")] = 1.0; self.cat_logits[2, :, CATS.index("memory")] = 0.9; self.cat_logits[2, :, CATS.index("repair")] = 0.9; self.cat_logits[-1, :, CATS.index("aggregate")] = 1.5

    def set_knobs(self, category_temp=None, primitive_temp=None):
        if category_temp is not None: self.category_temp = float(max(0.5, min(2.5, category_temp)))
        if primitive_temp is not None: self.primitive_temp = float(max(0.5, min(2.5, primitive_temp)))

    def export_arch_state(self):
        return {"category_temp": self.category_temp, "primitive_temp": self.primitive_temp, "role_prior": self.role_prior.cpu().tolist(), "cat_prior": self.cat_prior.cpu().tolist(), "op_prior": self.op_prior.cpu().tolist()}

    def load_arch_state(self, s):
        self.category_temp = float(s.get("category_temp", self.category_temp)); self.primitive_temp = float(s.get("primitive_temp", self.primitive_temp))
        with torch.no_grad(): self.role_prior.copy_(torch.tensor(s["role_prior"], device=self.role_prior.device)); self.cat_prior.copy_(torch.tensor(s["cat_prior"], device=self.cat_prior.device)); self.op_prior.copy_(torch.tensor(s["op_prior"], device=self.op_prior.device))

    def apply_action(self, action: str, strength: float = 0.05):
        with torch.no_grad():
            if action == "explore_categories": self.set_knobs(self.category_temp + 0.20, self.primitive_temp + 0.10); self.cat_prior.mul_(0.96)
            elif action == "sharpen_exploit": self.set_knobs(self.category_temp - 0.12, self.primitive_temp - 0.08)
            elif action == "boost_l2_memory" and self.layers > 2: self.cat_prior[2, :, CATS.index("memory")] += strength; self.op_prior[2, :, OPS.index("memory_read")] += strength; self.op_prior[2, :, OPS.index("memory_write")] += strength
            elif action == "boost_l2_repair" and self.layers > 2: self.cat_prior[2, :, CATS.index("repair")] += strength; self.op_prior[2, :, OPS.index("residual_repair")] += strength
            elif action == "boost_l1_compare" and self.layers > 1: self.cat_prior[1, :, CATS.index("compare")] += strength; self.op_prior[1, :, OPS.index("compare_mul")] += strength
            elif action == "boost_aggregate": self.cat_prior[-1, :, CATS.index("aggregate")] += strength; self.op_prior[-1, :, OPS.index("aggregate_merge")] += strength
            elif action == "reduce_priors": self.role_prior.mul_(0.85); self.cat_prior.mul_(0.85); self.op_prior.mul_(0.85)

    def attn(self, q, k, v):
        a = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1]), -1).to(q.dtype); return a @ v, a

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape; ev = self.evidence(x); task = self.task[None].expand(B, -1, -1); mem = self.mem0[None].expand(B, -1, -1); glob = self.glob0[None].expand(B, -1, -1); base = ev.mean(1); prev = base[:, None] + self.addr[0]
        blocks_all = []; role_log=[]; cat_log=[]; op_log=[]; route_log=[]; gate_log=[]
        for l in range(self.layers):
            h = self.norm((prev[:, :self.blocks] if prev.shape[1] >= self.blocks else base[:, None].expand(B, self.blocks, D)) + self.addr[l][None])
            bank = torch.cat([base[:, None], glob, prev], 1); route, ra = self.attn(self.qr(h), self.kr(bank), self.vr(bank)); evc,_ = self.attn(self.qe(h), self.ke(ev), self.ve(ev)); tc,_ = self.attn(self.qt(h), self.kt(task), self.vt(task)); mc,_ = self.attn(self.qm(h), self.km(mem), self.vm(mem)); gc = glob.mean(1, keepdim=True).expand(B,self.blocks,D); energy = ev.pow(2).mean(1, keepdim=True).expand(B,self.blocks,D)
            cand = torch.stack([h, evc, tc, route, h-route, h*torch.tanh(route), mc, h+0.5*mc, gc, h-gc, F.layer_norm(h,(D,)), energy, h+(evc-route), (h+route+evc+tc)/4], 2)
            role = torch.softmax((self.role_logits[l]+self.role_prior[l]).float(), -1).to(x.dtype); cat = torch.softmax(((self.cat_logits[l]+self.cat_prior[l])/self.category_temp).float(), -1).to(x.dtype); cat_prior = cat @ self.cat_op.to(x.device,x.dtype); logits = (h @ self.op_addr.t())/math.sqrt(D) + self.op_bias(torch.cat([h,route,evc,tc],-1)) + self.op_prior[l][None]; prim = torch.softmax((logits/self.primitive_temp).float(), -1).to(x.dtype); op = prim * cat_prior.clamp_min(1e-6); op = op / op.sum(-1, keepdim=True).clamp_min(1e-6)
            upd = (op[...,None]*cand).sum(2); g = torch.sigmoid(self.gate(torch.cat([h,upd,route,tc],-1))).squeeze(-1); h = self.norm(h + (0.15+0.85*g).unsqueeze(-1)*upd); blocks_all.append(h); prev = h; glob = glob + 0.05*h.mean(1,keepdim=True).expand_as(glob); mem = mem + 0.03*h.mean(1,keepdim=True).expand_as(mem)
            role_log.append(role.detach().cpu()); cat_log.append(cat.detach().mean(0).cpu()); op_log.append(op.detach().mean((0,1)).cpu()); route_log.append(entropy(ra.detach().float(),-1).mean().cpu()); gate_log.append(g.detach().mean(0).cpu())
        return {"blocks": torch.cat(blocks_all,1), "evidence": ev, "task": task, "memory": mem, "global": glob, "aux": {"role_mix_by_layer": torch.stack(role_log), "category_mix_by_layer": torch.stack(cat_log), "operator_mix_by_layer": torch.stack(op_log), "route_entropy_by_layer": torch.stack(route_log), "gate_by_layer_block": torch.stack(gate_log)}}


class MatrixSequenceHead(nn.Module):
    def __init__(self, dim: int, name: str) -> None:
        super().__init__(); self.name=name; self.q=nn.Linear(dim,dim,bias=False); self.kb=nn.Linear(dim,dim,bias=False); self.vb=nn.Linear(dim,dim,bias=False); self.ke=nn.Linear(dim,dim,bias=False); self.ve=nn.Linear(dim,dim,bias=False); self.km=nn.Linear(dim,dim,bias=False); self.vm=nn.Linear(dim,dim,bias=False); self.op_addr=nn.Parameter(torch.randn(9,dim)*0.02); self.out_norm=nn.LayerNorm(dim); self.gate=nn.Linear(dim*3,1)
    def read(self,q,k,v):
        a=torch.softmax((q@k.transpose(-1,-2)).float()/math.sqrt(q.shape[-1]),-1).to(q.dtype); return a@v,a
    def forward(self,x,core):
        q=self.q(x); bc,_=self.read(q,self.kb(core["blocks"]),self.vb(core["blocks"])); ec,_=self.read(q,self.ke(core["evidence"]),self.ve(core["evidence"])); mc,_=self.read(q,self.km(core["memory"]),self.vm(core["memory"])); gc=core["global"].mean(1,keepdim=True).expand_as(x); cand=torch.stack([x,bc,ec,mc,gc,x+bc,bc-x,x*torch.tanh(bc),F.layer_norm(bc,(x.shape[-1],))],2); logits=(q@self.op_addr.t())/math.sqrt(x.shape[-1]); op=torch.softmax(logits.float(),-1).to(x.dtype); upd=(op[...,None]*cand).sum(2); g=torch.sigmoid(self.gate(torch.cat([x,upd,bc],-1))); return self.out_norm(x + g*upd), op.detach().mean((0,1)).cpu()


class HFMatrixDropIn(nn.Module):
    def __init__(self, dim:int, args) -> None:
        super().__init__(); self.core=MatrixCore(dim,args.evidence_cells,args.task_cells,args.num_layers,args.blocks_per_layer,args.memory_cells,args.global_cells,args.category_temp,args.primitive_temp); self.heads=nn.ModuleDict({"block":MatrixSequenceHead(dim,"block"),"delta":MatrixSequenceHead(dim,"delta"),"norm":MatrixSequenceHead(dim,"norm")})
    @property
    def category_temp(self): return self.core.category_temp
    @property
    def primitive_temp(self): return self.core.primitive_temp
    def export_arch_state(self): return self.core.export_arch_state()
    def load_arch_state(self,s): return self.core.load_arch_state(s)
    def apply_action(self,a,strength=0.05): return self.core.apply_action(a,strength)
    def forward(self,x):
        core=self.core(x); out={}; hop={}
        for n,h in self.heads.items(): out[n], hop[n]=h(x,core)
        core["aux"]["head_operator_mix"]={k:{str(i):float(v) for i,v in enumerate(val)} for k,val in hop.items()}
        return out, core["aux"]


@dataclass
class Event: epoch:int; mode:str; action:str; score:float; best_score:float; reward:float; accepted:bool; category_temp:float; primitive_temp:float; reason:str; growth_recommendation:str


class SmartController:
    def __init__(self, patience:int, rollback:int, min_delta:float):
        self.patience=patience; self.rollback=rollback; self.min_delta=min_delta; self.prev=None; self.best=-1e9; self.best_state=None; self.bad=0; self.plateau=0; self.i=0; self.cool={}; self.actions=["explore_categories","boost_l2_memory","boost_l2_repair","boost_l1_compare","boost_aggregate","sharpen_exploit","reduce_priors"]
    def choose(self):
        for _ in range(len(self.actions)):
            a=self.actions[self.i%len(self.actions)]; self.i+=1
            if self.cool.get(a,0)<=0: return a
        return "explore_categories"
    def step(self,model,epoch,score,aux):
        for k in list(self.cool): self.cool[k]=max(0,self.cool[k]-1)
        reward=0.0 if self.prev is None else score-self.prev; accepted=score>self.best+self.min_delta; mode="WAIT"; action="noop"; reason="pending-effect buffer"; growth="none"
        if accepted:
            self.best=score; self.best_state=model.export_arch_state(); self.bad=0; self.plateau=0; mode="EXPLOIT"; action="sharpen_exploit" if epoch>3 else "noop"; reason="new best"
        else:
            if self.prev is not None and score<self.prev-self.min_delta: self.bad+=1
            else: self.plateau+=1
            if self.bad>=self.rollback and self.best_state is not None:
                model.load_arch_state(self.best_state); action="reduce_priors"; model.apply_action(action,0.04); self.cool[action]=2; mode="ROLLBACK"; reason="harmful action; restored best arch"; self.bad=0
            elif self.plateau>=self.patience:
                action=self.choose(); model.apply_action(action,0.05); mode="EXPLORE"; reason="plateau/slow growth; soft mutation"; self.plateau=0
        if action=="sharpen_exploit": model.apply_action(action,0.03)
        if self.plateau>=self.patience*2 or self.bad>=self.rollback:
            route=float(aux.get("route_entropy_mean",0.0)); growth="add_depth" if route>1.2 else "add_width_or_primitive"
        self.prev=score
        return Event(epoch,mode,action,score,self.best,reward,accepted,model.category_temp,model.primitive_temp,reason,growth)


def summarize(aux_list):
    role=torch.stack([a["role_mix_by_layer"] for a in aux_list]).mean(0); cat=torch.stack([a["category_mix_by_layer"] for a in aux_list]).mean(0); op=torch.stack([a["operator_mix_by_layer"] for a in aux_list]).mean(0); route=torch.stack([a["route_entropy_by_layer"] for a in aux_list]).mean(0); gate=torch.stack([a["gate_by_layer_block"] for a in aux_list]).mean(0)
    return {"role_mix_by_layer":[{"layer":i,"roles":{ROLES[j]:float(role[i,j]) for j in range(len(ROLES))}} for i in range(role.shape[0])],"category_mix_by_layer":[{"layer":i,"categories":{CATS[j]:float(cat[i,j]) for j in range(len(CATS))}} for i in range(cat.shape[0])],"operator_mix_by_layer":[{"layer":i,"ops":{OPS[j]:float(op[i,j]) for j in range(len(OPS))}} for i in range(op.shape[0])],"route_entropy_by_layer":[float(x) for x in route],"route_entropy_mean":float(route.mean()),"gate_by_layer_block":gate.tolist()}


def losses(pred,tgt,active,lc):
    loss=torch.zeros((),device=next(iter(pred.values())).device); m={}
    for n in active:
        mse=F.mse_loss(pred[n].float(),tgt[n].float()); cos=F.cosine_similarity(pred[n].float().flatten(1),tgt[n].float().flatten(1),-1).mean(); loss=loss+mse+lc*(1-cos); m[f"{n}_mse"]=float(mse.detach()); m[f"{n}_cos"]=float(cos.detach())
    return loss,m


def score_val(m,aux,active): return sum([-m[f"{n}_mse"]+0.2*m[f"{n}_cos"] for n in active])/max(1,len(active))+0.005*aux.get("route_entropy_mean",0.0)


def run(args):
    set_seed(args.seed); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); teacher=HFHiddenTeacher(args.model_name,args.layer_idx,args.device); args.dim=teacher.hidden_size; model=HFMatrixDropIn(args.dim,args).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9,0.95)); scaler=torch.amp.GradScaler("cuda",enabled=args.amp=="fp16" and args.device.startswith("cuda")); amp_dtype=torch.bfloat16 if args.amp=="bf16" else torch.float16; ctrl=SmartController(args.controller_patience,args.rollback_patience,args.controller_min_delta); active=[x.strip() for x in args.targets.split(",") if x.strip()]
    fields=["epoch","score","best_score","mode","action","reward","category_temp","primitive_temp","growth_recommendation","seconds"]+[f"train_{n}_{k}" for n in active for k in ("mse","cos")]+[f"val_{n}_{k}" for n in active for k in ("mse","cos")]
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writeheader()
    best={"score":-1e9,"epoch":0}
    for ep in range(1,args.epochs+1):
        model.train(); sums={}; t0=time.time()
        for _ in range(args.steps_per_epoch):
            b=teacher.batch(args.batch_size,args.seq_len,args.device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=devtype(args.device),dtype=amp_dtype,enabled=args.amp!="off" and args.device.startswith("cuda")):
                pred,_=model(b["x"]); loss,met=losses(pred,b,active,args.lambda_cos)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); scaler.step(opt); scaler.update()
            for k,v in met.items(): sums[k]=sums.get(k,0.0)+v
        train={k:v/args.steps_per_epoch for k,v in sums.items()}|{"seconds":time.time()-t0}
        model.eval(); sums={}; auxs=[]
        with torch.no_grad():
            for _ in range(args.val_steps):
                b=teacher.batch(args.eval_batch_size,args.seq_len,args.device)
                with torch.autocast(device_type=devtype(args.device),dtype=amp_dtype,enabled=args.amp!="off" and args.device.startswith("cuda")):
                    pred,aux=model(b["x"]); _,met=losses(pred,b,active,args.lambda_cos)
                for k,v in met.items(): sums[k]=sums.get(k,0.0)+v
                auxs.append(aux)
        val={k:v/args.val_steps for k,v in sums.items()}; aux=summarize(auxs); score=score_val(val,aux,active); ev=ctrl.step(model,ep,score,aux) if args.controller else Event(ep,"FIXED","noop",score,best["score"],0.0,False,model.category_temp,model.primitive_temp,"off","none")
        if score>best["score"]: best={"score":score,"epoch":ep,"val":val}
        row={"epoch":ep,"score":score,"best_score":best["score"],"mode":ev.mode,"action":ev.action,"reward":ev.reward,"category_temp":ev.category_temp,"primitive_temp":ev.primitive_temp,"growth_recommendation":ev.growth_recommendation,"seconds":train["seconds"]}
        for n in active:
            row[f"train_{n}_mse"]=train.get(f"{n}_mse",""); row[f"train_{n}_cos"]=train.get(f"{n}_cos",""); row[f"val_{n}_mse"]=val.get(f"{n}_mse",""); row[f"val_{n}_cos"]=val.get(f"{n}_cos","")
        with (out/"metrics.csv").open("a",newline="",encoding="utf-8") as f: csv.DictWriter(f,fieldnames=fields).writerow(row)
        analysis={"epoch":ep,"train":train,"val":val,"score":score,"best":best,"controller":asdict(ev),"matrix_program":aux,"arch_state":model.export_arch_state(),"model_name":args.model_name,"layer_idx":args.layer_idx,"targets":active}
        write_json(out/f"analysis_epoch_{ep:03d}.json",analysis)
        with (out/"controller_state.jsonl").open("a",encoding="utf-8") as f: f.write(json.dumps(asdict(ev),ensure_ascii=False)+"\n")
        print(f"ep {ep:03d}/{args.epochs} score={score:.5f} best={best['score']:.5f}@{best['epoch']} mode={ev.mode} action={ev.action} growth={ev.growth_recommendation} "+" ".join([f"{n}:mse={val.get(n+'_mse',0):.4g},cos={val.get(n+'_cos',0):.4f}" for n in active]),flush=True)
    write_json(out/"final_report.json",{"best":best,"args":vars(args),"architecture":"hf_matrix_dropin_probe_v1"})


def parser():
    p=argparse.ArgumentParser(); p.add_argument("--model-name",default="prajjwal1/bert-tiny"); p.add_argument("--layer-idx",type=int,default=0); p.add_argument("--device",default="cuda"); p.add_argument("--amp",choices=["bf16","fp16","off"],default="bf16"); p.add_argument("--seed",type=int,default=42); p.add_argument("--dim",type=int,default=0); p.add_argument("--seq-len",type=int,default=64); p.add_argument("--evidence-cells",type=int,default=40); p.add_argument("--task-cells",type=int,default=12); p.add_argument("--num-layers",type=int,default=4); p.add_argument("--blocks-per-layer",type=int,default=4); p.add_argument("--memory-cells",type=int,default=6); p.add_argument("--global-cells",type=int,default=3); p.add_argument("--category-temp",type=float,default=1.4); p.add_argument("--primitive-temp",type=float,default=1.2); p.add_argument("--targets",default="block,delta,norm"); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--eval-batch-size",type=int,default=128); p.add_argument("--steps-per-epoch",type=int,default=120); p.add_argument("--val-steps",type=int,default=20); p.add_argument("--epochs",type=int,default=40); p.add_argument("--lr",type=float,default=7e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-clip",type=float,default=0.7); p.add_argument("--lambda-cos",type=float,default=0.25); p.add_argument("--controller",action="store_true",default=True); p.add_argument("--no-controller",dest="controller",action="store_false"); p.add_argument("--controller-patience",type=int,default=2); p.add_argument("--rollback-patience",type=int,default=1); p.add_argument("--controller-min-delta",type=float,default=0.0015); p.add_argument("--out-dir",default="./runs/hf_matrix_dropin_probe_v1_40ep"); return p
if __name__=="__main__": run(parser().parse_args())
