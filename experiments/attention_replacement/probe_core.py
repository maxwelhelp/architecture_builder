from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROLES = ["extract", "contrast", "transform", "memory", "repair", "aggregate", "suppress"]
CATS = ["extract", "compare", "memory", "repair", "aggregate", "suppress", "transform"]
OPS = [
    "identity", "evidence_read", "task_read", "route_read", "delta_route",
    "compare_mul", "memory_read", "memory_write", "global_summary",
    "suppress_global", "normalize", "energy_count", "residual_repair", "aggregate_merge",
]


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = p.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=dim)


class StructuredSequenceGenerator:
    def __init__(self, seq_len: int, dim: int, device: str, seed: int = 42) -> None:
        self.seq_len, self.dim, self.device = seq_len, dim, device
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + 101)
        self.basis = (torch.randn(8, dim, generator=g) / math.sqrt(dim)).to(device)
        self.wave_proj = (torch.randn(4, dim, generator=g) / math.sqrt(dim)).to(device)

    def batch(self, batch_size: int) -> torch.Tensor:
        B, N, D, device = batch_size, self.seq_len, self.dim, self.device
        t = torch.linspace(0, 1, N, device=device).view(1, N, 1)
        freqs = torch.arange(1, 5, device=device).float().view(1, 1, 4)
        phase = torch.rand(B, 1, 4, device=device) * 2 * math.pi
        waves = torch.sin(2 * math.pi * freqs * t + phase)
        ids = torch.randint(0, 8, (B, N), device=device)
        x = self.basis[ids] + waves @ self.wave_proj
        pos = torch.randint(0, N, (B,), device=device)
        marker = self.basis[torch.randint(0, 8, (B,), device=device)] * 1.5
        x[torch.arange(B, device=device), pos] += marker
        return x + 0.05 * torch.randn_like(x)


class FrozenTeacherMHA(nn.Module):
    def __init__(self, dim: int, heads: int, seed: int = 123) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        y = self.norm(x + y)
        return self.norm(y + 0.25 * self.ff(y))


class SequenceEvidenceBuilder(nn.Module):
    def __init__(self, dim: int, evidence_cells: int) -> None:
        super().__init__()
        self.evidence_cells = evidence_cells
        self.type_emb = nn.Parameter(torch.randn(evidence_cells, dim) * 0.02)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        feats: List[torch.Tensor] = []
        local = min(24, max(8, self.evidence_cells - 8))
        for i in range(local):
            a = int(i * N / local)
            b = max(a + 1, int((i + 1) * N / local))
            feats.append(x[:, a:b].mean(1))
        q = max(1, N // 4)
        feats += [
            x.mean(1), x.std(1), x.amax(1),
            x[:, :q].mean(1), x[:, N // 2:max(N // 2 + 1, 3 * N // 4)].mean(1),
            x[:, -q:].mean(1), x[:, 1:].mean(1) - x[:, :-1].mean(1),
            x[:, -q:].mean(1) - x[:, :q].mean(1),
        ]
        orig = len(feats)
        while len(feats) < self.evidence_cells:
            feats.append(feats[len(feats) % orig])
        ev = torch.stack(feats[: self.evidence_cells], 1)
        return self.norm(self.proj(ev) + self.type_emb.view(1, self.evidence_cells, D))


class MatrixAttentionReplacementStudent(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        evidence_cells: int = 40,
        task_cells: int = 12,
        layers: int = 4,
        blocks: int = 4,
        memory_cells: int = 6,
        global_cells: int = 3,
        category_temp: float = 1.4,
        primitive_temp: float = 1.2,
    ) -> None:
        super().__init__()
        self.dim, self.layers, self.blocks = dim, layers, blocks
        self.category_temp, self.primitive_temp = float(category_temp), float(primitive_temp)
        self.evidence = SequenceEvidenceBuilder(dim, evidence_cells)
        self.task_emb = nn.Parameter(torch.randn(task_cells, dim) * 0.02)
        self.task_pressure = nn.Parameter(torch.zeros(task_cells, dim), requires_grad=False)
        self.block_addr = nn.Parameter(torch.randn(layers, blocks, dim) * 0.02)
        self.memory = nn.Parameter(torch.randn(memory_cells, dim) * 0.02)
        self.global_reg = nn.Parameter(torch.randn(global_cells, dim) * 0.02)
        self.qr = nn.Linear(dim, dim, bias=False); self.kr = nn.Linear(dim, dim, bias=False); self.vr = nn.Linear(dim, dim, bias=False)
        self.qe = nn.Linear(dim, dim, bias=False); self.ke = nn.Linear(dim, dim, bias=False); self.ve = nn.Linear(dim, dim, bias=False)
        self.qt = nn.Linear(dim, dim, bias=False); self.kt = nn.Linear(dim, dim, bias=False); self.vt = nn.Linear(dim, dim, bias=False)
        self.qm = nn.Linear(dim, dim, bias=False); self.km = nn.Linear(dim, dim, bias=False); self.vm = nn.Linear(dim, dim, bias=False)
        self.role_logits = nn.Parameter(torch.zeros(layers, len(ROLES)))
        self.cat_logits = nn.Parameter(torch.zeros(layers, blocks, len(CATS)))
        self.op_addr = nn.Parameter(torch.randn(len(OPS), dim) * 0.02)
        self.op_bias = nn.Linear(dim * 4, len(OPS))
        self.gate = nn.Linear(dim * 4, 1)
        self.norm = nn.LayerNorm(dim)
        self.token_q = nn.Linear(dim, dim, bias=False); self.block_k = nn.Linear(dim, dim, bias=False); self.block_v = nn.Linear(dim, dim, bias=False)
        self.seq_head = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.register_buffer("role_prior_delta", torch.zeros(layers, len(ROLES)))
        self.register_buffer("cat_prior_delta", torch.zeros(layers, blocks, len(CATS)))
        self.register_buffer("op_prior_delta", torch.zeros(layers, blocks, len(OPS)))
        cat_op = torch.zeros(len(CATS), len(OPS))
        def add(cat: str, ops: Tuple[str, ...]) -> None:
            c = CATS.index(cat)
            for op in ops:
                cat_op[c, OPS.index(op)] = 1.0
        add("extract", ("evidence_read", "delta_route", "energy_count", "identity"))
        add("compare", ("compare_mul", "delta_route", "route_read", "normalize", "residual_repair"))
        add("memory", ("memory_read", "memory_write", "aggregate_merge", "identity"))
        add("repair", ("residual_repair", "suppress_global", "normalize", "delta_route", "compare_mul"))
        add("aggregate", ("global_summary", "aggregate_merge", "energy_count", "memory_write"))
        add("suppress", ("suppress_global", "normalize", "delta_route"))
        add("transform", ("identity", "route_read", "evidence_read", "task_read", "aggregate_merge"))
        self.register_buffer("cat_op_basis", cat_op / cat_op.sum(-1, keepdim=True).clamp_min(1.0))
        with torch.no_grad():
            self.role_logits[0, ROLES.index("extract")] = 2.0
            self.role_logits[1, ROLES.index("contrast")] = 1.3; self.role_logits[1, ROLES.index("repair")] = 0.8
            self.role_logits[2, ROLES.index("memory")] = 1.0; self.role_logits[2, ROLES.index("repair")] = 1.0
            self.role_logits[-1, ROLES.index("aggregate")] = 2.0
            self.cat_logits[0, :, CATS.index("extract")] = 2.0
            self.cat_logits[1, :, CATS.index("repair")] = 1.2; self.cat_logits[1, :, CATS.index("compare")] = 0.8
            self.cat_logits[2, :, CATS.index("repair")] = 0.8; self.cat_logits[2, :, CATS.index("memory")] = 0.8
            self.cat_logits[-1, :, CATS.index("aggregate")] = 1.6

    def set_knobs(self, category_temp=None, primitive_temp=None) -> None:
        if category_temp is not None:
            self.category_temp = float(max(0.5, min(2.5, category_temp)))
        if primitive_temp is not None:
            self.primitive_temp = float(max(0.5, min(2.5, primitive_temp)))

    def export_arch_state(self) -> Dict[str, object]:
        return {
            "category_temp": self.category_temp,
            "primitive_temp": self.primitive_temp,
            "role_prior_delta": self.role_prior_delta.cpu().tolist(),
            "cat_prior_delta": self.cat_prior_delta.cpu().tolist(),
            "op_prior_delta": self.op_prior_delta.cpu().tolist(),
        }

    def load_arch_state(self, s: Dict[str, object]) -> None:
        self.category_temp = float(s.get("category_temp", self.category_temp))
        self.primitive_temp = float(s.get("primitive_temp", self.primitive_temp))
        with torch.no_grad():
            self.role_prior_delta.copy_(torch.tensor(s["role_prior_delta"], device=self.role_prior_delta.device))
            self.cat_prior_delta.copy_(torch.tensor(s["cat_prior_delta"], device=self.cat_prior_delta.device))
            self.op_prior_delta.copy_(torch.tensor(s["op_prior_delta"], device=self.op_prior_delta.device))

    def apply_action(self, action: str, strength: float = 0.06) -> None:
        with torch.no_grad():
            if action == "explore_categories":
                self.set_knobs(self.category_temp + 0.20, self.primitive_temp + 0.10); self.cat_prior_delta.mul_(0.95)
            elif action == "sharpen_exploit":
                self.set_knobs(self.category_temp - 0.15, self.primitive_temp - 0.10)
            elif action == "boost_l2_memory" and self.layers > 2:
                self.cat_prior_delta[2, :, CATS.index("memory")] += strength; self.op_prior_delta[2, :, OPS.index("memory_read")] += strength; self.op_prior_delta[2, :, OPS.index("memory_write")] += strength
            elif action == "boost_l2_repair" and self.layers > 2:
                self.cat_prior_delta[2, :, CATS.index("repair")] += strength; self.op_prior_delta[2, :, OPS.index("residual_repair")] += strength
            elif action == "boost_l1_compare" and self.layers > 1:
                self.cat_prior_delta[1, :, CATS.index("compare")] += strength; self.op_prior_delta[1, :, OPS.index("compare_mul")] += strength
            elif action == "boost_aggregate_memory":
                self.cat_prior_delta[-1, :, CATS.index("aggregate")] += strength; self.op_prior_delta[-1, :, OPS.index("aggregate_merge")] += strength
            elif action == "reduce_priors":
                self.role_prior_delta.mul_(0.85); self.cat_prior_delta.mul_(0.85); self.op_prior_delta.mul_(0.85)

    def attn(self, q, k, v):
        a = torch.softmax((q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1]), dim=-1).to(q.dtype)
        return a @ v, a

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        B, N, D = x.shape
        ev = self.evidence(x)
        task = self.task_emb[None].expand(B, -1, -1) + self.task_pressure[None]
        mem = self.memory[None].expand(B, -1, -1)
        glob = self.global_reg[None].expand(B, -1, -1)
        base = ev.mean(1)
        prev = base[:, None, :] + self.block_addr[0]
        blocks_all, role_log, cat_log, op_log, route_ent, gate_log = [], [], [], [], [], []
        for layer in range(self.layers):
            h0 = prev[:, :self.blocks] if prev.shape[1] >= self.blocks else base[:, None].expand(B, self.blocks, D)
            h = self.norm(h0 + self.block_addr[layer][None])
            route_bank = torch.cat([base[:, None], glob, prev], 1)
            route, ra = self.attn(self.qr(h), self.kr(route_bank), self.vr(route_bank))
            evc, _ = self.attn(self.qe(h), self.ke(ev), self.ve(ev))
            tc, _ = self.attn(self.qt(h), self.kt(task), self.vt(task))
            mc, _ = self.attn(self.qm(h), self.km(mem), self.vm(mem))
            gc = glob.mean(1, keepdim=True).expand(B, self.blocks, D)
            energy = ev.pow(2).mean(1, keepdim=True).expand(B, self.blocks, D)
            cand = torch.stack([h, evc, tc, route, h-route, h*torch.tanh(route), mc, h+0.5*mc, gc, h-gc, F.layer_norm(h, (D,)), energy, h+(evc-route), (h+route+evc+tc)/4], 2)
            role = torch.softmax((self.role_logits[layer] + self.role_prior_delta[layer]).float(), -1).to(x.dtype).expand(B, -1)
            cat = torch.softmax(((self.cat_logits[layer] + self.cat_prior_delta[layer]) / self.category_temp).float(), -1).to(x.dtype).expand(B, -1, -1)
            cat_prior = cat @ self.cat_op_basis.to(x.device, x.dtype)
            op_logits = (h @ self.op_addr.t()) / math.sqrt(D) + self.op_bias(torch.cat([h, route, evc, tc], -1)) + self.op_prior_delta[layer][None]
            prim = torch.softmax((op_logits / self.primitive_temp).float(), -1).to(x.dtype)
            op = prim * cat_prior.clamp_min(1e-6); op = op / op.sum(-1, keepdim=True).clamp_min(1e-6)
            upd = (op[..., None] * cand).sum(2)
            g = torch.sigmoid(self.gate(torch.cat([h, upd, route, tc], -1))).squeeze(-1)
            h = self.norm(h + (0.15 + 0.85 * g).unsqueeze(-1) * upd)
            blocks_all.append(h); prev = h
            glob = glob + 0.05 * h.mean(1, keepdim=True).expand_as(glob)
            mem = mem + 0.03 * h.mean(1, keepdim=True).expand_as(mem)
            role_log.append(role.detach().mean(0)); cat_log.append(cat.detach().mean((0, 1))); op_log.append(op.detach().mean((0, 1))); route_ent.append(entropy(ra.detach().float(), -1).mean()); gate_log.append(g.detach().mean(0))
        blocks = torch.cat(blocks_all, 1)
        ba = torch.softmax((self.token_q(x) @ self.block_k(blocks).transpose(-1, -2)).float() / math.sqrt(D), -1).to(x.dtype)
        ctx = ba @ self.block_v(blocks)
        out = self.seq_head(torch.cat([x, ctx, x * torch.tanh(ctx)], -1))
        aux = {}
        if return_aux:
            aux = {
                "role_mix_by_layer": torch.stack(role_log).cpu(),
                "category_mix_by_layer": torch.stack(cat_log).cpu(),
                "operator_mix_by_layer": torch.stack(op_log).cpu(),
                "route_entropy_by_layer": torch.stack(route_ent).cpu(),
                "gate_by_layer_block": torch.stack(gate_log).cpu(),
                "token_block_entropy": entropy(ba.detach().float(), -1).mean().cpu(),
            }
        return out, aux
