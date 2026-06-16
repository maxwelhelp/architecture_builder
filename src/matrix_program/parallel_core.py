from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import torch
import torch.nn as nn

from .torch_shadow_topk import dense_topk_weighted_sum

Mode = Literal["sequential_exact", "parallel_refine", "hybrid_groups"]


@dataclass
class ParallelSlotConfig:
    dim: int = 128
    layers: int = 4
    blocks: int = 4
    memory_cells: int = 6
    refine_iters: int = 2
    topk: int = 3
    mode: Mode = "parallel_refine"
    dropout: float = 0.05
    temperature: float = 1.0


class ParallelSlotCore(nn.Module):
    """v14 minimal [B,L,N,D] slot core.

    This is a separate experimental core, not a drop-in replacement for v13 yet.
    It tests the idea:

        init all layer/block slots -> parallel draft -> global/top-down refine

    Safe primitives are included so blocks can choose not to overwrite state.
    """

    OP_NAMES = ["noop", "identity", "small_refine", "local_mix", "global_bus", "memory_read", "matrix_lowrank", "normalize", "suppress"]

    def __init__(self, cfg: ParallelSlotConfig):
        super().__init__()
        self.cfg = cfg
        D, L, N = cfg.dim, cfg.layers, cfg.blocks
        self.layer_emb = nn.Parameter(torch.randn(L, D) * 0.02)
        self.block_emb = nn.Parameter(torch.randn(N, D) * 0.02)
        self.memory = nn.Parameter(torch.randn(cfg.memory_cells, D) * 0.02)
        self.in_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        self.norm = nn.LayerNorm(D)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.small = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))
        r = max(8, min(64, D // 4))
        self.low1 = nn.Linear(D, r, bias=False)
        self.low2 = nn.Linear(r, D, bias=False)
        self.op_score = nn.Sequential(nn.LayerNorm(D * 3), nn.Linear(D * 3, D), nn.GELU(), nn.Linear(D, len(self.OP_NAMES)))
        self.write_gate = nn.Sequential(nn.LayerNorm(D * 2), nn.Linear(D * 2, 1))
        self.drop = nn.Dropout(cfg.dropout)

    def init_slots(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,D] -> [B,L,N,D]
        base = self.in_proj(x).mean(dim=1)
        h = base[:, None, None, :] + self.layer_emb[None, :, None, :] + self.block_emb[None, None, :, :]
        return h

    def _lower_tri_ctx(self, h: torch.Tensor) -> torch.Tensor:
        # h [B,L,N,D], layer-level lower-triangular read over summaries.
        B, L, N, D = h.shape
        summary = h.mean(dim=2)  # [B,L,D]
        q = self.q(summary)
        k = self.k(summary)
        v = self.v(summary)
        scores = torch.einsum("bld,bmd->blm", q, k) / (D ** 0.5)
        mask = torch.tril(torch.ones(L, L, device=h.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask[None], -1e4)
        attn = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        ctx = torch.einsum("blm,bmd->bld", attn, v)
        return ctx[:, :, None, :].expand(B, L, N, D)

    def _topdown_ctx(self, h: torch.Tensor) -> torch.Tensor:
        # late aggregator summary broadcasts top-down need to earlier layers.
        agg = h[:, -1].mean(dim=1, keepdim=True)  # [B,1,D]
        return agg[:, None, :, :].expand(h.shape[0], h.shape[1], h.shape[2], h.shape[3])

    def _memory_ctx(self, h: torch.Tensor) -> torch.Tensor:
        B, L, N, D = h.shape
        mem = self.memory[None].expand(B, -1, -1)
        flat = h.reshape(B, L * N, D)
        scores = torch.einsum("bxd,bmd->bxm", self.q(flat), self.k(mem)) / (D ** 0.5)
        attn = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        ctx = torch.einsum("bxm,bmd->bxd", attn, self.v(mem))
        return ctx.reshape(B, L, N, D)

    def _candidates(self, h: torch.Tensor, ctx: torch.Tensor, bus: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        noop = torch.zeros_like(h)
        identity = h
        small = self.small(h)
        local = 0.5 * h + 0.25 * torch.roll(h, 1, dims=2) + 0.25 * torch.roll(h, -1, dims=2)
        global_bus = bus
        memory_read = mem
        low = self.low2(self.low1(h + ctx + mem))
        normalize = self.norm(h)
        suppress = h * torch.sigmoid(-self.small(h))
        return torch.stack([noop, identity, small, local, global_bus, memory_read, low, normalize, suppress], dim=-2)

    def _update_once(self, h: torch.Tensor, *, warmup_all: bool = False) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ctx = self._lower_tri_ctx(h)
        topdown = self._topdown_ctx(h)
        mem = self._memory_ctx(h)
        bus = (h.mean(dim=(1, 2), keepdim=True) + topdown).expand_as(h)
        features = torch.cat([h, ctx + topdown, mem + bus], dim=-1)
        scores = self.op_score(features)
        candidates = self._candidates(h, ctx + topdown, bus, mem)
        update, weights = dense_topk_weighted_sum(candidates, scores, self.cfg.topk, temperature=self.cfg.temperature, warmup_all=warmup_all)
        gate = torch.sigmoid(self.write_gate(torch.cat([h, update], dim=-1)))
        h2 = self.norm(h + gate * self.drop(update))
        aux = {"op_scores": scores.detach(), "op_weights": weights.detach(), "write_gate": gate.detach()}
        return h2, aux

    def forward(self, x: torch.Tensor, *, warmup_all: bool = False) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.init_slots(x)
        mode = self.cfg.mode
        aux: Dict[str, torch.Tensor] = {}
        if mode == "sequential_exact":
            # Still uses slots, but updates layers one at a time for a sequential baseline.
            outs = []
            gates = []
            weights = []
            scores = []
            cur = h
            for li in range(self.cfg.layers):
                updated, a = self._update_once(cur, warmup_all=warmup_all)
                cur = cur.clone()
                cur[:, li:li + 1] = updated[:, li:li + 1]
                outs.append(a["write_gate"][:, li].mean())
                gates.append(a["write_gate"][:, li:li + 1])
                weights.append(a["op_weights"][:, li:li + 1])
                scores.append(a["op_scores"][:, li:li + 1])
            h = cur
            aux["sequential_write_mean"] = torch.stack(outs).detach()
            aux["write_gate"] = torch.cat(gates, dim=1).detach()
            aux["op_weights"] = torch.cat(weights, dim=1).detach()
            aux["op_scores"] = torch.cat(scores, dim=1).detach()
        elif mode == "hybrid_groups":
            mid = max(1, self.cfg.layers // 2)
            h1, a1 = self._update_once(h, warmup_all=warmup_all)
            h = torch.cat([h1[:, :mid], h[:, mid:]], dim=1)
            h2, a2 = self._update_once(h, warmup_all=warmup_all)
            h = torch.cat([h[:, :mid], h2[:, mid:]], dim=1)
            aux.update({"op_weights": a2["op_weights"], "write_gate": a2["write_gate"], "op_scores": a2["op_scores"]})
        else:
            for _ in range(max(1, int(self.cfg.refine_iters))):
                h, aux = self._update_once(h, warmup_all=warmup_all)
        return h, aux

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=(1, 2))
