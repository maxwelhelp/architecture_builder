from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepBuilderConfig:
    dim: int = 128
    layers: int = 4
    blocks: int = 4
    steps: int = 4
    memory_cells: int = 6
    classes: int = 10
    dropout: float = 0.08
    read_temp: float = 1.1
    op_temp: float = 1.2
    write_bias: float = -0.35
    topk_ops: int = 3
    hard_topk_after: int = 0


class StepBuilderCore(nn.Module):
    """v14 explicit Layer/Block/Step program core.

    This is the first real step-program prototype:

        Layer -> Block -> Step -> read_source -> primitive -> write_gate

    v13 chose one soft primitive soup per block. This module makes each block a
    small algorithm made of causal steps. It is meant for utility tracking,
    skill extraction, and future grow/prune/replace controllers.
    """

    READ_NAMES = ["self", "prev_step", "prev_layer", "input", "global", "memory", "task"]
    OP_NAMES = [
        "noop", "identity", "keep_prev", "small_refine", "normalize", "suppress",
        "local_mix", "global_summary", "memory_read", "residual_delta", "matrix_lowrank", "class_hint",
    ]

    def __init__(self, cfg: StepBuilderConfig):
        super().__init__()
        self.cfg = cfg
        D, L, B, S = cfg.dim, cfg.layers, cfg.blocks, cfg.steps
        self.layer_emb = nn.Parameter(torch.randn(L, D) * 0.02)
        self.block_emb = nn.Parameter(torch.randn(B, D) * 0.02)
        self.step_emb = nn.Parameter(torch.randn(S, D) * 0.02)
        self.memory = nn.Parameter(torch.randn(cfg.memory_cells, D) * 0.02)
        self.task_bank = nn.Parameter(torch.randn(cfg.classes, D) * 0.02)
        self.input_proj = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        self.norm = nn.LayerNorm(D)
        self.read_score = nn.Sequential(nn.LayerNorm(D * 4), nn.Linear(D * 4, D), nn.GELU(), nn.Linear(D, len(self.READ_NAMES)))
        self.op_score = nn.Sequential(nn.LayerNorm(D * 4), nn.Linear(D * 4, D), nn.GELU(), nn.Linear(D, len(self.OP_NAMES)))
        self.write_gate = nn.Sequential(nn.LayerNorm(D * 3), nn.Linear(D * 3, 1))
        self.small = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D * 2), nn.GELU(), nn.Linear(D * 2, D))
        r = max(8, min(64, D // 4))
        self.low_a = nn.Linear(D, r, bias=False)
        self.low_b = nn.Linear(r, D, bias=False)
        self.q = nn.Linear(D, D, bias=False)
        self.k = nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, D, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        # Separate classifier readout remains outside if caller wants; built-in readout helps probes.
        self.classifier = nn.Linear(D, cfg.classes)

    def init_slots(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,D] -> H [B,L,N,D]
        base = self.input_proj(x).mean(dim=1)
        h = base[:, None, None, :] + self.layer_emb[None, :, None, :] + self.block_emb[None, None, :, :]
        return self.norm(h)

    def _memory_read(self, h: torch.Tensor) -> torch.Tensor:
        # h [B,L,N,D] -> [B,L,N,D]
        B, L, N, D = h.shape
        mem = self.memory[None].expand(B, -1, -1)
        flat = h.reshape(B, L * N, D)
        score = torch.einsum("bxd,bmd->bxm", self.q(flat), self.k(mem)) / (D ** 0.5)
        attn = torch.softmax(score.float(), dim=-1).to(h.dtype)
        out = torch.einsum("bxm,bmd->bxd", attn, self.v(mem)).reshape(B, L, N, D)
        return out

    def _task_read(self, h: torch.Tensor) -> torch.Tensor:
        B, L, N, D = h.shape
        task = self.task_bank[None].expand(B, -1, -1)
        flat = h.reshape(B, L * N, D)
        score = torch.einsum("bxd,bcd->bxc", self.q(flat), self.k(task)) / (D ** 0.5)
        attn = torch.softmax(score.float(), dim=-1).to(h.dtype)
        out = torch.einsum("bxc,bcd->bxd", attn, self.v(task)).reshape(B, L, N, D)
        return out

    def _read_sources(self, h: torch.Tensor, prev_step: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        # returns [B,L,N,R,D]
        B, L, N, D = h.shape
        prev_layer = torch.zeros_like(h)
        prev_layer[:, 1:] = h[:, :-1]
        global_ctx = h.mean(dim=(1, 2), keepdim=True).expand_as(h)
        mem_ctx = self._memory_read(h)
        task_ctx = self._task_read(h)
        return torch.stack([h, prev_step, prev_layer, base, global_ctx, mem_ctx, task_ctx], dim=-2)

    def _primitive_candidates(self, h: torch.Tensor, read: torch.Tensor, prev_step: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        # [B,L,N,O,D]
        noop = torch.zeros_like(h)
        identity = h
        keep_prev = prev_step
        small_refine = 0.10 * self.small(read)
        normalize = self.norm(h + read)
        suppress = h * torch.sigmoid(-self.small(read))
        local_mix = 0.5 * h + 0.25 * torch.roll(h, 1, dims=2) + 0.25 * torch.roll(h, -1, dims=2)
        global_summary = read.mean(dim=(1, 2), keepdim=True).expand_as(h)
        memory_read = self._memory_read(h + read)
        residual_delta = read - h
        matrix_lowrank = self.low_b(self.low_a(h + read))
        class_hint = self._task_read(h + read)
        return torch.stack([
            noop, identity, keep_prev, small_refine, normalize, suppress,
            local_mix, global_summary, memory_read, residual_delta, matrix_lowrank, class_hint,
        ], dim=-2)

    def _topk_weights(self, scores: torch.Tensor, temperature: float, topk: int, hard: bool) -> torch.Tensor:
        w = torch.softmax((scores / max(temperature, 1e-6)).float(), dim=-1).to(scores.dtype)
        if hard and topk > 0 and topk < scores.shape[-1]:
            vals, idx = torch.topk(w, k=topk, dim=-1)
            mask = torch.zeros_like(w).scatter_(-1, idx, 1.0)
            w_hard = w * mask
            w_hard = w_hard / w_hard.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            # straight-through: hard forward, soft backward
            w = w_hard.detach() - w.detach() + w
        return w

    def forward(self, x: torch.Tensor, *, epoch: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        Bsz = x.shape[0]
        h = self.init_slots(x)
        base = h.clone()
        prev_step = h
        aux = {
            "read_weights": [], "op_weights": [], "write_gate": [], "update_norm": [], "step_states": [],
        }
        hard = self.cfg.hard_topk_after > 0 and epoch >= self.cfg.hard_topk_after
        for si in range(self.cfg.steps):
            step_e = self.step_emb[si].view(1, 1, 1, -1).to(dtype=h.dtype, device=h.device)
            sources = self._read_sources(h, prev_step, base)
            features = torch.cat([h, prev_step, base, step_e.expand_as(h)], dim=-1)
            read_logits = self.read_score(features)
            read_w = torch.softmax((read_logits / max(self.cfg.read_temp, 1e-6)).float(), dim=-1).to(h.dtype)
            read_ctx = torch.einsum("blnr,blnrd->blnd", read_w, sources)
            op_logits = self.op_score(torch.cat([h, read_ctx, prev_step, step_e.expand_as(h)], dim=-1))
            op_w = self._topk_weights(op_logits, self.cfg.op_temp, self.cfg.topk_ops, hard)
            candidates = self._primitive_candidates(h, read_ctx, prev_step, base)
            update = torch.einsum("blno,blnod->blnd", op_w, candidates)
            gate_logits = self.write_gate(torch.cat([h, read_ctx, update], dim=-1)) + float(self.cfg.write_bias)
            gate = torch.sigmoid(gate_logits)
            prev_step = self.norm(h + gate * self.drop(update))
            h = prev_step
            aux["read_weights"].append(read_w.detach())
            aux["op_weights"].append(op_w.detach())
            aux["write_gate"].append(gate.detach())
            aux["update_norm"].append(update.detach().float().pow(2).sum(dim=-1).sqrt())
            aux["step_states"].append(h.detach())
        packed = {
            "read_weights": torch.stack(aux["read_weights"], dim=3),   # [B,L,N,S,R]
            "op_weights": torch.stack(aux["op_weights"], dim=3),       # [B,L,N,S,O]
            "write_gate": torch.stack(aux["write_gate"], dim=3),       # [B,L,N,S,1]
            "update_norm": torch.stack(aux["update_norm"], dim=3),     # [B,L,N,S]
            "step_states": torch.stack(aux["step_states"], dim=3),     # [B,L,N,S,D]
        }
        return h, packed

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return h.mean(dim=(1, 2))

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.readout(h))
