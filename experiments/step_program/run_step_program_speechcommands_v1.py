#!/usr/bin/env python3
"""
Clean differentiable StepProgram experiment for SpeechCommands.

This file intentionally does NOT reuse the large v13 logic.  It tests the core
idea directly:

  1. differentiable operations inside a step;
  2. differentiable transitions between primitives inside a step;
  3. differentiable transitions between steps;
  4. differentiable input/output reads;
  5. differentiable layer/block routes;
  6. explicit logging of which parts are used and by whom.

Addressing convention used in reports:

  L{layer}.B{block}.S{step}.P{primitive_slot}
  L{layer}.B{block}.S{step}.P{k}->P{k+1}
  L{layer}.B{block}.S{step}.read
  L{layer}.B{block}.route
  output.class_read

The goal is not to beat v13 immediately.  The goal is to verify that a small
step-program core can learn readable differentiable mini-programs end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import torchaudio
except Exception:  # pragma: no cover - allows synthetic smoke without torchaudio
    torchaudio = None


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_amp_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def entropy_from_probs(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    pp = p.float().clamp_min(1e-8)
    return -(pp * pp.log()).sum(dim=dim)


def top_items(names: Sequence[str], weights: torch.Tensor, k: int = 4) -> List[Dict[str, float]]:
    if weights.numel() == 0:
        return []
    vals, idxs = torch.topk(weights.float(), k=min(k, weights.numel()))
    out = []
    for v, j in zip(vals.tolist(), idxs.tolist()):
        jj = int(j)
        out.append({"name": names[jj] if jj < len(names) else f"idx_{jj}", "weight": float(v)})
    return out


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------


class SyntheticSpeechLike(Dataset):
    """Tiny deterministic dataset for smoke tests when real SpeechCommands is absent."""

    def __init__(self, n: int, num_classes: int, sample_rate: int = 16000, seconds: float = 1.0):
        self.n = int(n)
        self.num_classes = int(num_classes)
        self.sample_rate = int(sample_rate)
        self.length = int(sample_rate * seconds)
        t = torch.linspace(0, seconds, self.length)
        self.base = t

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        y = int(idx % self.num_classes)
        # Class-dependent frequency + phase; enough for smoke learning signal.
        freq = 180.0 + 55.0 * y
        phase = 0.17 * (idx % 11)
        wav = torch.sin(2 * math.pi * freq * self.base + phase)
        wav = wav + 0.05 * torch.randn_like(wav)
        return wav.unsqueeze(0), y


class SpeechCommandsFiltered(Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        classes: Sequence[str],
        limit: int = 0,
        download: bool = False,
    ):
        if torchaudio is None:
            raise RuntimeError("torchaudio is not available; use --synthetic for smoke tests")
        self.classes = list(classes)
        self.class_to_id = {c: i for i, c in enumerate(self.classes)}
        self.ds = torchaudio.datasets.SPEECHCOMMANDS(root=root, subset=subset, download=download)

        keep: List[int] = []
        for i in range(len(self.ds)):
            label = self._label_for_index(i)
            if label in self.class_to_id:
                keep.append(i)
                if limit and len(keep) >= limit:
                    break
        self.keep = keep

    def _label_for_index(self, i: int) -> str:
        # Fast path for torchaudio's SPEECHCOMMANDS internals.
        try:
            p = Path(self.ds._walker[i])
            return p.parent.name
        except Exception:
            item = self.ds[i]
            return str(item[2])

    def __len__(self) -> int:
        return len(self.keep)

    def __getitem__(self, j: int):
        item = self.ds[self.keep[j]]
        wav, sr, label = item[0], int(item[1]), str(item[2])
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        y = self.class_to_id[label]
        return wav, y


def collate_wavs(batch, sample_rate: int, seconds: float):
    target_len = int(sample_rate * seconds)
    wavs, ys = [], []
    for wav, y in batch:
        wav = wav.float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[-1] < target_len:
            wav = F.pad(wav, (0, target_len - wav.shape[-1]))
        elif wav.shape[-1] > target_len:
            wav = wav[..., :target_len]
        wavs.append(wav)
        ys.append(int(y))
    return torch.stack(wavs, dim=0), torch.tensor(ys, dtype=torch.long)


def make_loaders(args):
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.synthetic:
        train_ds = SyntheticSpeechLike(args.train_limit or 2048, len(classes), args.sample_rate, args.seconds)
        val_ds = SyntheticSpeechLike(args.val_limit or 512, len(classes), args.sample_rate, args.seconds)
    else:
        train_ds = SpeechCommandsFiltered(args.data_root, "training", classes, args.train_limit, args.download)
        val_ds = SpeechCommandsFiltered(args.data_root, "validation", classes, args.val_limit, args.download)

    collate = lambda b: collate_wavs(b, args.sample_rate, args.seconds)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        collate_fn=collate,
        drop_last=False,
    )
    return train_loader, val_loader, classes


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


@dataclass
class StepProgramAux:
    primitive_gates: torch.Tensor              # [B,L,N,S,K,O]
    primitive_transition_gates: torch.Tensor   # [B,L,N,S,K-1,T]
    step_read_gates: torch.Tensor              # [B,L,N,S,R]
    layer_route_gates: torch.Tensor            # [B,L,N,LR]
    step_write_gates: torch.Tensor             # [B,L,N,S]
    global_write_gates: torch.Tensor           # [B,L,N,S]
    memory_write_gates: torch.Tensor           # [B,L,N,S]
    class_read: torch.Tensor                   # [B,C,1+L*N*S]
    slot_states: torch.Tensor                  # [B,1+L*N*S,D]
    logits: torch.Tensor
    slot_names: List[str]


class AudioEvidence(nn.Module):
    """Turns waveform into a small EvidenceMatrix [B,E,D]."""

    def __init__(self, sample_rate: int, n_mels: int, hop_length: int, evidence_cells: int, dim: int):
        super().__init__()
        if torchaudio is not None:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=400,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
        else:
            self.mel = None
        self.evidence_cells = int(evidence_cells)
        self.scalar_proj = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.randn(evidence_cells, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav [B,1,T]
        B = wav.shape[0]
        if self.mel is not None:
            x = self.mel(wav.squeeze(1)).clamp_min(1e-8).log()  # [B,n_mels,Tm]
        else:
            # Fallback synthetic spectrogram from framed absolute waveform.
            x = wav.squeeze(1).unfold(-1, 320, 160).abs().mean(dim=-1).unsqueeze(1).repeat(1, 64, 1)
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)

        grid = F.adaptive_avg_pool2d(x.unsqueeze(1), (6, 6)).squeeze(1).flatten(1)  # [B,36]
        time3 = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (1, 3)).flatten(1)  # [B,3]
        freq4 = F.adaptive_avg_pool2d(x.mean(dim=2, keepdim=True), (4, 1)).flatten(1)  # [B,4]
        global_stats = torch.stack([
            x.mean(dim=(-2, -1)),
            x.std(dim=(-2, -1)),
            x.amax(dim=(-2, -1)),
            x.amin(dim=(-2, -1)),
        ], dim=1)
        delta_time = time3[:, 1:] - time3[:, :-1]
        feats = torch.cat([grid, time3, freq4, global_stats, delta_time], dim=1)
        if feats.shape[1] < self.evidence_cells:
            rep = math.ceil(self.evidence_cells / feats.shape[1])
            feats = feats.repeat(1, rep)
        feats = feats[:, : self.evidence_cells]
        ev = self.scalar_proj(feats.unsqueeze(-1)) + self.pos.view(1, self.evidence_cells, -1)
        return self.norm(ev)


class StepProgramNet(nn.Module):
    PRIMITIVE_NAMES = [
        "noop", "identity", "keep_state", "small_refine",
        "mlp", "matrix_mlp", "compare", "memory_read", "global_read", "normalize", "suppress",
    ]
    PRIMITIVE_COSTS = [0.05, 0.05, 0.05, 0.20, 1.00, 1.35, 0.75, 0.45, 0.40, 0.30, 0.45]
    PRIM_TRANSITION_NAMES = ["keep", "replace", "residual", "norm_residual", "product_gate", "compare_mix"]
    STEP_READ_NAMES = ["state", "input", "prev_step", "all_prev_steps", "layer_route", "global", "memory", "task"]
    LAYER_ROUTE_NAMES = ["input", "prev_same_block", "prev_layer_mean", "global_mean", "memory_mean"]

    def __init__(
        self,
        num_classes: int,
        dim: int = 128,
        evidence_cells: int = 48,
        num_layers: int = 4,
        blocks_per_layer: int = 4,
        steps_per_block: int = 4,
        primitive_slots: int = 3,
        global_cells: int = 3,
        memory_cells: int = 6,
        dropout: float = 0.05,
        sample_rate: int = 16000,
        n_mels: int = 64,
        hop_length: int = 160,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.evidence_cells = int(evidence_cells)
        self.L = int(num_layers)
        self.N = int(blocks_per_layer)
        self.S = int(steps_per_block)
        self.K = int(primitive_slots)
        self.G = int(global_cells)
        self.M = int(memory_cells)
        self.O = len(self.PRIMITIVE_NAMES)
        self.T = len(self.PRIM_TRANSITION_NAMES)
        self.R = len(self.STEP_READ_NAMES)
        self.LR = len(self.LAYER_ROUTE_NAMES)

        self.evidence = AudioEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)
        self.input_q = nn.Linear(dim, dim, bias=False)
        self.input_k = nn.Linear(dim, dim, bias=False)
        self.input_v = nn.Linear(dim, dim, bias=False)
        self.base_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

        self.layer_addr = nn.Parameter(torch.randn(self.L, dim) * 0.03)
        self.block_addr = nn.Parameter(torch.randn(self.N, dim) * 0.03)
        self.step_addr = nn.Parameter(torch.randn(self.S, dim) * 0.03)
        self.prim_slot_addr = nn.Parameter(torch.randn(self.K, dim) * 0.03)
        self.task_token = nn.Parameter(torch.randn(1, 1, dim) * 0.03)
        self.global_init = nn.Parameter(torch.randn(self.G, dim) * 0.03)
        self.memory_init = nn.Parameter(torch.randn(self.M, dim) * 0.03)

        self.layer_route_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, self.LR))
        self.step_read_net = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim), nn.GELU(), nn.Linear(dim, self.R))
        self.primitive_gate_net = nn.Sequential(nn.LayerNorm(dim * 6), nn.Linear(dim * 6, dim), nn.GELU(), nn.Linear(dim, self.O))
        self.prim_transition_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, self.T))
        self.write_gate_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))

        self.mlp_prim = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.small_refine = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.compare_prim = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.suppress_gate = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.suppress_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm_prim = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        self.matrix_q = nn.Linear(dim, dim, bias=False)
        self.matrix_k = nn.Linear(dim, dim, bias=False)
        self.matrix_v = nn.Linear(dim, dim, bias=False)
        self.matrix_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        self.global_q = nn.Linear(dim, dim, bias=False)
        self.global_k = nn.Linear(dim, dim, bias=False)
        self.global_v = nn.Linear(dim, dim, bias=False)
        self.memory_q = nn.Linear(dim, dim, bias=False)
        self.memory_k = nn.Linear(dim, dim, bias=False)
        self.memory_v = nn.Linear(dim, dim, bias=False)
        self.global_write_gate = nn.Linear(dim * 3, 1)
        self.memory_write_gate = nn.Linear(dim * 3, 1)
        self.global_write_val = nn.Linear(dim * 3, dim)
        self.memory_write_val = nn.Linear(dim * 3, dim)

        self.transition_compare = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.transition_value_gate = nn.Linear(dim * 3, dim)

        self.state_norm = nn.LayerNorm(dim)
        self.step_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)

        self.class_query = nn.Parameter(torch.randn(num_classes, dim) * 0.05)
        self.class_k = nn.Linear(dim, dim, bias=False)
        self.class_v = nn.Linear(dim, dim, bias=False)
        self.class_write = nn.Parameter(torch.randn(num_classes, dim) * 0.03)
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))
        self.register_buffer("primitive_cost", torch.tensor(self.PRIMITIVE_COSTS, dtype=torch.float32))

    def _attend_cells(self, q: torch.Tensor, cells: torch.Tensor, k_proj: nn.Linear, v_proj: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = 1.0 / math.sqrt(q.shape[-1])
        k = k_proj(cells)
        v = v_proj(cells)
        score = torch.einsum("bnd,bmd->bnm", q, k) * scale
        a = torch.softmax(score.float(), dim=-1).to(q.dtype)
        ctx = torch.einsum("bnm,bmd->bnd", a, v)
        return ctx, a

    def _input_read(self, state: torch.Tensor, evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = 1.0 / math.sqrt(state.shape[-1])
        q = self.input_q(state)
        k = self.input_k(evidence)
        v = self.input_v(evidence)
        score = torch.einsum("bnd,bed->bne", q, k) * scale
        a = torch.softmax(score.float(), dim=-1).to(state.dtype)
        ctx = torch.einsum("bne,bed->bnd", a, v)
        return ctx, a

    def _primitive_candidates(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        read_ctx: torch.Tensor,
        layer_blocks: torch.Tensor,
        global_ctx: torch.Tensor,
        memory_ctx: torch.Tensor,
        task_ctx: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape
        scale = 1.0 / math.sqrt(D)
        zeros = torch.zeros_like(x)
        keep_state = state
        small = 0.10 * self.small_refine(torch.cat([x, read_ctx], dim=-1))
        mlp = self.mlp_prim(torch.cat([x, read_ctx, global_ctx, memory_ctx], dim=-1))

        mq, mk, mv = self.matrix_q(x), self.matrix_k(layer_blocks), self.matrix_v(layer_blocks)
        ma = torch.softmax(torch.einsum("bnd,bmd->bnm", mq, mk).float() * scale, dim=-1).to(x.dtype)
        mctx = torch.einsum("bnm,bmd->bnd", ma, mv)
        matrix_mlp = self.matrix_out(torch.cat([x, mctx, global_ctx, memory_ctx], dim=-1))

        diff = x - state
        prod = x * state
        compare = self.compare_prim(torch.cat([diff, prod, read_ctx, task_ctx], dim=-1))
        normed = self.norm_prim(torch.cat([F.layer_norm(x, (D,)), F.layer_norm(read_ctx, (D,)), F.layer_norm(global_ctx + memory_ctx, (D,))], dim=-1))
        sg = torch.sigmoid(self.suppress_gate(torch.cat([x, global_ctx + memory_ctx, task_ctx], dim=-1)))
        suppress = self.suppress_out(x - sg * (global_ctx + memory_ctx))
        return torch.stack([
            zeros,
            x,
            keep_state,
            small,
            mlp,
            matrix_mlp,
            compare,
            memory_ctx,
            global_ctx,
            normed,
            suppress,
        ], dim=2)

    def forward(self, wav: torch.Tensor, primitive_temp: float = 1.0, read_temp: float = 1.0, return_aux: bool = True):
        B = wav.shape[0]
        device = wav.device
        dtype = wav.dtype
        D = self.dim
        scale = 1.0 / math.sqrt(D)

        evidence = self.evidence(wav)
        base = self.base_proj(evidence.mean(dim=1))
        task_ctx_base = self.task_token.to(device=device, dtype=dtype).expand(B, 1, D)
        global_cells = self.global_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) + base[:, None, :] * 0.10
        memory_cells = self.memory_init.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1) + base[:, None, :] * 0.10

        # Current layer states [B,N,D]. Blocks start as different views of the input.
        block_addr = self.block_addr.to(device=device, dtype=dtype).view(1, self.N, D)
        prev_layer = self.state_norm(base[:, None, :] + block_addr)
        all_slot_states: List[torch.Tensor] = []
        slot_names: List[str] = []

        primitive_gates_all = []
        prim_trans_all = []
        step_read_all = []
        layer_route_all = []
        step_write_all = []
        global_write_all = []
        memory_write_all = []

        for l in range(self.L):
            layer_addr = self.layer_addr[l].to(device=device, dtype=dtype).view(1, 1, D)
            state = self.state_norm(prev_layer + layer_addr + block_addr)

            # Layer/block route: differentiable operation between layers/blocks.
            global_mean = global_cells.mean(dim=1).unsqueeze(1).expand(-1, self.N, -1)
            memory_mean = memory_cells.mean(dim=1).unsqueeze(1).expand(-1, self.N, -1)
            prev_mean = prev_layer.mean(dim=1).unsqueeze(1).expand(-1, self.N, -1)
            route_logits = self.layer_route_net(torch.cat([state, prev_layer, global_mean, memory_mean], dim=-1))
            route_gates = torch.softmax(route_logits.float(), dim=-1).to(dtype)
            route_sources = torch.stack([
                base[:, None, :].expand(-1, self.N, -1),
                prev_layer,
                prev_mean,
                global_mean,
                memory_mean,
            ], dim=2)
            layer_route_ctx = torch.einsum("bnr,bnrd->bnd", route_gates, route_sources)
            layer_route_all.append(route_gates)

            prev_steps: List[torch.Tensor] = []
            for s in range(self.S):
                step_addr = self.step_addr[s].to(device=device, dtype=dtype).view(1, 1, D)
                input_ctx, _ = self._input_read(state + step_addr, evidence)
                global_ctx, _ = self._attend_cells(state + step_addr, global_cells, self.global_k, self.global_v)
                memory_ctx, _ = self._attend_cells(state + step_addr, memory_cells, self.memory_k, self.memory_v)
                task_ctx = task_ctx_base.expand(-1, self.N, -1)

                if prev_steps:
                    prev_step = prev_steps[-1]
                    all_prev = torch.stack(prev_steps, dim=2).mean(dim=2)
                else:
                    prev_step = torch.zeros_like(state)
                    all_prev = torch.zeros_like(state)

                read_logits = self.step_read_net(torch.cat([state, input_ctx, layer_route_ctx, global_ctx, memory_ctx], dim=-1))
                read_gates = torch.softmax((read_logits / max(1e-4, read_temp)).float(), dim=-1).to(dtype)
                read_sources = torch.stack([
                    state,
                    input_ctx,
                    prev_step,
                    all_prev,
                    layer_route_ctx,
                    global_ctx,
                    memory_ctx,
                    task_ctx,
                ], dim=2)
                read_ctx = torch.einsum("bnr,bnrd->bnd", read_gates, read_sources)
                step_read_all.append(read_gates)

                x = read_ctx
                prim_gates_slots = []
                trans_gates_slots = []
                for kslot in range(self.K):
                    prim_addr = self.prim_slot_addr[kslot].to(device=device, dtype=dtype).view(1, 1, D)
                    prim_logits = self.primitive_gate_net(torch.cat([x + prim_addr, state, read_ctx, task_ctx, global_ctx, memory_ctx], dim=-1))
                    prim_gates = torch.softmax((prim_logits / max(1e-4, primitive_temp)).float(), dim=-1).to(dtype)
                    cands = self._primitive_candidates(x, state, read_ctx, state, global_ctx, memory_ctx, task_ctx)
                    u = torch.einsum("bno,bnod->bnd", prim_gates, cands)
                    prim_gates_slots.append(prim_gates)

                    if kslot + 1 < self.K:
                        trans_logits = self.prim_transition_net(torch.cat([x, u, read_ctx, state], dim=-1))
                        trans_gates = torch.softmax(trans_logits.float(), dim=-1).to(dtype)
                        gate = torch.sigmoid(self.transition_value_gate(torch.cat([x, u, read_ctx], dim=-1)))
                        trans_cands = torch.stack([
                            x,
                            u,
                            x + u,
                            self.step_norm(x + u),
                            x * torch.sigmoid(u),
                            self.transition_compare(torch.cat([x - u, x * u, read_ctx, state], dim=-1)),
                        ], dim=2)
                        # gated residual is folded into residual candidate by value gate.
                        trans_cands[:, :, 2, :] = x + gate * u
                        x = torch.einsum("bnt,bntd->bnd", trans_gates, trans_cands)
                        trans_gates_slots.append(trans_gates)
                    else:
                        x = u

                prim_gates_step = torch.stack(prim_gates_slots, dim=2)  # [B,N,K,O]
                if trans_gates_slots:
                    trans_gates_step = torch.stack(trans_gates_slots, dim=2)  # [B,N,K-1,T]
                else:
                    trans_gates_step = torch.empty(B, self.N, 0, self.T, device=device, dtype=dtype)

                write_in = torch.cat([state, x, read_ctx, task_ctx], dim=-1)
                write_gate = torch.sigmoid(self.write_gate_net(write_in)).squeeze(-1)
                # If the step selected noop/identity/keep_state heavily, let it avoid writing.
                safe_mass = prim_gates_step[..., :3].mean(dim=2).sum(dim=-1).clamp(0.0, 1.0)
                write_gate = write_gate * (1.0 - 0.70 * safe_mass).clamp(0.05, 1.0)
                state = self.state_norm(state + write_gate.unsqueeze(-1) * x)

                # Differentiable output/register writes from this step.
                gw_in = torch.cat([state, x, read_ctx], dim=-1)
                mw_in = torch.cat([state, x, memory_ctx], dim=-1)
                global_write_gate = 0.10 * torch.sigmoid(self.global_write_gate(gw_in)).squeeze(-1)
                memory_write_gate = 0.10 * torch.sigmoid(self.memory_write_gate(mw_in)).squeeze(-1)
                # Tie writes to relevant primitive mass.
                global_mass = prim_gates_step[..., [8]].mean(dim=2).sum(dim=-1).clamp(0.0, 1.0)  # global_read as global/global op proxy
                memory_mass = prim_gates_step[..., [7]].mean(dim=2).sum(dim=-1).clamp(0.0, 1.0)
                global_write_gate = global_write_gate * (0.05 + 0.95 * global_mass)
                memory_write_gate = memory_write_gate * (0.05 + 0.95 * memory_mass)

                gw_val = self.global_write_val(gw_in)
                mw_val = self.memory_write_val(mw_in)
                gscore = torch.einsum("bnd,bgd->bng", self.global_q(state), self.global_k(global_cells)) * scale
                mscore = torch.einsum("bnd,bmd->bnm", self.memory_q(state), self.memory_k(memory_cells)) * scale
                ga = torch.softmax(gscore.float(), dim=-1).to(dtype)
                ma = torch.softmax(mscore.float(), dim=-1).to(dtype)
                global_cells = self.global_norm(global_cells + torch.einsum("bng,bnd->bgd", ga * global_write_gate.unsqueeze(-1), gw_val) / max(1, self.N))
                memory_cells = self.memory_norm(memory_cells + torch.einsum("bnm,bnd->bmd", ma * memory_write_gate.unsqueeze(-1), mw_val) / max(1, self.N))

                primitive_gates_all.append(prim_gates_step)
                prim_trans_all.append(trans_gates_step)
                step_write_all.append(write_gate)
                global_write_all.append(global_write_gate)
                memory_write_all.append(memory_write_gate)
                prev_steps.append(state)
                all_slot_states.append(state)
                for b in range(self.N):
                    slot_names.append(f"L{l}.B{b}.S{s}")

            prev_layer = state

        slot_states = torch.cat([base[:, None, :]] + [x for x in all_slot_states], dim=1)
        full_slot_names = ["input/base"] + slot_names
        q_cls = self.class_query.to(device=device, dtype=dtype).view(1, self.num_classes, D).expand(B, -1, -1)
        k_cls = self.class_k(slot_states)
        v_cls = self.class_v(slot_states)
        cls_score = torch.einsum("bcd,btd->bct", q_cls, k_cls) * scale
        cls_attn = torch.softmax(cls_score.float(), dim=-1).to(dtype)
        cls_ctx = torch.einsum("bct,btd->bcd", cls_attn, v_cls)
        logits = torch.einsum("bcd,cd->bc", cls_ctx, self.class_write.to(device=device, dtype=dtype)) + self.logit_bias.to(device=device, dtype=dtype)

        if not return_aux:
            return logits, None

        # Convert lists ordered by loop [L,S] with tensors [B,N,...] into [B,L,N,S,...].
        primitive_gates = torch.stack(primitive_gates_all, dim=1).view(B, self.L, self.S, self.N, self.K, self.O).permute(0, 1, 3, 2, 4, 5).contiguous()
        primitive_transition_gates = torch.stack(prim_trans_all, dim=1).view(B, self.L, self.S, self.N, max(0, self.K - 1), self.T).permute(0, 1, 3, 2, 4, 5).contiguous()
        step_read_gates = torch.stack(step_read_all, dim=1).view(B, self.L, self.S, self.N, self.R).permute(0, 1, 3, 2, 4).contiguous()
        step_write_gates = torch.stack(step_write_all, dim=1).view(B, self.L, self.S, self.N).permute(0, 1, 3, 2).contiguous()
        global_write_gates = torch.stack(global_write_all, dim=1).view(B, self.L, self.S, self.N).permute(0, 1, 3, 2).contiguous()
        memory_write_gates = torch.stack(memory_write_all, dim=1).view(B, self.L, self.S, self.N).permute(0, 1, 3, 2).contiguous()
        layer_route_gates = torch.stack(layer_route_all, dim=1)  # [B,L,N,LR]

        aux = StepProgramAux(
            primitive_gates=primitive_gates,
            primitive_transition_gates=primitive_transition_gates,
            step_read_gates=step_read_gates,
            layer_route_gates=layer_route_gates,
            step_write_gates=step_write_gates,
            global_write_gates=global_write_gates,
            memory_write_gates=memory_write_gates,
            class_read=cls_attn,
            slot_states=slot_states,
            logits=logits,
            slot_names=full_slot_names,
        )
        return logits, aux


# -----------------------------------------------------------------------------
# Reports / losses
# -----------------------------------------------------------------------------


def aux_losses(model: StepProgramNet, aux: StepProgramAux, epoch: int, epochs: int, args) -> Dict[str, torch.Tensor]:
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, epochs - 1)))
    sharpen = min(1.0, max(0.0, (progress - 0.30) / 0.55))
    explore = max(0.0, 1.0 - progress / 0.45)

    pg = aux.primitive_gates.float()
    pr_cost = model.primitive_cost.to(pg.device).view(1, 1, 1, 1, 1, -1)
    cost_loss = (pg * pr_cost).sum(dim=-1).mean()
    prim_entropy = entropy_from_probs(pg, dim=-1).mean()
    step_read_entropy = entropy_from_probs(aux.step_read_gates.float(), dim=-1).mean()
    trans_entropy = entropy_from_probs(aux.primitive_transition_gates.float(), dim=-1).mean() if aux.primitive_transition_gates.numel() else torch.zeros((), device=pg.device)
    write_l1 = aux.step_write_gates.float().mean()

    # Diversity over block final states: prevents all blocks becoming the same organ.
    slots = aux.slot_states[:, 1:, :].float()
    if slots.shape[1] > 1:
        z = F.normalize(slots.mean(dim=0), dim=-1)
        sim = z @ z.T
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
        slot_collapse = sim.masked_select(~eye).pow(2).mean()
    else:
        slot_collapse = torch.zeros((), device=pg.device)

    return {
        "cost_loss": cost_loss,
        "prim_entropy": prim_entropy,
        "step_read_entropy": step_read_entropy,
        "prim_transition_entropy": trans_entropy,
        "write_l1": write_l1,
        "slot_collapse": slot_collapse,
        "lambda_cost_eff": torch.tensor(float(args.lambda_cost) * (0.20 + 0.80 * sharpen), device=pg.device),
        "lambda_prim_entropy_eff": torch.tensor(float(args.lambda_primitive_entropy) * sharpen, device=pg.device),
        "lambda_read_entropy_eff": torch.tensor(float(args.lambda_read_entropy) * sharpen, device=pg.device),
        "lambda_transition_entropy_eff": torch.tensor(float(args.lambda_transition_entropy) * sharpen, device=pg.device),
        "lambda_write_l1_eff": torch.tensor(float(args.lambda_write_l1) * (0.25 + 0.75 * sharpen), device=pg.device),
        "lambda_slot_collapse_eff": torch.tensor(float(args.lambda_slot_collapse), device=pg.device),
        "explore_w": torch.tensor(float(explore), device=pg.device),
        "sharpen_w": torch.tensor(float(sharpen), device=pg.device),
    }


def make_program_report(model: StepProgramNet, aux: StepProgramAux, classes: Sequence[str]) -> Dict:
    with torch.no_grad():
        pg = aux.primitive_gates.float().mean(dim=0)          # [L,N,S,K,O]
        tg = aux.primitive_transition_gates.float().mean(dim=0) if aux.primitive_transition_gates.numel() else torch.empty(model.L, model.N, model.S, 0, model.T)
        rg = aux.step_read_gates.float().mean(dim=0)          # [L,N,S,R]
        lr = aux.layer_route_gates.float().mean(dim=0)        # [L,N,LR]
        wg = aux.step_write_gates.float().mean(dim=0)         # [L,N,S]
        gw = aux.global_write_gates.float().mean(dim=0)
        mw = aux.memory_write_gates.float().mean(dim=0)
        cls = aux.class_read.float().mean(dim=0)              # [C,T]
        slot_use = cls.mean(dim=0)                            # [T]

        levels = {
            "1_operations_inside_step": [],
            "2_operations_between_primitives_inside_step": [],
            "3_operations_between_steps_step_read": [],
            "4_operations_between_layers_blocks": [],
            "5_which_parts_are_needed_utility": [],
            "output_class_read": [],
        }

        for l in range(model.L):
            for b in range(model.N):
                levels["4_operations_between_layers_blocks"].append({
                    "address": f"L{l}.B{b}.route",
                    "top_routes": top_items(model.LAYER_ROUTE_NAMES, lr[l, b], 5),
                })
                for s in range(model.S):
                    step_addr = f"L{l}.B{b}.S{s}"
                    levels["3_operations_between_steps_step_read"].append({
                        "address": f"{step_addr}.read",
                        "top_reads": top_items(model.STEP_READ_NAMES, rg[l, b, s], 5),
                    })
                    slot_index = 1 + ((l * model.S + s) * model.N + b)
                    levels["5_which_parts_are_needed_utility"].append({
                        "address": step_addr,
                        "write_gate": float(wg[l, b, s]),
                        "global_write_gate": float(gw[l, b, s]),
                        "memory_write_gate": float(mw[l, b, s]),
                        "class_read_mass": float(slot_use[slot_index]) if slot_index < slot_use.numel() else 0.0,
                        "utility_proxy": float(wg[l, b, s] * (slot_use[slot_index] if slot_index < slot_use.numel() else 0.0)),
                    })
                    for k in range(model.K):
                        levels["1_operations_inside_step"].append({
                            "address": f"{step_addr}.P{k}",
                            "top_primitives": top_items(model.PRIMITIVE_NAMES, pg[l, b, s, k], 5),
                        })
                    for k in range(max(0, model.K - 1)):
                        levels["2_operations_between_primitives_inside_step"].append({
                            "address": f"{step_addr}.P{k}->P{k+1}",
                            "top_transitions": top_items(model.PRIM_TRANSITION_NAMES, tg[l, b, s, k], 4),
                        })

        for ci, cname in enumerate(classes):
            levels["output_class_read"].append({
                "class": cname,
                "top_slots": top_items(aux.slot_names, cls[ci], 8),
            })

        return {
            "primitive_names": list(model.PRIMITIVE_NAMES),
            "primitive_transition_names": list(model.PRIM_TRANSITION_NAMES),
            "step_read_names": list(model.STEP_READ_NAMES),
            "layer_route_names": list(model.LAYER_ROUTE_NAMES),
            "levels": levels,
        }


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, scaler, device, amp_dtype, epoch, args, classes):
    model.train()
    use_amp = device.startswith("cuda") and amp_dtype != torch.float32
    total_loss = total_ce = total_correct = total = 0.0
    loss_sums: Dict[str, float] = {}
    last_aux: Optional[StepProgramAux] = None
    t0 = time.time()

    progress = min(1.0, max(0.0, (epoch - 1) / max(1, args.epochs - 1)))
    primitive_temp = args.primitive_temp_start + (args.primitive_temp_end - args.primitive_temp_start) * progress
    read_temp = args.read_temp_start + (args.read_temp_end - args.read_temp_start) * progress

    for step, (wav, y) in enumerate(loader, start=1):
        if args.max_train_batches and step > args.max_train_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(wav, primitive_temp=primitive_temp, read_temp=read_temp, return_aux=True)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses(model, aux, epoch, args.epochs, args)
            loss = ce
            loss = loss + losses["lambda_cost_eff"] * losses["cost_loss"]
            loss = loss + losses["lambda_prim_entropy_eff"] * losses["prim_entropy"]
            loss = loss + losses["lambda_read_entropy_eff"] * losses["step_read_entropy"]
            loss = loss + losses["lambda_transition_entropy_eff"] * losses["prim_transition_entropy"]
            loss = loss + losses["lambda_write_l1_eff"] * losses["write_l1"]
            loss = loss + losses["lambda_slot_collapse_eff"] * losses["slot_collapse"]
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS: skip batch", flush=True)
            continue
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            bs = y.numel()
            total_loss += float(loss.detach().cpu()) * bs
            total_ce += float(ce.detach().cpu()) * bs
            total_correct += int((pred == y).sum().detach().cpu())
            total += bs
            for k, v in losses.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach().cpu()) * bs
            last_aux = StepProgramAux(
                primitive_gates=aux.primitive_gates.detach().cpu(),
                primitive_transition_gates=aux.primitive_transition_gates.detach().cpu(),
                step_read_gates=aux.step_read_gates.detach().cpu(),
                layer_route_gates=aux.layer_route_gates.detach().cpu(),
                step_write_gates=aux.step_write_gates.detach().cpu(),
                global_write_gates=aux.global_write_gates.detach().cpu(),
                memory_write_gates=aux.memory_write_gates.detach().cpu(),
                class_read=aux.class_read.detach().cpu(),
                slot_states=aux.slot_states.detach().cpu(),
                logits=aux.logits.detach().cpu(),
                slot_names=aux.slot_names,
            )

        if args.log_every and step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} step {step:05d} loss={total_loss/max(1,total):.4f} "
                f"ce={total_ce/max(1,total):.4f} acc={100*total_correct/max(1,total):.2f}% "
                f"seen={int(total)} t={time.time()-t0:.1f}s temp={primitive_temp:.2f}/{read_temp:.2f}",
                flush=True,
            )

    out = {
        "loss": total_loss / max(1, total),
        "ce": total_ce / max(1, total),
        "acc": total_correct / max(1, total),
        "n": int(total),
        "primitive_temp": primitive_temp,
        "read_temp": read_temp,
        "last_aux": last_aux,
    }
    for k, v in loss_sums.items():
        out[k] = v / max(1, total)
    return out


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, args, classes):
    model.eval()
    use_amp = device.startswith("cuda") and amp_dtype != torch.float32
    total_loss = total_correct = total = 0.0
    C = len(classes)
    conf = torch.zeros(C, C, dtype=torch.long)
    last_aux: Optional[StepProgramAux] = None
    for step, (wav, y) in enumerate(loader, start=1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(wav, primitive_temp=args.primitive_temp_end, read_temp=args.read_temp_end, return_aux=True)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(dim=-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        total_correct += int((pred == y).sum().detach().cpu())
        total += bs
        idx = y.detach().cpu() * C + pred.detach().cpu()
        conf += torch.bincount(idx, minlength=C * C).view(C, C)
        last_aux = StepProgramAux(
            primitive_gates=aux.primitive_gates.detach().cpu(),
            primitive_transition_gates=aux.primitive_transition_gates.detach().cpu(),
            step_read_gates=aux.step_read_gates.detach().cpu(),
            layer_route_gates=aux.layer_route_gates.detach().cpu(),
            step_write_gates=aux.step_write_gates.detach().cpu(),
            global_write_gates=aux.global_write_gates.detach().cpu(),
            memory_write_gates=aux.memory_write_gates.detach().cpu(),
            class_read=aux.class_read.detach().cpu(),
            slot_states=aux.slot_states.detach().cpu(),
            logits=aux.logits.detach().cpu(),
            slot_names=aux.slot_names,
        )
    return {"loss": total_loss / max(1, total), "acc": total_correct / max(1, total), "n": int(total), "confusion": conf, "last_aux": last_aux}


def top_confusions(conf: torch.Tensor, classes: Sequence[str], topn: int = 5) -> List[Dict]:
    out = []
    C = conf.shape[0]
    for i in range(C):
        for j in range(C):
            if i != j and int(conf[i, j]) > 0:
                out.append({"true": classes[i], "pred": classes[j], "n": int(conf[i, j])})
    out.sort(key=lambda x: x["n"], reverse=True)
    return out[:topn]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(args):
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    amp_dtype = get_amp_dtype(args.amp)
    out_dir = ensure_dir(Path(args.out_dir))

    train_loader, val_loader, classes = make_loaders(args)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)

    model = StepProgramNet(
        num_classes=len(classes),
        dim=args.dim,
        evidence_cells=args.evidence_cells,
        num_layers=args.layers,
        blocks_per_layer=args.blocks,
        steps_per_block=args.steps,
        primitive_slots=args.primitive_slots,
        global_cells=args.global_cells,
        memory_cells=args.memory_cells,
        dropout=args.dropout,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"StepProgramNet params={n_params} L={args.layers} B={args.blocks} S={args.steps} "
        f"K={args.primitive_slots} O={len(model.PRIMITIVE_NAMES)} device={device} amp={args.amp}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and amp_dtype == torch.float16))

    metrics_path = out_dir / "metrics.csv"
    fields = ["epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc", "primitive_temp", "read_temp", "cost_loss", "prim_entropy", "step_read_entropy", "prim_transition_entropy", "write_l1", "slot_collapse", "lambda_cost_eff", "lambda_prim_entropy_eff", "lambda_read_entropy_eff", "lambda_transition_entropy_eff", "lambda_write_l1_eff", "explore_w", "sharpen_w"]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best_acc = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, epoch, args, classes)
        val = evaluate(model, val_loader, device, amp_dtype, args, classes)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best_acc}, out_dir / "best.pt")
        torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best_acc}, out_dir / "last.pt")

        report_aux = val["last_aux"] or train["last_aux"]
        report = make_program_report(model, report_aux, classes) if report_aux is not None else {}
        analysis = {
            "epoch": epoch,
            "train": {k: v for k, v in train.items() if k != "last_aux"},
            "val": {k: v for k, v in val.items() if k not in ("last_aux", "confusion")},
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "top_confusions": top_confusions(val["confusion"], classes),
            "program_report": report,
            "notes": {
                "core": "clean differentiable step-program",
                "logs": [
                    "operations inside step: L.B.S.P primitive gates",
                    "operations between primitives: L.B.S.Pk->P{k+1}",
                    "operations between steps: L.B.S.read gates",
                    "operations input/output: evidence reads and output class_read",
                    "operations between layers/blocks: L.B.route gates",
                    "which parts needed: utility_proxy from write_gate and class_read_mass",
                ],
            },
        }
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", analysis)

        row = {
            "epoch": epoch,
            "train_loss": train.get("loss", 0.0),
            "train_ce": train.get("ce", 0.0),
            "train_acc": train.get("acc", 0.0),
            "val_loss": val.get("loss", 0.0),
            "val_acc": val.get("acc", 0.0),
            "best_acc": best_acc,
            "primitive_temp": train.get("primitive_temp", 0.0),
            "read_temp": train.get("read_temp", 0.0),
        }
        for f in fields:
            row.setdefault(f, train.get(f, 0.0))
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        print(
            f"epoch {epoch:03d}/{args.epochs} train={train['loss']:.4f}/{100*train['acc']:.2f}% "
            f"val={val['loss']:.4f}/{100*val['acc']:.2f}% best={100*best_acc:.2f}%@{best_epoch} "
            f"cost={train.get('cost_loss', 0):.3f} pent={train.get('prim_entropy', 0):.3f} "
            f"rent={train.get('step_read_entropy', 0):.3f} went={train.get('write_l1', 0):.3f} "
            f"conf={top_confusions(val['confusion'], classes, topn=3)}",
            flush=True,
        )

    write_json(out_dir / "final_report.json", {"best_acc": best_acc, "best_epoch": best_epoch, "args": vars(args), "classes": classes})
    print("done. out:", out_dir, flush=True)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data/speechcommands")
    p.add_argument("--download", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="use synthetic data for compile/smoke test")
    p.add_argument("--classes", type=str, default="yes,no,up,down,left,right,on,off,stop,go")
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=160)

    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--primitive-slots", type=int, default=3)
    p.add_argument("--global-cells", type=int, default=3)
    p.add_argument("--memory-cells", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.05)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=0.7)
    p.add_argument("--amp", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "off"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--primitive-temp-start", type=float, default=1.50)
    p.add_argument("--primitive-temp-end", type=float, default=0.70)
    p.add_argument("--read-temp-start", type=float, default=1.25)
    p.add_argument("--read-temp-end", type=float, default=0.85)
    p.add_argument("--lambda-cost", type=float, default=0.005)
    p.add_argument("--lambda-primitive-entropy", type=float, default=0.002)
    p.add_argument("--lambda-read-entropy", type=float, default=0.001)
    p.add_argument("--lambda-transition-entropy", type=float, default=0.001)
    p.add_argument("--lambda-write-l1", type=float, default=0.002)
    p.add_argument("--lambda-slot-collapse", type=float, default=0.004)
    p.add_argument("--out-dir", type=str, default="./runs/step_program_v1")
    return p


if __name__ == "__main__":
    run(build_argparser().parse_args())
