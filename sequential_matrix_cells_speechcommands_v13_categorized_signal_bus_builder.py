#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequential_matrix_cells_speechcommands_v9_6_matrix_register_ops.py

v9.6 Matrix Register Operators: MechanismMatrix layers/blocks + GlobalMatrix/MemoryMatrix + aggregation/repair operators.

Purpose:
  Step back from chaotic all-to-all "water" and test a simple causal basis.

Architecture:
  waveform -> mel tokens -> 4x4 evidence grid (+ globals)
  evidence cells -> sequential matrix cells
  each cell:
      reads evidence by attention
      receives previous hidden state
      receives train-pressure context
      applies gated update
  class queries read all intermediate states
  logits

No:
  - blocks/GrowerBlock
  - ADD_ROUTE / ADD_BLOCK
  - all-to-all cell attention
  - task/output recurrent cells
  - pair_allowed graph
  - dynamic_edge_gate

Adaptive parts:
  - each stage chooses evidence by attention
  - each stage has dynamic update gate + static capacity
  - class readout chooses which stage to read
  - active depth can be inspected via gates
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
except Exception as e:
    raise RuntimeError("This script needs torchaudio installed.") from e


# ---------------------------
# utils
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    def clean(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, (float, int, str, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return str(x)
    Path(path).write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def get_amp_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def parse_classes(s: str) -> List[str]:
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def top_confusions(conf: torch.Tensor, classes: List[str], topn: int = 12) -> List[Dict[str, Any]]:
    rows = []
    C = len(classes)
    conf = conf.detach().cpu()
    for i in range(C):
        total = int(conf[i].sum().item())
        if total <= 0:
            continue
        for j in range(C):
            if i == j:
                continue
            n = int(conf[i, j].item())
            if n:
                rows.append({
                    "true": classes[i],
                    "pred": classes[j],
                    "n": n,
                    "rate_of_true": n / max(1, total),
                })
    rows.sort(key=lambda r: (r["rate_of_true"], r["n"]), reverse=True)
    return rows[:topn]


# ---------------------------
# dataset
# ---------------------------

class SpeechCommandsFiltered(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        subset: str,
        classes: List[str],
        download: bool = False,
        limit: int = 0,
        seed: int = 42,
    ):
        super().__init__()
        self.classes = classes
        self.class_to_id = {c: i for i, c in enumerate(classes)}

        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        if not os.access(root_path, os.W_OK):
            raise RuntimeError(f"SpeechCommands root is not writable: {root_path}")

        self.base = SPEECHCOMMANDS(root=str(root_path), download=download, subset=subset)

        # Fast filtering: do not decode audio here.
        walker = getattr(self.base, "_walker", None)
        if walker is None:
            raise RuntimeError("torchaudio SPEECHCOMMANDS internals changed: _walker not found")

        idxs: List[int] = []
        for i, file_path in enumerate(walker):
            label = Path(str(file_path)).parent.name
            if label in self.class_to_id:
                idxs.append(i)

        rng = random.Random(seed + {"training": 0, "validation": 1, "testing": 2}.get(subset, 3))
        rng.shuffle(idxs)
        if limit and limit > 0:
            idxs = idxs[:limit]
        self.idxs = idxs

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int):
        wav, sr, label, speaker, utt = self.base[self.idxs[idx]]
        y = self.class_to_id[label]
        return wav, sr, y


def collate_waveforms(batch, seconds: float = 1.0, sample_rate: int = 16000):
    target_len = int(seconds * sample_rate)
    wavs = []
    ys = []
    for wav, sr, y in batch:
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)
        wav = wav.mean(dim=0)
        if wav.numel() < target_len:
            wav = F.pad(wav, (0, target_len - wav.numel()))
        elif wav.numel() > target_len:
            wav = wav[:target_len]
        wavs.append(wav)
        ys.append(y)
    return torch.stack(wavs, dim=0), torch.tensor(ys, dtype=torch.long)


def make_loaders(args):
    classes = parse_classes(args.classes)
    train_ds = SpeechCommandsFiltered(
        root=args.data_root,
        subset="training",
        classes=classes,
        download=args.download,
        limit=args.train_limit,
        seed=args.seed,
    )
    val_ds = SpeechCommandsFiltered(
        root=args.data_root,
        subset="validation",
        classes=classes,
        download=False,
        limit=args.val_limit,
        seed=args.seed,
    )
    collate = lambda b: collate_waveforms(b, seconds=args.seconds, sample_rate=args.sample_rate)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=True,
        persistent_workers=(args.workers > 0),
        collate_fn=collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        persistent_workers=(args.workers > 0),
        collate_fn=collate,
    )
    return train_loader, val_loader, classes


# ---------------------------
# frontend
# ---------------------------

class MelPatchFrontend(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 400,
        hop_length: int = 160,
        patch_time: int = 4,
        patch_freq: int = 4,
        max_tokens: int = 512,
    ):
        super().__init__()
        self.dim = dim
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.patch = nn.Conv2d(1, dim, kernel_size=(patch_freq, patch_time), stride=(patch_freq, patch_time))
        self.pos = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        # Keep mel in fp32 for stability.
        devtype = wav.device.type
        with torch.autocast(device_type=devtype, enabled=False):
            wav32 = wav.float()
            mel = self.mel(wav32)
            mel = self.db(mel).float()
            mel = torch.nan_to_num(mel, nan=0.0, posinf=80.0, neginf=-80.0)
            mean = mel.mean(dim=(1, 2), keepdim=True)
            std = mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-4)
            mel = ((mel - mean) / std).clamp(-8.0, 8.0)

            z = self.patch(mel.unsqueeze(1))  # [B,D,Fp,Tp]
            B, D, Fp, Tp = z.shape
            z = z.permute(0, 3, 2, 1).contiguous()
            x = z.view(B, Tp * Fp, D)
            if x.shape[1] > self.pos.shape[1]:
                raise RuntimeError(f"Too many tokens {x.shape[1]} > max_tokens={self.pos.shape[1]}")
            x = self.norm(x + self.pos[:, :x.shape[1], :])
        return x, (Tp, Fp)


# ---------------------------
# v9 sequential cells
# ---------------------------

@dataclasses.dataclass
class SeqAux:
    evidence_attention: torch.Tensor      # [B,N,E] proxy: 0.5*onset + 0.5*raw evidence
    time_attention: torch.Tensor          # [B,N,T_time]
    freq_attention: torch.Tensor          # [B,N,T_freq]
    onset_attention: torch.Tensor         # [B,N,E]
    energy_attention: torch.Tensor        # [B,N,T_energy]
    global_attention: torch.Tensor        # [B,N,G]
    memory_attention: torch.Tensor        # [B,N,M]
    task_attention: torch.Tensor          # [B,N,T_task]
    route_attention: torch.Tensor         # [B,N,2+blocks_per_layer] = base/global/prev blocks
    operator_gates: torch.Tensor          # [B,N,num_ops]
    category_gates: torch.Tensor          # [B,N,num_categories]
    task_matrix_norm: torch.Tensor        # [T_task]
    global_matrix_norm: torch.Tensor      # [global_cells]
    memory_matrix_norm: torch.Tensor      # [memory_cells]
    stage_states: torch.Tensor            # [B,N+1,D] block summaries
    dynamic_stage_gate: torch.Tensor      # [B,N]
    effective_stage_gate: torch.Tensor    # [B,N]
    static_stage_gate: torch.Tensor       # [N]
    class_stage_attention: torch.Tensor   # [B,C,N+1]
    pressure_features: torch.Tensor
    read_entropy_loss: torch.Tensor
    read_diversity_loss: torch.Tensor
    stage_load_balance_loss: torch.Tensor
    evidence_source_balance_loss: torch.Tensor
    operator_balance_loss: torch.Tensor
    register_diversity_loss: torch.Tensor
    global_diversity_loss: torch.Tensor
    memory_diversity_loss: torch.Tensor
    operator_entropy_per_block_loss: torch.Tensor
    block_diversity_loss: torch.Tensor
    route_diversity_loss: torch.Tensor
    adapter_diversity_loss: torch.Tensor
    phase_role_loss: torch.Tensor
    global_skip_penalty_loss: torch.Tensor
    late_write_loss: torch.Tensor
    late_dyn_gate_loss: torch.Tensor
    weak_class_late_read_loss: torch.Tensor
    role_diversity_loss: torch.Tensor
    role_anchor_loss: torch.Tensor
    category_diversity_loss: torch.Tensor
    category_entropy_loss: torch.Tensor
    signal_op_bias_norm: torch.Tensor
    role_mix_by_layer: torch.Tensor
    category_mix_by_layer: torch.Tensor
    signal_summary: torch.Tensor
    update_norm_by_block: torch.Tensor
    write_delta_norm_by_block: torch.Tensor
    logits_cell_norm: torch.Tensor


class SequentialMatrixCellsCore(nn.Module):
    """
    v13 Categorized SignalBus Matrix Builder.

    This is a cleaner architecture rather than a v9 patch:
      * EvidenceMatrix is intended to be 6x6+feature cells (wrapper builds it).
      * Mechanism is processed in causal phases, not block-by-block Python loops.
      * Multiple typed matrix-attention heads are computed per phase:
          route, task, time, freq, onset, energy, global, memory.
      * OperatorBank remains matrix-space and vectorized:
          candidates [B, N_blocks, N_ops, D] -> einsum mix.
      * Shared operators get block/phase specialization through low-rank-like FiLM adapters:
          per phase/block/op scale and bias gates; no ModuleList per block.
      * Registers are matrix spaces: GlobalMatrix and MemoryMatrix.
    """

    OP_NAMES = [
        "noop", "identity", "keep_prev", "small_refine",
        "mlp", "matrix_mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",
        "global_summary", "memory_read", "memory_write", "ema_memory", "class_pair_memory", "residual_refdelta",
        "block_compare", "class_pair_contrast", "late_read_repair", "suppress", "normalize", "energy_count",
    ]

    HEAD_NAMES = [
        "route", "task", "time", "freq", "onset", "energy", "global", "memory",
        "compare", "suppress",
    ]

    ROLE_NAMES = [
        "extract", "contrast", "transform", "memory", "repair", "aggregate", "suppress",
    ]

    CATEGORY_NAMES = [
        "extract", "compare", "memory", "repair", "aggregate", "suppress", "transform",
    ]

    def __init__(
        self,
        dim: int,
        num_classes: int,
        evidence_cells: int = 48,
        chain_depth: int = 16,
        num_layers: int = 4,
        blocks_per_layer: int = 4,
        block_cells: int = 2,
        global_cells: int = 3,
        memory_cells: int = 6,
        pressure_alpha: float = 0.20,
        dropout: float = 0.05,
        class_query_std: float = 0.15,
        class_stage_bias_std: float = 0.35,
        stage_gate_bias_init: float = 1.0,
        gate_floor: float = 0.15,
        late_gate_floor: float = 0.25,
        late_write_target: float = 0.35,
        late_dyn_gate_target: float = 0.15,
        weak_class_late_read_target: float = 0.35,
        role_signal_scale: float = 0.35,
        signal_op_bias_scale: float = 0.20,
        category_temp: float = 1.0,
        primitive_temp: float = 1.0,
        lock_last_role: bool = True,
        task_attn_temp: float = 0.85,
        evidence_attn_temp: float = 1.0,
        operator_temp: float = 1.25,
        topology: str = "grid",
        skip_connections: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.evidence_cells = int(evidence_cells)
        self.topology = str(topology)
        self.skip_connections = bool(skip_connections)

        # v10 default: 4 causal phases with 4 parallel matrix branches.
        if self.topology == "two_squares_merge":
            self.topology_plan = [[0, 1, 2, 3], [0, 1, 2, 3], [0]]
            self.num_layers = 3
            self.blocks_per_layer = 4
        elif self.topology == "two_by_two_by_two_merge":
            self.topology_plan = [[0, 1], [0, 1], [0, 1], [0]]
            self.num_layers = 4
            self.blocks_per_layer = 2
        elif self.topology in ("grid_3x3x3x3",):
            self.num_layers = 4
            self.blocks_per_layer = 3
            self.topology_plan = [list(range(self.blocks_per_layer)) for _ in range(self.num_layers)]
        else:
            self.num_layers = int(num_layers)
            self.blocks_per_layer = int(blocks_per_layer)
            self.topology_plan = [list(range(self.blocks_per_layer)) for _ in range(self.num_layers)]

        self.block_cells = int(block_cells)
        self.global_cells = int(global_cells)
        self.memory_cells = int(memory_cells)
        # chain_depth is derived from topology_plan; run(args) mirrors this into args.chain_depth.
        self.chain_depth = sum(len(row) for row in self.topology_plan)
        self.pressure_alpha = float(pressure_alpha)
        self.gate_floor = float(gate_floor)
        self.late_gate_floor = float(late_gate_floor)
        self.late_write_target = float(late_write_target)
        self.late_dyn_gate_target = float(late_dyn_gate_target)
        self.weak_class_late_read_target = float(weak_class_late_read_target)
        self.role_signal_scale = float(role_signal_scale)
        self.signal_op_bias_scale = float(signal_op_bias_scale)
        self.category_temp = float(category_temp)
        self.primitive_temp = float(primitive_temp)
        self.lock_last_role = bool(lock_last_role)
        self.task_attn_temp = float(task_attn_temp)
        self.evidence_attn_temp = float(evidence_attn_temp)
        self.operator_temp = float(operator_temp)
        self.num_ops = len(self.OP_NAMES)
        self.num_heads = len(self.HEAD_NAMES)

        # Train-side task pressure.
        self.pressure_dim = 2 + 3 * self.num_classes
        self.register_buffer("pressure_ema", torch.zeros(self.pressure_dim))
        self.register_buffer("confusion_ema", torch.eye(self.num_classes) / max(1, self.num_classes))

        # TaskMatrix cells: global loss/dist + class errors + confusion rows.
        self.global_task_cells = 2
        self.class_task_cells = self.num_classes
        self.confusion_task_cells = self.num_classes
        self.pair_task_cells = min(6, self.num_classes)
        self.task_cells = self.global_task_cells + self.class_task_cells + self.confusion_task_cells + self.pair_task_cells

        # Task projection.
        self.task_k = nn.Linear(dim, dim, bias=False)
        self.task_v = nn.Linear(dim, dim, bias=False)
        self.global_loss_proj = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.global_dist_proj = nn.Sequential(nn.LayerNorm(2 * num_classes), nn.Linear(2 * num_classes, dim), nn.GELU(), nn.Linear(dim, dim))
        self.class_task_proj = nn.Sequential(nn.LayerNorm(3), nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.confusion_task_proj = nn.Sequential(nn.LayerNorm(num_classes), nn.Linear(num_classes, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pair_task_proj = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.task_type_emb = nn.Embedding(4, dim)
        # Separate slots for global task cells:
        # slot0 = loss/error pressure, slot1 = true/pred distribution pressure.
        # Do NOT reuse task_type_emb[1], because type 1 is class-pressure cells.
        self.global_task_slot_emb = nn.Embedding(2, dim)
        self.task_class_emb = nn.Embedding(num_classes, dim)
        self.task_norm = nn.LayerNorm(dim)

        # Evidence typed projections. One shared K/V plus head-specific queries.
        self.ev_k = nn.Linear(dim, dim, bias=False)
        self.ev_v = nn.Linear(dim, dim, bias=False)
        self.time_q = nn.Linear(dim, dim, bias=False)
        self.freq_q = nn.Linear(dim, dim, bias=False)
        self.onset_q = nn.Linear(dim, dim, bias=False)
        self.energy_q = nn.Linear(dim, dim, bias=False)

        # Mechanism/read projections.
        self.route_q = nn.Linear(dim, dim, bias=False)
        self.route_k = nn.Linear(dim, dim, bias=False)
        self.route_v = nn.Linear(dim, dim, bias=False)
        self.task_q = nn.Linear(dim, dim, bias=False)
        self.global_q = nn.Linear(dim, dim, bias=False)
        self.global_k = nn.Linear(dim, dim, bias=False)
        self.global_v = nn.Linear(dim, dim, bias=False)
        self.memory_q = nn.Linear(dim, dim, bias=False)
        self.memory_k = nn.Linear(dim, dim, bias=False)
        self.memory_v = nn.Linear(dim, dim, bias=False)

        # Initial evidence -> base mechanism state.
        self.init_mlp = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

        # Matrix addresses.
        self.layer_addr = nn.Parameter(torch.randn(self.num_layers, dim) * 0.04)
        self.block_addr = nn.Parameter(torch.randn(self.blocks_per_layer, dim) * 0.04)
        self.cell_addr = nn.Parameter(torch.randn(self.block_cells, dim) * 0.04)
        self.global_addr = nn.Parameter(torch.randn(self.global_cells, dim) * 0.04)
        self.memory_addr = nn.Parameter(torch.randn(self.memory_cells, dim) * 0.04)
        self.head_addr = nn.Parameter(torch.randn(self.num_heads, dim) * 0.04)
        self.operator_addr = nn.Parameter(torch.randn(self.num_ops, dim) * 0.04)

        self.init_block_region = nn.Parameter(torch.randn(self.blocks_per_layer, self.block_cells, dim) * 0.03)
        self.init_global = nn.Parameter(torch.randn(self.global_cells, dim) * 0.03)
        self.init_memory = nn.Parameter(torch.randn(self.memory_cells, dim) * 0.03)

        # Shared operator bank. These are matrix-to-matrix row transforms, vectorized over blocks.
        self.mlp_op = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        # MatrixMLP primitive: unlike channel-only MLP, this mixes the active block axis
        # through a learned block-to-block matrix and then applies a channel MLP.
        self.matrix_mlp_q = nn.Linear(dim, dim, bias=False)
        self.matrix_mlp_k = nn.Linear(dim, dim, bias=False)
        self.matrix_mlp_v = nn.Linear(dim, dim, bias=False)
        self.matrix_mlp_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.bilin_h = nn.Linear(dim, dim, bias=False)
        self.bilin_e = nn.Linear(dim, dim, bias=False)
        self.bilin_t = nn.Linear(dim, dim, bias=True)
        self.bilin_out = nn.Linear(dim, dim, bias=True)
        self.delta_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.short_onset_out = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.global_summary_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.memory_read_out = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.memory_write_out = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.ema_memory_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.class_pair_memory_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.residual_refdelta_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.block_compare_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.class_pair_contrast_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.late_read_repair_out = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.offset_out = nn.Sequential(nn.LayerNorm(dim * 5), nn.Linear(dim * 5, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.suppress_gate = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.suppress_out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.normalize_out = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.energy_count_out = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        # Operator attention/gating. Keys are operator addresses, query is block+task+register.
        self.op_query = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.op_key = nn.Linear(dim, dim, bias=False)
        self.op_bias_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, self.num_ops))

        # Block adapters: cheap specialization while keeping shared vectorized kernels.
        self.block_op_scale = nn.Parameter(torch.zeros(self.num_layers, self.blocks_per_layer, self.num_ops))
        self.block_op_bias = nn.Parameter(torch.zeros(self.num_layers, self.blocks_per_layer, self.num_ops))
        self.block_state_film = nn.Parameter(torch.zeros(self.num_layers, self.blocks_per_layer, dim))

        # Soft phase priors. Not hard roles; gradients can override.
        self.layer_op_prior = nn.Parameter(torch.zeros(self.num_layers, self.num_ops))
        with torch.no_grad():
            def bump(layer: int, names: Tuple[str, ...], value: float = 0.12) -> None:
                for name in names:
                    if name in self.OP_NAMES:
                        self.layer_op_prior[layer, self.OP_NAMES.index(name)] += value
            if self.num_layers >= 1:
                bump(0, ("time", "freq", "delta", "short_onset", "offset", "bilinear"), 0.28)
            if self.num_layers >= 2:
                bump(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "matrix_mlp", "normalize"), 0.24)
            if self.num_layers >= 3:
                bump(2, ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "matrix_mlp", "suppress", "normalize", "residual_refdelta", "late_read_repair"), 0.24)
            if self.num_layers >= 4:
                bump(3, ("global_summary", "energy_count", "suppress", "memory_write", "ema_memory", "block_compare"), 0.28)

            # FMG-style block start: shared kernels, but each block begins with a different role.
            for layer in range(self.num_layers):
                for block in range(self.blocks_per_layer):
                    role = block % 4
                    if role == 0:
                        names = ("time", "short_onset", "offset", "delta")
                    elif role == 1:
                        names = ("freq", "bilinear", "block_compare")
                    elif role == 2:
                        names = ("class_pair_contrast", "residual_refdelta", "normalize")
                    else:
                        names = ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "suppress", "global_summary")
                    for name in names:
                        if name in self.OP_NAMES:
                            self.block_op_bias[layer, block, self.OP_NAMES.index(name)] += 0.08

        # Phase-role targets for differentiable regularization.
        phase_target = torch.zeros(self.num_layers, self.num_ops)
        def target(layer: int, names: Tuple[str, ...]) -> None:
            if layer >= self.num_layers:
                return
            for name in names:
                if name in self.OP_NAMES:
                    phase_target[layer, self.OP_NAMES.index(name)] = 1.0
        target(0, ("time", "freq", "delta", "short_onset", "offset", "bilinear"))
        target(1, ("block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "matrix_mlp", "normalize"))
        target(2, ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "suppress", "normalize", "residual_refdelta", "late_read_repair"))
        target(3, ("global_summary", "energy_count", "suppress", "memory_write", "ema_memory", "block_compare"))
        phase_target = phase_target / phase_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("phase_op_target", phase_target)

        # v12 SignalBus + RolePlanner.
        # Data/task/mechanism signals softly choose role mixtures per layer.
        self.num_roles = len(self.ROLE_NAMES)
        self.role_logits = nn.Parameter(torch.zeros(self.num_layers, self.num_roles))
        with torch.no_grad():
            def role_bias(layer: int, role: str, val: float) -> None:
                if layer < self.num_layers and role in self.ROLE_NAMES:
                    self.role_logits[layer, self.ROLE_NAMES.index(role)] += val
            role_bias(0, "extract", 2.0)
            role_bias(1, "contrast", 1.4); role_bias(1, "transform", 0.6); role_bias(1, "repair", 0.3)
            role_bias(2, "memory", 1.2); role_bias(2, "repair", 0.9); role_bias(2, "suppress", 0.4)
            if self.num_layers >= 4:
                role_bias(3, "aggregate", 2.0); role_bias(3, "suppress", 0.6)

        role_op = torch.zeros(self.num_roles, self.num_ops)
        def role_ops(role: str, names: Tuple[str, ...]) -> None:
            r = self.ROLE_NAMES.index(role)
            for name in names:
                if name in self.OP_NAMES:
                    role_op[r, self.OP_NAMES.index(name)] = 1.0
        role_ops("extract", ("time", "freq", "delta", "short_onset", "offset", "bilinear"))
        role_ops("contrast", ("class_pair_contrast", "block_compare", "bilinear", "residual_refdelta", "normalize"))
        role_ops("transform", ("matrix_mlp", "bilinear", "normalize", "residual_refdelta", "time", "freq"))
        role_ops("memory", ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "matrix_mlp", "residual_refdelta", "normalize"))
        role_ops("repair", ("residual_refdelta", "suppress", "normalize", "class_pair_contrast", "late_read_repair", "class_pair_memory"))
        role_ops("aggregate", ("global_summary", "energy_count", "memory_write", "ema_memory", "suppress", "block_compare"))
        role_ops("suppress", ("suppress", "normalize", "block_compare", "global_summary"))
        role_op = role_op / role_op.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("role_op_basis", role_op)

        self.signal_norm = nn.LayerNorm(dim * 4)
        self.signal_context_net = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.signal_role_net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.num_roles))
        self.signal_op_bias_net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.num_ops))
        self.signal_category_bias_net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, len(self.CATEGORY_NAMES)))
        self.signal_gate_bias_net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

        # v13 hierarchical OperatorCategoryPlanner.
        self.num_categories = len(self.CATEGORY_NAMES)
        self.category_bias_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, self.num_categories))
        self.block_category_bias = nn.Parameter(torch.zeros(self.num_layers, self.blocks_per_layer, self.num_categories))
        self.layer_category_prior = nn.Parameter(torch.zeros(self.num_layers, self.num_categories))

        op_cat = torch.zeros(self.num_categories, self.num_ops)
        def category_ops(category: str, names: Tuple[str, ...]) -> None:
            c = self.CATEGORY_NAMES.index(category)
            for name in names:
                if name in self.OP_NAMES:
                    op_cat[c, self.OP_NAMES.index(name)] = 1.0
        category_ops("extract", ("time", "freq", "delta", "short_onset", "offset", "energy_count"))
        category_ops("compare", ("bilinear", "block_compare", "class_pair_contrast"))
        category_ops("memory", ("memory_read", "memory_write", "ema_memory", "class_pair_memory", "global_summary", "matrix_mlp"))
        category_ops("repair", ("residual_refdelta", "class_pair_contrast", "class_pair_memory", "late_read_repair", "suppress", "normalize"))
        category_ops("aggregate", ("global_summary", "energy_count", "memory_write", "ema_memory", "block_compare"))
        category_ops("suppress", ("suppress", "normalize", "block_compare"))
        category_ops("transform", ("mlp", "matrix_mlp", "bilinear", "normalize", "time", "freq", "offset", "residual_refdelta"))
        # Ensure every primitive is reachable.
        for oi in range(self.num_ops):
            if float(op_cat[:, oi].sum().item()) == 0.0:
                op_cat[self.CATEGORY_NAMES.index("transform"), oi] = 1.0
        self.register_buffer("op_category_mask", op_cat.bool())

        role_cat = torch.zeros(self.num_roles, self.num_categories)
        def role_cats(role: str, cats: Tuple[str, ...]) -> None:
            r = self.ROLE_NAMES.index(role)
            for cat in cats:
                if cat in self.CATEGORY_NAMES:
                    role_cat[r, self.CATEGORY_NAMES.index(cat)] = 1.0
        role_cats("extract", ("extract", "transform"))
        role_cats("contrast", ("compare", "repair", "transform"))
        role_cats("transform", ("transform", "extract", "repair"))
        role_cats("memory", ("memory", "repair", "aggregate"))
        role_cats("repair", ("repair", "suppress", "compare"))
        role_cats("aggregate", ("aggregate", "memory", "suppress"))
        role_cats("suppress", ("suppress", "repair", "aggregate"))
        role_cat = role_cat / role_cat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("role_category_basis", role_cat)
        with torch.no_grad():
            def cat_bump(layer: int, cats: Tuple[str, ...], value: float) -> None:
                if layer >= self.num_layers:
                    return
                for cat in cats:
                    if cat in self.CATEGORY_NAMES:
                        self.layer_category_prior[layer, self.CATEGORY_NAMES.index(cat)] += value
            cat_bump(0, ("extract", "transform"), 0.35)
            cat_bump(1, ("compare", "repair", "transform"), 0.30)
            cat_bump(2, ("memory", "repair", "suppress"), 0.30)
            cat_bump(3, ("aggregate", "suppress", "memory"), 0.35)
            for layer in range(self.num_layers):
                for block in range(self.blocks_per_layer):
                    cat = self.CATEGORY_NAMES[block % self.num_categories]
                    self.block_category_bias[layer, block, self.CATEGORY_NAMES.index(cat)] += 0.06

        # Write and register update.
        self.stage_gate_net = nn.Sequential(nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))
        last = self.stage_gate_net[-1]
        if isinstance(last, nn.Linear) and last.bias is not None:
            nn.init.zeros_(last.weight)
            last.bias.data.fill_(float(stage_gate_bias_init))
        self.cell_write_net = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cell_gate_net = nn.Sequential(nn.LayerNorm(dim * 3), nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.block_norm = nn.LayerNorm(dim)
        self.cell_norm = nn.LayerNorm(dim)
        self.global_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)

        self.global_write_q = nn.Linear(dim, dim, bias=False)
        self.global_write_val = nn.Linear(dim * 3, dim, bias=True)
        self.global_write_gate = nn.Linear(dim * 3, 1, bias=True)
        self.memory_write_q = nn.Linear(dim, dim, bias=False)
        self.memory_write_val = nn.Linear(dim * 3, dim, bias=True)
        self.memory_write_gate = nn.Linear(dim * 3, 1, bias=True)

        self.register_buffer("static_stage_gate_buffer", torch.ones(self.chain_depth))

        # Class read.
        self.class_query = nn.Parameter(torch.randn(num_classes, dim) * float(class_query_std))
        self.cls_q = nn.Linear(dim, dim, bias=False)
        self.cls_k = nn.Linear(dim, dim, bias=False)
        self.cls_v = nn.Linear(dim, dim, bias=False)
        self.class_stage_bias = nn.Parameter(torch.randn(num_classes, self.chain_depth + 1) * float(class_stage_bias_std))
        self.class_layer_read_bias = nn.Parameter(torch.zeros(self.num_layers, self.blocks_per_layer))
        with torch.no_grad():
            if self.num_layers > 1:
                self.class_layer_read_bias[-1].fill_(0.25)
                self.class_layer_read_bias[0].fill_(-0.08)
        self.class_write = nn.Parameter(torch.randn(num_classes, dim) * 0.05)
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))

    @torch.no_grad()
    def update_pressure_ema(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        logits = logits.detach().float()
        y = y.detach()
        C = self.num_classes
        pred = logits.argmax(dim=-1)
        err = (pred != y).float()
        loss_mean = F.cross_entropy(logits, y, reduction="none").mean()
        err_mean = err.mean()
        true_dist = torch.bincount(y, minlength=C).float(); true_dist = true_dist / true_dist.sum().clamp_min(1.0)
        pred_dist = torch.bincount(pred, minlength=C).float(); pred_dist = pred_dist / pred_dist.sum().clamp_min(1.0)
        class_err = torch.zeros(C, device=logits.device)
        for c in range(C):
            m = (y == c)
            if bool(m.any()):
                class_err[c] = err[m].mean()
        feat = torch.cat([loss_mean.view(1), err_mean.view(1), class_err, true_dist.to(logits.device), pred_dist.to(logits.device)], dim=0)
        self.pressure_ema.mul_(1.0 - self.pressure_alpha).add_(self.pressure_alpha * feat.to(self.pressure_ema.device))
        idx = y * C + pred
        cm = torch.bincount(idx.detach().cpu(), minlength=C * C).float().view(C, C).to(self.confusion_ema.device)
        cm = cm / cm.sum(dim=1, keepdim=True).clamp_min(1.0)
        self.confusion_ema.mul_(1.0 - self.pressure_alpha).add_(self.pressure_alpha * cm)

    def build_task_matrix(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        p = self.pressure_ema.detach().to(device=device, dtype=dtype).clone()
        C = self.num_classes
        loss_err = p[:2].view(1, 2).expand(batch_size, -1)
        class_err = p[2:2 + C]
        true_dist = p[2 + C:2 + 2 * C]
        pred_dist = p[2 + 2 * C:2 + 3 * C]
        dist_pair = torch.cat([true_dist, pred_dist], dim=0).view(1, 2 * C).expand(batch_size, -1)
        global_type = self.task_type_emb.weight[0].to(device=device, dtype=dtype).view(1, -1)
        global0 = self.global_loss_proj(loss_err) + global_type + self.global_task_slot_emb.weight[0].to(device=device, dtype=dtype).view(1, -1)
        global1 = self.global_dist_proj(dist_pair) + global_type + self.global_task_slot_emb.weight[1].to(device=device, dtype=dtype).view(1, -1)
        class_idx = torch.arange(C, device=device)
        triplet = torch.stack([class_err, true_dist, pred_dist], dim=-1).to(device=device, dtype=dtype)
        class_cells = self.class_task_proj(triplet).unsqueeze(0).expand(batch_size, -1, -1)
        class_cells = class_cells + self.task_type_emb.weight[1].to(device=device, dtype=dtype).view(1, 1, -1)
        class_cells = class_cells + self.task_class_emb(class_idx).to(dtype).view(1, C, -1)
        cm = self.confusion_ema.detach().to(device=device, dtype=dtype).clone()
        conf_cells = self.confusion_task_proj(cm).unsqueeze(0).expand(batch_size, -1, -1)
        conf_cells = conf_cells + self.task_type_emb.weight[2].to(device=device, dtype=dtype).view(1, 1, -1)
        conf_cells = conf_cells + self.task_class_emb(class_idx).to(dtype).view(1, C, -1)

        # Dynamic top-confusion pair cells: true->pred pairs with diagonal removed.
        # This gives class_pair_contrast targeted pressure for current mistakes.
        cm_pairs = cm.clone()
        cm_pairs.fill_diagonal_(0.0)
        k_pairs = min(self.pair_task_cells, C * C)
        vals, flat = torch.topk(cm_pairs.reshape(-1), k=k_pairs)
        true_idx = torch.div(flat, C, rounding_mode="floor")
        pred_idx = flat % C
        pair_feat = torch.stack([
            class_err[true_idx],
            true_dist[true_idx],
            pred_dist[pred_idx],
            vals.to(device=device, dtype=dtype),
        ], dim=-1).to(device=device, dtype=dtype)
        pair_cells = self.pair_task_proj(pair_feat).unsqueeze(0).expand(batch_size, -1, -1)
        pair_cells = pair_cells + self.task_type_emb.weight[3].to(device=device, dtype=dtype).view(1, 1, -1)
        pair_cells = pair_cells + self.task_class_emb(true_idx.to(device)).to(dtype).view(1, k_pairs, -1)
        pair_cells = pair_cells - 0.50 * self.task_class_emb(pred_idx.to(device)).to(dtype).view(1, k_pairs, -1)

        task = torch.cat([global0.unsqueeze(1), global1.unsqueeze(1), class_cells, conf_cells, pair_cells], dim=1)
        return self.task_norm(task)

    def _route_prior_for_topology(self, layer: int, active_blocks: List[int], prev_active_blocks: List[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        N = len(active_blocks); R = 2 + len(prev_active_blocks)
        prior = torch.zeros(N, R, device=device, dtype=dtype)
        if self.topology == "two_squares_merge":
            if layer == 1:
                for n, b in enumerate(active_blocks):
                    group = {0, 1} if b in (0, 1) else {2, 3}
                    for j, pb in enumerate(prev_active_blocks):
                        prior[n, 2 + j] += 0.45 if pb in group else -0.20
                    if self.skip_connections: prior[n, 0] += 0.10
            elif layer >= 2:
                if self.skip_connections: prior[:, 1] += 0.25
                if R > 2: prior[:, 2:] += 0.15
        elif self.topology == "two_by_two_by_two_merge":
            if layer in (1, 2):
                for n, b in enumerate(active_blocks):
                    for j, pb in enumerate(prev_active_blocks):
                        prior[n, 2 + j] += 0.45 if pb == b else -0.15
                    if self.skip_connections: prior[n, 0] += 0.10
            elif layer >= 3:
                if self.skip_connections: prior[:, 1] += 0.25
                if R > 2: prior[:, 2:] += 0.20
        if not self.skip_connections:
            prior[:, :2] = -1.0e4
        return prior

    def _select_prev_regions(self, prev_regions: torch.Tensor, active_blocks: List[int], prev_active_blocks: List[int]) -> torch.Tensor:
        out = []
        mean_region = prev_regions.mean(dim=1)
        for b in active_blocks:
            if b in prev_active_blocks:
                out.append(prev_regions[:, prev_active_blocks.index(b), :, :])
            else:
                out.append(mean_region)
        return torch.stack(out, dim=1)

    def _rms_norm_vec(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        return x / x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()

    def _evidence_slices(self, evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Evidence cells are expected as 6x6 grid followed by extra features. Fallback works for smaller E.
        B, E, D = evidence.shape
        if E >= 36:
            grid = evidence[:, :36, :].view(B, 6, 6, D)
            time_tokens = torch.cat([grid.mean(dim=2), grid[:, 1:, :, :].mean(dim=2) - grid[:, :-1, :, :].mean(dim=2)], dim=1)  # [B,11,D]
            freq_tokens = torch.cat([grid.mean(dim=1), grid[:, :, 1:, :].mean(dim=1) - grid[:, :, :-1, :].mean(dim=1)], dim=1)  # [B,11,D]
            early = grid[:, 0:2, :, :].mean(dim=(1,2)); mid = grid[:, 2:4, :, :].mean(dim=(1,2)); late = grid[:, 4:6, :, :].mean(dim=(1,2))
            onset = grid[:, 1:3, :, :].mean(dim=(1,2)) - grid[:, 0:2, :, :].mean(dim=(1,2))
            offset = grid[:, 4:6, :, :].mean(dim=(1,2)) - grid[:, 3:5, :, :].mean(dim=(1,2))
            energy_tokens = torch.cat([grid.mean(dim=(1,2), keepdim=False).unsqueeze(1), evidence[:, 36:, :]], dim=1) if E > 36 else grid.mean(dim=(1,2)).unsqueeze(1)
            delta_feat = torch.cat([late - early, mid - early, late - mid, onset], dim=-1)
            short_feat = torch.cat([early, mid, late, onset, offset], dim=-1)
        else:
            # Compatibility fallback.
            time_tokens = evidence
            freq_tokens = evidence
            energy_tokens = evidence
            early = evidence[:, 0, :]
            mid = evidence[:, E//2, :]
            late = evidence[:, -1, :]
            onset = mid - early
            offset = late - mid
            delta_feat = torch.cat([late - early, mid - early, late - mid, onset], dim=-1)
            short_feat = torch.cat([early, mid, late, onset, offset], dim=-1)
        return time_tokens, freq_tokens, energy_tokens, delta_feat, short_feat

    def forward(self, evidence: torch.Tensor) -> Tuple[torch.Tensor, SeqAux]:
        B, E, D = evidence.shape
        dtype = evidence.dtype; device = evidence.device
        scale = 1.0 / math.sqrt(D)
        task_matrix = self.build_task_matrix(B, device, dtype)
        pressure = self.pressure_ema.detach().to(device=device, dtype=dtype).clone()

        ev_mean = evidence.mean(dim=1); ev_std = evidence.std(dim=1); ev_max = evidence.amax(dim=1)
        base = self.init_mlp(torch.cat([ev_mean, ev_std, ev_max], dim=-1))
        time_tokens, freq_tokens, energy_tokens, delta_feat, short_feat = self._evidence_slices(evidence)

        # v12 SignalBus: data + task pressure + mechanism seed.
        task_seed = task_matrix.mean(dim=1)
        signal_raw = torch.cat([ev_mean, ev_std, task_seed, base], dim=-1)
        signal_ctx = self.signal_context_net(self.signal_norm(signal_raw))
        signal_role_bias = self.signal_role_net(signal_ctx)              # [B,R]
        signal_category_bias = self.signal_category_bias_net(signal_ctx)  # [B,Cat]
        signal_op_bias = self.signal_op_bias_net(signal_ctx)             # [B,O]
        signal_gate_bias = self.signal_gate_bias_net(signal_ctx).squeeze(-1)  # [B]

        global_matrix = base[:, None, :] + self.init_global.to(device=device, dtype=dtype).unsqueeze(0) + self.global_addr.to(device=device, dtype=dtype).view(1, self.global_cells, D)
        memory_matrix = 0.15 * base[:, None, :] + self.init_memory.to(device=device, dtype=dtype).unsqueeze(0) + self.memory_addr.to(device=device, dtype=dtype).view(1, self.memory_cells, D)
        prev_regions = (
            base[:, None, None, :]
            + self.init_block_region.to(device=device, dtype=dtype).unsqueeze(0)
            + self.block_addr.to(device=device, dtype=dtype).view(1, self.blocks_per_layer, 1, D)
            + self.cell_addr.to(device=device, dtype=dtype).view(1, 1, self.block_cells, D)
        )

        K_ev = self.ev_k(evidence); V_ev = self.ev_v(evidence)
        K_time = self.ev_k(time_tokens); V_time = self.ev_v(time_tokens)
        K_freq = self.ev_k(freq_tokens); V_freq = self.ev_v(freq_tokens)
        K_energy = self.ev_k(energy_tokens); V_energy = self.ev_v(energy_tokens)
        K_task = self.task_k(task_matrix); V_task = self.task_v(task_matrix)

        states = [base]
        evid_attns = []
        time_attns = []; freq_attns = []; onset_attns = []; energy_attns = []
        global_attns = []; memory_attns = []
        task_attns = []; route_attns = []; op_gates_all = []
        dyn_gates = []; eff_gates = []
        update_norms = []; write_delta_norms = []
        live_write_delta_norms = []
        role_mixes_live = []
        role_mixes_report = []
        category_gates_all = []
        category_mixes = []
        signal_op_bias_norms = []
        static_gates = self.static_stage_gate_buffer.to(device=device, dtype=dtype)
        prev_active_blocks = list(range(prev_regions.shape[1]))

        for layer, active_blocks in enumerate(self.topology_plan):
            N = len(active_blocks)
            prev_summaries = prev_regions.mean(dim=2)
            global_summary = global_matrix.mean(dim=1)
            active_block_ids = torch.tensor(active_blocks, device=device, dtype=torch.long)
            addr = self.layer_addr[layer].to(device=device, dtype=dtype).view(1,1,D) + self.block_addr[active_block_ids].to(device=device, dtype=dtype).view(1,N,D)
            film = torch.tanh(self.block_state_film[layer, active_block_ids].to(device=device, dtype=dtype)).view(1,N,D)

            route_tokens = torch.cat([base[:, None, :], global_summary[:, None, :], prev_summaries], dim=1)
            prev_region_for_active = self._select_prev_regions(prev_regions, active_blocks, prev_active_blocks)
            block_seed = prev_region_for_active.mean(dim=2)
            q_route = self.route_q(block_seed + addr + 0.10 * film)
            K_route = self.route_k(route_tokens); V_route = self.route_v(route_tokens)
            route_prior = self._route_prior_for_topology(layer, active_blocks, prev_active_blocks, device, dtype).view(1,N,-1)
            score_route = torch.einsum('bnd,brd->bnr', q_route, K_route) * scale + route_prior
            A_route = torch.softmax(score_route.float(), dim=-1).to(dtype)
            route_ctx = torch.einsum('bnr,brd->bnd', A_route, V_route)
            h = route_ctx + 0.05 * film
            q_base = h + addr

            q_task = self.task_q(q_base)
            score_task = torch.einsum('bnd,btd->bnt', q_task, K_task) * (scale / max(1e-4, self.task_attn_temp))
            A_task = torch.softmax(score_task.float(), dim=-1).to(dtype)
            task_ctx = torch.einsum('bnt,btd->bnd', A_task, V_task)

            # Typed Evidence -> Mechanism heads.
            q_time = self.time_q(q_base + task_ctx + self.head_addr[self.HEAD_NAMES.index('time')].to(device=device, dtype=dtype).view(1,1,D))
            score_time = torch.einsum('bnd,btd->bnt', q_time, K_time) * (scale / max(1e-4,self.evidence_attn_temp))
            A_time = torch.softmax(score_time.float(), dim=-1).to(dtype)
            time_ctx = torch.einsum('bnt,btd->bnd', A_time, V_time)

            q_freq = self.freq_q(q_base + task_ctx + self.head_addr[self.HEAD_NAMES.index('freq')].to(device=device, dtype=dtype).view(1,1,D))
            score_freq = torch.einsum('bnd,bfd->bnf', q_freq, K_freq) * (scale / max(1e-4,self.evidence_attn_temp))
            A_freq = torch.softmax(score_freq.float(), dim=-1).to(dtype)
            freq_ctx = torch.einsum('bnf,bfd->bnd', A_freq, V_freq)

            q_onset = self.onset_q(q_base + task_ctx + self.head_addr[self.HEAD_NAMES.index('onset')].to(device=device, dtype=dtype).view(1,1,D))
            score_onset = torch.einsum('bnd,bsd->bns', q_onset, K_ev) * (scale / max(1e-4,self.evidence_attn_temp))
            A_onset = torch.softmax(score_onset.float(), dim=-1).to(dtype)
            onset_ctx = torch.einsum('bns,bsd->bnd', A_onset, V_ev)

            q_energy = self.energy_q(q_base + task_ctx + self.head_addr[self.HEAD_NAMES.index('energy')].to(device=device, dtype=dtype).view(1,1,D))
            score_energy = torch.einsum('bnd,bqd->bnq', q_energy, K_energy) * (scale / max(1e-4,self.evidence_attn_temp))
            A_energy = torch.softmax(score_energy.float(), dim=-1).to(dtype)
            energy_ctx = torch.einsum('bnq,bqd->bnd', A_energy, V_energy)

            # Use average evidence attention for existing analyzer shape [B,N,E].
            score_ev_raw = torch.einsum('bnd,bsd->bns', q_base, K_ev) * scale
            A_ev = 0.5 * A_onset + 0.5 * torch.softmax(score_ev_raw.float(), dim=-1).to(dtype)
            ev_mix = 0.30 * time_ctx + 0.30 * freq_ctx + 0.25 * onset_ctx + 0.15 * energy_ctx

            # Register reads.
            K_global = self.global_k(global_matrix); V_global = self.global_v(global_matrix)
            q_global = self.global_q(q_base + task_ctx)
            score_global = torch.einsum('bnd,bgd->bng', q_global, K_global) * scale
            A_global = torch.softmax(score_global.float(), dim=-1).to(dtype)
            global_ctx = torch.einsum('bng,bgd->bnd', A_global, V_global)

            K_mem = self.memory_k(memory_matrix); V_mem = self.memory_v(memory_matrix)
            q_mem = self.memory_q(q_base + task_ctx)
            score_mem = torch.einsum('bnd,bmd->bnm', q_mem, K_mem) * scale
            A_mem = torch.softmax(score_mem.float(), dim=-1).to(dtype)
            mem_ctx = torch.einsum('bnm,bmd->bnd', A_mem, V_mem)

            # Operator candidates.
            u_mlp = self.mlp_op(torch.cat([h, ev_mix, task_ctx, global_ctx, mem_ctx], dim=-1))
            # MatrixMLP: block/state-axis mixer + channel mixer. It is still fully differentiable:
            # early training gives gradient to all blocks/primitives; later entropy schedule can sharpen.
            mm_src = block_seed + 0.50 * route_ctx + 0.25 * mem_ctx
            mm_q = self.matrix_mlp_q(h + task_ctx)
            mm_k = self.matrix_mlp_k(mm_src)
            mm_v = self.matrix_mlp_v(mm_src)
            mm_score = torch.einsum('bnd,bmd->bnm', mm_q, mm_k) * scale
            mm_attn = torch.softmax(mm_score.float(), dim=-1).to(dtype)
            mm_ctx = torch.einsum('bnm,bmd->bnd', mm_attn, mm_v)
            u_matrix_mlp = self.matrix_mlp_out(torch.cat([h, mm_ctx, task_ctx, global_ctx + mem_ctx], dim=-1))
            u_bilin = self.bilin_out(self.bilin_h(h) * self.bilin_e(ev_mix) * torch.sigmoid(self.bilin_t(task_ctx)))
            u_time = time_ctx
            u_freq = freq_ctx
            u_delta = self.delta_out(delta_feat).unsqueeze(1).expand(-1, N, -1)
            u_short = self.short_onset_out(short_feat).unsqueeze(1).expand(-1, N, -1)
            prev_mean = prev_summaries.mean(dim=1).unsqueeze(1).expand(-1,N,-1)
            prev_max = prev_summaries.amax(dim=1).unsqueeze(1).expand(-1,N,-1)
            prev_energy = prev_summaries.pow(2).mean(dim=1).unsqueeze(1).expand(-1,N,-1)
            u_global = self.global_summary_out(torch.cat([prev_mean, prev_max, prev_energy, global_ctx], dim=-1))
            u_mem_read = self.memory_read_out(torch.cat([h, mem_ctx, task_ctx], dim=-1))
            surprise = h - global_ctx
            u_mem_write = self.memory_write_out(torch.cat([surprise, mem_ctx, task_ctx], dim=-1))
            mem_mean_ctx = memory_matrix.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
            u_ema_memory = self.ema_memory_out(torch.cat([h, mem_ctx, mem_mean_ctx, global_ctx], dim=-1))
            ref = base[:, None, :].expand(-1,N,-1)
            residual = h - ref; delta_m = h - block_seed
            u_refdelta = self.residual_refdelta_out(torch.cat([residual, delta_m, block_seed, task_ctx], dim=-1))
            comp_diff = h - route_ctx; comp_prod = h * route_ctx; comp_abs = comp_diff.abs()
            u_compare = self.block_compare_out(torch.cat([comp_diff, comp_prod, comp_abs, task_ctx], dim=-1))

            # Class-pair contrast: TaskMatrix confusion rows guide repair for confused class pairs
            # such as no/on/go/down and left/right. Still matrix-space: it reads a confusion submatrix.
            class_cells = task_matrix[:, 2:2+self.num_classes, :]
            conf_cells = task_matrix[:, 2+self.num_classes:2+2*self.num_classes, :]
            pair_cells = task_matrix[:, 2+2*self.num_classes:, :]
            class_ctx_pair = class_cells.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
            if pair_cells.numel() > 0:
                conf_ctx_pair = pair_cells.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
            else:
                conf_ctx_pair = conf_cells.mean(dim=1).unsqueeze(1).expand(-1, N, -1)
            u_pair = self.class_pair_contrast_out(torch.cat([h - conf_ctx_pair, h * conf_ctx_pair, task_ctx - class_ctx_pair, conf_ctx_pair], dim=-1))
            u_class_pair_memory = self.class_pair_memory_out(torch.cat([h, conf_ctx_pair, mem_ctx, task_ctx], dim=-1))
            u_late_read_repair = self.late_read_repair_out(torch.cat([h - conf_ctx_pair, task_ctx, global_ctx, mem_ctx], dim=-1))

            common = prev_mean + 0.25 * global_ctx
            sg = torch.sigmoid(self.suppress_gate(torch.cat([h, common, task_ctx], dim=-1)))
            u_suppress = self.suppress_out(h - sg * common)
            u_norm = self.normalize_out(torch.cat([self._rms_norm_vec(h), self._rms_norm_vec(ev_mix), self._rms_norm_vec(global_ctx + mem_ctx)], dim=-1))
            ev_energy = evidence.pow(2).mean(dim=1).unsqueeze(1).expand(-1,N,-1)
            mem_energy = memory_matrix.pow(2).mean(dim=1).unsqueeze(1).expand(-1,N,-1)
            u_energy = self.energy_count_out(torch.cat([prev_energy, ev_energy, mem_energy], dim=-1))
            offset_ctx = short_feat[:, -D:].unsqueeze(1).expand(-1, N, -1)
            u_offset = self.offset_out(torch.cat([h, ev_mix, offset_ctx, task_ctx, global_ctx], dim=-1))
            u_noop = torch.zeros_like(h)
            u_identity = h
            u_keep_prev = block_seed
            u_small_refine = 0.10 * u_mlp
            candidates = torch.stack([u_noop,u_identity,u_keep_prev,u_small_refine,u_mlp,u_matrix_mlp,u_bilin,u_time,u_freq,u_delta,u_short,u_offset,u_global,u_mem_read,u_mem_write,u_ema_memory,u_class_pair_memory,u_refdelta,u_compare,u_pair,u_late_read_repair,u_suppress,u_norm,u_energy], dim=2)

            # RolePlanner: layer role mix -> operator prior. SignalBus adds data/task/mechanism bias.
            role_logits = self.role_logits[layer].to(device=device, dtype=dtype).view(1, self.num_roles).expand(B, -1)
            role_bias = self.role_signal_scale * signal_role_bias
            if self.lock_last_role and layer == (self.num_layers - 1):
                # Last phase may still learn inside aggregate/suppress, but does not fully drift into extractor.
                role_bias = 0.25 * role_bias
            role_mix = torch.softmax((role_logits + role_bias).float(), dim=-1).to(dtype)  # [B,R]
            role_prior = torch.matmul(role_mix, self.role_op_basis.to(device=device, dtype=dtype))  # [B,O]
            role_category_prior = torch.matmul(role_mix, self.role_category_basis.to(device=device, dtype=dtype))  # [B,Cat]
            role_mixes_live.append(role_mix.float().mean(dim=0))
            role_mixes_report.append(role_mix.detach().float().mean(dim=0))
            signal_op_bias_norms.append(signal_op_bias.detach().float().norm(dim=-1).mean())

            # CategoryPlanner: choose logical primitive family before choosing primitive inside it.
            category_logits = self.category_bias_net(torch.cat([h, task_ctx, global_ctx, mem_ctx], dim=-1))
            category_logits = category_logits + self.layer_category_prior[layer].to(device=device, dtype=dtype).view(1,1,self.num_categories)
            category_logits = category_logits + role_category_prior.view(B, 1, self.num_categories)
            category_logits = category_logits + self.signal_op_bias_scale * signal_category_bias.to(dtype).view(B, 1, self.num_categories)
            category_logits = category_logits + self.block_category_bias[layer, active_block_ids].to(device=device, dtype=dtype).view(1,N,self.num_categories)
            category_gates = torch.softmax((category_logits / max(1e-4, self.category_temp)).float(), dim=-1).to(dtype)
            category_mixes.append(category_gates.detach().float().mean(dim=(0,1)))

            # Operator attention: query block state -> operator address keys + learned block adapters.
            op_q = self.op_query(torch.cat([h, task_ctx, global_ctx, mem_ctx], dim=-1))
            op_keys = self.op_key(self.operator_addr.to(device=device, dtype=dtype))
            op_logits = torch.einsum('bnd,od->bno', op_q, op_keys) * scale
            op_logits = op_logits + self.op_bias_net(torch.cat([h, task_ctx, global_ctx, mem_ctx], dim=-1))
            op_logits = op_logits + self.layer_op_prior[layer].to(device=device, dtype=dtype).view(1,1,-1)
            op_logits = op_logits + role_prior.view(B, 1, self.num_ops)
            op_logits = op_logits + self.signal_op_bias_scale * signal_op_bias.to(dtype).view(B, 1, self.num_ops)
            op_logits = op_logits + self.block_op_bias[layer, active_block_ids].to(device=device, dtype=dtype).view(1,N,self.num_ops)
            op_logits = op_logits / max(1e-4, self.operator_temp)
            # PrimitivePlanner inside each category. Same primitive can belong to multiple categories.
            cat_mask = self.op_category_mask.to(device=device).view(1, 1, self.num_categories, self.num_ops)
            primitive_logits = (op_logits.float() / max(1e-4, self.primitive_temp)).unsqueeze(2).masked_fill(~cat_mask, -1e4)
            primitive_gates = torch.softmax(primitive_logits, dim=-1).to(dtype)  # [B,N,Cat,O]
            op_gates = (category_gates.unsqueeze(-1) * primitive_gates).sum(dim=2)  # [B,N,O]
            op_gates = op_gates / op_gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            def _op_mass(names: Tuple[str, ...]) -> torch.Tensor:
                idxs = [self.OP_NAMES.index(nm) for nm in names if nm in self.OP_NAMES]
                if not idxs:
                    return torch.zeros(B, N, device=device, dtype=dtype)
                return op_gates.index_select(-1, torch.tensor(idxs, device=device)).sum(dim=-1).clamp(0.0, 1.0)
            safe_write_mass = _op_mass(("noop", "identity", "keep_prev"))
            global_write_mass = _op_mass(("global_summary", "energy_count"))
            mem_write_mass = _op_mass(("memory_write", "ema_memory", "class_pair_memory"))

            op_scale = 1.0 + 0.20 * torch.tanh(self.block_op_scale[layer, active_block_ids].to(device=device, dtype=dtype)).view(1,N,self.num_ops,1)
            candidates = candidates * op_scale
            update = torch.einsum('bno,bnod->bnd', op_gates, candidates)

            gate_in = torch.cat([h, update, task_ctx, global_ctx + mem_ctx], dim=-1)
            dyn_gate_logit = self.stage_gate_net(gate_in).squeeze(-1) + 0.15 * signal_gate_bias.to(dtype).view(B, 1)
            dyn_gate = torch.sigmoid(dyn_gate_logit)
            # Phase-aware gate floor: Layer0 extracts strongly; later layers must still
            # write enough to actually transform/repair/aggregate, not only look pretty in op gates.
            phase_floor = self.gate_floor if layer == 0 else self.late_gate_floor
            eff_gate = phase_floor + (1.0 - phase_floor) * dyn_gate
            eff_gate = eff_gate * (1.0 - 0.75 * safe_write_mass).clamp(0.10, 1.0)
            update_expand = update[:, :, None, :].expand(-1,-1,self.block_cells,-1)
            task_expand = task_ctx[:, :, None, :].expand(-1,-1,self.block_cells,-1)
            cell_in = torch.cat([prev_region_for_active, update_expand, task_expand], dim=-1)
            cell_delta = self.cell_write_net(cell_in)
            cell_gate = torch.sigmoid(self.cell_gate_net(cell_in))
            write_delta = eff_gate[:,:,None,None] * cell_gate * (cell_delta + update_expand)
            new_regions = self.cell_norm(prev_region_for_active + write_delta)
            summaries = self.block_norm(new_regions.mean(dim=2) + addr)

            # Register writes from all blocks in the phase.
            write_in = torch.cat([summaries, task_ctx, ev_mix], dim=-1)
            gw_val = self.global_write_val(write_in)
            gw_gate = 0.15 * torch.sigmoid(self.global_write_gate(write_in))
            gw_gate = gw_gate * (0.05 + 0.95 * global_write_mass.unsqueeze(-1))
            gw_q = self.global_write_q(summaries)
            K_global2 = self.global_k(global_matrix)
            score_gw = torch.einsum('bnd,bgd->bng', gw_q, K_global2) * scale
            gw = torch.softmax(score_gw.float(), dim=-1).to(dtype)
            global_delta = torch.einsum('bng,bnd->bgd', gw * gw_gate, gw_val) / max(1,N)
            global_matrix = self.global_norm(global_matrix + global_delta)

            mw_val = self.memory_write_val(torch.cat([u_mem_write, task_ctx, ev_mix], dim=-1))
            mw_gate = 0.20 * torch.sigmoid(self.memory_write_gate(torch.cat([summaries, task_ctx, mem_ctx], dim=-1)))
            mw_gate = mw_gate * (0.05 + 0.95 * mem_write_mass.unsqueeze(-1))
            mw_q = self.memory_write_q(summaries + u_mem_write)
            K_mem2 = self.memory_k(memory_matrix)
            score_mw = torch.einsum('bnd,bmd->bnm', mw_q, K_mem2) * scale
            mw = torch.softmax(score_mw.float(), dim=-1).to(dtype)
            memory_delta = torch.einsum('bnm,bnd->bmd', mw * mw_gate, mw_val) / max(1,N)
            memory_matrix = self.memory_norm(memory_matrix + memory_delta)

            prev_regions = new_regions
            prev_active_blocks = list(active_blocks)
            states.append(summaries)
            evid_attns.append(A_ev)
            time_attns.append(A_time); freq_attns.append(A_freq); onset_attns.append(A_onset); energy_attns.append(A_energy)
            global_attns.append(A_global); memory_attns.append(A_mem)
            task_attns.append(A_task); route_attns.append(A_route); op_gates_all.append(op_gates); category_gates_all.append(category_gates)
            dyn_gates.append(dyn_gate); eff_gates.append(eff_gate)
            update_norms.append(update.detach().float().norm(dim=-1))
            write_delta_norm_block_live = write_delta.float().norm(dim=-1).mean(dim=-1)
            live_write_delta_norms.append(write_delta_norm_block_live)
            write_delta_norms.append(write_delta_norm_block_live.detach())

        stage_states = torch.cat([base[:, None, :]] + states[1:], dim=1)
        evidence_attention = torch.cat(evid_attns, dim=1) if evid_attns else torch.empty(B,0,E,device=device,dtype=dtype)
        time_attention = torch.cat(time_attns, dim=1) if time_attns else torch.empty(B,0,time_tokens.shape[1],device=device,dtype=dtype)
        freq_attention = torch.cat(freq_attns, dim=1) if freq_attns else torch.empty(B,0,freq_tokens.shape[1],device=device,dtype=dtype)
        onset_attention = torch.cat(onset_attns, dim=1) if onset_attns else torch.empty(B,0,E,device=device,dtype=dtype)
        energy_attention = torch.cat(energy_attns, dim=1) if energy_attns else torch.empty(B,0,energy_tokens.shape[1],device=device,dtype=dtype)
        global_attention = torch.cat(global_attns, dim=1) if global_attns else torch.empty(B,0,self.global_cells,device=device,dtype=dtype)
        memory_attention = torch.cat(memory_attns, dim=1) if memory_attns else torch.empty(B,0,self.memory_cells,device=device,dtype=dtype)
        task_attention = torch.cat(task_attns, dim=1) if task_attns else torch.empty(B,0,self.task_cells,device=device,dtype=dtype)
        route_attention = torch.cat(route_attns, dim=1) if route_attns else torch.empty(B,0,self.blocks_per_layer+2,device=device,dtype=dtype)
        operator_gates = torch.cat(op_gates_all, dim=1) if op_gates_all else torch.empty(B,0,self.num_ops,device=device,dtype=dtype)
        category_gates_all_tensor = torch.cat(category_gates_all, dim=1) if category_gates_all else torch.empty(B,0,self.num_categories,device=device,dtype=dtype)
        category_mix_by_layer = torch.stack(category_mixes, dim=0) if category_mixes else torch.empty(0, self.num_categories, device=device)
        dynamic_stage_gate = torch.cat(dyn_gates, dim=1) if dyn_gates else torch.empty(B,0,device=device,dtype=dtype)
        effective_stage_gate = torch.cat(eff_gates, dim=1) if eff_gates else torch.empty(B,0,device=device,dtype=dtype)
        update_norm_by_block = torch.cat(update_norms, dim=1) if update_norms else torch.empty(B, 0, device=device)
        write_delta_norm_by_block = torch.cat(write_delta_norms, dim=1) if write_delta_norms else torch.empty(B, 0, device=device)
        live_write_delta_norm_by_block = torch.cat(live_write_delta_norms, dim=1) if live_write_delta_norms else torch.empty(B, 0, device=device)
        role_mix_by_layer_live = torch.stack(role_mixes_live, dim=0) if role_mixes_live else torch.empty(0, self.num_roles, device=device)
        role_mix_by_layer = torch.stack(role_mixes_report, dim=0) if role_mixes_report else torch.empty(0, self.num_roles, device=device)
        signal_summary = torch.stack([
            signal_ctx.detach().float().norm(dim=-1).mean(),
            signal_role_bias.detach().float().norm(dim=-1).mean(),
            signal_category_bias.detach().float().norm(dim=-1).mean(),
            signal_op_bias.detach().float().norm(dim=-1).mean(),
            signal_gate_bias.detach().float().abs().mean(),
        ]).to(device)
        signal_op_bias_norm = torch.stack(signal_op_bias_norms).mean() if signal_op_bias_norms else torch.zeros((), device=device)

        address_states = [torch.zeros(1,D,device=device,dtype=dtype)]
        for layer, active_blocks in enumerate(self.topology_plan):
            for block in active_blocks:
                address_states.append((self.layer_addr[layer] + self.block_addr[block]).to(device=device,dtype=dtype).view(1,-1))
        addr_tensor = torch.stack(address_states, dim=1)
        keyed_states = stage_states + addr_tensor
        q = self.cls_q(self.class_query.to(device=device,dtype=dtype)).view(1,self.num_classes,D).expand(B,-1,-1)
        k = self.cls_k(keyed_states); v = self.cls_v(stage_states)
        score_cls = torch.einsum('bcd,bld->bcl', q, k) * scale
        score_cls = score_cls + self.class_stage_bias.to(device=device,dtype=dtype).view(1,self.num_classes,self.chain_depth+1)
        layer_bias_items = []
        for layer, active_blocks in enumerate(self.topology_plan):
            for block in active_blocks:
                layer_bias_items.append(self.class_layer_read_bias[layer, block])
        layer_bias_flat = torch.stack(layer_bias_items).to(device=device,dtype=dtype)
        stage_bias = torch.cat([torch.zeros(1,device=device,dtype=dtype), layer_bias_flat], dim=0)
        score_cls = score_cls + stage_bias.view(1,1,-1)
        A_cls = torch.softmax(score_cls.float(), dim=-1).to(dtype)
        class_ctx = torch.einsum('bcl,bld->bcd', A_cls, v)
        logits_cell = torch.einsum('bcd,cd->bc', class_ctx, self.class_write.to(device=device,dtype=dtype)) + self.logit_bias.to(device=device,dtype=dtype)

        read_entropy_loss = -(A_cls.float().clamp_min(1e-12) * A_cls.float().clamp_min(1e-12).log()).sum(dim=-1).mean()
        rr = F.normalize(A_cls.float().mean(dim=0), dim=-1)
        sim = rr @ rr.T
        eye = torch.eye(self.num_classes, device=device, dtype=torch.bool)
        offdiag = sim.masked_select(~eye.expand_as(sim))
        read_diversity_loss = F.relu(offdiag - 0.30).mean()
        stage_usage = A_cls.float().mean(dim=(0,1)); stage_usage = stage_usage / stage_usage.sum().clamp_min(1e-8)
        ent = -(stage_usage * stage_usage.clamp_min(1e-8).log()).sum()
        stage_load_balance_loss = (math.log(self.chain_depth + 1) - ent) / math.log(self.chain_depth + 1)
        if evidence_attention.numel() > 0:
            ev_usage = evidence_attention.float().mean(dim=(0,1)); ev_usage = ev_usage / ev_usage.sum().clamp_min(1e-8)
            ent_ev = -(ev_usage * ev_usage.clamp_min(1e-8).log()).sum()
            evidence_source_balance_loss = (math.log(E) - ent_ev) / math.log(max(2,E))
        else:
            evidence_source_balance_loss = torch.zeros((),device=device)
        if operator_gates.numel() > 0:
            op_usage = operator_gates.float().mean(dim=(0,1)); op_usage = op_usage / op_usage.sum().clamp_min(1e-8)
            ent_op = -(op_usage * op_usage.clamp_min(1e-8).log()).sum()
            operator_balance_loss = (math.log(self.num_ops) - ent_op) / math.log(self.num_ops)
            op_by_block = operator_gates.float().mean(dim=0)
            ent_block = -(op_by_block.clamp_min(1e-8) * op_by_block.clamp_min(1e-8).log()).sum(dim=-1) / math.log(self.num_ops)
            operator_entropy_per_block_loss = F.relu(0.55 - ent_block).mean()
        else:
            operator_balance_loss = torch.zeros((),device=device)
            operator_entropy_per_block_loss = torch.zeros((),device=device)
        def matrix_diversity_loss(mat: torch.Tensor) -> torch.Tensor:
            n = mat.shape[1]
            if n <= 1: return torch.zeros((),device=device)
            x = F.normalize(mat.float(), dim=-1)
            sim = torch.einsum('bid,bjd->bij', x, x)
            eye = torch.eye(n, device=device, dtype=torch.bool).view(1,n,n)
            return F.relu(sim.masked_select(~eye.expand_as(sim)) - 0.15).mean()
        global_diversity_loss = matrix_diversity_loss(global_matrix)
        memory_diversity_loss = matrix_diversity_loss(memory_matrix)
        register_diversity_loss = global_diversity_loss + memory_diversity_loss

        def block_like_diversity(x: torch.Tensor, margin: float = 0.65) -> torch.Tensor:
            # x [B,N,D] or [B,N,R]. Squared off-diagonal cosine keeps pressure
            # active when blocks/routes are still similar; no silent zero at 0.94 cosine.
            n = x.shape[1]
            if n <= 1:
                return torch.zeros((), device=device)
            z = F.normalize(x.float().mean(dim=0), dim=-1)  # [N,D/R]
            sim = z @ z.T
            eye = torch.eye(n, device=device, dtype=torch.bool)
            off = sim.masked_select(~eye)
            return off.pow(2).mean()


        block_div_losses = []
        route_div_losses = []
        offset = 1
        route_offset = 0
        for layer, active_blocks in enumerate(self.topology_plan):
            n = len(active_blocks)
            if n > 1:
                block_div_losses.append(block_like_diversity(stage_states[:, offset:offset+n, :], margin=0.65))
                if route_attention.numel() > 0:
                    route_div_losses.append(block_like_diversity(route_attention[:, route_offset:route_offset+n, :], margin=0.55))
            offset += n
            route_offset += n
        block_diversity_loss = torch.stack(block_div_losses).mean() if block_div_losses else torch.zeros((), device=device)
        route_diversity_loss = torch.stack(route_div_losses).mean() if route_div_losses else torch.zeros((), device=device)

        # Prevent late layers from solving everything through GLOBAL_REGISTER_SKIP.
        global_skip_losses = []
        if route_attention.numel() > 0 and route_attention.shape[-1] > 1:
            ro = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                if layer >= 2:
                    global_skip_losses.append(F.relu(route_attention[:, ro:ro+n, 1].float() - 0.80).mean())
                ro += n
        global_skip_penalty_loss = torch.stack(global_skip_losses).mean() if global_skip_losses else torch.zeros((), device=device)

        adapter_div_losses = []
        for layer, active_blocks in enumerate(self.topology_plan):
            if len(active_blocks) <= 1:
                continue
            idx = torch.tensor(active_blocks, device=self.block_op_scale.device, dtype=torch.long)
            a = self.block_op_scale[layer, idx, :] + self.block_op_bias[layer, idx, :]
            z = F.normalize(a.float() + 1e-4 * torch.arange(a.shape[-1], device=a.device).float().view(1, -1), dim=-1)
            sim = z @ z.T
            eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
            off = sim.masked_select(~eye)
            adapter_div_losses.append(off.pow(2).mean())
        adapter_diversity_loss = torch.stack(adapter_div_losses).mean() if adapter_div_losses else torch.zeros((), device=device)

        phase_losses = []
        offset = 0
        for layer, active_blocks in enumerate(self.topology_plan):
            n = len(active_blocks)
            if operator_gates.numel() > 0 and layer < self.phase_op_target.shape[0]:
                usage = operator_gates[:, offset:offset+n, :].float().mean(dim=(0, 1))
                usage = usage / usage.sum().clamp_min(1e-8)
                tgt = self.phase_op_target[layer].to(device=device).float()
                if float(tgt.sum().item()) > 0:
                    # Interpretable role pressure: loss is 1 - mass on expected phase operators.
                    # This matches phase_role_diagnostics.match_mass instead of raw CE scale.
                    expected_mass = (usage * (tgt > 0).float()).sum()
                    phase_losses.append((1.0 - expected_mass).clamp_min(0.0))
            offset += n
        phase_role_loss = torch.stack(phase_losses).mean() if phase_losses else torch.zeros((), device=device)

        # Late write repair: layers 1..end must write some non-trivial delta.
        # This fixes v11.1d: good op programs but near-zero write_delta in late layers.
        late_write_losses = []
        if live_write_delta_norm_by_block.numel() > 0:
            off = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                if layer >= 1:
                    wn = live_write_delta_norm_by_block[:, off:off+n].mean(dim=1)
                    late_write_losses.append(F.relu(self.late_write_target - wn).pow(2).mean())
                off += n
        late_write_loss = torch.stack(late_write_losses).mean() if late_write_losses else torch.zeros((), device=device)

        # Teach late dynamic gates to open by signal, not only via late_gate_floor.
        late_dyn_gate_losses = []
        if dynamic_stage_gate.numel() > 0:
            off = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                if layer >= 1:
                    dg = dynamic_stage_gate[:, off:off+n].mean(dim=1)
                    late_dyn_gate_losses.append(F.relu(self.late_dyn_gate_target - dg).pow(2).mean())
                off += n
        late_dyn_gate_loss = torch.stack(late_dyn_gate_losses).mean() if late_dyn_gate_losses else torch.zeros((), device=device)

        # Weak-class late-read repair: high-error classes should not be solved only by BASE/L0.
        # Uses train-side pressure_ema, no validation leakage.
        class_err_for_read = pressure[2:2+self.num_classes].float()
        weak_w = F.relu(class_err_for_read - class_err_for_read.mean())
        weak_w = weak_w / weak_w.sum().clamp_min(1e-8)
        late_mask = torch.zeros(self.chain_depth + 1, device=device)
        pos = 1
        for layer, active_blocks in enumerate(self.topology_plan):
            n = len(active_blocks)
            if layer >= 1:
                late_mask[pos:pos+n] = 1.0
            pos += n
        late_mass = (A_cls.float() * late_mask.view(1, 1, -1)).sum(dim=-1).mean(dim=0)  # [C]
        weak_class_late_read_loss = (weak_w * F.relu(self.weak_class_late_read_target - late_mass).pow(2)).sum()

        # RolePlanner regularization.
        if role_mix_by_layer_live.numel() > 0:
            # Encourage adjacent layers to use different role mixtures, not one universal solution.
            if role_mix_by_layer_live.shape[0] > 1:
                rn = F.normalize(role_mix_by_layer_live.float(), dim=-1)
                adj_sim = (rn[:-1] * rn[1:]).sum(dim=-1)
                role_diversity_loss = F.relu(adj_sim - 0.65).pow(2).mean()
            else:
                role_diversity_loss = torch.zeros((), device=device)
            # Keep final phase mostly aggregate/suppress unless data strongly proves otherwise.
            if self.lock_last_role and role_mix_by_layer_live.shape[0] >= 1:
                last = role_mix_by_layer_live[-1].float()
                agg_idx = self.ROLE_NAMES.index("aggregate")
                sup_idx = self.ROLE_NAMES.index("suppress")
                role_anchor_loss = F.relu(0.65 - (last[agg_idx] + 0.35 * last[sup_idx])).pow(2)
            else:
                role_anchor_loss = torch.zeros((), device=device)
        else:
            role_diversity_loss = torch.zeros((), device=device)
            role_anchor_loss = torch.zeros((), device=device)

        # Category health: avoid same category mix in all blocks, but keep category choices reasonably sharp.
        category_div_losses = []
        if category_gates_all_tensor.numel() > 0:
            off = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                cg = category_gates_all_tensor[:, off:off+n, :].float().mean(dim=0)  # [n,Cat]
                if n > 1:
                    zn = F.normalize(cg, dim=-1)
                    sim = zn @ zn.T
                    eye = torch.eye(n, device=device, dtype=torch.bool)
                    category_div_losses.append(sim.masked_select(~eye).pow(2).mean())
                off += n
            ent = -(category_gates_all_tensor.float().clamp_min(1e-12) * category_gates_all_tensor.float().clamp_min(1e-12).log()).sum(dim=-1).mean()
            category_entropy_loss = ent / math.log(max(2, self.num_categories))
        else:
            category_entropy_loss = torch.zeros((), device=device)
        category_diversity_loss = torch.stack(category_div_losses).mean() if category_div_losses else torch.zeros((), device=device)

        if 'update_norm_by_block' not in locals():
            update_norm_by_block = torch.empty(B, 0, device=device)
        if 'write_delta_norm_by_block' not in locals():
            write_delta_norm_by_block = torch.empty(B, 0, device=device)

        aux = SeqAux(
            evidence_attention=evidence_attention.detach(),
            time_attention=time_attention.detach(),
            freq_attention=freq_attention.detach(),
            onset_attention=onset_attention.detach(),
            energy_attention=energy_attention.detach(),
            global_attention=global_attention.detach(),
            memory_attention=memory_attention.detach(),
            task_attention=task_attention.detach(),
            route_attention=route_attention.detach(),
            operator_gates=operator_gates.detach(),
            category_gates=category_gates_all_tensor.detach(),
            task_matrix_norm=task_matrix.detach().float().norm(dim=-1).mean(dim=0),
            global_matrix_norm=global_matrix.detach().float().norm(dim=-1).mean(dim=0),
            memory_matrix_norm=memory_matrix.detach().float().norm(dim=-1).mean(dim=0),
            stage_states=stage_states.detach(),
            dynamic_stage_gate=dynamic_stage_gate.detach(),
            effective_stage_gate=effective_stage_gate.detach(),
            static_stage_gate=self.static_stage_gate_buffer.to(device=device,dtype=dtype).detach(),
            class_stage_attention=A_cls.detach(),
            pressure_features=pressure.detach(),
            read_entropy_loss=read_entropy_loss,
            read_diversity_loss=read_diversity_loss,
            stage_load_balance_loss=stage_load_balance_loss,
            evidence_source_balance_loss=evidence_source_balance_loss,
            operator_balance_loss=operator_balance_loss,
            register_diversity_loss=register_diversity_loss,
            global_diversity_loss=global_diversity_loss,
            memory_diversity_loss=memory_diversity_loss,
            operator_entropy_per_block_loss=operator_entropy_per_block_loss,
            block_diversity_loss=block_diversity_loss,
            route_diversity_loss=route_diversity_loss,
            adapter_diversity_loss=adapter_diversity_loss,
            phase_role_loss=phase_role_loss,
            global_skip_penalty_loss=global_skip_penalty_loss,
            late_write_loss=late_write_loss,
            late_dyn_gate_loss=late_dyn_gate_loss,
            weak_class_late_read_loss=weak_class_late_read_loss,
            role_diversity_loss=role_diversity_loss,
            role_anchor_loss=role_anchor_loss,
            category_diversity_loss=category_diversity_loss,
            category_entropy_loss=category_entropy_loss,
            signal_op_bias_norm=signal_op_bias_norm.detach(),
            role_mix_by_layer=role_mix_by_layer.detach(),
            category_mix_by_layer=category_mix_by_layer.detach(),
            signal_summary=signal_summary.detach(),
            update_norm_by_block=update_norm_by_block.detach(),
            write_delta_norm_by_block=write_delta_norm_by_block.detach(),
            logits_cell_norm=logits_cell.detach().float().norm(dim=-1),
        )
        return logits_cell, aux

    def regularization(self, lambda_stage: float = 0.0, lambda_addr: float = 0.0) -> torch.Tensor:
        reg = torch.zeros((), device=self.layer_addr.device)
        if lambda_addr > 0:
            addr = torch.cat([
                self.layer_addr.float(), self.block_addr.float(), self.cell_addr.float(),
                self.global_addr.float(), self.memory_addr.float(), self.head_addr.float(), self.operator_addr.float()
            ], dim=0)
            addr = F.normalize(addr, dim=-1)
            sim = addr @ addr.T
            eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
            reg = reg + float(lambda_addr) * sim.masked_select(~eye).pow(2).mean()
        if lambda_stage > 0:
            terms = [q.float().pow(2).mean() for q in self.stage_gate_net.parameters()]
            terms += [self.class_stage_bias.float().pow(2).mean(), self.class_layer_read_bias.float().pow(2).mean()]
            reg = reg + float(lambda_stage) * torch.stack(terms).mean()
        return reg


class SequentialMatrixCellsModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dim: int = 128,
        sample_rate: int = 16000,
        n_mels: int = 64,
        hop_length: int = 160,
        evidence_cells: int = 48,
        chain_depth: int = 12,
        num_layers: int = 4,
        blocks_per_layer: int = 3,
        block_cells: int = 2,
        global_cells: int = 2,
        memory_cells: int = 4,
        dropout: float = 0.05,
        pressure_alpha: float = 0.20,
        evidence_dropout: float = 0.10,
        class_stage_bias_std: float = 0.35,
        stage_gate_bias_init: float = 1.0,
        gate_floor: float = 0.15,
        late_gate_floor: float = 0.25,
        late_write_target: float = 0.35,
        late_dyn_gate_target: float = 0.15,
        weak_class_late_read_target: float = 0.35,
        role_signal_scale: float = 0.35,
        signal_op_bias_scale: float = 0.20,
        category_temp: float = 1.0,
        primitive_temp: float = 1.0,
        lock_last_role: bool = True,
        task_attn_temp: float = 0.85,
        evidence_attn_temp: float = 1.0,
        operator_temp: float = 1.15,
        topology: str = "grid",
        skip_connections: bool = True,
    ):
        super().__init__()
        self.evidence_cells = int(evidence_cells)
        self.evidence_dropout = float(evidence_dropout)
        self.frontend = MelPatchFrontend(dim=dim, sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length)
        self.evidence_norm = nn.LayerNorm(dim)
        self.core = SequentialMatrixCellsCore(
            dim=dim,
            num_classes=num_classes,
            evidence_cells=evidence_cells,
            chain_depth=chain_depth,
            num_layers=num_layers,
            blocks_per_layer=blocks_per_layer,
            block_cells=block_cells,
            global_cells=global_cells,
            memory_cells=memory_cells,
            pressure_alpha=pressure_alpha,
            dropout=dropout,
            class_stage_bias_std=class_stage_bias_std,
            stage_gate_bias_init=stage_gate_bias_init,
            gate_floor=gate_floor,
            late_gate_floor=late_gate_floor,
            late_write_target=late_write_target,
            late_dyn_gate_target=late_dyn_gate_target,
            weak_class_late_read_target=weak_class_late_read_target,
            role_signal_scale=role_signal_scale,
            signal_op_bias_scale=signal_op_bias_scale,
            category_temp=category_temp,
            primitive_temp=primitive_temp,
            lock_last_role=lock_last_role,
            task_attn_temp=task_attn_temp,
            evidence_attn_temp=evidence_attn_temp,
            operator_temp=operator_temp,
            topology=topology,
            skip_connections=skip_connections,
        )

    @torch.no_grad()
    def update_train_pressure(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        self.core.update_pressure_ema(logits, y)

    def _build_evidence(self, x: torch.Tensor, grid: Tuple[int, int]) -> torch.Tensor:
        """Build richer EvidenceMatrix: 6x6 time-frequency grid + global/onset/duration cells."""
        B, N, D = x.shape
        T, Freq = grid
        z = x.view(B, T, Freq, D)
        feats: List[torch.Tensor] = []

        # 6x6 time-frequency grid = 36 cells.
        T_SPLITS, F_SPLITS = 6, 6
        for ti in range(T_SPLITS):
            for fi in range(F_SPLITS):
                t0 = int(ti / T_SPLITS * T)
                t1 = max(t0 + 1, int((ti + 1) / T_SPLITS * T)); t1 = min(t1, T)
                f0 = int(fi / F_SPLITS * Freq)
                f1 = max(f0 + 1, int((fi + 1) / F_SPLITS * Freq)); f1 = min(f1, Freq)
                feats.append(z[:, t0:t1, f0:f1, :].mean(dim=(1, 2)))

        # Global and acoustic shape cells.
        global_mean = z.mean(dim=(1, 2))
        global_std = z.std(dim=(1, 2))
        global_max = z.amax(dim=(1, 2))
        feats.extend([global_mean, global_std, global_max])

        # Time/frequency marginals for short commands.
        time_rows = z.mean(dim=2)  # [B,T,D]
        freq_cols = z.mean(dim=1)  # [B,F,D]
        early = time_rows[:, :max(1, T // 4), :].mean(dim=1)
        mid = time_rows[:, max(0, T // 3):max(1, 2 * T // 3), :].mean(dim=1)
        late = time_rows[:, max(0, T - max(1, T // 4)):, :].mean(dim=1)
        onset = mid - early
        offset = late - mid
        low = freq_cols[:, :max(1, Freq // 3), :].mean(dim=1)
        high = freq_cols[:, max(0, 2 * Freq // 3):, :].mean(dim=1)
        feats.extend([early, mid, late, onset, offset, low, high, high - low])

        n_orig = len(feats)
        while len(feats) < self.evidence_cells:
            feats.append(feats[len(feats) % n_orig])
        evidence = torch.stack(feats[:self.evidence_cells], dim=1)
        return self.evidence_norm(evidence)

    def forward(self, wav: torch.Tensor, return_aux: bool = False):
        x, grid = self.frontend(wav)
        evidence = self._build_evidence(x, grid)
        if self.training and self.evidence_dropout > 0:
            keep = (torch.rand(evidence.shape[0], evidence.shape[1], 1, device=evidence.device) > self.evidence_dropout).to(evidence.dtype)
            evidence = evidence * keep / max(1e-6, (1.0 - self.evidence_dropout))
        logits, aux = self.core(evidence)
        if return_aux:
            return logits, {"tokens": x, "evidence": evidence, "grid": grid, "seq": aux}
        return logits

    def regularization(self, args) -> torch.Tensor:
        return self.core.regularization(lambda_stage=args.lambda_stage, lambda_addr=args.lambda_addr)


# ---------------------------
# analysis
# ---------------------------

class SeqAccumulator:
    def __init__(self, num_classes: int, evidence_cells: int, chain_depth: int, num_layers: int = 4, blocks_per_layer: int = 3, decomp_every: int = 25, topology_plan: List[List[int]] | None = None):
        self.num_classes = num_classes
        self.evidence_cells = evidence_cells
        self.num_layers = int(num_layers)
        self.blocks_per_layer = int(blocks_per_layer)
        self.topology_plan = [list(x) for x in topology_plan] if topology_plan is not None else [list(range(self.blocks_per_layer)) for _ in range(self.num_layers)]
        self.flat_to_layer_block = [(li, bi) for li, blocks in enumerate(self.topology_plan) for bi in blocks]
        self.chain_depth = len(self.flat_to_layer_block)
        self.decomp_every = int(decomp_every)
        self.reset()

    def block_label(self, flat_idx: int) -> Dict[str, Any]:
        flat_idx = int(flat_idx)
        if 0 <= flat_idx < len(self.flat_to_layer_block):
            layer, block = self.flat_to_layer_block[flat_idx]
        else:
            layer = flat_idx // max(1, self.blocks_per_layer)
            block = flat_idx % max(1, self.blocks_per_layer)
        return {"stage": flat_idx, "layer": int(layer), "block": int(block), "name": f"L{int(layer)}.B{int(block)}"}

    def read_label(self, read_idx: int) -> Dict[str, Any]:
        if int(read_idx) == 0:
            return {"stage": 0, "layer": -1, "block": -1, "name": "BASE"}
        out = self.block_label(int(read_idx) - 1)
        out["stage"] = int(read_idx)
        return out

    def reset(self):
        self.count = 0
        self.evidence_attention = None
        self.time_attention = None
        self.freq_attention = None
        self.onset_attention = None
        self.energy_attention = None
        self.global_attention_by_block = None
        self.memory_attention_by_block = None
        self.task_attention = None
        self.route_attention = None
        self.operator_gates = None
        self.category_gates = None
        self.task_matrix_norm = None
        self.global_matrix_norm = None
        self.memory_matrix_norm = None
        self.dynamic_stage_gate = None
        self.effective_stage_gate = None
        self.static_stage_gate = None
        self.class_stage_attention = None
        self.stage_state_norm = None
        self.pressure_features = None
        self.logits_cell_norm = 0.0
        self.read_entropy = 0.0
        self.read_diversity = 0.0
        self.stage_load_balance = 0.0
        self.evidence_source_balance = 0.0
        self.operator_balance = 0.0
        self.register_diversity = 0.0
        self.global_diversity = 0.0
        self.memory_diversity = 0.0
        self.operator_entropy_per_block = 0.0
        self.block_diversity = 0.0
        self.route_diversity = 0.0
        self.adapter_diversity = 0.0
        self.phase_role = 0.0
        self.global_skip_penalty = 0.0
        self.late_write = 0.0
        self.late_dyn_gate = 0.0
        self.weak_class_late_read = 0.0
        self.role_diversity = 0.0
        self.role_anchor = 0.0
        self.category_diversity = 0.0
        self.category_entropy = 0.0
        self.signal_op_bias_norm = 0.0
        self.role_mix_by_layer = None
        self.category_mix_by_layer = None
        self.signal_summary = None
        self.update_norm_by_block = None
        self.write_delta_norm_by_block = None
        self.layer_rank_sum = [0.0 for _ in range(self.num_layers)]
        self.layer_cos_sum = [0.0 for _ in range(self.num_layers)]
        self.layer_rank_count = 0
        self.attn_corr_gap_sum = [0.0 for _ in range(self.chain_depth)]
        self.attn_corr_gap_count = 0

    @torch.no_grad()
    def add(self, aux: SeqAux):
        def acc(old, x):
            x = x.detach().float().cpu()
            return x if old is None else old + x

        self.count += 1
        self.evidence_attention = acc(self.evidence_attention, aux.evidence_attention.mean(dim=0))  # [L,S] proxy
        self.time_attention = acc(self.time_attention, aux.time_attention.mean(dim=0))
        self.freq_attention = acc(self.freq_attention, aux.freq_attention.mean(dim=0))
        self.onset_attention = acc(self.onset_attention, aux.onset_attention.mean(dim=0))
        self.energy_attention = acc(self.energy_attention, aux.energy_attention.mean(dim=0))
        self.global_attention_by_block = acc(self.global_attention_by_block, aux.global_attention.mean(dim=0))
        self.memory_attention_by_block = acc(self.memory_attention_by_block, aux.memory_attention.mean(dim=0))
        self.task_attention = acc(self.task_attention, aux.task_attention.mean(dim=0))  # [L,T_task]
        self.route_attention = acc(self.route_attention, aux.route_attention.mean(dim=0))  # [L,R]
        self.operator_gates = acc(self.operator_gates, aux.operator_gates.mean(dim=0))  # [L,O]
        self.category_gates = acc(self.category_gates, aux.category_gates.mean(dim=0))  # [L,Cat]
        self.task_matrix_norm = acc(self.task_matrix_norm, aux.task_matrix_norm)  # [T_task]
        self.global_matrix_norm = acc(self.global_matrix_norm, aux.global_matrix_norm)  # [G]
        self.memory_matrix_norm = acc(self.memory_matrix_norm, aux.memory_matrix_norm)  # [M]
        self.dynamic_stage_gate = acc(self.dynamic_stage_gate, aux.dynamic_stage_gate.mean(dim=0))  # [L]
        self.effective_stage_gate = acc(self.effective_stage_gate, aux.effective_stage_gate.mean(dim=0))  # [L]
        self.update_norm_by_block = acc(self.update_norm_by_block, aux.update_norm_by_block.mean(dim=0))  # [L]
        self.write_delta_norm_by_block = acc(self.write_delta_norm_by_block, aux.write_delta_norm_by_block.mean(dim=0))  # [L]
        self.static_stage_gate = aux.static_stage_gate.detach().float().cpu()
        self.class_stage_attention = acc(self.class_stage_attention, aux.class_stage_attention.mean(dim=0))  # [C,L+1]
        self.stage_state_norm = acc(self.stage_state_norm, aux.stage_states.detach().float().norm(dim=-1).mean(dim=0))  # [L+1]

        # Matrix decomposition analytics are intentionally sampled.
        # Full SVD/rank every batch forces GPU->CPU sync and can dominate epoch time.
        do_decomp = (self.decomp_every > 0 and (self.count % self.decomp_every == 0))
        if do_decomp:
            st = aux.stage_states.detach().float().cpu()  # [B,N+1,D], index0=base
            ev_attn_b = aux.evidence_attention.detach().float().cpu()  # [B,N,E]
            offset = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                s = offset + 1
                e = s + n
                offset += n
                if n > 0 and e <= st.shape[1]:
                    x = st[:, s:e, :].reshape(-1, st.shape[-1])
                    x = x - x.mean(dim=0, keepdim=True)
                    try:
                        sv = torch.linalg.svdvals(x)
                        p = sv / sv.sum().clamp_min(1e-8)
                        er = float(torch.exp(-(p * p.clamp_min(1e-8).log()).sum()).cpu())
                    except Exception:
                        er = 0.0
                    block_mean = st[:, s:e, :].mean(dim=0)  # [blocks,D]
                    z = F.normalize(block_mean, dim=-1)
                    sim = z @ z.T
                    eye = torch.eye(sim.shape[0], dtype=torch.bool)
                    cos = float(sim.masked_select(~eye).mean().cpu()) if sim.numel() > sim.shape[0] else 0.0
                    self.layer_rank_sum[layer] += er
                    self.layer_cos_sum[layer] += cos
            self.layer_rank_count += 1

            # Attention concentration proxy sampled with rank.
            if ev_attn_b.numel() > 0:
                p_ev = ev_attn_b.mean(dim=0).clamp_min(1e-8)  # [N,E]
                ent_ev = -(p_ev * p_ev.log()).sum(dim=-1) / math.log(max(2, p_ev.shape[-1]))
                gap = (1.0 - ent_ev).tolist()
                for i, g in enumerate(gap[:self.chain_depth]):
                    self.attn_corr_gap_sum[i] += float(g)
                self.attn_corr_gap_count += 1

        self.pressure_features = aux.pressure_features.detach().float().cpu()
        self.logits_cell_norm += float(aux.logits_cell_norm.detach().float().mean().cpu())
        self.read_entropy += float(aux.read_entropy_loss.detach().float().cpu())
        self.read_diversity += float(aux.read_diversity_loss.detach().float().cpu())
        self.stage_load_balance += float(aux.stage_load_balance_loss.detach().float().cpu())
        self.evidence_source_balance += float(aux.evidence_source_balance_loss.detach().float().cpu())
        self.operator_balance += float(aux.operator_balance_loss.detach().float().cpu())
        self.register_diversity += float(aux.register_diversity_loss.detach().float().cpu())
        self.global_diversity += float(aux.global_diversity_loss.detach().float().cpu())
        self.memory_diversity += float(aux.memory_diversity_loss.detach().float().cpu())
        self.operator_entropy_per_block += float(aux.operator_entropy_per_block_loss.detach().float().cpu())
        self.block_diversity += float(aux.block_diversity_loss.detach().float().cpu())
        self.route_diversity += float(aux.route_diversity_loss.detach().float().cpu())
        self.adapter_diversity += float(aux.adapter_diversity_loss.detach().float().cpu())
        self.phase_role += float(aux.phase_role_loss.detach().float().cpu())
        self.global_skip_penalty += float(aux.global_skip_penalty_loss.detach().float().cpu())
        self.late_write += float(aux.late_write_loss.detach().float().cpu())
        self.late_dyn_gate += float(aux.late_dyn_gate_loss.detach().float().cpu())
        self.weak_class_late_read += float(aux.weak_class_late_read_loss.detach().float().cpu())
        self.role_diversity += float(aux.role_diversity_loss.detach().float().cpu())
        self.role_anchor += float(aux.role_anchor_loss.detach().float().cpu())
        self.category_diversity += float(aux.category_diversity_loss.detach().float().cpu())
        self.category_entropy += float(aux.category_entropy_loss.detach().float().cpu())
        self.signal_op_bias_norm += float(aux.signal_op_bias_norm.detach().float().cpu())
        self.role_mix_by_layer = acc(self.role_mix_by_layer, aux.role_mix_by_layer)
        self.category_mix_by_layer = acc(self.category_mix_by_layer, aux.category_mix_by_layer)
        self.signal_summary = acc(self.signal_summary, aux.signal_summary)

    @torch.no_grad()
    def summary(self, classes: List[str]) -> Dict[str, Any]:
        c = max(1, self.count)
        ev_attn = self.evidence_attention / c if self.evidence_attention is not None else torch.zeros(self.chain_depth, self.evidence_cells)
        channel_attn = {
            "time": self.time_attention / c if self.time_attention is not None else None,
            "freq": self.freq_attention / c if self.freq_attention is not None else None,
            "onset": self.onset_attention / c if self.onset_attention is not None else None,
            "energy": self.energy_attention / c if self.energy_attention is not None else None,
            "global": self.global_attention_by_block / c if self.global_attention_by_block is not None else None,
            "memory": self.memory_attention_by_block / c if self.memory_attention_by_block is not None else None,
        }
        cls_attn = self.class_stage_attention / c if self.class_stage_attention is not None else torch.zeros(self.num_classes, self.chain_depth + 1)

        top_evidence_by_stage = []
        for i in range(ev_attn.shape[0]):
            vals, idxs = torch.topk(ev_attn[i], k=min(6, ev_attn.shape[1]))
            item = self.block_label(i)
            item["top_evidence"] = [{"cell": int(j), "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
            top_evidence_by_stage.append(item)

        task_attn = self.task_attention / c if self.task_attention is not None else torch.zeros(self.chain_depth, 1)
        attention_channels_by_stage = []
        for channel_name, attn in channel_attn.items():
            if attn is None or attn.numel() == 0:
                continue
            for i in range(attn.shape[0]):
                vals, idxs = torch.topk(attn[i], k=min(5, attn.shape[1]))
                item = self.block_label(i)
                item["channel"] = channel_name
                item["top"] = [{"cell": int(j), "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
                attention_channels_by_stage.append(item)

        def top_attn_items(attn, i: int, k: int = 4) -> List[Dict[str, float]]:
            if attn is None or attn.numel() == 0 or i >= attn.shape[0]:
                return []
            vals, idxs = torch.topk(attn[i], k=min(k, attn.shape[1]))
            return [{"cell": int(j), "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]

        top_task_by_stage = []
        for i in range(task_attn.shape[0]):
            vals, idxs = torch.topk(task_attn[i], k=min(8, task_attn.shape[1]))
            item = self.block_label(i)
            item["top_task"] = [{"cell": int(j), "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
            top_task_by_stage.append(item)

        route_attn = self.route_attention / c if self.route_attention is not None else torch.zeros(self.chain_depth, self.blocks_per_layer + 2)
        route_usage_by_block = []
        for i in range(route_attn.shape[0]):
            if 0 <= int(i) < len(self.flat_to_layer_block):
                layer, _block = self.flat_to_layer_block[int(i)]
            else:
                layer = int(i) // max(1, self.blocks_per_layer)
            prev_layer = int(layer) - 1
            prev_blocks = self.topology_plan[prev_layer] if 0 <= prev_layer < len(self.topology_plan) else list(range(self.blocks_per_layer))
            vals, idxs = torch.topk(route_attn[i], k=min(5, route_attn.shape[1]))
            route_items = []
            for v, j in zip(vals.tolist(), idxs.tolist()):
                jj = int(j)
                if jj == 0:
                    name = "BASE_SKIP"
                elif jj == 1:
                    name = "GLOBAL_REGISTER_SKIP"
                else:
                    route_pos = jj - 2
                    if prev_layer < 0:
                        name = f"INIT.B{route_pos}"
                    elif 0 <= route_pos < len(prev_blocks):
                        name = f"L{prev_layer}.B{int(prev_blocks[route_pos])}"
                    else:
                        name = f"L{prev_layer}.B?{route_pos}"
                route_items.append({"route": jj, "name": name, "weight": float(v)})
            item = self.block_label(i)
            item["routes"] = route_items
            route_usage_by_block.append(item)

        op_gates = self.operator_gates / c if self.operator_gates is not None else torch.zeros(self.chain_depth, 5)
        op_names = [
            "mlp", "matrix_mlp", "bilinear", "time", "freq", "delta", "short_onset", "offset",
            "global_summary", "memory_read", "memory_write", "ema_memory", "class_pair_memory",
            "residual_refdelta", "block_compare", "class_pair_contrast", "late_read_repair", "suppress", "normalize", "energy_count",
        ]
        operator_gates_by_stage = []
        for i in range(op_gates.shape[0]):
            vals, idxs = torch.topk(op_gates[i], k=min(5, op_gates.shape[1]))
            item = self.block_label(i)
            item["ops"] = [{"op": op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}", "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
            operator_gates_by_stage.append(item)

        class_reads = []
        for ci, cname in enumerate(classes):
            vals, idxs = torch.topk(cls_attn[ci], k=min(6, cls_attn.shape[1]))
            class_reads.append({
                "class": cname,
                "top_blocks": [dict(self.read_label(int(j)), weight=float(v)) for v, j in zip(vals.tolist(), idxs.tolist())],
            })

        def evidence_name(cell: int) -> str:
            # v11 evidence: 6x6 time-frequency grid (0..35) + acoustic shape cells.
            if cell < 36:
                return f"tf6_t{cell // 6}_f{cell % 6}"
            extra = {
                36: "global_mean",
                37: "global_std",
                38: "global_max_peak",
                39: "early_time",
                40: "mid_time",
                41: "late_time",
                42: "onset_mid_minus_early",
                43: "offset_late_minus_mid",
                44: "low_freq_band",
                45: "high_freq_band",
                46: "high_minus_low_freq",
            }
            if cell in extra:
                return extra[cell]
            rep = cell % 47
            return f"repeat_of:{evidence_name(rep)}"

        def task_name(cell: int) -> str:
            if cell == 0:
                return "global_loss_error"
            if cell == 1:
                return "global_true_pred_distribution"
            cnum = self.num_classes
            if 2 <= cell < 2 + cnum:
                return f"class_pressure:{classes[cell - 2] if cell - 2 < len(classes) else cell - 2}"
            if 2 + cnum <= cell < 2 + 2 * cnum:
                return f"confusion_row:{classes[cell - 2 - cnum] if cell - 2 - cnum < len(classes) else cell - 2 - cnum}"
            if 2 + 2 * cnum <= cell:
                return f"top_confusion_pair_cell_{cell - (2 + 2 * cnum)}"
            return f"task_{cell}"

        def role_from_ops(ops: List[Dict[str, float]]) -> str:
            if not ops:
                return "unknown"
            top = ops[0]["op"]
            op_set = {o["op"] for o in ops[:3]}
            if "global_summary" in op_set or "energy_count" in op_set:
                return "aggregate/global_summary"
            if "memory_write" in op_set or "memory_read" in op_set:
                return "memory/register"
            if "block_compare" in op_set or "class_pair_contrast" in op_set:
                return "compare/contrast"
            if "suppress" in op_set:
                return "suppress/common_mode_removal"
            if "residual_refdelta" in op_set or "normalize" in op_set:
                return "repair/normalize"
            if top in ("time", "freq", "delta", "short_onset", "offset"):
                return "extract/temporal_spectral"
            if top == "bilinear":
                return "bilinear_interaction"
            if top == "mlp":
                return "fallback_mlp"
            return "mixed"

        category_names = ["extract", "compare", "memory", "repair", "aggregate", "suppress", "transform"]
        cat_gates = self.category_gates / c if self.category_gates is not None else torch.zeros(self.chain_depth, len(category_names))
        category_gates_by_stage = []
        for i in range(cat_gates.shape[0]):
            vals, idxs = torch.topk(cat_gates[i], k=min(4, cat_gates.shape[1]))
            item = self.block_label(i)
            item["categories"] = [{"category": category_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
            category_gates_by_stage.append(item)

        operator_program_by_block = []
        for i in range(op_gates.shape[0]):
            vals, idxs = torch.topk(op_gates[i], k=min(6, op_gates.shape[1]))
            ops = [{"op": op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}", "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
            ev_vals, ev_idxs = torch.topk(ev_attn[i], k=min(5, ev_attn.shape[1]))
            task_vals, task_idxs = torch.topk(task_attn[i], k=min(5, task_attn.shape[1]))

            # Which classes read this block summary? class read index i+1 maps to block i.
            if cls_attn.shape[1] > i + 1:
                class_weights = cls_attn[:, i + 1]
                cv, ci = torch.topk(class_weights, k=min(4, class_weights.shape[0]))
                top_classes = [{"class": classes[int(j)], "weight": float(v)} for v, j in zip(cv.tolist(), ci.tolist())]
            else:
                top_classes = []

            item = self.block_label(i)
            item.update({
                "role": role_from_ops(ops),
                "operator_program": ops,
                "category_program": category_gates_by_stage[i]["categories"] if i < len(category_gates_by_stage) else [],
                "evidence_patterns": [{"cell": int(j), "name": evidence_name(int(j)), "weight": float(v)} for v, j in zip(ev_vals.tolist(), ev_idxs.tolist())],
                "attention_channel_patterns": {name: top_attn_items(attn, i, 4) for name, attn in channel_attn.items()},
                "task_patterns": [{"cell": int(j), "name": task_name(int(j)), "weight": float(v)} for v, j in zip(task_vals.tolist(), task_idxs.tolist())],
                "top_classes_using_block": top_classes,
                "effective_gate": float((self.effective_stage_gate / c)[i]) if self.effective_stage_gate is not None and i < self.effective_stage_gate.shape[0] else None,
            })
            operator_program_by_block.append(item)

        program_summary_by_layer = []
        for layer in range(self.num_layers):
            blocks = [b for b in operator_program_by_block if b["layer"] == layer]
            role_counts = {}
            for b in blocks:
                role_counts[b["role"]] = role_counts.get(b["role"], 0) + 1
            program_summary_by_layer.append({
                "layer": layer,
                "blocks": [{"name": b["name"], "role": b["role"], "top_op": b["operator_program"][0]["op"] if b["operator_program"] else "none"} for b in blocks],
                "role_counts": role_counts,
            })

        op_mean_named = []
        if op_gates.numel():
            for name, val in zip(op_names, op_gates.mean(dim=0).tolist()):
                op_mean_named.append({"op": name, "weight": float(val)})

        category_names = ["extract", "compare", "memory", "repair", "aggregate", "suppress", "transform"]
        cat_gates = self.category_gates / c if self.category_gates is not None else torch.zeros(self.chain_depth, len(category_names))
        category_gate_mean_named = []
        if cat_gates.numel():
            for name, val in zip(category_names, cat_gates.mean(dim=0).tolist()):
                category_gate_mean_named.append({"category": name, "weight": float(val)})
        category_gates_by_stage = []
        for i in range(cat_gates.shape[0]):
            vals, idxs = torch.topk(cat_gates[i], k=min(4, cat_gates.shape[1]))
            item = self.block_label(i)
            item["categories"] = [{"category": category_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
            category_gates_by_stage.append(item)

        rr = F.normalize(cls_attn.float(), dim=-1)
        sim = rr @ rr.T
        eye = torch.eye(self.num_classes, dtype=torch.bool)
        offdiag_cos = float(sim.masked_select(~eye).mean()) if self.num_classes > 1 else 0.0

        matrix_decomposition = []
        rank_count = max(1, self.layer_rank_count)
        stage_norm_avg = self.stage_state_norm / c if self.stage_state_norm is not None else torch.zeros(self.chain_depth + 1)
        offset = 0
        for layer, active_blocks in enumerate(self.topology_plan):
            blocks = []
            for local_i, block in enumerate(active_blocks):
                flat = offset + local_i
                read_idx = flat + 1
                blocks.append({
                    "name": f"L{layer}.B{block}",
                    "stage": flat,
                    "state_norm": float(stage_norm_avg[read_idx]) if read_idx < len(stage_norm_avg) else 0.0,
                    "attention_concentration_proxy": self.attn_corr_gap_sum[flat] / max(1, self.attn_corr_gap_count) if flat < len(self.attn_corr_gap_sum) else 0.0,
                })
            offset += len(active_blocks)
            matrix_decomposition.append({
                "layer": layer,
                "effective_rank": self.layer_rank_sum[layer] / rank_count,
                "block_cosine_collapse": self.layer_cos_sum[layer] / rank_count,
                "blocks": blocks,
            })

        update_norm_avg = self.update_norm_by_block / c if self.update_norm_by_block is not None else torch.zeros(self.chain_depth)
        write_delta_norm_avg = self.write_delta_norm_by_block / c if self.write_delta_norm_by_block is not None else torch.zeros(self.chain_depth)

        gate_write_by_block = []
        for i in range(self.chain_depth):
            item = self.block_label(i)
            item.update({
                "effective_gate": float((self.effective_stage_gate / c)[i]) if self.effective_stage_gate is not None and i < len(self.effective_stage_gate) else 0.0,
                "dynamic_gate": float((self.dynamic_stage_gate / c)[i]) if self.dynamic_stage_gate is not None and i < len(self.dynamic_stage_gate) else 0.0,
                "update_norm": float(update_norm_avg[i]) if i < len(update_norm_avg) else 0.0,
                "write_delta_norm": float(write_delta_norm_avg[i]) if i < len(write_delta_norm_avg) else 0.0,
            })
            gate_write_by_block.append(item)

        phase_role_diagnostics = []
        if op_gates.numel():
            offset = 0
            for layer, active_blocks in enumerate(self.topology_plan):
                n = len(active_blocks)
                usage = op_gates[offset:offset+n, :].mean(dim=0) if n > 0 and offset + n <= op_gates.shape[0] else torch.zeros(op_gates.shape[1])
                usage = usage / usage.sum().clamp_min(1e-8)
                if layer == 0:
                    expected = {"time", "freq", "delta", "short_onset", "bilinear"}
                elif layer == 1:
                    expected = {"block_compare", "class_pair_contrast", "residual_refdelta", "bilinear", "normalize"}
                elif layer == 2:
                    expected = {"memory_read", "memory_write", "suppress", "normalize", "residual_refdelta"}
                else:
                    expected = {"global_summary", "energy_count", "memory_write", "suppress", "block_compare"}
                match = sum(float(usage[j]) for j, name in enumerate(op_names) if name in expected)
                vals, idxs = torch.topk(usage, k=min(5, usage.shape[0]))
                top = [{"op": op_names[int(j)] if int(j) < len(op_names) else f"op_{int(j)}", "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
                phase_role_diagnostics.append({"layer": layer, "expected_ops": sorted(expected), "match_mass": match, "top_ops": top})
                offset += n

        role_names = ["extract", "contrast", "transform", "memory", "repair", "aggregate", "suppress"]
        role_mix_avg = self.role_mix_by_layer / c if self.role_mix_by_layer is not None else torch.zeros(self.num_layers, len(role_names))
        role_mix_by_layer = []
        for layer in range(role_mix_avg.shape[0]):
            vals, idxs = torch.topk(role_mix_avg[layer], k=min(4, role_mix_avg.shape[1]))
            role_mix_by_layer.append({
                "layer": layer,
                "top_roles": [{"role": role_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())],
                "all_roles": {role_names[i]: float(role_mix_avg[layer, i]) for i in range(len(role_names))},
            })

        category_mix_avg = self.category_mix_by_layer / c if self.category_mix_by_layer is not None else torch.zeros(self.num_layers, len(category_names))
        category_mix_by_layer = []
        for layer in range(category_mix_avg.shape[0]):
            vals, idxs = torch.topk(category_mix_avg[layer], k=min(4, category_mix_avg.shape[1]))
            category_mix_by_layer.append({
                "layer": layer,
                "top_categories": [{"category": category_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())],
                "all_categories": {category_names[i]: float(category_mix_avg[layer, i]) for i in range(len(category_names))},
            })

        signal_summary_avg = (self.signal_summary / c).tolist() if self.signal_summary is not None else []
        signal_bus_summary = {
            "signal_ctx_norm": float(signal_summary_avg[0]) if len(signal_summary_avg) > 0 else 0.0,
            "role_bias_norm": float(signal_summary_avg[1]) if len(signal_summary_avg) > 1 else 0.0,
            "category_bias_norm": float(signal_summary_avg[2]) if len(signal_summary_avg) > 2 else 0.0,
            "op_bias_norm": float(signal_summary_avg[3]) if len(signal_summary_avg) > 3 else 0.0,
            "gate_bias_abs": float(signal_summary_avg[4]) if len(signal_summary_avg) > 4 else 0.0,
            "signal_op_bias_norm": self.signal_op_bias_norm / c,
        }

        return {
            "count": self.count,
            "dynamic_stage_gate": (self.dynamic_stage_gate / c).tolist() if self.dynamic_stage_gate is not None else [],
            "effective_stage_gate": (self.effective_stage_gate / c).tolist() if self.effective_stage_gate is not None else [],
            "static_stage_gate": self.static_stage_gate.tolist() if self.static_stage_gate is not None else [],
            "stage_state_norm": (self.stage_state_norm / c).tolist() if self.stage_state_norm is not None else [],
            "top_evidence_by_stage": top_evidence_by_stage,
            "attention_channels_by_stage": attention_channels_by_stage,
            "top_task_by_stage": top_task_by_stage,
            "operator_gates_by_stage": operator_gates_by_stage,
            "operator_gate_mean": (op_gates.mean(dim=0).tolist() if op_gates.numel() else []),
            "operator_gate_mean_named": op_mean_named,
            "category_gate_mean_named": category_gate_mean_named,
            "category_gates_by_stage": category_gates_by_stage,
            "operator_program_by_block": operator_program_by_block,
            "program_summary_by_layer": program_summary_by_layer,
            "route_usage_by_block": route_usage_by_block,
            "matrix_decomposition": matrix_decomposition,
            "gate_write_by_block": gate_write_by_block,
            "phase_role_diagnostics": phase_role_diagnostics,
            "role_mix_by_layer": role_mix_by_layer,
            "category_mix_by_layer": category_mix_by_layer,
            "signal_bus_summary": signal_bus_summary,
            "evidence_cell_legend": {str(i): evidence_name(i) for i in range(self.evidence_cells)},
            "task_cell_legend": {str(i): task_name(i) for i in range(task_attn.shape[1])},
            "task_matrix_norm": (self.task_matrix_norm / c).tolist() if self.task_matrix_norm is not None else [],
            "global_matrix_norm": (self.global_matrix_norm / c).tolist() if self.global_matrix_norm is not None else [],
            "memory_matrix_norm": (self.memory_matrix_norm / c).tolist() if self.memory_matrix_norm is not None else [],
            "class_reads": class_reads,
            "class_read_offdiag_cos": offdiag_cos,
            "logits_cell_norm_mean": self.logits_cell_norm / c,
            "read_entropy_loss": self.read_entropy / c,
            "read_diversity_loss": self.read_diversity / c,
            "stage_load_balance_loss": self.stage_load_balance / c,
            "evidence_source_balance_loss": self.evidence_source_balance / c,
            "operator_balance_loss": self.operator_balance / c,
            "register_diversity_loss": self.register_diversity / c,
            "global_diversity_loss": self.global_diversity / c,
            "memory_diversity_loss": self.memory_diversity / c,
            "operator_entropy_per_block_loss": self.operator_entropy_per_block / c,
            "block_diversity_loss": self.block_diversity / c,
            "route_diversity_loss": self.route_diversity / c,
            "adapter_diversity_loss": self.adapter_diversity / c,
            "phase_role_loss": self.phase_role / c,
            "global_skip_penalty_loss": self.global_skip_penalty / c,
            "late_write_loss": self.late_write / c,
            "late_dyn_gate_loss": self.late_dyn_gate / c,
            "weak_class_late_read_loss": self.weak_class_late_read / c,
            "role_diversity_loss": self.role_diversity / c,
            "role_anchor_loss": self.role_anchor / c,
            "category_diversity_loss": self.category_diversity / c,
            "category_entropy_loss": self.category_entropy / c,
            "signal_op_bias_norm": self.signal_op_bias_norm / c,
            "pressure_features": self.pressure_features.tolist() if self.pressure_features is not None else [],
        }


@torch.no_grad()
def evaluate(model, loader, device: str, amp_dtype: torch.dtype, classes: List[str], args):
    model.eval()
    use_amp = device.startswith("cuda") and amp_dtype != torch.float32
    total_loss = 0.0
    total_correct = 0
    total = 0
    C = len(classes)
    conf = torch.zeros(C, C, dtype=torch.long)
    # In validation we force decomposition on sampled/early batches; val has too few batches for decomp_every=25.
    acc = SeqAccumulator(num_classes=C, evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=1, topology_plan=model.core.topology_plan)

    for bi, (wav, y) in enumerate(loader, start=1):
        if args.max_val_batches and bi > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        t_fwd0 = time.perf_counter()
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(wav, return_aux=True)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(dim=-1)
        total_loss += float(loss.detach().cpu()) * y.numel()
        total_correct += int((pred == y).sum().detach().cpu())
        total += y.numel()
        idx = y.detach().cpu() * C + pred.detach().cpu()
        conf += torch.bincount(idx, minlength=C * C).view(C, C)
        acc.add(aux["seq"])

    if prof is not None:
        prof.__exit__(None, None, None)
        try:
            profile_dir = ensure_dir(Path(args.out_dir) / "profiles")
            sort_key = "cuda_time_total" if device.startswith("cuda") else "cpu_time_total"
            table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
            profile_path = profile_dir / f"profile_epoch_{int(epoch):03d}.txt"
            profile_path.write_text(table, encoding="utf-8")
            print(f"torch_profiler: wrote {profile_path}", flush=True)
        except Exception as e:
            print(f"torch_profiler: failed to export profile table: {e}", flush=True)

    return {
        "loss": total_loss / max(1, total),
        "acc": total_correct / max(1, total),
        "n": total,
        "confusion": conf,
        "seq": acc.summary(classes),
    }


def train_one_epoch(model, loader, optimizer, scaler, device: str, amp_dtype: torch.dtype, epoch: int, args, classes: List[str]):
    model.train()
    use_amp = device.startswith("cuda") and amp_dtype != torch.float32

    prof = None
    if getattr(args, "torch_profiler", False) and int(getattr(args, "profile_epoch", 1)) == int(epoch):
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        prof = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        prof.__enter__()
    total_loss = 0.0
    total_ce = 0.0
    total_reg = 0.0
    total_correct = 0
    total = 0
    t0 = time.time()
    last_end = time.perf_counter()
    speed_data = 0.0
    speed_fwd = 0.0
    speed_bwd = 0.0
    speed_opt = 0.0
    speed_count = 0
    acc = SeqAccumulator(num_classes=len(classes), evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=args.decomp_every, topology_plan=model.core.topology_plan)

    for step, (wav, y) in enumerate(loader, start=1):
        if args.max_train_batches and step > args.max_train_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if getattr(args, "speed_sync", False) and device.startswith("cuda"):
            torch.cuda.synchronize()
        t_data_done = time.perf_counter()
        speed_data += t_data_done - last_end

        optimizer.zero_grad(set_to_none=True)
        t_fwd0 = time.perf_counter()
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype, enabled=use_amp):
            logits, aux = model(wav, return_aux=True)
            ce = F.cross_entropy(logits.float(), y)
            reg = model.regularization(args)
            read_entropy = aux["seq"].read_entropy_loss
            read_div = aux["seq"].read_diversity_loss
            stage_lb = aux["seq"].stage_load_balance_loss
            ev_lb = aux["seq"].evidence_source_balance_loss
            op_lb = aux["seq"].operator_balance_loss
            reg_div = aux["seq"].register_diversity_loss
            glob_div = aux["seq"].global_diversity_loss
            mem_div = aux["seq"].memory_diversity_loss
            op_entropy_block = aux["seq"].operator_entropy_per_block_loss
            block_div = aux["seq"].block_diversity_loss
            route_div = aux["seq"].route_diversity_loss
            adapter_div = aux["seq"].adapter_diversity_loss
            phase_role = aux["seq"].phase_role_loss
            global_skip_penalty = aux["seq"].global_skip_penalty_loss
            late_write = aux["seq"].late_write_loss
            late_dyn_gate = aux["seq"].late_dyn_gate_loss
            weak_class_late_read = aux["seq"].weak_class_late_read_loss
            role_diversity = aux["seq"].role_diversity_loss
            role_anchor = aux["seq"].role_anchor_loss
            category_diversity = aux["seq"].category_diversity_loss
            category_entropy = aux["seq"].category_entropy_loss
            # vNext schedule: early soft exploration, late sharper readable primitive program.
            # This preserves gradient from all candidates early, then stops forcing every block to mix everything.
            denom_epochs = max(1, int(getattr(args, "epochs", 1)) - 1)
            train_progress = min(1.0, max(0.0, (float(epoch) - 1.0) / float(denom_epochs)))
            explore_w = max(0.0, 1.0 - train_progress / 0.45)
            sharpen_w = min(1.0, max(0.0, (train_progress - 0.25) / 0.55))
            lambda_operator_balance_eff = args.lambda_operator_balance * explore_w
            lambda_operator_entropy_per_block_eff = args.lambda_operator_entropy_per_block * max(0.05, explore_w)
            lambda_category_entropy_eff = args.lambda_category_entropy * sharpen_w
            loss = (
                ce + reg
                + args.lambda_read_entropy * read_entropy
                + args.lambda_read_diversity * read_div
                + args.lambda_stage_load_balance * stage_lb
                + args.lambda_evidence_source_balance * ev_lb
                + lambda_operator_balance_eff * op_lb
                # Deprecated aggregate register loss; default is 0.0 because
                # global/memory diversity are controlled separately below.
                + args.lambda_register_diversity * reg_div
                + args.lambda_global_diversity * glob_div
                + args.lambda_memory_diversity * mem_div
                + lambda_operator_entropy_per_block_eff * op_entropy_block
                + args.lambda_block_diversity * block_div
                + args.lambda_route_diversity * route_div
                + args.lambda_adapter_diversity * adapter_div
                + args.lambda_phase_role * phase_role
                + args.lambda_global_skip_penalty * global_skip_penalty
                + args.lambda_late_write * late_write
                + args.lambda_late_dyn_gate * late_dyn_gate
                + args.lambda_weak_class_late_read * weak_class_late_read
                + args.lambda_role_diversity * role_diversity
                + args.lambda_role_anchor * role_anchor
                + args.lambda_category_diversity * category_diversity
                + lambda_category_entropy_eff * category_entropy
            )

        if getattr(args, "speed_sync", False) and device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()
        speed_fwd += t_fwd1 - t_fwd0

        if not torch.isfinite(loss):
            print("NONFINITE_LOSS: skip batch", flush=True)
            last_end = time.perf_counter()
            continue

        acc.add(aux["seq"])

        t_bwd0 = time.perf_counter()
        scaler.scale(loss).backward()
        if getattr(args, "speed_sync", False) and device.startswith("cuda"):
            torch.cuda.synchronize()
        t_bwd1 = time.perf_counter()
        speed_bwd += t_bwd1 - t_bwd0
        did_step = True
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(norm):
                print("NONFINITE_GRAD: skip optimizer step", flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                did_step = False
        t_opt0 = time.perf_counter()
        if did_step:
            scaler.step(optimizer)
            scaler.update()
            model.update_train_pressure(logits.detach(), y.detach())
        if getattr(args, "speed_sync", False) and device.startswith("cuda"):
            torch.cuda.synchronize()
        t_opt1 = time.perf_counter()
        speed_opt += t_opt1 - t_opt0
        speed_count += 1
        if prof is not None:
            prof.step()
            if int(getattr(args, "profile_steps", 0)) > 0 and step >= int(getattr(args, "profile_steps", 0)):
                print(f"torch_profiler: reached profile_steps={int(getattr(args, 'profile_steps', 0))}; stopping epoch early", flush=True)
                last_end = time.perf_counter()
                break
        last_end = time.perf_counter()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            total_loss += float(loss.detach().cpu()) * y.numel()
            total_ce += float(ce.detach().cpu()) * y.numel()
            total_reg += float(reg.detach().cpu()) * y.numel()
            total_correct += int((pred == y).sum().detach().cpu())
            total += y.numel()

        if args.log_every and step % args.log_every == 0:
            speed_msg = ""
            if getattr(args, "speed_log", False) and speed_count > 0:
                speed_msg = (
                    f" data={1000*speed_data/speed_count:.1f}ms"
                    f" fwd={1000*speed_fwd/speed_count:.1f}ms"
                    f" bwd={1000*speed_bwd/speed_count:.1f}ms"
                    f" opt={1000*speed_opt/speed_count:.1f}ms"
                )
            print(
                f"epoch {epoch:03d} step {step:05d} "
                f"loss={total_loss/max(1,total):.4f} ce={total_ce/max(1,total):.4f} "
                f"acc={100*total_correct/max(1,total):.2f}% seen={total} "
                f"t={time.time()-t0:.1f}s{speed_msg}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, total),
        "ce": total_ce / max(1, total),
        "reg": total_reg / max(1, total),
        "acc": total_correct / max(1, total),
        "n": total,
        "seq": acc.summary(classes),
        "speed": {
            "data_ms": 1000 * speed_data / max(1, speed_count),
            "forward_ms": 1000 * speed_fwd / max(1, speed_count),
            "backward_ms": 1000 * speed_bwd / max(1, speed_count),
            "optimizer_ms": 1000 * speed_opt / max(1, speed_count),
        },
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_acc: float, classes: List[str], args) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "classes": classes,
        "args": vars(args),
    }, path)



def compute_epoch_lr(base_lr: float, epoch: int, total_epochs: int, warmup_epochs: int, min_lr_ratio: float, mode: str) -> float:
    if mode == "none":
        return float(base_lr)
    warm = max(0, int(warmup_epochs))
    if warm > 0 and epoch <= warm:
        return float(base_lr) * float(epoch) / float(warm)
    if mode == "cosine":
        denom = max(1, int(total_epochs) - warm)
        progress = min(1.0, max(0.0, float(epoch - warm) / float(denom)))
        return float(base_lr) * (float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress)))
    return float(base_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def compute_late_gate_floor(args, epoch: int) -> float:
    start = float(args.late_gate_floor)
    end = float(args.late_gate_floor_end)
    decay_start = int(args.late_gate_floor_decay_start)
    decay_epochs = max(1, int(args.late_gate_floor_decay_epochs))
    if epoch <= decay_start:
        return start
    t = min(1.0, max(0.0, float(epoch - decay_start) / float(decay_epochs)))
    return start + (end - start) * t



def _iter_arch_memory_records(path: str):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        print(f"[arch-memory] not found: {path}", flush=True)
        return []
    records = []
    try:
        if p.suffix.lower() == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        else:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                records.extend(obj)
            else:
                records.append(obj)
    except Exception as exc:
        print(f"[arch-memory] failed to read {path}: {exc}", flush=True)
    return records


def apply_architecture_memory_prior(model: nn.Module, path: str, strength: float) -> None:
    records = _iter_arch_memory_records(path)
    if not records:
        return
    core = model.core
    op_names = list(getattr(core, "OP_NAMES", []))
    role_names = list(getattr(core, "ROLE_NAMES", []))
    added = 0
    with torch.no_grad():
        for rec in records[-32:]:
            seq = rec.get("seq_space", rec.get("seq", rec))
            weight = float(rec.get("best_val_acc", rec.get("val_acc", rec.get("score", 0.5))))
            weight = max(0.1, min(1.0, weight)) * float(strength)
            # phase_role_diagnostics from previous reports
            for item in seq.get("phase_role_diagnostics", []):
                layer = int(item.get("layer", -1))
                if 0 <= layer < core.num_layers:
                    for top in item.get("top_ops", [])[:5]:
                        op = top.get("op")
                        if op in op_names:
                            core.layer_op_prior[layer, op_names.index(op)] += weight * float(top.get("weight", 0.1))
                            added += 1
            # role_mix_by_layer from v12/v13 reports
            for item in seq.get("role_mix_by_layer", []):
                layer = int(item.get("layer", -1))
                if 0 <= layer < core.num_layers and hasattr(core, "role_logits"):
                    for rname, val in item.get("all_roles", {}).items():
                        if rname in role_names:
                            core.role_logits[layer, role_names.index(rname)] += weight * float(val)
                            added += 1
            # category_mix_by_layer from v13 reports
            cat_names = list(getattr(core, "CATEGORY_NAMES", []))
            for item in seq.get("category_mix_by_layer", []):
                layer = int(item.get("layer", -1))
                if 0 <= layer < core.num_layers and hasattr(core, "layer_category_prior"):
                    for cname, val in item.get("all_categories", {}).items():
                        if cname in cat_names:
                            core.layer_category_prior[layer, cat_names.index(cname)] += weight * float(val)
                            added += 1
    print(f"[arch-memory] applied {added} soft priors from {path}", flush=True)


def compact_architecture_record(analysis: Dict[str, Any], best_acc: float, best_epoch: int) -> Dict[str, Any]:
    seq = analysis.get("seq_space", {})
    return {
        "run_epoch": analysis.get("epoch"),
        "best_val_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "val_acc": float(analysis.get("val", {}).get("acc", 0.0)),
        "classes": analysis.get("classes", []),
        "class_accuracy": analysis.get("class_accuracy", {}),
        "top_confusions": analysis.get("top_confusions", [])[:12],
        "seq_space": {
            "program_summary_by_layer": seq.get("program_summary_by_layer", []),
            "phase_role_diagnostics": seq.get("phase_role_diagnostics", []),
            "role_mix_by_layer": seq.get("role_mix_by_layer", []),
            "category_mix_by_layer": seq.get("category_mix_by_layer", []),
            "category_gate_mean_named": seq.get("category_gate_mean_named", []),
            "operator_gate_mean_named": seq.get("operator_gate_mean_named", {}),
            "matrix_decomposition": seq.get("matrix_decomposition", []),
            "gate_write_by_block": seq.get("gate_write_by_block", [])[:32],
            "signal_bus_summary": seq.get("signal_bus_summary", {}),
        },
        "notes": analysis.get("notes", {}),
    }


def run(args):
    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_dir = ensure_dir(args.out_dir)
    amp_dtype = get_amp_dtype(args.amp)

    train_loader, val_loader, classes = make_loaders(args)
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)}", flush=True)

    if args.topology == "two_squares_merge":
        args.num_layers = 3
        args.blocks_per_layer = 4
        args.chain_depth = 9
    elif args.topology == "two_by_two_by_two_merge":
        args.num_layers = 4
        args.blocks_per_layer = 2
        args.chain_depth = 7
    elif args.topology == "grid_3x3x3x3":
        args.num_layers = 4
        args.blocks_per_layer = 3
        args.chain_depth = 12
    else:
        args.chain_depth = int(args.num_layers) * int(args.blocks_per_layer)

    model = SequentialMatrixCellsModel(
        num_classes=len(classes),
        dim=args.dim,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        hop_length=args.hop_length,
        evidence_cells=args.evidence_cells,
        chain_depth=args.chain_depth,
        num_layers=args.num_layers,
        blocks_per_layer=args.blocks_per_layer,
        block_cells=args.block_cells,
        global_cells=args.global_cells,
        memory_cells=args.memory_cells,
        dropout=args.dropout,
        pressure_alpha=args.pressure_alpha_start,
        evidence_dropout=args.evidence_dropout,
        class_stage_bias_std=args.class_stage_bias_std,
        stage_gate_bias_init=args.stage_gate_bias_init,
        gate_floor=args.gate_floor,
        late_gate_floor=args.late_gate_floor,
        late_write_target=args.late_write_target,
        late_dyn_gate_target=args.late_dyn_gate_target,
        weak_class_late_read_target=args.weak_class_late_read_target,
        role_signal_scale=args.role_signal_scale,
        signal_op_bias_scale=args.signal_op_bias_scale,
        category_temp=args.category_temp,
        primitive_temp=args.primitive_temp,
        lock_last_role=args.lock_last_role,
        task_attn_temp=args.task_attn_temp,
        evidence_attn_temp=args.evidence_attn_temp,
        operator_temp=args.operator_temp,
        topology=args.topology,
        skip_connections=(args.skip_mode != "off"),
    ).to(device)

    if args.architecture_memory:
        apply_architecture_memory_prior(model, args.architecture_memory, args.architecture_memory_strength)

    # Separate head decay.
    head_keys = ("class_write", "logit_bias", "class_stage_bias")
    head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in head_keys):
            head_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": args.weight_decay},
            {"params": head_params, "weight_decay": args.head_weight_decay},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda") and amp_dtype == torch.float16))

    n_params = sum(p.numel() for p in model.parameters())
    print("classes:", classes, flush=True)
    print("params:", n_params, flush=True)
    print(
        f"matrix layers/blocks/registers: topology={args.topology} skip_mode={args.skip_mode} evidence={args.evidence_cells} task_matrix={getattr(model.core, 'task_cells', 2 + 2 * len(classes))} layers={args.num_layers} blocks_per_layer={args.blocks_per_layer} block_cells={args.block_cells} global_cells={args.global_cells} memory_cells={args.memory_cells} total_blocks={args.chain_depth}",
        flush=True,
    )
    print("device:", device, "amp:", args.amp, flush=True)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_ce", "train_reg", "train_acc",
            "val_loss", "val_acc", "best_acc", "class_cos",
            "logits_cell_norm", "stage_lb", "evidence_lb", "operator_lb", "register_diversity", "global_diversity", "memory_diversity", "operator_entropy_per_block",
            "block_diversity", "route_diversity", "adapter_diversity", "phase_role", "global_skip_penalty", "late_write", "late_dyn_gate", "weak_class_late_read",
            "role_diversity", "role_anchor", "category_diversity", "category_entropy", "signal_op_bias_norm",
            "lr", "current_late_gate_floor",
            "speed_data_ms", "speed_forward_ms", "speed_backward_ms", "speed_optimizer_ms",
        ])
        w.writeheader()

    best_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.pressure_alpha_fast_epochs:
            model.core.pressure_alpha = args.pressure_alpha_start
        else:
            model.core.pressure_alpha = args.pressure_alpha_end

        current_lr = compute_epoch_lr(args.lr, epoch, args.epochs, args.warmup_epochs, args.min_lr_ratio, args.lr_scheduler)
        set_optimizer_lr(optimizer, current_lr)

        current_late_floor = compute_late_gate_floor(args, epoch)
        model.core.late_gate_floor = current_late_floor

        train = train_one_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, epoch, args, classes)
        train["lr"] = current_lr
        train["late_gate_floor"] = current_late_floor
        val = evaluate(model, val_loader, device, amp_dtype, classes, args)

        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, best_acc, classes, args)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, best_acc, classes, args)

        conf = val["confusion"]
        row_sums = conf.sum(dim=1).clamp_min(1)
        class_acc = (conf.diag().float() / row_sums.float()).tolist()
        seq = val["seq"]

        analysis = {
            "epoch": epoch,
            "train": train,
            "val": {"loss": val["loss"], "acc": val["acc"], "n": val["n"]},
            "classes": classes,
            "class_accuracy": {classes[i]: class_acc[i] for i in range(len(classes))},
            "top_confusions": top_confusions(conf, classes),
            "seq_space": seq,
            "notes": {
                "architecture": "sequential_matrix_cells_v13_categorized_signal_bus_builder",
                "no_all_to_all": True,
                "uses_matrix_blocks": True,
                "no_old_blockless_water": True,
                "causal_chain": True,
                "pressure_source": "train batches only; val is report only",
            },
        }
        write_json(out_dir / f"seq_analysis_epoch_{epoch:03d}.json", analysis)
        if args.export_architecture_memory and epoch == best_epoch:
            mem_record = compact_architecture_record(analysis, best_acc, best_epoch)
            with (out_dir / "architecture_memory.jsonl").open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(mem_record, ensure_ascii=False) + "\n")

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "epoch", "train_loss", "train_ce", "train_reg", "train_acc",
                "val_loss", "val_acc", "best_acc", "class_cos",
                "logits_cell_norm", "stage_lb", "evidence_lb", "operator_lb", "register_diversity", "global_diversity", "memory_diversity", "operator_entropy_per_block",
                "block_diversity", "route_diversity", "adapter_diversity", "phase_role", "global_skip_penalty", "late_write", "late_dyn_gate", "weak_class_late_read",
                "role_diversity", "role_anchor", "category_diversity", "category_entropy", "signal_op_bias_norm",
                "lr", "current_late_gate_floor",
                "speed_data_ms", "speed_forward_ms", "speed_backward_ms", "speed_optimizer_ms",
            ])
            w.writerow({
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_ce": train["ce"],
                "train_reg": train["reg"],
                "train_acc": train["acc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "best_acc": best_acc,
                "class_cos": seq.get("class_read_offdiag_cos", 0.0),
                "logits_cell_norm": seq.get("logits_cell_norm_mean", 0.0),
                "stage_lb": seq.get("stage_load_balance_loss", 0.0),
                "evidence_lb": seq.get("evidence_source_balance_loss", 0.0),
                "operator_lb": seq.get("operator_balance_loss", 0.0),
                "register_diversity": seq.get("register_diversity_loss", 0.0),
                "global_diversity": seq.get("global_diversity_loss", 0.0),
                "memory_diversity": seq.get("memory_diversity_loss", 0.0),
                "operator_entropy_per_block": seq.get("operator_entropy_per_block_loss", 0.0),
                "block_diversity": seq.get("block_diversity_loss", 0.0),
                "route_diversity": seq.get("route_diversity_loss", 0.0),
                "adapter_diversity": seq.get("adapter_diversity_loss", 0.0),
                "phase_role": seq.get("phase_role_loss", 0.0),
                "global_skip_penalty": seq.get("global_skip_penalty_loss", 0.0),
                "late_write": seq.get("late_write_loss", 0.0),
                "late_dyn_gate": seq.get("late_dyn_gate_loss", 0.0),
                "weak_class_late_read": seq.get("weak_class_late_read_loss", 0.0),
                "role_diversity": seq.get("role_diversity_loss", 0.0),
                "role_anchor": seq.get("role_anchor_loss", 0.0),
                "category_diversity": seq.get("category_diversity_loss", 0.0),
                "category_entropy": seq.get("category_entropy_loss", 0.0),
                "signal_op_bias_norm": seq.get("signal_op_bias_norm", 0.0),
                "lr": train.get("lr", args.lr),
                "current_late_gate_floor": train.get("late_gate_floor", args.late_gate_floor),
                "speed_data_ms": train.get("speed", {}).get("data_ms", 0.0),
                "speed_forward_ms": train.get("speed", {}).get("forward_ms", 0.0),
                "speed_backward_ms": train.get("speed", {}).get("backward_ms", 0.0),
                "speed_optimizer_ms": train.get("speed", {}).get("optimizer_ms", 0.0),
            })

        gate = seq.get("effective_stage_gate", [])
        dyn_gate = seq.get("dynamic_stage_gate", [])
        gate_msg = (
            f"eff_gate={sum(gate)/max(1,len(gate)):.3f} "
            f"dyn_gate={sum(dyn_gate)/max(1,len(dyn_gate)):.3f}"
        ) if gate else "eff_gate=0.000 dyn_gate=0.000"
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train={train['loss']:.4f}/{100*train['acc']:.2f}% "
            f"val={val['loss']:.4f}/{100*val['acc']:.2f}% "
            f"best={100*best_acc:.2f}%@{best_epoch} "
            f"class_cos={seq.get('class_read_offdiag_cos', 0):.3f} "
            f"{gate_msg} "
            f"logit_norm={seq.get('logits_cell_norm_mean', 0):.3f} "
            f"blkdiv={seq.get('block_diversity_loss', 0):.3f} "
            f"routediv={seq.get('route_diversity_loss', 0):.3f} "
            f"phase={seq.get('phase_role_loss', 0):.3f} "
            f"gskip={seq.get('global_skip_penalty_loss', 0):.3f} "
            f"latew={seq.get('late_write_loss', 0):.3f} "
            f"ldyn={seq.get('late_dyn_gate_loss', 0):.3f} "
            f"weaklate={seq.get('weak_class_late_read_loss', 0):.3f} "
            f"rolediv={seq.get('role_diversity_loss', 0):.3f} "
            f"catdiv={seq.get('category_diversity_loss', 0):.3f} "
            f"catent={seq.get('category_entropy_loss', 0):.3f} "
            f"sigop={seq.get('signal_op_bias_norm', 0):.2f} "
            f"lr={train.get('lr', args.lr):.2e} "
            f"lgfloor={train.get('late_gate_floor', args.late_gate_floor):.3f} "
            f"conf={top_confusions(conf, classes, topn=3)}",
            flush=True,
        )

    write_json(out_dir / "final_report.json", {
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "classes": classes,
        "args": vars(args),
        "architecture": "sequential_matrix_cells_v13_categorized_signal_bus_builder",
    })
    print("done. out:", out_dir, flush=True)


def build_argparser():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data-root", type=str, default="./data/speechcommands")
    p.add_argument("--download", action="store_true")
    p.add_argument("--classes", type=str, default="yes,no,up,down,left,right,on,off,stop,go")
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=160)

    # model
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--chain-depth", type=int, default=12)  # compatibility: num_layers * blocks_per_layer
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--blocks-per-layer", type=int, default=4)
    p.add_argument("--block-cells", type=int, default=2)
    p.add_argument("--global-cells", type=int, default=3)
    p.add_argument("--memory-cells", type=int, default=6)
    p.add_argument("--topology", type=str, default="grid", choices=["grid", "grid_3x3x3x3", "two_squares_merge", "two_by_two_by_two_merge"])
    p.add_argument("--skip-mode", type=str, default="on", choices=["on", "off"])
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--evidence-dropout", type=float, default=0.10)
    p.add_argument("--class-stage-bias-std", type=float, default=0.35)
    p.add_argument("--stage-gate-bias-init", type=float, default=1.0)
    p.add_argument("--gate-floor", type=float, default=0.15)
    p.add_argument("--late-gate-floor", type=float, default=0.25)
    p.add_argument("--late-write-target", type=float, default=0.35)
    p.add_argument("--late-dyn-gate-target", type=float, default=0.15)
    p.add_argument("--weak-class-late-read-target", type=float, default=0.35)
    p.add_argument("--role-signal-scale", type=float, default=0.35)
    p.add_argument("--signal-op-bias-scale", type=float, default=0.20)
    p.add_argument("--category-temp", type=float, default=1.0)
    p.add_argument("--primitive-temp", type=float, default=1.0)
    p.add_argument("--lock-last-role", action="store_true", default=True)
    p.add_argument("--no-lock-last-role", dest="lock_last_role", action="store_false")
    p.add_argument("--late-gate-floor-end", type=float, default=0.15)
    p.add_argument("--late-gate-floor-decay-start", type=int, default=5)
    p.add_argument("--late-gate-floor-decay-epochs", type=int, default=8)
    p.add_argument("--task-attn-temp", type=float, default=0.85)
    p.add_argument("--evidence-attn-temp", type=float, default=1.0)
    p.add_argument("--operator-temp", type=float, default=1.25)

    # optimization
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--lr-scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--min-lr-ratio", type=float, default=0.15)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--head-weight-decay", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=0.7)
    p.add_argument("--amp", type=str, default="fp16", choices=["fp16", "bf16", "fp32", "off"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # pressure schedule
    p.add_argument("--pressure-alpha-start", type=float, default=0.20)
    p.add_argument("--pressure-alpha-end", type=float, default=0.05)
    p.add_argument("--pressure-alpha-fast-epochs", type=int, default=3)

    # regularization
    p.add_argument("--lambda-stage", type=float, default=0.0)
    p.add_argument("--lambda-addr", type=float, default=1e-4)
    p.add_argument("--lambda-read-entropy", type=float, default=0.0)
    p.add_argument("--lambda-read-diversity", type=float, default=0.04)
    p.add_argument("--lambda-stage-load-balance", type=float, default=0.004)
    p.add_argument("--lambda-evidence-source-balance", type=float, default=0.02)
    p.add_argument("--lambda-operator-balance", type=float, default=0.02)
    p.add_argument("--lambda-register-diversity", type=float, default=0.0)  # deprecated: use split losses below
    p.add_argument("--lambda-global-diversity", type=float, default=0.008)
    p.add_argument("--lambda-memory-diversity", type=float, default=0.004)
    p.add_argument("--lambda-operator-entropy-per-block", type=float, default=0.003)
    p.add_argument("--lambda-block-diversity", type=float, default=0.020)
    p.add_argument("--lambda-route-diversity", type=float, default=0.010)
    p.add_argument("--lambda-adapter-diversity", type=float, default=0.008)
    p.add_argument("--lambda-phase-role", type=float, default=0.020)
    p.add_argument("--lambda-global-skip-penalty", type=float, default=0.004)
    p.add_argument("--lambda-late-write", type=float, default=0.015)
    p.add_argument("--lambda-late-dyn-gate", type=float, default=0.010)
    p.add_argument("--lambda-weak-class-late-read", type=float, default=0.020)
    p.add_argument("--lambda-role-diversity", type=float, default=0.006)
    p.add_argument("--lambda-role-anchor", type=float, default=0.010)
    p.add_argument("--lambda-category-diversity", type=float, default=0.006)
    p.add_argument("--lambda-category-entropy", type=float, default=0.002)

    # runtime
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--decomp-every", type=int, default=25, help="sample matrix decomposition every N batches; 0 disables rank/SVD analytics")
    p.add_argument("--torch-profiler", action="store_true", help="write PyTorch CPU/CUDA profiler table for one epoch")
    p.add_argument("--profile-epoch", type=int, default=1)
    p.add_argument("--profile-steps", type=int, default=20)
    p.add_argument("--speed-log", action="store_true", help="print averaged data/forward/backward/optimizer times")
    p.add_argument("--speed-sync", action="store_true", help="cuda synchronize timing; slower but accurate")
    p.add_argument("--architecture-memory", type=str, default="", help="optional json/jsonl memory from previous best architecture reports")
    p.add_argument("--architecture-memory-strength", type=float, default=0.08)
    p.add_argument("--export-architecture-memory", action="store_true", default=True)
    p.add_argument("--no-export-architecture-memory", dest="export_architecture_memory", action="store_false")
    p.add_argument("--out-dir", type=str, default="./runs/sequential_matrix_cells_v13_categorized_signal_bus_builder")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
