from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class StepUtilityConfig:
    dead_gate: float = 0.04
    dead_update: float = 0.02
    overloaded_gate: float = 0.82
    overloaded_entropy: float = 1.5
    high_update: float = 0.75
    cost_per_step: float = 1.0


def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def summarize_step_utility(aux: Dict[str, torch.Tensor], cfg: StepUtilityConfig | None = None) -> Dict[str, Any]:
    """Summarize utility for explicit step programs.

    Expected shapes from StepBuilderCore:
      write_gate  [B,L,N,S,1]
      op_weights  [B,L,N,S,O]
      read_weights[B,L,N,S,R]
      update_norm [B,L,N,S]

    Returns aggregate health plus per-step action suggestions. This is detached
    reporting, not a differentiable loss.
    """

    cfg = cfg or StepUtilityConfig()
    wg = aux["write_gate"].float().squeeze(-1)
    ow = aux["op_weights"].float()
    rw = aux["read_weights"].float()
    un = aux["update_norm"].float()
    op_ent = _entropy(ow, dim=-1)
    read_ent = _entropy(rw, dim=-1)
    # Utility proxy: useful work / cost, but penalize fully saturated gates.
    utility = wg * un * (0.5 + 0.5 * op_ent / max(1e-6, float(torch.log(torch.tensor(ow.shape[-1], device=ow.device)))))
    utility = utility / max(cfg.cost_per_step, 1e-6)

    B, L, N, S = wg.shape
    per: List[Dict[str, Any]] = []
    for li in range(L):
        for bi in range(N):
            for si in range(S):
                gate = float(wg[:, li, bi, si].mean().item())
                upd = float(un[:, li, bi, si].mean().item())
                oe = float(op_ent[:, li, bi, si].mean().item())
                re = float(read_ent[:, li, bi, si].mean().item())
                util = float(utility[:, li, bi, si].mean().item())
                if gate < cfg.dead_gate and upd < cfg.dead_update:
                    status = "dead/prune_candidate"
                    action = "soft_prune_or_replace"
                elif gate > cfg.overloaded_gate and oe > cfg.overloaded_entropy and upd > cfg.high_update:
                    status = "overloaded/add_step_candidate"
                    action = "add_step_or_split_block"
                elif gate > 0.9:
                    status = "gate_saturated"
                    action = "apply_gate_budget_or_freeze"
                elif util > 0.20:
                    status = "useful/skill_candidate"
                    action = "archive_if_stable"
                else:
                    status = "learning"
                    action = "keep_training"
                per.append({
                    "layer": li, "block": bi, "step": si,
                    "gate": gate, "update_norm": upd,
                    "op_entropy": oe, "read_entropy": re,
                    "utility": util, "status": status, "action": action,
                })
    return {
        "write_gate_mean": float(wg.mean().item()),
        "write_gate_max": float(wg.max().item()),
        "op_entropy_mean": float(op_ent.mean().item()),
        "read_entropy_mean": float(read_ent.mean().item()),
        "update_norm_mean": float(un.mean().item()),
        "utility_mean": float(utility.mean().item()),
        "dead_steps": sum(1 for x in per if x["status"].startswith("dead")),
        "saturated_steps": sum(1 for x in per if x["status"] == "gate_saturated"),
        "skill_candidates": sum(1 for x in per if x["status"].startswith("useful")),
        "overloaded_steps": sum(1 for x in per if x["status"].startswith("overloaded")),
        "per_step": per,
    }
