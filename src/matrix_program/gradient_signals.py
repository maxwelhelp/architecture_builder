from __future__ import annotations

from typing import Dict

import torch


def entropy_from_positive(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = x.clamp_min(0)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(eps)
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=dim)


def slot_gradient_health(slot_grad: torch.Tensor | None, eps: float = 1e-8) -> Dict[str, float]:
    """Summarize dLoss/dSlots for H[B,L,N,D].

    This is diagnostic/health information. It is intentionally detached and
    should be used as a next-step/controller/skill signal, not as a direct
    differentiable forward input in the same step.
    """

    if slot_grad is None:
        return {
            "grad_slot_norm_mean": 0.0,
            "grad_slot_norm_max": 0.0,
            "grad_layer_entropy": 0.0,
            "grad_block_entropy": 0.0,
            "grad_dead_slot_frac": 1.0,
        }
    g = slot_grad.detach().float()
    # [B,L,N]
    n = g.pow(2).sum(dim=-1).sqrt()
    layer_energy = n.mean(dim=(0, 2))  # [L]
    block_energy = n.mean(dim=(0, 1))  # [N]
    dead = (n < (n.mean() * 0.05 + eps)).float().mean()
    return {
        "grad_slot_norm_mean": float(n.mean().item()),
        "grad_slot_norm_max": float(n.max().item()),
        "grad_layer_entropy": float(entropy_from_positive(layer_energy, dim=0, eps=eps).item()),
        "grad_block_entropy": float(entropy_from_positive(block_energy, dim=0, eps=eps).item()),
        "grad_dead_slot_frac": float(dead.item()),
    }
