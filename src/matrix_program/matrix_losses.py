from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MatrixLossConfig:
    lambda_write_budget: float = 0.0
    write_target: float = 0.70
    lambda_op_entropy_floor: float = 0.0
    op_min_entropy: float = 0.35
    lambda_op_usage_balance: float = 0.0
    lambda_slot_diversity: float = 0.0


def _entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def slot_diversity_loss(h: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Discourage all slots from becoming the same vector.

    h: [B,L,N,D]. Computes squared off-diagonal cosine similarity per batch.
    """

    b, l, n, d = h.shape
    x = F.normalize(h.reshape(b, l * n, d).float(), dim=-1, eps=eps)
    sim = torch.matmul(x, x.transpose(-1, -2))
    eye = torch.eye(l * n, device=h.device, dtype=torch.bool)[None]
    off = sim.masked_select(~eye)
    return off.pow(2).mean().to(h.dtype)


def matrix_aux_losses(
    h: torch.Tensor,
    aux: Dict[str, torch.Tensor],
    cfg: MatrixLossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Differentiable matrix losses over slots/gates/operator matrices.

    These losses are normal first-order auxiliary losses. They do not use raw
    gradients. Raw dL/dH should be logged separately as gradient health.
    """

    total = h.new_tensor(0.0)
    logs: Dict[str, float] = {}

    wg = aux.get("write_gate_live", aux.get("write_gate"))
    ow = aux.get("op_weights_live", aux.get("op_weights"))

    if wg is not None and cfg.lambda_write_budget > 0:
        loss = (wg.float().mean() - float(cfg.write_target)).pow(2)
        total = total + float(cfg.lambda_write_budget) * loss.to(h.dtype)
        logs["loss_write_budget"] = float(loss.detach().item())

    if ow is not None:
        ent = _entropy(ow.float(), dim=-1)
        logs["op_entropy_mean"] = float(ent.detach().mean().item())
        if cfg.lambda_op_entropy_floor > 0:
            loss = torch.relu(float(cfg.op_min_entropy) - ent).pow(2).mean()
            total = total + float(cfg.lambda_op_entropy_floor) * loss.to(h.dtype)
            logs["loss_op_entropy_floor"] = float(loss.detach().item())
        if cfg.lambda_op_usage_balance > 0:
            usage = ow.float().mean(dim=(0, 1, 2))
            usage_ent = _entropy(usage, dim=0)
            max_ent = torch.log(torch.tensor(float(usage.numel()), device=usage.device))
            loss = torch.relu(0.80 * max_ent - usage_ent).pow(2)
            total = total + float(cfg.lambda_op_usage_balance) * loss.to(h.dtype)
            logs["loss_op_usage_balance"] = float(loss.detach().item())
            logs["op_usage_entropy"] = float(usage_ent.detach().item())

    if cfg.lambda_slot_diversity > 0:
        loss = slot_diversity_loss(h)
        total = total + float(cfg.lambda_slot_diversity) * loss
        logs["loss_slot_diversity"] = float(loss.detach().item())

    logs["matrix_aux_loss"] = float(total.detach().item())
    return total, logs
