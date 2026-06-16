from __future__ import annotations

import torch


def topk_mask(scores: torch.Tensor, k: int, *, warmup_all: bool = False) -> torch.Tensor:
    """Return a binary top-k mask with the same shape as scores.

    scores: [..., O]
    mask:   [..., O]
    """

    if warmup_all or int(k) <= 0 or int(k) >= scores.shape[-1]:
        return torch.ones_like(scores, dtype=scores.dtype)
    idx = torch.topk(scores, k=int(k), dim=-1).indices
    mask = torch.zeros_like(scores, dtype=scores.dtype)
    return mask.scatter(-1, idx, 1.0)


def straight_through_topk_weights(
    scores: torch.Tensor,
    k: int,
    *,
    temperature: float = 1.0,
    warmup_all: bool = False,
) -> torch.Tensor:
    """Hard top-k in forward, softmax gradient in backward.

    This is useful when the executor computes only selected branches or when a
    dense prototype wants to emulate hard selection before sparse execution.
    """

    soft = torch.softmax(scores.float() / max(float(temperature), 1e-6), dim=-1).to(scores.dtype)
    mask = topk_mask(scores, k, warmup_all=warmup_all)
    hard = soft * mask
    hard = hard / hard.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return hard.detach() - soft.detach() + soft


def dense_topk_weighted_sum(
    candidates: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    *,
    temperature: float = 1.0,
    warmup_all: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prototype executor for candidate tensor [..., O, D].

    It still receives precomputed candidates, so it is not a speed win by itself.
    It is useful for testing top-k semantics before moving heavy operators behind
    lazy branch functions.
    """

    weights = straight_through_topk_weights(scores, k, temperature=temperature, warmup_all=warmup_all)
    out = torch.einsum("...o,...od->...d", weights, candidates)
    return out, weights


def selected_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    if int(k) <= 0 or int(k) >= scores.shape[-1]:
        return torch.arange(scores.shape[-1], device=scores.device).expand(*scores.shape[:-1], scores.shape[-1])
    return torch.topk(scores, k=int(k), dim=-1).indices
