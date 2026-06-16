from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class GrowPruneConfig:
    plateau_epochs: int = 3
    overfit_gap: float = 0.18
    min_epoch_for_growth: int = 5
    max_actions: int = 6


def decide_step_actions(history: List[Dict[str, Any]], utility_report: Dict[str, Any], cfg: GrowPruneConfig | None = None) -> Dict[str, Any]:
    """Decide architecture actions from metric history and per-step utility.

    This controller is intentionally conservative. It returns *actions to try*,
    not in-place architecture mutations. The next stage can execute them with a
    grace period / shadow replacement.
    """

    cfg = cfg or GrowPruneConfig()
    actions: List[Dict[str, Any]] = []
    if not history:
        return {"mode": "wait", "reason": "no_history", "actions": actions}

    last = history[-1]
    epoch = int(last.get("epoch", len(history)))
    train_acc = float(last.get("train_acc", last.get("train", 0.0)) or 0.0)
    val_acc = float(last.get("val_acc", last.get("val", 0.0)) or 0.0)
    overfit = train_acc - val_acc > cfg.overfit_gap
    recent = history[-cfg.plateau_epochs:]
    vals = [float(x.get("val_acc", x.get("val", 0.0)) or 0.0) for x in recent]
    plateau = len(vals) >= cfg.plateau_epochs and max(vals) - min(vals) < 0.006

    per = list(utility_report.get("per_step", []))
    dead = [x for x in per if str(x.get("status", "")).startswith("dead")]
    overloaded = [x for x in per if str(x.get("status", "")).startswith("overloaded")]
    saturated = [x for x in per if str(x.get("status", "")) == "gate_saturated"]

    if overfit:
        mode = "compress"
        reason = "train_val_gap_overfit"
        for x in saturated[: cfg.max_actions]:
            actions.append({"action": "gate_budget_or_freeze", "target": _target(x), "why": "gate saturated under overfit"})
        for x in dead[: cfg.max_actions - len(actions)]:
            actions.append({"action": "soft_prune", "target": _target(x), "why": "dead step under overfit"})
    elif plateau and epoch >= cfg.min_epoch_for_growth:
        mode = "grow_or_replace"
        reason = "validation_plateau"
        for x in overloaded[: cfg.max_actions]:
            actions.append({"action": "add_step_after", "target": _target(x), "why": "overloaded step"})
        for x in dead[: cfg.max_actions - len(actions)]:
            actions.append({"action": "replace_step_shadow", "target": _target(x), "why": "dead step on plateau"})
    else:
        mode = "train"
        reason = "no_arch_action"

    if not actions and dead and epoch >= cfg.min_epoch_for_growth:
        actions.append({"action": "mark_prune_candidate", "target": _target(dead[0]), "why": "low utility"})

    return {
        "mode": mode,
        "reason": reason,
        "overfit": overfit,
        "plateau": plateau,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "dead_steps": len(dead),
        "overloaded_steps": len(overloaded),
        "saturated_steps": len(saturated),
        "actions": actions,
    }


def _target(x: Dict[str, Any]) -> Dict[str, int]:
    return {"layer": int(x.get("layer", -1)), "block": int(x.get("block", -1)), "step": int(x.get("step", -1))}
