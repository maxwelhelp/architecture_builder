#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

TARGET = Path("sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        print(f"SKIP {name}: pattern not found or already patched")
        return text
    print(f"PATCH {name}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # 1) Role regularization must use live tensors, while reports stay detached.
    text = replace_once(
        text,
        "        role_mixes = []\n        category_gates_all = []",
        "        role_mixes_live = []\n        role_mixes_report = []\n        category_gates_all = []",
        "role buffers",
    )
    text = replace_once(
        text,
        "            role_mixes.append(role_mix.detach().float().mean(dim=0))",
        "            role_mixes_live.append(role_mix.float().mean(dim=0))\n            role_mixes_report.append(role_mix.detach().float().mean(dim=0))",
        "role live/report append",
    )
    text = replace_once(
        text,
        "        role_mix_by_layer = torch.stack(role_mixes, dim=0) if role_mixes else torch.empty(0, self.num_roles, device=device)",
        "        role_mix_by_layer_live = torch.stack(role_mixes_live, dim=0) if role_mixes_live else torch.empty(0, self.num_roles, device=device)\n        role_mix_by_layer = torch.stack(role_mixes_report, dim=0) if role_mixes_report else torch.empty(0, self.num_roles, device=device)",
        "role live/report stack",
    )
    text = replace_once(
        text,
        "        if role_mix_by_layer.numel() > 0:\n            # Encourage adjacent layers to use different role mixtures, not one universal solution.\n            if role_mix_by_layer.shape[0] > 1:\n                rn = F.normalize(role_mix_by_layer.float(), dim=-1)\n                adj_sim = (rn[:-1] * rn[1:]).sum(dim=-1)",
        "        if role_mix_by_layer_live.numel() > 0:\n            # Encourage adjacent layers to use different role mixtures, not one universal solution.\n            if role_mix_by_layer_live.shape[0] > 1:\n                rn = F.normalize(role_mix_by_layer_live.float(), dim=-1)\n                adj_sim = (rn[:-1] * rn[1:]).sum(dim=-1)",
        "role diversity live loss",
    )
    text = replace_once(
        text,
        "            if self.lock_last_role and role_mix_by_layer.shape[0] >= 1:\n                last = role_mix_by_layer[-1].float()",
        "            if self.lock_last_role and role_mix_by_layer_live.shape[0] >= 1:\n                last = role_mix_by_layer_live[-1].float()",
        "role anchor live loss",
    )

    # 2) Fix one-way store_true flag if present in local version.
    text = text.replace(
        'p.add_argument("--export-architecture-memory", action="store_true", default=True)',
        'p.add_argument("--export-architecture-memory", action="store_true", default=True)\n    p.add_argument("--no-export-architecture-memory", dest="export_architecture_memory", action="store_false")',
    )

    if text == original:
        print("No changes made.")
        return
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_role_live_patch")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backup written: {backup}")
    TARGET.write_text(text, encoding="utf-8")
    print("Done. Run: python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")


if __name__ == "__main__":
    main()
