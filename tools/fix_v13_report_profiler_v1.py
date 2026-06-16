#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

TARGET = Path("sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")


def replace_once(text: str, old: str, new: str, name: str) -> str:
    if old not in text:
        print(f"SKIP {name}: not found or already patched")
        return text
    print(f"PATCH {name}")
    return text.replace(old, new, 1)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # 1) Remove duplicate attention_channels_by_stage block left by earlier patch layers.
    duplicate_block = '''        attention_channels_by_stage = []
        for channel_name, attn in channel_attn.items():
            if attn is None or attn.numel() == 0:
                continue
            for i in range(attn.shape[0]):
                vals, idxs = torch.topk(attn[i], k=min(5, attn.shape[1]))
                item = self.block_label(i)
                item["channel"] = channel_name
                item["top"] = [{"cell": int(j), "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
                attention_channels_by_stage.append(item)

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
'''
    single_block = '''        attention_channels_by_stage = []
        for channel_name, attn in channel_attn.items():
            if attn is None or attn.numel() == 0:
                continue
            for i in range(attn.shape[0]):
                vals, idxs = torch.topk(attn[i], k=min(5, attn.shape[1]))
                item = self.block_label(i)
                item["channel"] = channel_name
                item["top"] = [{"cell": int(j), "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]
                attention_channels_by_stage.append(item)
'''
    text = replace_once(text, duplicate_block, single_block, "deduplicate attention_channels_by_stage")

    # 2) Make route report topology-aware for merge topologies.
    old_route = '''        route_attn = self.route_attention / c if self.route_attention is not None else torch.zeros(self.chain_depth, self.blocks_per_layer + 2)
        route_usage_by_block = []
        for i in range(route_attn.shape[0]):
            layer = i // max(1, self.blocks_per_layer)
            vals, idxs = torch.topk(route_attn[i], k=min(5, route_attn.shape[1]))
            route_items = []
            for v, j in zip(vals.tolist(), idxs.tolist()):
                jj = int(j)
                if jj == 0:
                    name = "BASE_SKIP"
                elif jj == 1:
                    name = "GLOBAL_REGISTER_SKIP"
                else:
                    prev_block = jj - 2
                    prev_layer = layer - 1
                    name = f"INIT.B{prev_block}" if prev_layer < 0 else f"L{prev_layer}.B{prev_block}"
                route_items.append({"route": jj, "name": name, "weight": float(v)})
            item = self.block_label(i)
            item["routes"] = route_items
            route_usage_by_block.append(item)
'''
    new_route = '''        route_attn = self.route_attention / c if self.route_attention is not None else torch.zeros(self.chain_depth, self.blocks_per_layer + 2)
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
'''
    text = replace_once(text, old_route, new_route, "topology-aware route_usage_by_block")

    # 3) Make profiler actually step/close/export a table.
    step_marker = '''        speed_count += 1
        last_end = time.perf_counter()
'''
    step_patch = '''        speed_count += 1
        if prof is not None:
            prof.step()
            if int(getattr(args, "profile_steps", 0)) > 0 and step >= int(getattr(args, "profile_steps", 0)):
                print(f"torch_profiler: reached profile_steps={int(getattr(args, 'profile_steps', 0))}; stopping epoch early", flush=True)
                last_end = time.perf_counter()
                break
        last_end = time.perf_counter()
'''
    if "torch_profiler: reached profile_steps" not in text:
        text = replace_once(text, step_marker, step_patch, "profiler step/profile_steps")
    else:
        print("SKIP profiler step/profile_steps: already patched")

    return_marker = '''    return {
        "loss": total_loss / max(1, total),
'''
    profile_close = '''    if prof is not None:
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
'''
    if "torch_profiler: wrote" not in text:
        text = replace_once(text, return_marker, profile_close, "profiler close/export")
    else:
        print("SKIP profiler close/export: already patched")

    if text == original:
        print("No changes made.")
        return
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_v13_report_profiler_v1")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backup written: {backup}")
    TARGET.write_text(text, encoding="utf-8")
    print("Done. Run:")
    print("  python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")
    print("  PYTHONPATH=\"$PWD\" python tools/smoke_v13_topologies.py")


if __name__ == "__main__":
    main()
