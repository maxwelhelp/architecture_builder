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


def replace_all(text: str, old: str, new: str, name: str) -> str:
    n = text.count(old)
    if n == 0:
        print(f"SKIP {name}: pattern not found or already patched")
        return text
    print(f"PATCH {name}: {n} occurrence(s)")
    return text.replace(old, new)


def main() -> None:
    if not TARGET.exists():
        raise SystemExit(f"Missing {TARGET}")
    text = TARGET.read_text(encoding="utf-8")
    original = text

    # ------------------------------------------------------------------
    # 1) Role regularization: live tensors for losses, detached tensors for report.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2) fp16 safety on P40: compute all softmaxes from fp32 logits, cast back.
    # ------------------------------------------------------------------
    text = replace_once(
        text,
        "            A_route = torch.softmax(torch.einsum('bnd,brd->bnr', q_route, K_route) * scale + route_prior, dim=-1).to(dtype)\n            route_ctx = torch.einsum('bnr,brd->bnd', A_route, V_route)",
        "            score_route = torch.einsum('bnd,brd->bnr', q_route, K_route) * scale + route_prior\n            A_route = torch.softmax(score_route.float(), dim=-1).to(dtype)\n            route_ctx = torch.einsum('bnr,brd->bnd', A_route, V_route)",
        "fp32 A_route softmax",
    )
    text = replace_once(
        text,
        "            A_time = torch.softmax(torch.einsum('bnd,btd->bnt', q_time, K_time) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)\n            time_ctx = torch.einsum('bnt,btd->bnd', A_time, V_time)",
        "            score_time = torch.einsum('bnd,btd->bnt', q_time, K_time) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_time = torch.softmax(score_time.float(), dim=-1).to(dtype)\n            time_ctx = torch.einsum('bnt,btd->bnd', A_time, V_time)",
        "fp32 A_time softmax",
    )
    text = replace_once(
        text,
        "            A_freq = torch.softmax(torch.einsum('bnd,bfd->bnf', q_freq, K_freq) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)\n            freq_ctx = torch.einsum('bnf,bfd->bnd', A_freq, V_freq)",
        "            score_freq = torch.einsum('bnd,bfd->bnf', q_freq, K_freq) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_freq = torch.softmax(score_freq.float(), dim=-1).to(dtype)\n            freq_ctx = torch.einsum('bnf,bfd->bnd', A_freq, V_freq)",
        "fp32 A_freq softmax",
    )
    text = replace_once(
        text,
        "            A_onset = torch.softmax(torch.einsum('bnd,bsd->bns', q_onset, K_ev) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)\n            onset_ctx = torch.einsum('bns,bsd->bnd', A_onset, V_ev)",
        "            score_onset = torch.einsum('bnd,bsd->bns', q_onset, K_ev) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_onset = torch.softmax(score_onset.float(), dim=-1).to(dtype)\n            onset_ctx = torch.einsum('bns,bsd->bnd', A_onset, V_ev)",
        "fp32 A_onset softmax",
    )
    text = replace_once(
        text,
        "            A_energy = torch.softmax(torch.einsum('bnd,bqd->bnq', q_energy, K_energy) * (scale / max(1e-4,self.evidence_attn_temp)), dim=-1).to(dtype)\n            energy_ctx = torch.einsum('bnq,bqd->bnd', A_energy, V_energy)",
        "            score_energy = torch.einsum('bnd,bqd->bnq', q_energy, K_energy) * (scale / max(1e-4,self.evidence_attn_temp))\n            A_energy = torch.softmax(score_energy.float(), dim=-1).to(dtype)\n            energy_ctx = torch.einsum('bnq,bqd->bnd', A_energy, V_energy)",
        "fp32 A_energy softmax",
    )
    text = replace_once(
        text,
        "            A_ev = 0.5 * A_onset + 0.5 * torch.softmax(torch.einsum('bnd,bsd->bns', q_base, K_ev) * scale, dim=-1).to(dtype)\n            ev_mix = 0.30 * time_ctx + 0.30 * freq_ctx + 0.25 * onset_ctx + 0.15 * energy_ctx",
        "            score_ev_raw = torch.einsum('bnd,bsd->bns', q_base, K_ev) * scale\n            A_ev = 0.5 * A_onset + 0.5 * torch.softmax(score_ev_raw.float(), dim=-1).to(dtype)\n            ev_mix = 0.30 * time_ctx + 0.30 * freq_ctx + 0.25 * onset_ctx + 0.15 * energy_ctx",
        "fp32 raw evidence softmax",
    )
    text = replace_once(
        text,
        "            A_global = torch.softmax(torch.einsum('bnd,bgd->bng', q_global, K_global) * scale, dim=-1).to(dtype)\n            global_ctx = torch.einsum('bng,bgd->bnd', A_global, V_global)",
        "            score_global = torch.einsum('bnd,bgd->bng', q_global, K_global) * scale\n            A_global = torch.softmax(score_global.float(), dim=-1).to(dtype)\n            global_ctx = torch.einsum('bng,bgd->bnd', A_global, V_global)",
        "fp32 A_global softmax",
    )
    text = replace_once(
        text,
        "            A_mem = torch.softmax(torch.einsum('bnd,bmd->bnm', q_mem, K_mem) * scale, dim=-1).to(dtype)\n            mem_ctx = torch.einsum('bnm,bmd->bnd', A_mem, V_mem)",
        "            score_mem = torch.einsum('bnd,bmd->bnm', q_mem, K_mem) * scale\n            A_mem = torch.softmax(score_mem.float(), dim=-1).to(dtype)\n            mem_ctx = torch.einsum('bnm,bmd->bnd', A_mem, V_mem)",
        "fp32 A_mem softmax",
    )
    text = replace_once(
        text,
        "            gw = torch.softmax(torch.einsum('bnd,bgd->bng', gw_q, K_global2) * scale, dim=-1).to(dtype)\n            global_delta = torch.einsum('bng,bnd->bgd', gw * gw_gate, gw_val) / max(1,N)",
        "            score_gw = torch.einsum('bnd,bgd->bng', gw_q, K_global2) * scale\n            gw = torch.softmax(score_gw.float(), dim=-1).to(dtype)\n            global_delta = torch.einsum('bng,bnd->bgd', gw * gw_gate, gw_val) / max(1,N)",
        "fp32 global write softmax",
    )
    text = replace_once(
        text,
        "            mw = torch.softmax(torch.einsum('bnd,bmd->bnm', mw_q, K_mem2) * scale, dim=-1).to(dtype)\n            memory_delta = torch.einsum('bnm,bnd->bmd', mw * mw_gate, mw_val) / max(1,N)",
        "            score_mw = torch.einsum('bnd,bmd->bnm', mw_q, K_mem2) * scale\n            mw = torch.softmax(score_mw.float(), dim=-1).to(dtype)\n            memory_delta = torch.einsum('bnm,bnd->bmd', mw * mw_gate, mw_val) / max(1,N)",
        "fp32 memory write softmax",
    )

    # ------------------------------------------------------------------
    # 3) Evidence report channels. Add real channel reporting if not already present.
    # ------------------------------------------------------------------
    text = replace_once(
        text,
        "    evidence_attention: torch.Tensor      # [B,N,E]\n    task_attention: torch.Tensor          # [B,N,T_task]",
        "    evidence_attention: torch.Tensor      # [B,N,E] proxy: 0.5*onset + 0.5*raw evidence\n    time_attention: torch.Tensor          # [B,N,T_time]\n    freq_attention: torch.Tensor          # [B,N,T_freq]\n    onset_attention: torch.Tensor         # [B,N,E]\n    energy_attention: torch.Tensor        # [B,N,T_energy]\n    global_attention: torch.Tensor        # [B,N,G]\n    memory_attention: torch.Tensor        # [B,N,M]\n    task_attention: torch.Tensor          # [B,N,T_task]",
        "SeqAux attention fields",
    )
    text = replace_once(
        text,
        "        states = [base]\n        evid_attns = []; task_attns = []; route_attns = []; op_gates_all = []",
        "        states = [base]\n        evid_attns = []\n        time_attns = []; freq_attns = []; onset_attns = []; energy_attns = []\n        global_attns = []; memory_attns = []\n        task_attns = []; route_attns = []; op_gates_all = []",
        "attention buffers",
    )
    text = replace_once(
        text,
        "            evid_attns.append(A_ev); task_attns.append(A_task); route_attns.append(A_route); op_gates_all.append(op_gates); category_gates_all.append(category_gates)",
        "            evid_attns.append(A_ev)\n            time_attns.append(A_time); freq_attns.append(A_freq); onset_attns.append(A_onset); energy_attns.append(A_energy)\n            global_attns.append(A_global); memory_attns.append(A_mem)\n            task_attns.append(A_task); route_attns.append(A_route); op_gates_all.append(op_gates); category_gates_all.append(category_gates)",
        "attention append",
    )
    text = replace_once(
        text,
        "        evidence_attention = torch.cat(evid_attns, dim=1) if evid_attns else torch.empty(B,0,E,device=device,dtype=dtype)\n        task_attention = torch.cat(task_attns, dim=1) if task_attns else torch.empty(B,0,self.task_cells,device=device,dtype=dtype)",
        "        evidence_attention = torch.cat(evid_attns, dim=1) if evid_attns else torch.empty(B,0,E,device=device,dtype=dtype)\n        time_attention = torch.cat(time_attns, dim=1) if time_attns else torch.empty(B,0,time_tokens.shape[1],device=device,dtype=dtype)\n        freq_attention = torch.cat(freq_attns, dim=1) if freq_attns else torch.empty(B,0,freq_tokens.shape[1],device=device,dtype=dtype)\n        onset_attention = torch.cat(onset_attns, dim=1) if onset_attns else torch.empty(B,0,E,device=device,dtype=dtype)\n        energy_attention = torch.cat(energy_attns, dim=1) if energy_attns else torch.empty(B,0,energy_tokens.shape[1],device=device,dtype=dtype)\n        global_attention = torch.cat(global_attns, dim=1) if global_attns else torch.empty(B,0,self.global_cells,device=device,dtype=dtype)\n        memory_attention = torch.cat(memory_attns, dim=1) if memory_attns else torch.empty(B,0,self.memory_cells,device=device,dtype=dtype)\n        task_attention = torch.cat(task_attns, dim=1) if task_attns else torch.empty(B,0,self.task_cells,device=device,dtype=dtype)",
        "attention cat tensors",
    )
    text = replace_once(
        text,
        "        aux = SeqAux(\n            evidence_attention=evidence_attention.detach(),\n            task_attention=task_attention.detach(),",
        "        aux = SeqAux(\n            evidence_attention=evidence_attention.detach(),\n            time_attention=time_attention.detach(),\n            freq_attention=freq_attention.detach(),\n            onset_attention=onset_attention.detach(),\n            energy_attention=energy_attention.detach(),\n            global_attention=global_attention.detach(),\n            memory_attention=memory_attention.detach(),\n            task_attention=task_attention.detach(),",
        "SeqAux attention args",
    )
    text = replace_once(
        text,
        "        self.evidence_attention = None\n        self.task_attention = None",
        "        self.evidence_attention = None\n        self.time_attention = None\n        self.freq_attention = None\n        self.onset_attention = None\n        self.energy_attention = None\n        self.global_attention_by_block = None\n        self.memory_attention_by_block = None\n        self.task_attention = None",
        "acc reset attention fields",
    )
    text = replace_once(
        text,
        "        self.evidence_attention = acc(self.evidence_attention, aux.evidence_attention.mean(dim=0))  # [L,S]\n        self.task_attention = acc(self.task_attention, aux.task_attention.mean(dim=0))  # [L,T_task]",
        "        self.evidence_attention = acc(self.evidence_attention, aux.evidence_attention.mean(dim=0))  # [L,S] proxy\n        self.time_attention = acc(self.time_attention, aux.time_attention.mean(dim=0))\n        self.freq_attention = acc(self.freq_attention, aux.freq_attention.mean(dim=0))\n        self.onset_attention = acc(self.onset_attention, aux.onset_attention.mean(dim=0))\n        self.energy_attention = acc(self.energy_attention, aux.energy_attention.mean(dim=0))\n        self.global_attention_by_block = acc(self.global_attention_by_block, aux.global_attention.mean(dim=0))\n        self.memory_attention_by_block = acc(self.memory_attention_by_block, aux.memory_attention.mean(dim=0))\n        self.task_attention = acc(self.task_attention, aux.task_attention.mean(dim=0))  # [L,T_task]",
        "acc add attention fields",
    )
    text = replace_once(
        text,
        "        ev_attn = self.evidence_attention / c if self.evidence_attention is not None else torch.zeros(self.chain_depth, self.evidence_cells)\n        cls_attn = self.class_stage_attention / c if self.class_stage_attention is not None else torch.zeros(self.num_classes, self.chain_depth + 1)",
        "        ev_attn = self.evidence_attention / c if self.evidence_attention is not None else torch.zeros(self.chain_depth, self.evidence_cells)\n        channel_attn = {\n            \"time\": self.time_attention / c if self.time_attention is not None else None,\n            \"freq\": self.freq_attention / c if self.freq_attention is not None else None,\n            \"onset\": self.onset_attention / c if self.onset_attention is not None else None,\n            \"energy\": self.energy_attention / c if self.energy_attention is not None else None,\n            \"global\": self.global_attention_by_block / c if self.global_attention_by_block is not None else None,\n            \"memory\": self.memory_attention_by_block / c if self.memory_attention_by_block is not None else None,\n        }\n        cls_attn = self.class_stage_attention / c if self.class_stage_attention is not None else torch.zeros(self.num_classes, self.chain_depth + 1)",
        "summary channel_attn",
    )
    text = replace_once(
        text,
        "        top_task_by_stage = []\n        for i in range(task_attn.shape[0]):",
        "        attention_channels_by_stage = []\n        for channel_name, attn in channel_attn.items():\n            if attn is None or attn.numel() == 0:\n                continue\n            for i in range(attn.shape[0]):\n                vals, idxs = torch.topk(attn[i], k=min(5, attn.shape[1]))\n                item = self.block_label(i)\n                item[\"channel\"] = channel_name\n                item[\"top\"] = [{\"cell\": int(j), \"weight\": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]\n                attention_channels_by_stage.append(item)\n\n        def top_attn_items(attn, i: int, k: int = 4) -> List[Dict[str, float]]:\n            if attn is None or attn.numel() == 0 or i >= attn.shape[0]:\n                return []\n            vals, idxs = torch.topk(attn[i], k=min(k, attn.shape[1]))\n            return [{\"cell\": int(j), \"weight\": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())]\n\n        top_task_by_stage = []\n        for i in range(task_attn.shape[0]):",
        "summary attention_channels_by_stage",
    )
    # If previous patch inserted a brittle zip(*topk) expression, replace it.
    text = replace_once(
        text,
        "                \"attention_channel_patterns\": {\n                    name: (\n                        [{\"cell\": int(j), \"weight\": float(v)} for v, j in zip(*[list(x) for x in torch.topk(attn[i], k=min(4, attn.shape[1]))])]\n                        if attn is not None and attn.numel() > 0 and i < attn.shape[0] else []\n                    )\n                    for name, attn in channel_attn.items()\n                },",
        "                \"attention_channel_patterns\": {name: top_attn_items(attn, i, 4) for name, attn in channel_attn.items()},",
        "simplify attention_channel_patterns",
    )
    text = replace_once(
        text,
        "                \"evidence_patterns\": [{\"cell\": int(j), \"name\": evidence_name(int(j)), \"weight\": float(v)} for v, j in zip(ev_vals.tolist(), ev_idxs.tolist())],\n                \"task_patterns\":",
        "                \"evidence_patterns\": [{\"cell\": int(j), \"name\": evidence_name(int(j)), \"weight\": float(v)} for v, j in zip(ev_vals.tolist(), ev_idxs.tolist())],\n                \"attention_channel_patterns\": {name: top_attn_items(attn, i, 4) for name, attn in channel_attn.items()},\n                \"task_patterns\":",
        "operator block attention channels",
    )
    text = replace_once(
        text,
        "            \"top_evidence_by_stage\": top_evidence_by_stage,\n            \"top_task_by_stage\": top_task_by_stage,",
        "            \"top_evidence_by_stage\": top_evidence_by_stage,\n            \"attention_channels_by_stage\": attention_channels_by_stage,\n            \"top_task_by_stage\": top_task_by_stage,",
        "return attention_channels_by_stage",
    )

    # ------------------------------------------------------------------
    # 4) SeqAccumulator topology-awareness for merge topologies.
    # ------------------------------------------------------------------
    text = replace_once(
        text,
        "    def __init__(self, num_classes: int, evidence_cells: int, chain_depth: int, num_layers: int = 4, blocks_per_layer: int = 3, decomp_every: int = 25):\n        self.num_classes = num_classes\n        self.evidence_cells = evidence_cells\n        self.chain_depth = chain_depth\n        self.num_layers = int(num_layers)\n        self.blocks_per_layer = int(blocks_per_layer)\n        self.decomp_every = int(decomp_every)",
        "    def __init__(self, num_classes: int, evidence_cells: int, chain_depth: int, num_layers: int = 4, blocks_per_layer: int = 3, decomp_every: int = 25, topology_plan: List[List[int]] | None = None):\n        self.num_classes = num_classes\n        self.evidence_cells = evidence_cells\n        self.num_layers = int(num_layers)\n        self.blocks_per_layer = int(blocks_per_layer)\n        self.topology_plan = [list(x) for x in topology_plan] if topology_plan is not None else [list(range(self.blocks_per_layer)) for _ in range(self.num_layers)]\n        self.flat_to_layer_block = [(li, bi) for li, blocks in enumerate(self.topology_plan) for bi in blocks]\n        self.chain_depth = len(self.flat_to_layer_block)\n        self.decomp_every = int(decomp_every)",
        "SeqAccumulator topology init",
    )
    text = replace_once(
        text,
        "    def block_label(self, flat_idx: int) -> Dict[str, Any]:\n        layer = int(flat_idx) // max(1, self.blocks_per_layer)\n        block = int(flat_idx) % max(1, self.blocks_per_layer)\n        return {\"stage\": int(flat_idx), \"layer\": layer, \"block\": block, \"name\": f\"L{layer}.B{block}\"}",
        "    def block_label(self, flat_idx: int) -> Dict[str, Any]:\n        flat_idx = int(flat_idx)\n        if 0 <= flat_idx < len(self.flat_to_layer_block):\n            layer, block = self.flat_to_layer_block[flat_idx]\n        else:\n            layer = flat_idx // max(1, self.blocks_per_layer)\n            block = flat_idx % max(1, self.blocks_per_layer)\n        return {\"stage\": flat_idx, \"layer\": int(layer), \"block\": int(block), \"name\": f\"L{int(layer)}.B{int(block)}\"}",
        "topology block_label",
    )
    text = replace_once(
        text,
        "            for layer in range(self.num_layers):\n                s = layer * self.blocks_per_layer + 1\n                e = s + self.blocks_per_layer\n                if e <= st.shape[1]:",
        "            offset = 0\n            for layer, active_blocks in enumerate(self.topology_plan):\n                n = len(active_blocks)\n                s = offset + 1\n                e = s + n\n                offset += n\n                if n > 0 and e <= st.shape[1]:",
        "topology decomp stage slices",
    )
    text = replace_once(
        text,
        "        for layer in range(self.num_layers):\n            blocks = []\n            for block in range(self.blocks_per_layer):\n                flat = layer * self.blocks_per_layer + block\n                read_idx = flat + 1\n                blocks.append({\n                    \"name\": f\"L{layer}.B{block}\",\n                    \"stage\": flat,\n                    \"state_norm\": float(stage_norm_avg[read_idx]) if read_idx < len(stage_norm_avg) else 0.0,\n                    \"attention_concentration_proxy\": self.attn_corr_gap_sum[flat] / max(1, self.attn_corr_gap_count),\n                })\n            matrix_decomposition.append({",
        "        offset = 0\n        for layer, active_blocks in enumerate(self.topology_plan):\n            blocks = []\n            for local_i, block in enumerate(active_blocks):\n                flat = offset + local_i\n                read_idx = flat + 1\n                blocks.append({\n                    \"name\": f\"L{layer}.B{block}\",\n                    \"stage\": flat,\n                    \"state_norm\": float(stage_norm_avg[read_idx]) if read_idx < len(stage_norm_avg) else 0.0,\n                    \"attention_concentration_proxy\": self.attn_corr_gap_sum[flat] / max(1, self.attn_corr_gap_count) if flat < len(self.attn_corr_gap_sum) else 0.0,\n                })\n            offset += len(active_blocks)\n            matrix_decomposition.append({",
        "topology matrix_decomposition loop",
    )
    text = replace_once(
        text,
        "            for layer in range(self.num_layers):\n                n = self.blocks_per_layer\n                usage = op_gates[offset:offset+n, :].mean(dim=0) if offset + n <= op_gates.shape[0] else torch.zeros(op_gates.shape[1])",
        "            for layer, active_blocks in enumerate(self.topology_plan):\n                n = len(active_blocks)\n                usage = op_gates[offset:offset+n, :].mean(dim=0) if n > 0 and offset + n <= op_gates.shape[0] else torch.zeros(op_gates.shape[1])",
        "topology phase_role_diagnostics loop",
    )
    text = replace_once(
        text,
        "        acc = SeqAccumulator(num_classes=len(classes), evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=args.decomp_every)",
        "        acc = SeqAccumulator(num_classes=len(classes), evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=args.decomp_every, topology_plan=model.core.topology_plan)",
        "train accumulator topology_plan",
    )
    text = replace_once(
        text,
        "    acc = SeqAccumulator(num_classes=C, evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=1)",
        "    acc = SeqAccumulator(num_classes=C, evidence_cells=args.evidence_cells, chain_depth=args.chain_depth, num_layers=args.num_layers, blocks_per_layer=args.blocks_per_layer, decomp_every=1, topology_plan=model.core.topology_plan)",
        "eval accumulator topology_plan",
    )

    # ------------------------------------------------------------------
    # 5) Idempotent export-memory flag.
    # ------------------------------------------------------------------
    if "--no-export-architecture-memory" not in text:
        text = replace_once(
            text,
            '    p.add_argument("--export-architecture-memory", action="store_true", default=True)',
            '    p.add_argument("--export-architecture-memory", action="store_true", default=True)\n    p.add_argument("--no-export-architecture-memory", dest="export_architecture_memory", action="store_false")',
            "add no-export-architecture-memory flag",
        )
    else:
        print("SKIP no-export flag: already present")

    if text == original:
        print("No changes made.")
        return
    backup = TARGET.with_suffix(TARGET.suffix + ".bak_before_v13_stability_v2")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
        print(f"Backup written: {backup}")
    TARGET.write_text(text, encoding="utf-8")
    print("Done. Run:")
    print("  python -m py_compile sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py")
    print("  python tools/smoke_v13_topologies.py")


if __name__ == "__main__":
    main()
