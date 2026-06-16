#!/usr/bin/env python3
from pathlib import Path

p = Path('sequential_matrix_cells_speechcommands_v13_categorized_signal_bus_builder.py')
text = p.read_text(encoding='utf-8')
orig = text
changes = []

def rep(old, new, name):
    global text
    if old in text:
        text = text.replace(old, new, 1)
        changes.append(name)

# 1) SeqAux fields.
old = '    category_entropy_loss: torch.Tensor\n    signal_op_bias_norm: torch.Tensor'
new = '    category_entropy_loss: torch.Tensor\n    gate_budget_loss: torch.Tensor\n    signal_op_bias_norm_loss: torch.Tensor\n    signal_op_bias_norm: torch.Tensor'
if 'gate_budget_loss: torch.Tensor' not in text:
    rep(old, new, 'seqaux_saturation_fields')

# 2) Live saturation losses in forward, before SeqAux construction.
old = '''        category_diversity_loss = torch.stack(category_div_losses).mean() if category_div_losses else torch.zeros((), device=device)

        if 'update_norm_by_block' not in locals():'''
new = '''        category_diversity_loss = torch.stack(category_div_losses).mean() if category_div_losses else torch.zeros((), device=device)

        # Saturation control: previous late_dyn loss only opened gates, it never
        # punished always-open gates. These are live first-order losses.
        if effective_stage_gate.numel() > 0:
            eff_mean_live = effective_stage_gate.float().mean()
            dyn_mean_live = dynamic_stage_gate.float().mean() if dynamic_stage_gate.numel() > 0 else eff_mean_live
            gate_budget_loss = F.relu(eff_mean_live - 0.82).pow(2) + 0.50 * F.relu(dyn_mean_live - 0.86).pow(2)
        else:
            gate_budget_loss = torch.zeros((), device=device)
        signal_op_bias_norm_loss = F.relu(signal_op_bias_norm.float() - 8.0).pow(2) / 64.0

        if 'update_norm_by_block' not in locals():'''
if 'gate_budget_loss = F.relu(eff_mean_live - 0.82)' not in text:
    rep(old, new, 'live_saturation_losses')

# 3) SeqAux construction.
old = '''            category_diversity_loss=category_diversity_loss,
            category_entropy_loss=category_entropy_loss,
            signal_op_bias_norm=signal_op_bias_norm.detach(),'''
new = '''            category_diversity_loss=category_diversity_loss,
            category_entropy_loss=category_entropy_loss,
            gate_budget_loss=gate_budget_loss,
            signal_op_bias_norm_loss=signal_op_bias_norm_loss,
            signal_op_bias_norm=signal_op_bias_norm.detach(),'''
if 'gate_budget_loss=gate_budget_loss' not in text:
    rep(old, new, 'seqaux_saturation_values')

# 4) Train loop extracts and adds losses.
old = '''            category_diversity = aux["seq"].category_diversity_loss
            category_entropy = aux["seq"].category_entropy_loss
            # vNext schedule: early soft exploration, late sharper readable primitive program.'''
new = '''            category_diversity = aux["seq"].category_diversity_loss
            category_entropy = aux["seq"].category_entropy_loss
            gate_budget = aux["seq"].gate_budget_loss
            signal_op_norm_loss = aux["seq"].signal_op_bias_norm_loss
            # vNext schedule: early soft exploration, late sharper readable primitive program.'''
if 'gate_budget = aux["seq"].gate_budget_loss' not in text:
    rep(old, new, 'train_extract_saturation_losses')

old = '''                + args.lambda_category_diversity * category_diversity
                + lambda_category_entropy_eff * category_entropy
            )'''
new = '''                + args.lambda_category_diversity * category_diversity
                + lambda_category_entropy_eff * category_entropy
                + args.lambda_gate_budget * gate_budget
                + args.lambda_signal_op_norm * signal_op_norm_loss
            )'''
if 'args.lambda_gate_budget * gate_budget' not in text:
    rep(old, new, 'train_add_saturation_losses')

# 5) Metrics row and print.
old = '''                "category_entropy": seq.get("category_entropy_loss", 0.0),
                "signal_op_bias_norm": seq.get("signal_op_bias_norm", 0.0),'''
new = '''                "category_entropy": seq.get("category_entropy_loss", 0.0),
                "gate_budget": seq.get("gate_budget_loss", 0.0),
                "signal_op_norm_loss": seq.get("signal_op_bias_norm_loss", 0.0),
                "signal_op_bias_norm": seq.get("signal_op_bias_norm", 0.0),'''
if '"gate_budget": seq.get("gate_budget_loss"' not in text:
    rep(old, new, 'metrics_saturation_losses')

old = '''            f"sigop={seq.get('signal_op_bias_norm', 0):.2f} "
            f"lr={train.get('lr', args.lr):.2e} "'''
new = '''            f"sigop={seq.get('signal_op_bias_norm', 0):.2f} "
            f"gbud={seq.get('gate_budget_loss', 0):.3f} "
            f"signl={seq.get('signal_op_bias_norm_loss', 0):.3f} "
            f"lr={train.get('lr', args.lr):.2e} "'''
if "gbud={seq.get('gate_budget_loss'" not in text:
    rep(old, new, 'print_saturation_losses')

# 6) Parser args.
old = '''    p.add_argument("--lambda-category-diversity", type=float, default=0.006)
    p.add_argument("--lambda-category-entropy", type=float, default=0.002)'''
new = '''    p.add_argument("--lambda-category-diversity", type=float, default=0.006)
    p.add_argument("--lambda-category-entropy", type=float, default=0.002)
    p.add_argument("--lambda-gate-budget", type=float, default=0.0)
    p.add_argument("--lambda-signal-op-norm", type=float, default=0.0)'''
if '--lambda-gate-budget' not in text:
    rep(old, new, 'parser_saturation_loss_args')

if text == orig:
    print('No changes; saturation losses already patched or patterns changed.')
else:
    bak = p.with_suffix(p.suffix + '.bak_before_saturation_losses')
    if not bak.exists():
        bak.write_text(orig, encoding='utf-8')
    p.write_text(text, encoding='utf-8')
    print('Applied:', ', '.join(changes))
    print('Backup:', bak)
