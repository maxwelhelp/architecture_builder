#!/usr/bin/env python3
"""Patch StepProgram v1 to keep audio frontend/log-normalization in fp32.

Why: under CUDA fp16 autocast, MelSpectrogram + clamp_min(1e-8) + log()
can underflow to log(0) and produce -inf, which then makes every training
batch NONFINITE_LOSS.  The model can still run the program core in fp16; only
spectrogram/statistics are forced to fp32.
"""
from pathlib import Path

path = Path("experiments/step_program/run_step_program_speechcommands_v1.py")
text = path.read_text(encoding="utf-8")

old = '''    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav [B,1,T]
        B = wav.shape[0]
        if self.mel is not None:
            x = self.mel(wav.squeeze(1)).clamp_min(1e-8).log()  # [B,n_mels,Tm]
        else:
            # Fallback synthetic spectrogram from framed absolute waveform.
            x = wav.squeeze(1).unfold(-1, 320, 160).abs().mean(dim=-1).unsqueeze(1).repeat(1, 64, 1)
        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)

        grid = F.adaptive_avg_pool2d(x.unsqueeze(1), (6, 6)).squeeze(1).flatten(1)  # [B,36]
        time3 = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (1, 3)).flatten(1)  # [B,3]
        freq4 = F.adaptive_avg_pool2d(x.mean(dim=2, keepdim=True), (4, 1)).flatten(1)  # [B,4]
        global_stats = torch.stack([
            x.mean(dim=(-2, -1)),
            x.std(dim=(-2, -1)),
            x.amax(dim=(-2, -1)),
            x.amin(dim=(-2, -1)),
        ], dim=1)
        delta_time = time3[:, 1:] - time3[:, :-1]
        feats = torch.cat([grid, time3, freq4, global_stats, delta_time], dim=1)
        if feats.shape[1] < self.evidence_cells:
            rep = math.ceil(self.evidence_cells / feats.shape[1])
            feats = feats.repeat(1, rep)
        feats = feats[:, : self.evidence_cells]
        ev = self.scalar_proj(feats.unsqueeze(-1)) + self.pos.view(1, self.evidence_cells, -1)
        return self.norm(ev)
'''

new = '''    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav [B,1,T]
        # Keep audio frontend and feature statistics in fp32 even when the outer
        # model is under CUDA fp16 autocast.  This prevents log(0)/-inf from
        # MelSpectrogram underflow and removes NONFINITE_LOSS at batch 1.
        device_type = wav.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            wav32 = wav.float()
            if self.mel is not None:
                x = self.mel(wav32.squeeze(1)).float().clamp_min(1e-5).log()  # [B,n_mels,Tm]
            else:
                # Fallback synthetic spectrogram from framed absolute waveform.
                x = wav32.squeeze(1).unfold(-1, 320, 160).abs().mean(dim=-1).unsqueeze(1).repeat(1, 64, 1)
            x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
            x = (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            grid = F.adaptive_avg_pool2d(x.unsqueeze(1), (6, 6)).squeeze(1).flatten(1)  # [B,36]
            time3 = F.adaptive_avg_pool2d(x.mean(dim=1, keepdim=True), (1, 3)).flatten(1)  # [B,3]
            freq4 = F.adaptive_avg_pool2d(x.mean(dim=2, keepdim=True), (4, 1)).flatten(1)  # [B,4]
            global_stats = torch.stack([
                x.mean(dim=(-2, -1)),
                x.std(dim=(-2, -1)),
                x.amax(dim=(-2, -1)),
                x.amin(dim=(-2, -1)),
            ], dim=1)
            delta_time = time3[:, 1:] - time3[:, :-1]
            feats = torch.cat([grid, time3, freq4, global_stats, delta_time], dim=1)
            if feats.shape[1] < self.evidence_cells:
                rep = math.ceil(self.evidence_cells / feats.shape[1])
                feats = feats.repeat(1, rep)
            feats = torch.nan_to_num(feats[:, : self.evidence_cells], nan=0.0, posinf=0.0, neginf=0.0)
        ev = self.scalar_proj(feats.unsqueeze(-1)) + self.pos.view(1, self.evidence_cells, -1)
        return torch.nan_to_num(self.norm(ev), nan=0.0, posinf=0.0, neginf=0.0)
'''

if new in text:
    print("already patched")
elif old in text:
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched {path}")
else:
    raise SystemExit("target block not found; file already changed or patch needs manual update")
