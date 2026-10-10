#!/usr/bin/env python3
"""Week 3: log-Mel frontend + SpecAugment.

Feature path (training):
    waveform -> log-Mel -> (featurise) -> SpecAugment -> model
Feature path (validation/test): waveform -> log-Mel -> model
(SpecAugment is constructed with train=True only for training.)

Normalisation: per-utterance log10 energy scaled to zero mean / unit variance
per frequency band, the standard wav2vec-style log10f normalisation. This
keeps features in a stable range on top of the heterogeneous augmentation
gains.

Output tensor layout: (time, n_mels) per utterance; batches become
(B, 1, T, F) for the CNN frontend.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from config import FeatureConfig, SpecAugmentConfig


class LogMelFrontend(nn.Module):
    """Waveform (T,) or (B, T) -> log-Mel (T', n_mels) or (B, T', n_mels)."""

    def __init__(self, cfg: FeatureConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
        )

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        single = wav.dim() == 1
        if single:
            wav = wav.unsqueeze(0)
        mel = self.melspec(wav)                    # (B, n_mels, T)
        logmel = torch.log10(mel.clamp_min(1e-10))  # log10f energy
        logmel = (logmel - logmel.mean(dim=-1, keepdim=True)) / (
            logmel.std(dim=-1, keepdim=True) + 1e-5
        )
        feats = logmel.transpose(1, 2)             # (B, T, n_mels)
        return feats[0] if single else feats


class SpecAugment(nn.Module):
    """Frequency + time masking applied to (B, T, F) features, train-only."""

    def __init__(self, cfg: SpecAugmentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.freq_mask = torchaudio.transforms.FrequencyMasking(cfg.freq_mask_width, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(cfg.time_mask_width, iid_masks=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # torchaudio masking ops expect (B, F, T); convert from (B, T, F).
        x = feats.transpose(1, 2)
        if self.training:
            for _ in range(self.cfg.n_freq_masks):
                x = self.freq_mask(x)
            for _ in range(self.cfg.n_time_masks):
                x = self.time_mask(x)
        return x.transpose(1, 2)
