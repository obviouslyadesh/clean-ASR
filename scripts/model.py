#!/usr/bin/env python3
"""Week 3 Day 4-6: small CNN-BiLSTM + CTC baseline model.

Tensor shapes through the network (batch B, frames T, mels F=80, classes C):

    input          (B, 1, T, F)
    Conv2d xN      (B, ch[-1], T//2^N, F//2)   stride (2, 2), same padding
    permute/reshape (B, T//2^N, ch[-1] * F//2)
    Linear         (B, T//2^N, hidden_dim)
    BiLSTM xL      (B, T//2^N, 2 * rnn_hidden)
    Linear (out)   (B, T//2^N, C)              logits, not softmaxed
    CTC loss       F.ctc_loss expects (T, B, C) log-probs -> we transpose

With F=80, two stride-2 convs -> F' = 80/4 = 20 time-frequency channels per
frame; the CNN therefore also gives a 4x time downsample, which keeps the
LSTM sequences short enough for a fast pilot on MPS.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig


class ConvFrontend(nn.Module):
    def __init__(self, n_mels: int, channels: tuple[int, ...], kernel: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel, stride=(2, 2), padding=kernel // 2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.freq_out = n_mels // (2 ** len(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, T, F)
        return self.net(x)


class ASRModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = ConvFrontend(cfg.n_mels, cfg.cnn_channels, cfg.cnn_kernel)
        cnn_flat = cfg.cnn_channels[-1] * self.cnn.freq_out
        self.project = nn.Linear(cnn_flat, cfg.hidden_dim)
        self.rnn = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.rnn_dropout if cfg.rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(2 * cfg.rnn_hidden, cfg.vocab_size)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """features (B, T, F), lengths (B,) frame counts.

        Returns (log_probs (T', B, C), out_lengths (B,)) ready for ctc_loss.
        """
        x = features.unsqueeze(1)                          # (B, 1, T, F)
        x = self.cnn(x)                                     # (B, ch, T', F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)      # (B, T', ch*F')
        x = self.project(x)                                 # (B, T', hidden)
        x, _ = self.rnn(x)                                  # (B, T', 2*hidden)
        logits = self.classifier(x)                         # (B, T', C)
        log_probs = logits.log_softmax(dim=-1)
        out_lengths = lengths // (2 ** len(self.cfg.cnn_channels))
        return log_probs.transpose(0, 1), out_lengths       # (T', B, C), (B,)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
