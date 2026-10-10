#!/usr/bin/env python3
"""Week 3: training/augmentation configuration (single source of truth).

Loaded by the augmentation, dataset, model and training modules. Values are
defaults; every knob can be overridden from the `train.py` CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 400          # 25 ms window at 16 kHz
    hop_length: int = 160     # 10 ms frame shift
    win_length: int = 400
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0     # nyquist; log10f energy normalisation applied after


@dataclass
class AugmentConfig:
    # Probability that a training utterance receives ANY waveform augmentation.
    # ~0.5 leaves the other half of training batches clean.
    prob: float = 0.5
    # Probability of each augmentation *given* that augmentation is active.
    noise_prob: float = 0.8
    rir_prob: float = 0.4
    speed_prob: float = 0.5
    gain_prob: float = 0.5
    # Additive noise: SNR drawn uniformly (dB) - configurable, not hardcoded.
    snr_low_db: float = 0.0
    snr_high_db: float = 20.0
    # Speed perturbation factor drawn uniformly.
    speed_min: float = 0.9
    speed_max: float = 1.1
    # Gain perturbation in dB, drawn uniformly.
    gain_low_db: float = -6.0
    gain_high_db: float = 6.0
    # Which MUSAN categories to sample noise from (babble = speech subset).
    noise_categories: tuple[str, ...] = ("noise", "music", "speech")
    # Independent pitch shifting is intentionally NOT implemented: the reliable
    # options (librosa/sox) are not in this environment and would add a
    # dependency + significant CPU cost for marginal pilot value. Speed
    # perturbation already shifts pitch proportionally.
    pitch_enabled: bool = False


@dataclass
class SpecAugmentConfig:
    enabled: bool = True
    n_freq_masks: int = 2
    freq_mask_width: int = 20    # max mask width in mel bins
    n_time_masks: int = 2
    time_mask_width: int = 60    # max mask width in frames (100 ms x 60 = 0.6 s)


@dataclass
class ModelConfig:
    n_mels: int = 80
    cnn_channels: tuple[int, ...] = (32, 64)
    cnn_kernel: int = 3
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.1
    hidden_dim: int = 256
    vocab_size: int = 91         # overridden from manifests/vocab.json at load
    blank_id: int = 0


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 16
    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    num_workers: int = 4
    subset_hours: float = 8.0    # pilot size cap; None/0 = full train split
    device: str = "mps"
    checkpoint_dir: str = "checkpoints/pilot"
    val_max_utts: int | None = 250  # cap validation cost for the pilot; 0/None = full val.csv
    # Optional fixed validation manifest (relative to PROJECT_ROOT). When set,
    # it overrides val.csv + val_max_utts entirely: the exact rows in the file
    # are used, with no re-shuffling or re-selection. Used for the final run's
    # deterministic 2,000-utterance validation set (manifests/val_2000.csv).
    val_manifest: str | None = None
    max_utt_seconds: float = 16.0  # skip absurdly long utts in the pilot subset
    zero_infinity: bool = True
    # --- Week 6 hyperparameter-screening knobs (defaults = clean baseline) ---
    scheduler: str = "none"          # none | plateau | cosine
    sched_factor: float = 0.5        # ReduceLROnPlateau factor
    sched_patience: int = 2          # ReduceLROnPlateau patience (epochs)
    min_lr: float = 1e-6             # scheduler floor
    early_stop_patience: int = 0     # epochs without val-loss improvement before stopping (0 = off)
    max_seconds: float = 0.0         # wall-clock cap per run in seconds (0 = off)


@dataclass
class Config:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        for section, values in d.items():
            target = getattr(cfg, section)
            for key, val in values.items():
                if hasattr(target, key):
                    cur = getattr(target, key)
                    if isinstance(cur, tuple) and not isinstance(val, tuple):
                        val = tuple(val)
                    setattr(target, key, val)
        return cfg

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def vocab_size_from_manifest(path: Path = PROJECT_ROOT / "manifests" / "vocab.json") -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return int(data["vocab_size"])
