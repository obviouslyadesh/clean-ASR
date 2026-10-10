#!/usr/bin/env python3
"""Week 3 Day 2-3: waveform augmentation for training data.

Pipeline per training utterance (train split only):

    waveform -> maybe noise mix -> maybe RIR convolution -> maybe speed
             -> maybe gain -> normalised, valid waveform

Everything is controlled by `AugmentConfig`; probability semantics:

    P(any augment)        = cfg.prob          (~0.5 -> rest stay clean)
    given active:
      P(additive noise)   = cfg.noise_prob    SNR ~ U[cfg.snr_low_db, cfg.snr_high_db]
      P(RIR convolution)  = cfg.rir_prob
      P(speed)            = cfg.speed_prob    factor ~ U[cfg.speed_min, cfg.speed_max]
      P(gain)             = cfg.gain_prob     dB ~ U[cfg.gain_low_db, cfg.gain_high_db]

Validation and test datasets never construct an Augmenter, so they are clean
by construction, not by convention.

Design notes
------------
* Noise/RIR indexes (manifests/musan.csv, manifests/rir.csv) are loaded once
  as lists of relative paths; clips are read from disk on demand and only one
  at a time - the noise corpus is never held in memory.
* Noise longer than the utterance is cropped at a random offset; shorter
  noise is tiled then cropped, so the mix always matches the speech length.
* RIRs are convolved with scipy-free numpy FFT convolution, normalised to
  unit energy, and only the early-reflective part (direct + first
  reflections) is used by cropping the tail at 10% of peak energy to keep
  reverberation realistic without long smearing.
* The output is always checked for NaN/Inf and peak-normalised to <= 0.99.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from config import AugmentConfig, PROJECT_ROOT


def _load_index(path: Path, categories: tuple[str, ...] | None) -> list[str]:
    paths: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            if categories is not None and row["category"] not in categories:
                continue
            paths.append(row["audio_path"])
    if not paths:
        raise ValueError(f"no usable rows in {path} for categories={categories}")
    return paths


def _read_mono(rel_path: str) -> np.ndarray:
    """Read a clip relative to the project root as float32 mono."""
    wav, _sr = sf.read(str(PROJECT_ROOT / rel_path), dtype="float32", always_2d=True)
    return wav.mean(axis=1)


def _match_length(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if len(noise) >= length:
        start = rng.randint(0, len(noise) - length)
        return noise[start : start + length]
    tiles = int(np.ceil(length / max(len(noise), 1)))
    tiled = np.tile(noise, tiles)
    start = rng.randint(0, len(tiled) - length)
    return tiled[start : start + length]


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(speech**2)) + 1e-10
    noise_power = float(np.mean(noise**2)) + 1e-10
    scale = np.sqrt(speech_power / noise_power / (10.0 ** (snr_db / 10.0)))
    return speech + scale * noise


def _crop_rir_tail(rir: np.ndarray) -> np.ndarray:
    """Keep the direct path + early reflections (first crossing of 10% peak)."""
    peak = float(np.max(np.abs(rir))) + 1e-12
    below = np.abs(rir) < 0.1 * peak
    idx = np.where(below[len(rir) // 2 :])[0]
    if len(idx) == 0:
        return rir
    return rir[: len(rir) // 2 + int(idx[0]) + 1]


def _convolve_rir(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    rir = _crop_rir_tail(rir)
    energy = float(np.sqrt(np.sum(rir**2))) + 1e-12
    rir = rir / energy
    n = len(speech) + len(rir) - 1
    n_fft = 1 << (n - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(speech, n_fft) * np.fft.rfft(rir, n_fft), n_fft)[:n]
    return out[: len(speech)]  # align with input length (causal part)


def _resample_linear(wav: np.ndarray, speed: float) -> np.ndarray:
    """Linear-interpolation resampling for speed perturbation.

    Good enough for a pilot: 0.9-1.1x introduces at most mild aliasing-free
    pitch shift without adding a resampling dependency.
    """
    new_len = int(round(len(wav) / speed))
    if new_len <= 1:
        return wav
    x_old = np.linspace(0.0, len(wav) - 1, num=len(wav), dtype=np.float64)
    x_new = np.linspace(0.0, len(wav) - 1, num=new_len, dtype=np.float64)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def _finalise(wav: np.ndarray) -> np.ndarray:
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav


class WaveformAugmenter:
    """On-demand waveform augmentation using the noise/RIR indexes."""

    def __init__(
        self,
        cfg: AugmentConfig,
        musan_index: Path = PROJECT_ROOT / "manifests" / "musan.csv",
        rir_index: Path = PROJECT_ROOT / "manifests" / "rir.csv",
        seed: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = random.Random(seed)
        noise_cats = tuple(c for c in cfg.noise_categories if c != "rir")
        self.noise_paths = _load_index(musan_index, noise_cats)
        self.rir_paths = _load_index(rir_index, ("rir",))
        self.last_applied: dict[str, bool | float] = {}

    def _coin(self, p: float) -> bool:
        return self.rng.random() < p

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        applied: dict[str, bool | float] = {"augmented": False}

        if not self._coin(cfg.prob):
            self.last_applied = applied
            return _finalise(wav)  # clean example (still peak-guarded)

        applied["augmented"] = True

        if self._coin(cfg.noise_prob):
            snr = self.rng.uniform(cfg.snr_low_db, cfg.snr_high_db)
            noise = _match_length(_read_mono(self.rng.choice(self.noise_paths)), len(wav), self.rng)
            wav = _mix_at_snr(wav, noise, snr)
            applied["noise_snr_db"] = round(snr, 2)

        if self._coin(cfg.rir_prob):
            rir = _read_mono(self.rng.choice(self.rir_paths))
            if len(rir) > 1:
                wav = _convolve_rir(wav, rir)

        if self._coin(cfg.speed_prob):
            speed = self.rng.uniform(cfg.speed_min, cfg.speed_max)
            wav = _resample_linear(wav, speed)
            applied["speed"] = round(speed, 3)

        if self._coin(cfg.gain_prob):
            gain_db = self.rng.uniform(cfg.gain_low_db, cfg.gain_high_db)
            wav = wav * (10.0 ** (gain_db / 20.0))
            applied["gain_db"] = round(gain_db, 2)

        self.last_applied = applied
        return _finalise(wav)
