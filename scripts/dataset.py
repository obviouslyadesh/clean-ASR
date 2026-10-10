#!/usr/bin/env python3
"""Week 3: Dataset / DataLoader built on the Week 2 splits.

    train.csv -> waveform -> waveform augmentation -> log-Mel -> SpecAugment -> model
    val.csv   -> waveform -> log-Mel -> model            (no augmentation)

* Validation/test datasets are constructed without an Augmenter and without
  SpecAugment in train mode, so they are clean by construction.
* Files are decoded with soundfile at __getitem__ time (FLAC -> float32 mono);
  only one waveform per worker is in memory at a time.
* The pilot subset is drawn uniformly at random (seeded) up to a duration
  budget, then sorted by duration so batches have low padding waste.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from augment import WaveformAugmenter
from config import Config, PROJECT_ROOT

DATA_ROOT = PROJECT_ROOT / "asr_nepali_data"


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_durations(validation_csv: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with validation_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                durations[r["utterance_id"]] = float(r["duration_seconds"])
            except (KeyError, ValueError):
                pass
    return durations


class ManifestDataset(Dataset):
    """Utterance-level dataset over a manifest CSV."""

    def __init__(
        self,
        manifest: Path,
        cfg: Config,
        char_to_id: dict[str, int],
        augmenter: WaveformAugmenter | None = None,
        max_utts: int | None = None,
        subset_hours: float | None = None,
        max_utt_seconds: float | None = None,
        order_seed: int | None = None,
        validation_csv: Path = PROJECT_ROOT / "manifests" / "audio_validation.csv",
    ) -> None:
        rows = load_manifest_rows(manifest)
        durations = load_durations(validation_csv)

        if max_utt_seconds is not None:
            keep_ids = {
                r["utterance_id"]
                for r in rows
                if durations.get(r["utterance_id"], 0.0) <= max_utt_seconds
            }
            rows = [r for r in rows if r["utterance_id"] in keep_ids]

        # Random (seeded) window into the corpus, honouring the duration cap.
        if order_seed is not None:
            rng = random.Random(order_seed)
            rng.shuffle(rows)

        if subset_hours is not None and subset_hours > 0:
            budget = subset_hours * 3600.0
            kept: list[dict[str, str]] = []
            total = 0.0
            for row in rows:
                kept.append(row)
                total += durations.get(row["utterance_id"], 0.0)
                if total >= budget:
                    break
            rows = kept

        if max_utts is not None:
            rows = rows[:max_utts]

        # Duration grouping for efficient batching (padding minimisation).
        rows.sort(key=lambda r: durations.get(r["utterance_id"], 0.0))

        self.rows = rows
        self.cfg = cfg
        self.char_to_id = char_to_id
        self.augmenter = augmenter
        self.durations = durations
        self.total_hours = sum(durations.get(r["utterance_id"], 0.0) for r in rows) / 3600.0
        self._frontend = None  # lazily built per DataLoader worker

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, transcript: str) -> list[int]:
        ids = [self.char_to_id.get(ch, -1) for ch in transcript]
        return [i for i in ids if i > 0]  # blank is id 0; never a CTC target

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        wav, sr = sf.read(str(DATA_ROOT / row["audio_path"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == self.cfg.features.sample_rate, f"unexpected sample rate {sr}"

        applied = {"augmented": False}
        if self.augmenter is not None:
            wav = self.augmenter(wav)
            applied = dict(self.augmenter.last_applied)

        if self._frontend is None:
            from frontend import LogMelFrontend

            self._frontend = LogMelFrontend(self.cfg.features)
        feats = self._frontend(torch.from_numpy(wav))  # (T, n_mels)

        target = self._encode(row["transcript"])
        if not target:  # degenerate guard: corpus has text on every row
            target = [1]

        return {
            "features": feats,
            "targets": torch.tensor(target, dtype=torch.long),
            "input_lengths": feats.shape[0],
            "target_lengths": len(target),
            "utterance_id": row["utterance_id"],
            "augmented": bool(applied.get("augmented", False)),
        }


class CTCBatchCollator:
    """Pads a batch and produces the exact tensors ctc_loss needs."""

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        features = [b["features"] for b in batch]
        targets = [b["targets"] for b in batch]
        input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        max_t = max(f.shape[0] for f in features)
        max_u = int(max(len(t) for t in targets))

        feat_dim = features[0].shape[1]
        feat_pad = torch.zeros(len(batch), max_t, feat_dim)
        for i, f in enumerate(features):
            feat_pad[i, : f.shape[0]] = f

        tgt_pad = torch.ones(len(batch), max_u, dtype=torch.long)  # pad with non-blank id 1
        for i, t in enumerate(targets):
            tgt_pad[i, : len(t)] = t

        return {
            "features": feat_pad,
            "targets": tgt_pad,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
            "utterance_ids": [b["utterance_id"] for b in batch],
            "augmented": torch.tensor([bool(b["augmented"]) for b in batch], dtype=torch.bool),
        }


def load_vocab(path: Path = PROJECT_ROOT / "manifests" / "vocab.json") -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
