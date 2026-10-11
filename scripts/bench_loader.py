#!/usr/bin/env python3
"""Week 4 Stage 4: loader-only benchmark.

Iterates the training DataLoader (FLAC decode -> waveform augmentation ->
log-Mel -> collate) WITHOUT the model, so loader throughput can be compared
with end-to-end training throughput. If loader-only s/batch is well below the
training s/batch (~0.11 s/batch on MPS), the CPU pipeline is not the
bottleneck; if comparable or higher, MPS is waiting on data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from torch.utils.data import DataLoader

from config import Config, PROJECT_ROOT
from dataset import ManifestDataset, CTCBatchCollator, load_vocab
from augment import WaveformAugmenter

MANIFESTS = PROJECT_ROOT / "manifests"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-hours", type=float, default=16.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 6])
    args = parser.parse_args()

    cfg = Config()
    cfg.train.subset_hours = args.subset_hours
    vocab = load_vocab(MANIFESTS / "vocab.json")
    results = []

    for w in args.workers:
        augmenter = WaveformAugmenter(cfg.augment, seed=cfg.train.seed)
        ds = ManifestDataset(
            MANIFESTS / "train.csv",
            cfg,
            vocab["char_to_id"],
            augmenter=augmenter,
            subset_hours=args.subset_hours,
            max_utt_seconds=cfg.train.max_utt_seconds,
            order_seed=cfg.train.seed,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,          # measure pure pipeline speed, no reshuffle cost
            num_workers=w,
            collate_fn=CTCBatchCollator(),
            persistent_workers=w > 0,
            prefetch_factor=2 if w > 0 else None,
        )
        t0 = time.time()
        n = 0
        for batch in loader:
            n += 1
        dt = time.time() - t0
        r = {
            "workers": w,
            "batches": n,
            "seconds": round(dt, 1),
            "s_per_batch": round(dt / n, 4),
            "batches_per_s": round(n / dt, 1),
            "utt_per_s": round(len(ds) / dt, 1),
            "augment_prob": cfg.augment.prob,
        }
        results.append(r)
        print(
            f"workers={w}: {n} batches in {dt:.1f}s -> "
            f"{r['s_per_batch']:.4f} s/batch ({r['utt_per_s']:.0f} utt/s loader-only)"
        )
        del loader, ds, augmenter

    Path("reports").mkdir(exist_ok=True)
    Path("reports/loader_bench.json").write_text(json.dumps(results, indent=2))
    print("written: reports/loader_bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
