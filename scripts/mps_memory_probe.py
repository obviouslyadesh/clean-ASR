#!/usr/bin/env python3
"""Week 4 Stage 3: MPS memory probe for candidate batch sizes.

Builds a worst-case batch (longest allowed utterances: 16 s -> ~1,601 frames)
for batch sizes 8/16/32, runs forward+backward on MPS, and reports allocated
and driver-level memory. Answers "does batch 32 fit in 24 GB unified memory?"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, vocab_size_from_manifest
from model import ASRModel

DEVICE = torch.device("mps")


def probe(batch_size: int, frames: int = 1601) -> dict[str, float]:
    torch.mps.empty_cache()
    cfg = Config()
    cfg.model.vocab_size = vocab_size_from_manifest()
    model = ASRModel(cfg.model).to(DEVICE)
    criterion = torch.nn.CTCLoss(blank=cfg.model.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    feats = torch.randn(batch_size, frames, cfg.model.n_mels, device=DEVICE)
    targets = torch.randint(1, cfg.model.vocab_size, (batch_size, 100), device=DEVICE)
    in_lens = torch.full((batch_size,), frames, dtype=torch.long, device=DEVICE)
    tgt_lens = torch.full((batch_size,), 100, dtype=torch.long, device=DEVICE)

    peak = 0
    for _ in range(3):  # a few steps to reach steady state
        log_probs, out_lens = model(feats, in_lens)
        loss = criterion(log_probs, targets, out_lens, tgt_lens)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        peak = max(peak, torch.mps.current_allocated_memory())

    driver_bytes = 0
    try:
        from torch.mps import recommended_max_memory  # availability varies

        driver_bytes = recommended_max_memory()
    except Exception:
        pass

    out = {
        "batch_size": batch_size,
        "frames": frames,
        "allocated_current_mb": round(torch.mps.current_allocated_memory() / 2**20, 1),
        "allocated_peak_mb": round(peak / 2**20, 1),
        "driver_limit_mb": round(driver_bytes / 2**20, 1) if driver_bytes else -1,
    }
    del model, optimizer, feats, targets, log_probs, loss
    torch.mps.empty_cache()
    return out


def main() -> int:
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return 1
    print(f"torch {torch.__version__}, device: {DEVICE}")
    print("worst case: 16 s utterances -> 1,601 frames per sample\n")
    results = []
    for bs in (8, 16, 32):
        try:
            r = probe(bs)
            results.append(r)
            print(
                f"bs={bs:>2}: allocated now {r['allocated_current_mb']:>8.1f} MB, "
                f"peak {r['allocated_peak_mb']:>8.1f} MB, "
                f"driver limit {r['driver_limit_mb']:>8.1f} MB"
            )
        except RuntimeError as exc:
            results.append({"batch_size": bs, "error": str(exc)})
            print(f"bs={bs:>2}: RuntimeError: {exc}")
    import json

    Path("reports").mkdir(exist_ok=True)
    Path("reports/mps_memory_probe.json").write_text(json.dumps(results, indent=2))
    print("\nwritten: reports/mps_memory_probe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
