#!/usr/bin/env python3
"""Standalone checkpoint re-evaluation on the validation split (no training).

Loads one or more saved checkpoints and reports, for each:
  * CTC validation loss
  * greedy CER / WER over the FULL validation manifest (or a --val-max-utts slice)
  * optional sample reference/prediction pairs

Reuses the exact validation + decoding code paths from train.py
(same frontend, same val dataset construction, same greedy_decode /
edit_distance / word_error helpers), so numbers are directly comparable
with training-run reports.

Usage:
    python scripts/eval_val.py \
        --checkpoint checkpoints/clean_baseline/clean_full_overnight_best.pt \
        --checkpoint checkpoints/clean_baseline/clean_full_overnight_latest.pt \
        --tag val_full_reeval --eval-samples 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Config, PROJECT_ROOT
from dataset import ManifestDataset, CTCBatchCollator, load_vocab
from model import ASRModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import (  # noqa: E402  (reuse train.py helpers verbatim)
    edit_distance,
    evaluate_cer_wer,
    greedy_decode,
    ids_to_text,
    load_checkpoint,
    make_val_dataset,
    move_batch,
    validate,
    word_error,
)

MANIFESTS = PROJECT_ROOT / "manifests"
REPORTS = PROJECT_ROOT / "reports"


def build_val_loader(cfg: Config) -> tuple[DataLoader, dict, ManifestDataset]:
    """Val loader identical to train.py's build_loaders val path."""
    vocab = load_vocab(MANIFESTS / "vocab.json")
    cfg.model.vocab_size = int(vocab["vocab_size"])
    cfg.model.blank_id = int(vocab["blank_id"])

    if cfg.train.val_manifest:
        # Fixed validation manifest: the exact rows in the file, no filters,
        # no re-selection - identical to the training run's validation set.
        val_ds = make_val_dataset(cfg, vocab)
    else:
        val_ds = ManifestDataset(
            MANIFESTS / "val.csv",
            cfg,
            vocab["char_to_id"],
            augmenter=None,               # clean by construction
            max_utts=cfg.train.val_max_utts,
            max_utt_seconds=cfg.train.max_utt_seconds,
            order_seed=cfg.train.seed,
        )
    loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=CTCBatchCollator(),
    )
    return loader, vocab, val_ds


def main() -> int:
    parser = argparse.ArgumentParser(description="No-training checkpoint re-evaluation on val split.")
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="path to a .pt checkpoint (repeatable)")
    parser.add_argument("--val-max-utts", type=int, default=0,
                        help="cap on val utterances (0 = use the FULL val.csv)")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="fixed validation manifest CSV (e.g. manifests/val_2000.csv); "
                             "overrides val.csv + --val-max-utts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--tag", type=str, default="val_reeval",
                        help="report filename tag: reports/<tag>.json")
    parser.add_argument("--eval-samples", type=int, default=16)
    args = parser.parse_args()

    cfg = Config()
    cfg.train.val_max_utts = args.val_max_utts if args.val_max_utts > 0 else None
    cfg.train.val_manifest = args.val_manifest or None
    cfg.train.batch_size = args.batch_size
    device = torch.device(args.device)

    loader, vocab, val_ds = build_val_loader(cfg)
    id_to_char = {i: c for c, i in vocab["char_to_id"].items()}
    speakers = sorted({r["speaker_id"] for r in val_ds.rows})

    print("=" * 70)
    print("CHECKPOINT RE-EVALUATION (no training)")
    print("=" * 70)
    print(f"device            : {device}")
    print(f"val manifest      : {cfg.train.val_manifest or MANIFESTS / 'val.csv'}")
    print(f"val utterances    : {len(val_ds):,} of {sum(1 for _ in open(MANIFESTS / 'val.csv')) - 1:,} available")
    print(f"val hours         : {val_ds.total_hours:.2f}")
    print(f"val speakers      : {len(speakers)}")
    print(f"batch size        : {cfg.train.batch_size}")
    print(f"vocab size        : {cfg.model.vocab_size} (blank id {cfg.model.blank_id})")

    results = []
    for ckpt_path in args.checkpoint:
        print("-" * 70)
        print(f"checkpoint: {ckpt_path}")
        model = ASRModel(cfg.model).to(device)
        payload = load_checkpoint(Path(ckpt_path), model)
        saved_epoch = int(payload.get("epoch", -1))
        saved_val = float(payload.get("val_loss", float("nan")))
        saved_tag = payload.get("config", {}).get("train", {}).get("tag", "?")

        t0 = time.time()
        val_loss = validate(model, loader, device, cfg)
        t1 = time.time()
        metrics = evaluate_cer_wer(model, loader, device, id_to_char, cfg.model.blank_id)
        t2 = time.time()
        print(f"  saved epoch       : {saved_epoch} (run tag: {saved_tag})")
        print(f"  val_loss          : {val_loss:.4f}  (saved: {saved_val:.4f})")
        print(f"  greedy CER        : {metrics['cer']:.4f}")
        print(f"  greedy WER        : {metrics['wer']:.4f}")
        print(f"  timing            : loss {t1 - t0:.1f}s + decode {t2 - t1:.1f}s")

        samples = []
        if args.eval_samples > 0:
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    batch = move_batch(batch, device)
                    log_probs, out_lengths = model(batch["features"], batch["input_lengths"])
                    hyps = greedy_decode(log_probs, out_lengths, cfg.model.blank_id)
                    for j, hyp_ids in enumerate(hyps):
                        ref = "".join(
                            id_to_char.get(int(t), "")
                            for t in batch["targets"][j][: int(batch["target_lengths"][j])]
                        )
                        hyp = ids_to_text(hyp_ids, id_to_char)
                        samples.append({
                            "utterance_id": batch["utterance_ids"][j],
                            "reference": ref,
                            "prediction": hyp,
                            "cer": round(edit_distance(ref, hyp) / max(len(ref), 1), 4),
                        })
                        if len(samples) >= args.eval_samples:
                            break
                    if len(samples) >= args.eval_samples:
                        break

        results.append({
            "checkpoint": str(ckpt_path),
            "saved_epoch": saved_epoch,
            "saved_val_loss": round(saved_val, 4),
            "val_loss": round(val_loss, 4),
            "cer_greedy": round(metrics["cer"], 4),
            "wer_greedy": round(metrics["wer"], 4),
            "loss_seconds": round(t1 - t0, 1),
            "decode_seconds": round(t2 - t1, 1),
            "samples": samples,
        })

    report = {
        "tag": args.tag,
        "device": str(device),
        "val_max_utts": args.val_max_utts if args.val_max_utts > 0 else "full",
        "val_manifest": cfg.train.val_manifest or "manifests/val.csv",
        "val_utterances": len(val_ds),
        "val_hours": round(val_ds.total_hours, 2),
        "val_speakers": len(speakers),
        "batch_size": cfg.train.batch_size,
        "results": results,
    }
    REPORTS.mkdir(exist_ok=True)
    report_path = REPORTS / f"{args.tag}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 70)
    print("SUMMARY (full val set, greedy decoding)")
    print("=" * 70)
    print(f"{'checkpoint':<58} {'epoch':>5} {'val_loss':>9} {'CER':>8} {'WER':>8}")
    for r in results:
        name = Path(r["checkpoint"]).name
        print(f"{name:<58} {r['saved_epoch']:>5} {r['val_loss']:>9.4f} "
              f"{r['cer_greedy']:>8.4f} {r['wer_greedy']:>8.4f}")
    print(f"report   -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
