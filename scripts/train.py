#!/usr/bin/env python3
"""Week 3 Day 6-7: pilot training loop for the CNN-BiLSTM-CTC model.

Responsibilities:
    * build the pilot train subset and a clean validation set from the splits
    * run training epochs with augmentation + SpecAugment (train only)
    * run a clean validation loop (no augmentation, no SpecAugment)
    * checkpoint (model, optimizer, epoch, losses, config) and reload-verify
    * greedy-decode CER/WER on a small validation slice at the end
    * write reports/pilot_metrics.json

Usage:
    python scripts/train.py                       # pilot: 8 h, 3 epochs
    python scripts/train.py --epochs 1 --subset-hours 1   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
import numpy as np
from pathlib import Path
import soundfile
import torchaudio

import torch
from torch.utils.data import DataLoader

from config import Config, PROJECT_ROOT, vocab_size_from_manifest
from dataset import ManifestDataset, CTCBatchCollator, load_vocab
from augment import WaveformAugmenter
from frontend import LogMelFrontend, SpecAugment
from model import ASRModel

MANIFESTS = PROJECT_ROOT / "manifests"
REPORTS = PROJECT_ROOT / "reports"


def ids_to_text(ids: list[int], id_to_char: dict[int, str]) -> str:
    return "".join(id_to_char.get(int(i), "") for i in ids)


def greedy_decode(log_probs: torch.Tensor, out_lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """log_probs (T, B, C) -> list of B collapsed id sequences."""
    best = log_probs.argmax(dim=-1).transpose(0, 1)  # (B, T)
    sequences: list[list[int]] = []
    for row, length in zip(best, out_lengths):
        prev = blank
        seq: list[int] = []
        for t in row[: int(length)]:
            token = int(t)
            if token != prev and token != blank:
                seq.append(token)
            prev = token
        sequences.append(seq)
    return sequences


def edit_distance(ref: str, hyp: str) -> int:
    """Character-level Levenshtein distance."""
    if ref == hyp:
        return 0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def word_error(ref: str, hyp: str) -> int:
    """Word-level Levenshtein distance (insert/delete/substitute words)."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if ref_words == hyp_words:
        return 0
    prev = list(range(len(hyp_words) + 1))
    for i, r in enumerate(ref_words, 1):
        cur = [i]
        for j, h in enumerate(hyp_words, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def move_batch(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ("features", "targets", "input_lengths", "target_lengths"):
        out[key] = batch[key].to(device)
    return out


def make_val_dataset(cfg: Config, vocab: dict) -> ManifestDataset:
    """Validation dataset - always clean (no augmenter, no SpecAugment).

    When `cfg.train.val_manifest` is set, the exact rows of that fixed
    manifest are used (no re-shuffling / re-selection), so every run and
    every re-evaluation sees the identical validation utterances.
    Otherwise the original seeded slice of val.csv is built as before.
    """
    if cfg.train.val_manifest:
        return ManifestDataset(
            PROJECT_ROOT / cfg.train.val_manifest,
            cfg,
            vocab["char_to_id"],
            augmenter=None,               # clean by construction
        )
    return ManifestDataset(
        MANIFESTS / "val.csv",
        cfg,
        vocab["char_to_id"],
        augmenter=None,                   # clean by construction
        max_utts=cfg.train.val_max_utts or None,  # 0/None = full val.csv
        order_seed=cfg.train.seed,
    )


def build_loaders(cfg: Config, device: torch.device) -> tuple[DataLoader, DataLoader, dict, list[dict]]:
    vocab = load_vocab(MANIFESTS / "vocab.json")
    cfg.model.vocab_size = int(vocab["vocab_size"])
    cfg.model.blank_id = int(vocab["blank_id"])

    augmenter = (
        WaveformAugmenter(cfg.augment, seed=cfg.train.seed)
        if cfg.augment.prob > 0
        else None                 # clean baseline: no waveform augmentation at all
    )
    train_ds = ManifestDataset(
        MANIFESTS / "train.csv",
        cfg,
        vocab["char_to_id"],
        augmenter=augmenter,
        subset_hours=cfg.train.subset_hours,
        max_utt_seconds=cfg.train.max_utt_seconds,
        order_seed=cfg.train.seed,
    )
    val_ds = make_val_dataset(cfg, vocab)
    collator = CTCBatchCollator()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collator,
        pin_memory=False,
        persistent_workers=cfg.train.num_workers > 0,
        prefetch_factor=2 if cfg.train.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collator,
    )
    return train_loader, val_loader, vocab, train_ds.rows


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config):
    """Return the configured LR scheduler, or None for a flat LR (clean baseline)."""
    if cfg.train.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.train.sched_factor,
            patience=cfg.train.sched_patience,
            min_lr=cfg.train.min_lr,
        )
    if cfg.train.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.min_lr
        )
    return None


def save_checkpoint(
    path: Path,
    model: ASRModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    cfg: Config,
    extra: dict | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})
    if "rng_state" not in extra:
        rng_state = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        try:
            if torch.backends.mps.is_available():
                rng_state["torch_mps"] = torch.mps.get_rng_state()
        except Exception:  # noqa: BLE001 - RNG state capture is best-effort
            pass
        extra["rng_state"] = rng_state
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": cfg.to_dict(),
        "extra": extra,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic: an interrupted write never corrupts the previous file


def load_checkpoint(path: Path, model: ASRModel, optimizer: torch.optim.Optimizer | None = None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def validate(model: ASRModel, loader: DataLoader, device: torch.device, cfg: Config) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0
    criterion = torch.nn.CTCLoss(blank=cfg.model.blank_id, zero_infinity=True)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            log_probs, out_lengths = model(batch["features"], batch["input_lengths"])
            loss = criterion(
                log_probs, batch["targets"], out_lengths, batch["target_lengths"]
            )
            n = int(batch["target_lengths"].sum())
            total_loss += float(loss) * n
            total_weight += n
    return total_loss / max(total_weight, 1)


def validate_with_metrics(
    model: ASRModel,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    id_to_char: dict[int, str],
) -> dict[str, float]:
    """One clean validation pass returning CTC loss AND corpus-level greedy
    CER/WER in a single forward pass (edit errors and reference lengths are
    aggregated across the corpus, never averaged per utterance).

    Used at the end of every epoch: validation CER is the primary
    model-selection metric; val_loss and WER are recorded alongside it.
    """
    model.eval()
    total_loss = 0.0
    total_weight = 0
    cer_num = cer_den = wer_num = wer_den = 0
    criterion = torch.nn.CTCLoss(blank=cfg.model.blank_id, zero_infinity=True)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            log_probs, out_lengths = model(batch["features"], batch["input_lengths"])
            loss = criterion(log_probs, batch["targets"], out_lengths, batch["target_lengths"])
            n = int(batch["target_lengths"].sum())
            total_loss += float(loss) * n
            total_weight += n
            hyps = greedy_decode(log_probs, out_lengths, cfg.model.blank_id)
            for j, hyp_ids in enumerate(hyps):
                ref = "".join(
                    id_to_char.get(int(t), "")
                    for t in batch["targets"][j][: int(batch["target_lengths"][j])]
                )
                hyp = ids_to_text(hyp_ids, id_to_char)
                cer_num += edit_distance(ref, hyp)
                cer_den += max(len(ref), 1)
                wer_num += word_error(ref, hyp)
                wer_den += max(len(ref.split()), 1)
    return {
        "val_loss": total_loss / max(total_weight, 1),
        "cer": cer_num / max(cer_den, 1),
        "wer": wer_num / max(wer_den, 1),
    }


def evaluate_cer_wer(
    model: ASRModel,
    loader: DataLoader,
    device: torch.device,
    id_to_char: dict[int, str],
    blank: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    cer_num = cer_den = wer_num = wer_den = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = move_batch(batch, device)
            log_probs, out_lengths = model(batch["features"], batch["input_lengths"])
            hyps = greedy_decode(log_probs, out_lengths, blank)
            for j, hyp_ids in enumerate(hyps):
                ref = "".join(
                    id_to_char.get(int(t), "") for t in batch["targets"][j][: int(batch["target_lengths"][j])]
                )
                hyp = ids_to_text(hyp_ids, id_to_char)
                cer_num += edit_distance(ref, hyp)
                cer_den += max(len(ref), 1)
                wer_num += word_error(ref, hyp)
                wer_den += max(len(ref.split()), 1)
    return {"cer": cer_num / max(cer_den, 1), "wer": wer_num / max(wer_den, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 3 pilot training run.")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file loaded as the base config (CLI flags override)")
    parser.add_argument("--epochs", type=int, default=None, help="override cfg epochs")
    parser.add_argument("--subset-hours", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="AdamW weight decay (default = config value)",
    )
    parser.add_argument(
        "--scheduler", type=str, default=None, choices=["none", "plateau", "cosine"],
        help="LR scheduler: none (flat, baseline) | plateau | cosine",
    )
    parser.add_argument("--sched-factor", type=float, default=None,
                        help="ReduceLROnPlateau factor")
    parser.add_argument("--sched-patience", type=int, default=None,
                        help="ReduceLROnPlateau patience in epochs")
    parser.add_argument("--min-lr", type=float, default=None, help="scheduler LR floor")
    parser.add_argument(
        "--patience", type=int, default=None,
        help="early stopping: stop after N epochs without validation-CER improvement (0 = off)",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None,
        help="wall-clock cap: stop after the first epoch that pushes elapsed past this (0 = off)",
    )
    parser.add_argument("--val-max-utts", type=int, default=None,
                        help="cap val utterances (<=0 = full val.csv)")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="fixed validation manifest CSV (e.g. manifests/val_2000.csv); "
                             "overrides val.csv + --val-max-utts")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tag", type=str, default="pilot", help="checkpoint filename tag")
    parser.add_argument(
        "--augment",
        action="store_true",
        default=None,
        help="enable waveform augmentation (default = config value)",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="disable waveform augmentation (clean baseline)",
    )
    parser.add_argument(
        "--spec-augment",
        action="store_true",
        default=None,
        help="enable SpecAugment (default = config value)",
    )
    parser.add_argument(
        "--no-spec-augment",
        dest="spec_augment",
        action="store_false",
        help="disable SpecAugment (clean baseline)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        metavar="N",
        help="print N reference/prediction sample pairs after training",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <checkpoint-dir>/<tag>_latest.pt; --epochs is the target epoch count",
    )
    args = parser.parse_args()

    cfg = Config.load(Path(args.config)) if args.config else Config()
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.subset_hours is not None:
        cfg.train.subset_hours = args.subset_hours
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.scheduler is not None:
        cfg.train.scheduler = args.scheduler
    if args.sched_factor is not None:
        cfg.train.sched_factor = args.sched_factor
    if args.sched_patience is not None:
        cfg.train.sched_patience = args.sched_patience
    if args.min_lr is not None:
        cfg.train.min_lr = args.min_lr
    if args.patience is not None:
        cfg.train.early_stop_patience = args.patience
    if args.max_seconds is not None:
        cfg.train.max_seconds = args.max_seconds
    if args.val_max_utts is not None:
        cfg.train.val_max_utts = args.val_max_utts if args.val_max_utts > 0 else None
    if args.val_manifest is not None:
        cfg.train.val_manifest = args.val_manifest or None
    if args.checkpoint_dir is not None:
        cfg.train.checkpoint_dir = args.checkpoint_dir
    if args.device is not None:
        cfg.train.device = args.device
    if args.augment is not None:
        cfg.augment.prob = 0.0 if not args.augment else cfg.augment.prob
    if args.spec_augment is not None:
        cfg.spec_augment.enabled = args.spec_augment

    torch.manual_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)
    if device.type == "mps":
        assert torch.backends.mps.is_available(), "MPS requested but not available"

    train_loader, val_loader, vocab, train_rows = build_loaders(cfg, device)
    id_to_char = {i: c for c, i in vocab["char_to_id"].items()}
    print("=" * 70)
    print("WEEK 3 PILOT TRAINING")
    print("=" * 70)
    print(f"device            : {device} (mps available: {torch.backends.mps.is_available()})")
    print(f"train utterances  : {len(train_loader.dataset):,} ({train_loader.dataset.total_hours:.2f} h)")
    print(f"val utterances    : {len(val_loader.dataset):,} "
          f"(manifest {cfg.train.val_manifest or 'manifests/val.csv seeded slice'})")
    print(f"selection metric  : validation CER (corpus-level greedy), primary")
    print(f"batch size        : {cfg.train.batch_size}")
    print(f"epochs            : {cfg.train.epochs}")
    print(f"lr                : {cfg.train.lr}")
    print(f"weight_decay      : {cfg.train.weight_decay}")
    print(f"scheduler         : {cfg.train.scheduler}")
    print(f"early stop        : patience {cfg.train.early_stop_patience}")
    print(f"wall-clock cap    : {cfg.train.max_seconds or 'off'}")
    print(f"num_workers       : {cfg.train.num_workers}")
    print(f"waveform augment  : {'ON (p=%.2f)' % cfg.augment.prob if cfg.augment.prob > 0 else 'OFF (clean)'}")
    print(f"SpecAugment       : {'ON' if cfg.spec_augment.enabled else 'OFF (clean)'}")
    print(f"vocab size        : {cfg.model.vocab_size} (blank id {cfg.model.blank_id})")

    model = ASRModel(cfg.model).to(device)
    print(f"model parameters  : {model.num_parameters():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = torch.nn.CTCLoss(blank=cfg.model.blank_id, zero_infinity=cfg.train.zero_infinity)
    spec_augment = SpecAugment(cfg.spec_augment).to(device) if cfg.spec_augment.enabled else None

    ckpt_dir = PROJECT_ROOT / cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{args.tag}_best.pt"
    latest_path = ckpt_dir / f"{args.tag}_latest.pt"
    epochs_path = ckpt_dir / f"{args.tag}_epochs.jsonl"   # per-epoch metrics, appended live
    cfg.save(ckpt_dir / f"{args.tag}_config.json")        # resolved config saved up front

    history: list[dict] = []
    lr_history: list[float] = []
    scheduler_reductions = 0
    best_val = float("inf")    # val_loss at the best epoch
    best_cer = float("inf")    # primary model-selection metric (corpus-level greedy CER)
    best_wer = float("inf")
    best_epoch = 0
    no_improve = 0
    start_epoch = 1
    if args.resume and latest_path.is_file():
        payload = load_checkpoint(latest_path, model, optimizer)
        start_epoch = int(payload["epoch"]) + 1
        extra = payload.get("extra", {})
        history = list(extra.get("history", []))
        lr_history = list(extra.get("lr_history", []))
        scheduler_reductions = int(extra.get("scheduler_reductions", 0))
        if scheduler is not None and payload.get("scheduler_state"):
            scheduler.load_state_dict(payload["scheduler_state"])
        if history:
            # Model selection: validation CER (primary). Checkpoints that predate
            # per-epoch CER recording fall back to val_loss.
            sel_key = "val_cer" if all("val_cer" in h for h in history) else "val_loss"
            best_entry = min(history, key=lambda h: h[sel_key])
            best_val = float(best_entry["val_loss"])
            best_cer = float(best_entry.get("val_cer", float("inf")))
            best_wer = float(best_entry.get("val_wer", float("inf")))
            best_epoch = int(best_entry["epoch"])
            no_improve = int(history[-1]["epoch"]) - best_epoch
        rng_state = extra.get("rng_state")
        if rng_state:
            try:
                torch.set_rng_state(rng_state["torch"])
                np.random.set_state(rng_state["numpy"])
                random.setstate(rng_state["python"])
                if "torch_mps" in rng_state:
                    torch.mps.set_rng_state(rng_state["torch_mps"])
            except Exception as exc:  # noqa: BLE001 - never abort a resume over RNG state
                print(f"rng restore skipped: {exc}")
        print(
            f"resumed: epoch {payload['epoch']} done, best val {best_val:.4f} @ ep {best_epoch}; "
            f"target epochs {cfg.train.epochs}"
        )
    epoch = start_epoch - 1
    early_stopped = False
    wall_clock_cap_hit = False
    t_start = time.time()

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_weight = 0
        n_batches = 0
        t_epoch = time.time()
        n_total_batches = len(train_loader)
        n_aug_total = 0
        n_samples_total = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            if spec_augment is not None:
                batch["features"] = spec_augment(batch["features"])  # train-mode masking
            log_probs, out_lengths = model(batch["features"], batch["input_lengths"])
            loss = criterion(log_probs, batch["targets"], out_lengths, batch["target_lengths"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            n = int(batch["target_lengths"].sum())
            epoch_loss += float(loss.detach()) * n
            epoch_weight += n
            n_batches += 1
            n_aug_total += int(batch["augmented"].sum())
            n_samples_total += len(batch["augmented"])

            if n_batches % 500 == 0:
                print(
                    f"  epoch {epoch}: batch {n_batches}/{n_total_batches}  "
                    f"running_loss {epoch_loss / max(epoch_weight, 1):.4f}  "
                    f"aug_frac {n_aug_total / max(n_samples_total, 1):.2f}  "
                    f"({time.time() - t_epoch:.0f}s)",
                    flush=True,
                )

        train_loss = epoch_loss / max(epoch_weight, 1)
        val_metrics = validate_with_metrics(model, val_loader, device, cfg, id_to_char)
        val_loss = val_metrics["val_loss"]
        val_cer = val_metrics["cer"]
        val_wer = val_metrics["wer"]
        aug_frac = n_aug_total / max(n_samples_total, 1)
        dt = time.time() - t_epoch

        improved = val_cer < best_cer - 1e-6   # primary selection: validation CER
        if improved:
            best_cer = val_cer
            best_wer = val_wer
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            prev_lr = optimizer.param_groups[0]["lr"]
            if cfg.train.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
            if optimizer.param_groups[0]["lr"] < prev_lr:
                scheduler_reductions += 1
        lr_now = float(optimizer.param_groups[0]["lr"])
        lr_history.append(lr_now)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_cer": round(val_cer, 6),
                "val_wer": round(val_wer, 6),
                "lr": lr_now,
                "epoch_seconds": round(dt, 1),
                "cumulative_seconds": round(time.time() - t_start, 1),
                "batches": n_batches,
                "augmented_fraction": round(aug_frac, 4),
                "best_epoch": best_epoch,
            }
        )
        print(
            f"epoch {epoch}/{cfg.train.epochs}  train_loss {train_loss:.4f}  "
            f"val_loss {val_loss:.4f}  val_CER {val_cer:.4f}  val_WER {val_wer:.4f}  "
            f"lr {lr_now:.2e}  ({dt:.1f}s, {n_batches} batches, aug {aug_frac:.2f})"
            + ("  *new best CER*" if improved else ""),
            flush=True,
        )

        extra = {"history": history, "lr_history": lr_history,
                 "scheduler_reductions": scheduler_reductions}
        save_checkpoint(
            latest_path, model, optimizer, epoch, train_loss, val_loss, cfg,
            extra=extra, scheduler=scheduler,
        )
        if improved:
            save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss, cfg,
                            extra=extra, scheduler=scheduler)

        # persistent per-epoch metrics (survives an interrupted run)
        epoch_record = {
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_cer": round(val_cer, 6),
            "val_wer": round(val_wer, 6),
            "lr": lr_now,
            "epoch_seconds": round(dt, 1),
            "cumulative_seconds": round(time.time() - t_start, 1),
            "batches": n_batches,
            "augmented_fraction": round(aug_frac, 4),
            "best_epoch": best_epoch,
            "best_val_cer": round(best_cer, 6),
            "best_val_loss": round(best_val, 6),
            "best_checkpoint": str(best_path),
            "latest_checkpoint": str(latest_path),
            "no_improve_epochs": no_improve,
        }
        with epochs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(epoch_record) + "\n")

        elapsed = time.time() - t_start
        if cfg.train.early_stop_patience > 0 and no_improve >= cfg.train.early_stop_patience:
            early_stopped = True
            print(f"early stopping: no validation-CER improvement for {no_improve} epochs "
                  f"(patience {cfg.train.early_stop_patience}); "
                  f"best CER {best_cer:.4f} @ epoch {best_epoch}", flush=True)
            break
        if cfg.train.max_seconds > 0 and elapsed >= cfg.train.max_seconds:
            wall_clock_cap_hit = True
            print(f"wall-clock cap reached after epoch {epoch} "
                  f"({elapsed:.0f}s >= {cfg.train.max_seconds:.0f}s)")
            break

    total_time = time.time() - t_start
    if start_epoch > cfg.train.epochs:
        print("all requested epochs already complete; skipping to evaluation")

    # ----- checkpoint round-trip verification -------------------------------
    reload_model = ASRModel(cfg.model).to(device)
    reload_opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    payload = load_checkpoint(best_path, reload_model, reload_opt)
    reloaded_val = validate(reload_model, val_loader, device, cfg)
    ckpt_ok = abs(reloaded_val - best_val) < 1e-3
    print(f"checkpoint reload : val_loss {reloaded_val:.4f} vs saved best {best_val:.4f} -> {'OK' if ckpt_ok else 'MISMATCH'}")
    print(f"best epoch {best_epoch}: val_CER {best_cer:.4f}  val_WER {best_wer:.4f}  val_loss {best_val:.4f}")

    # ----- greedy CER/WER on a validation slice ------------------------------
    id_to_char = {i: c for c, i in vocab["char_to_id"].items()}
    metrics = evaluate_cer_wer(reload_model, val_loader, device, id_to_char, cfg.model.blank_id)
    print(f"greedy CER {metrics['cer']:.4f} | WER {metrics['wer']:.4f} (best checkpoint, validation slice)")
    print(f"best-checkpoint round-trip: CER {metrics['cer']:.4f} / WER {metrics['wer']:.4f} "
          f"vs recorded best {best_cer:.4f} / {best_wer:.4f}")
    metrics_final = evaluate_cer_wer(model, val_loader, device, id_to_char, cfg.model.blank_id)
    print(f"greedy CER {metrics_final['cer']:.4f} | WER {metrics_final['wer']:.4f} (final checkpoint, validation slice)")

    # ----- optional qualitative samples --------------------------------------
    samples: list[dict] = []
    if args.eval_samples:
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                log_probs, out_lengths = model(batch["features"], batch["input_lengths"])
                hyps = greedy_decode(log_probs, out_lengths, cfg.model.blank_id)
                for j, hyp_ids in enumerate(hyps):
                    ref = "".join(
                        id_to_char.get(int(t), "")
                        for t in batch["targets"][j][: int(batch["target_lengths"][j])]
                    )
                    hyp = ids_to_text(hyp_ids, id_to_char)
                    samples.append(
                        {
                            "utterance_id": batch["utterance_ids"][j],
                            "reference": ref,
                            "prediction": hyp,
                            "cer": round(
                                edit_distance(ref, hyp) / max(len(ref), 1), 4
                            ),
                        }
                    )
                    if len(samples) >= args.eval_samples:
                        break
                if len(samples) >= args.eval_samples:
                    break
        print(f"\nsample predictions ({len(samples)}):")
        for s in samples:
            print(f"  [{s['utterance_id']}] CER {s['cer']:.2f}")
            print(f"    ref: {s['reference']}")
            print(f"    hyp: {s['prediction']}")

    # ----- report ------------------------------------------------------------
    REPORTS.mkdir(exist_ok=True)
    report = {
        "tag": args.tag,
        "device": str(device),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "model_parameters": model.num_parameters(),
        "vocab_size": cfg.model.vocab_size,
        "train_utterances": len(train_loader.dataset),
        "train_hours": round(train_loader.dataset.total_hours, 2),
        "val_utterances": len(val_loader.dataset),
        "batch_size": cfg.train.batch_size,
        "epochs": cfg.train.epochs,
        "epochs_completed": len(history),
        "learning_rate": cfg.train.lr,
        "weight_decay": cfg.train.weight_decay,
        "scheduler": cfg.train.scheduler,
        "scheduler_config": {
            "factor": cfg.train.sched_factor,
            "patience": cfg.train.sched_patience,
            "min_lr": cfg.train.min_lr,
        },
        "optimizer": "AdamW",
        "num_workers": cfg.train.num_workers,
        "val_max_utts": cfg.train.val_max_utts,
        "history": history,
        "lr_history": lr_history,
        "scheduler_reductions": scheduler_reductions,
        "final_lr": lr_history[-1] if lr_history else cfg.train.lr,
        "best_val_loss": round(best_val, 4),
        "best_val_cer": round(best_cer, 6),
        "best_val_wer": round(best_wer, 6),
        "best_epoch": best_epoch,
        "selection_policy": "best checkpoint = lowest validation CER (corpus-level greedy); val_loss and WER recorded",
        "val_manifest": cfg.train.val_manifest or "manifests/val.csv (seeded slice)",
        "epochs_metrics_jsonl": str(epochs_path.relative_to(PROJECT_ROOT)),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "total_training_seconds": round(total_time, 1),
        "early_stopped": early_stopped,
        "wall_clock_cap_hit": wall_clock_cap_hit,
        "cer_greedy": round(metrics["cer"], 4),
        "wer_greedy": round(metrics["wer"], 4),
        "cer_greedy_final": round(metrics_final["cer"], 4),
        "wer_greedy_final": round(metrics_final["wer"], 4),
        "checkpoint_reload_ok": ckpt_ok,
        "waveform_augmentation": cfg.augment.prob > 0,
        "spec_augment": cfg.spec_augment.enabled,
        "samples": samples,
        "config": cfg.to_dict(),
    }
    report_path = REPORTS / f"{args.tag}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg.save(PROJECT_ROOT / cfg.train.checkpoint_dir / f"{args.tag}_config.json")
    print(f"report   -> {report_path}")
    print(f"total    : {total_time:.1f}s")
    return 0 if ckpt_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
