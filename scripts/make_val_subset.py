#!/usr/bin/env python3
"""Construct a fixed, deterministic 2,000-utterance validation subset.

Selection rule (identical to how train.py builds its validation set):
    rows = load manifests/val.csv in file order
    random.Random(seed).shuffle(rows)
    subset = rows[:2000]

The resulting subset is speaker-disjoint from train/test by construction
(it is carved out of the already speaker-disjoint val split), is fixed for
the whole experiment, and never changes between epochs.

Writes:
    manifests/val_2000.csv          the exact validation manifest
    manifests/val_2000.sha256       sha256 of the manifest file
    manifests/val_2000_summary.json validation/leakage report

Usage:
    python scripts/make_val_subset.py --n 2000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from config import PROJECT_ROOT
from dataset import load_durations, load_manifest_rows

MANIFESTS = PROJECT_ROOT / "manifests"
DATA_ROOT = PROJECT_ROOT / "asr_nepali_data"  # same root as dataset.py


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fixed validation subset manifest.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(MANIFESTS / "val_2000.csv"))
    args = parser.parse_args()

    out_path = Path(args.out)
    val_rows = load_manifest_rows(MANIFESTS / "val.csv")
    train_rows = load_manifest_rows(MANIFESTS / "train.csv")
    test_rows = load_manifest_rows(MANIFESTS / "test.csv")
    durations = load_durations(MANIFESTS / "audio_validation.csv")
    vocab = json.loads((MANIFESTS / "vocab.json").read_text(encoding="utf-8"))
    char_to_id = vocab["char_to_id"]

    # --- deterministic selection, exactly as ManifestDataset(order_seed=seed) does
    rows = list(val_rows)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    subset = rows[: args.n]
    assert len(subset) == args.n, f"only {len(subset)} rows available in val.csv"

    # --- validation checks ---------------------------------------------------
    problems: list[str] = []
    missing_audio = []
    empty_transcript = []
    no_target = []
    unknown_chars: set[str] = set()
    for r in subset:
        audio = DATA_ROOT / r["audio_path"]
        if not audio.is_file():
            missing_audio.append(r["utterance_id"])
        if not r["transcript"].strip():
            empty_transcript.append(r["utterance_id"])
        # same encoding rule as ManifestDataset._encode (blank/unknown dropped)
        ids = [char_to_id.get(ch, -1) for ch in r["transcript"]]
        kept = [i for i in ids if i > 0]
        if not kept:
            no_target.append(r["utterance_id"])
        unknown_chars.update(ch for ch in r["transcript"] if ch not in char_to_id)

    if missing_audio:
        problems.append(f"{len(missing_audio)} missing audio files")
    if empty_transcript:
        problems.append(f"{len(empty_transcript)} empty transcripts")
    if no_target:
        problems.append(f"{len(no_target)} transcripts with no valid target ids")

    # --- leakage checks ------------------------------------------------------
    val_ids = {r["utterance_id"] for r in subset}
    train_ids = {r["utterance_id"] for r in train_rows}
    test_ids = {r["utterance_id"] for r in test_rows}
    train_speakers = {r["speaker_id"] for r in train_rows}
    test_speakers = {r["speaker_id"] for r in test_rows}
    subset_speakers = {r["speaker_id"] for r in subset}

    overlap_train = sorted(val_ids & train_ids)
    overlap_test = sorted(val_ids & test_ids)
    speaker_overlap_train = sorted(subset_speakers & train_speakers)
    speaker_overlap_test = sorted(subset_speakers & test_speakers)
    if overlap_train:
        problems.append(f"{len(overlap_train)} utterances leak into train.csv")
    if overlap_test:
        problems.append(f"{len(overlap_test)} utterances leak into test.csv")
    if speaker_overlap_train:
        problems.append(f"{len(speaker_overlap_train)} speakers shared with train")
    if speaker_overlap_test:
        problems.append(f"{len(speaker_overlap_test)} speakers shared with test")

    # --- reproducibility check ----------------------------------------------
    rows2 = list(val_rows)
    random.Random(args.seed).shuffle(rows2)
    reproducible = [r["utterance_id"] for r in rows2[: args.n]] == [
        r["utterance_id"] for r in subset
    ]
    if not reproducible:
        problems.append("selection not reproducible")

    # --- write manifest ------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(subset[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)

    raw = out_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    (out_path.with_suffix(".sha256")).write_text(f"{sha}  {out_path.name}\n", encoding="utf-8")

    durs = [durations.get(r["utterance_id"], 0.0) for r in subset]
    summary = {
        "manifest": str(out_path.relative_to(PROJECT_ROOT)),
        "sha256": sha,
        "source": "manifests/val.csv",
        "selection": f"random.Random({args.seed}).shuffle(val.csv rows in file order) -> first {args.n}",
        "seed": args.seed,
        "n_utterances": len(subset),
        "n_speakers": len(subset_speakers),
        "duration_seconds_total": round(sum(durs), 1),
        "hours": round(sum(durs) / 3600.0, 2),
        "min_duration_s": round(min(durs), 2),
        "max_duration_s": round(max(durs), 2),
        "utterance_overlap_train": len(overlap_train),
        "utterance_overlap_test": len(overlap_test),
        "speaker_overlap_train": len(speaker_overlap_train),
        "speaker_overlap_test": len(speaker_overlap_test),
        "subset_speakers_in_val_split": len(subset_speakers),
        "missing_audio": missing_audio,
        "empty_transcripts": empty_transcript,
        "no_valid_target_ids": no_target,
        "unknown_chars_not_in_vocab": sorted(unknown_chars),
        "reproducible": reproducible,
        "problems": problems,
        "ok": not problems,
    }
    summary_path = out_path.with_name(out_path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
