#!/usr/bin/env python3
"""
Week 1 - Goal 1: discover and index every audio file in the dataset.

Scans the dataset root recursively for FLAC files and writes a portable audio
index (paths are relative to the dataset root).

The dataset uses the OpenSLR SLR54 layout:

    <data_root>/asr_nepali_<N>/data/<2-hex>/<10-hex>.flac

`utterance_id` is derived from the FLAC file stem (the filename without the
`.flac` extension). This is *not* assumed - the script reports the observed
naming pattern, extension distribution and any filename that does not fit it.

Outputs
-------
manifests/audio_index.csv          utterance_id, audio_path, filename, dataset, shard
manifests/audio_index_duplicates.csv  (only written when duplicate ids exist)

Example
-------
    python scripts/build_audio_index.py --data-root asr_nepali_data
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files the dataset is known to ship in non-audio subdirectories. They are
# reported if found and never treated as audio.
EXPECTED_ID_RE = re.compile(r"^[0-9a-f]{10}$")


def discover_audio(data_root: Path) -> list[Path]:
    """Recursively collect every .flac file under `data_root`."""
    return sorted(p for p in data_root.rglob("*") if p.suffix.lower() == ".flac")


def derive_utterance_id(path: Path) -> str:
    """utterance_id == file stem, e.g. .../0d/0df34c8402.flac -> 0df34c8402."""
    return path.stem


def relative_parts(path: Path, data_root: Path) -> tuple[str, str]:
    """Return (dataset, shard) for a shard-layout path, else best-effort values."""
    rel = path.relative_to(data_root)
    parts = rel.parts
    dataset = parts[0] if len(parts) >= 1 else ""
    # .../asr_nepali_1/data/0d/file.flac -> shard is the directory holding the file
    shard = path.parent.name if path.parent != data_root else ""
    return dataset, shard


def main() -> int:
    parser = argparse.ArgumentParser(description="Index all audio files in the dataset.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "asr_nepali_data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "manifests" / "audio_index.csv")
    parser.add_argument(
        "--duplicates-output",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "audio_index_duplicates.csv",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root not found: {data_root}")
        return 2

    print("=" * 70)
    print("WEEK 1 - AUDIO INDEXING")
    print("=" * 70)
    print(f"data root : {data_root}")

    files = discover_audio(data_root)
    total = len(files)
    print(f"FLAC files: {total:,}")

    # ----------------------------------------------------------------
    # Diagnostics: ids, duplicates, extensions, unexpected entries
    # ----------------------------------------------------------------
    seen: dict[str, Path] = {}
    duplicate_rows: list[dict[str, str]] = []
    stem_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    shard_counts: Counter[str] = Counter()
    irregular_names: list[str] = []

    for path in files:
        utter_id = derive_utterance_id(path)
        stem_counts[utter_id] += 1
        extension_counts[path.suffix.lower()] += 1

        dataset, shard = relative_parts(path, data_root)
        dataset_counts[dataset] += 1
        shard_counts[f"{dataset}/{shard}"] += 1

        if utter_id in seen:
            duplicate_rows.append(
                {
                    "utterance_id": utter_id,
                    "first_path": seen[utter_id].relative_to(data_root).as_posix(),
                    "duplicate_path": path.relative_to(data_root).as_posix(),
                }
            )
        else:
            seen[utter_id] = path

        if not EXPECTED_ID_RE.match(utter_id):
            irregular_names.append(path.relative_to(data_root).as_posix())

    unique_ids = len(seen)
    duplicate_ids = sum(1 for c in stem_counts.values() if c > 1)

    # Non-flac files, so nothing is silently ignored.
    other_files = sorted(
        p.relative_to(data_root).as_posix()
        for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() != ".flac"
    )

    # ----------------------------------------------------------------
    # Write the index
    # ----------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "audio_path", "filename", "dataset", "shard"])
        for path in files:
            utter_id = derive_utterance_id(path)
            dataset, shard = relative_parts(path, data_root)
            writer.writerow(
                [
                    utter_id,
                    path.relative_to(data_root).as_posix(),
                    path.name,
                    dataset,
                    shard,
                ]
            )

    if duplicate_rows:
        args.duplicates_output.parent.mkdir(parents=True, exist_ok=True)
        with args.duplicates_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["utterance_id", "first_path", "duplicate_path"]
            )
            writer.writeheader()
            writer.writerows(duplicate_rows)

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print()
    print("-" * 70)
    print("AUDIO INDEX SUMMARY")
    print("-" * 70)
    print(f"total .flac files        : {total:,}")
    print(f"unique utterance ids     : {unique_ids:,}")
    print(f"duplicate utterance ids  : {duplicate_ids:,}")
    print(f"irregular (non-10hex) ids: {len(irregular_names):,}")
    if irregular_names[:10]:
        for name in irregular_names[:10]:
            print(f"    {name}")
    print(f"extension distribution   : {dict(extension_counts)}")
    print(f"non-flac files           : {len(other_files):,}")
    for name in other_files[:10]:
        print(f"    {name}")

    print()
    print(f"shard directories (dataset/shard) : {len(shard_counts):,}")
    print("files per dataset:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"    {dataset:>16}: {count:,}")

    if duplicate_rows:
        print()
        print(f"WARNING: {len(duplicate_rows)} duplicate utterance id(s); see {args.duplicates_output}")

    print()
    print(f"index written: {args.output}")
    return 1 if duplicate_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
