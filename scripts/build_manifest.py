#!/usr/bin/env python3
"""
Week 1 - Goals 2 & 3: parse the dataset metadata and build the canonical manifest.

OpenSLR SLR54 metadata ships as UTF-8, tab-separated files named
`utt_spk_text.tsv` with three columns and no header:

    <utterance_id> \t <speaker_id> \t <transcript>

This script does NOT assume there is exactly one such file, and it does NOT
assume that several files are independent partitions. It discovers every
`utt_spk_text.tsv` under the dataset root, analyses each one, measures the
overlap between them, and only then builds the manifest from unique utterances.

Transcripts are preserved *exactly* as they appear in the TSV (whitespace,
including non-breaking spaces, is not stripped or normalised). Only the line
terminator is removed.

Outputs
-------
manifests/all.csv                     utterance_id, speaker_id, audio_path, transcript
manifests/metadata_file_summary.csv   per-TSV row/column/duplicate statistics
manifests/metadata_malformed.csv      rows that could not be parsed (if any)
manifests/metadata_conflicts.csv      duplicate ids with differing content (if any)
manifests/metadata_without_audio.csv  metadata ids with no matching audio file
manifests/audio_without_metadata.csv  audio files with no matching metadata row
manifests/week1_manifest_stats.json   machine-readable summary of all counters

Example
-------
    python scripts/build_manifest.py --data-root asr_nepali_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_COLUMNS = 3
METADATA_FILENAME = "utt_spk_text.tsv"


# ----------------------------------------------------------------------
# Loading inputs
# ----------------------------------------------------------------------
def load_audio_index(path: Path) -> dict[str, dict[str, str]]:
    """Load the audio index keyed by utterance_id, rejecting duplicate ids."""
    index: dict[str, dict[str, str]] = {}
    duplicates: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            utt_id = row["utterance_id"]
            if utt_id in index:
                duplicates.append((utt_id, index[utt_id]["audio_path"], row["audio_path"]))
                continue
            index[utt_id] = row
    if duplicates:
        raise ValueError(
            f"audio index contains {len(duplicates)} duplicate utterance_id(s); "
            f"first: {duplicates[0]}"
        )
    return index


def find_metadata_files(data_root: Path) -> list[Path]:
    return sorted(p for p in data_root.rglob(METADATA_FILENAME) if p.is_file())


# ----------------------------------------------------------------------
# Parsing a single TSV
# ----------------------------------------------------------------------
def parse_tsv(path: Path) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Parse one TSV, preserving transcripts exactly.

    Returns (records, malformed_rows). Malformed rows are reported, never
    silently dropped.
    """
    records: list[dict[str, str]] = []
    malformed: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line
            if line.endswith("\n"):
                line = line[:-1]
            if line.endswith("\r"):
                line = line[:-1]

            if line == "":
                malformed.append(
                    {
                        "file": path.as_posix(),
                        "line_number": line_number,
                        "columns": 0,
                        "reason": "empty line",
                    }
                )
                continue

            columns = line.split("\t")
            if len(columns) != EXPECTED_COLUMNS:
                malformed.append(
                    {
                        "file": path.as_posix(),
                        "line_number": line_number,
                        "columns": len(columns),
                        "reason": f"expected {EXPECTED_COLUMNS} tab-separated columns",
                    }
                )
                continue

            utt_id = columns[0].strip()
            speaker_id = columns[1].strip()
            transcript = columns[2]  # preserved exactly

            if not utt_id:
                malformed.append(
                    {
                        "file": path.as_posix(),
                        "line_number": line_number,
                        "columns": len(columns),
                        "reason": "empty utterance_id",
                    }
                )
                continue
            if not speaker_id:
                malformed.append(
                    {
                        "file": path.as_posix(),
                        "line_number": line_number,
                        "columns": len(columns),
                        "reason": "empty speaker_id",
                    }
                )
                continue
            if transcript == "":
                malformed.append(
                    {
                        "file": path.as_posix(),
                        "line_number": line_number,
                        "columns": len(columns),
                        "reason": "empty transcript",
                    }
                )
                continue

            records.append(
                {
                    "utterance_id": utt_id,
                    "speaker_id": speaker_id,
                    "transcript": transcript,
                    "source_file": path.as_posix(),
                }
            )

    return records, malformed


def id_set_hash(ids) -> str:
    """Stable fingerprint of an id collection (order independent)."""
    digest = hashlib.sha256()
    for utt_id in sorted(ids):
        digest.update(utt_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical utterance manifest.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "asr_nepali_data")
    parser.add_argument(
        "--audio-index", type=Path, default=PROJECT_ROOT / "manifests" / "audio_index.csv"
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "manifests" / "all.csv")
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "week1_manifest_stats.json",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    out_dir = args.output.parent

    print("=" * 70)
    print("WEEK 1 - METADATA PARSING & MANIFEST CONSTRUCTION")
    print("=" * 70)

    if not args.audio_index.is_file():
        print(f"ERROR: audio index not found: {args.audio_index}")
        print("Run scripts/build_audio_index.py first.")
        return 2

    audio_index = load_audio_index(args.audio_index)
    print(f"audio index records : {len(audio_index):,}")

    metadata_files = find_metadata_files(data_root)
    print(f"metadata TSV files  : {len(metadata_files):,}")
    for path in metadata_files:
        print(f"    {path.relative_to(data_root).as_posix()}")

    # ------------------------------------------------------------------
    # Parse every TSV, recording per-file statistics.
    # ------------------------------------------------------------------
    all_records: list[dict[str, str]] = []
    malformed_rows: list[dict[str, object]] = []
    file_summaries: list[dict[str, object]] = []

    for path in metadata_files:
        records, malformed = parse_tsv(path)
        all_records.extend(records)
        malformed_rows.extend(malformed)

        ids = [r["utterance_id"] for r in records]
        id_counts = Counter(ids)
        complete = Counter((r["utterance_id"], r["speaker_id"], r["transcript"]) for r in records)
        speakers = {r["speaker_id"] for r in records}

        file_summaries.append(
            {
                "file": path.relative_to(data_root).as_posix(),
                "rows": len(ids),
                "malformed_rows": len(malformed),
                "unique_utterance_ids": len(id_counts),
                "duplicate_utterance_ids": sum(1 for c in id_counts.values() if c > 1),
                "duplicate_complete_records": sum(1 for c in complete.values() if c > 1),
                "unique_speakers": len(speakers),
                "id_fingerprint": id_set_hash(ids),
            }
        )

    total_metadata_rows = len(all_records) + len(malformed_rows)

    # ------------------------------------------------------------------
    # Global uniqueness / duplication analysis.
    # ------------------------------------------------------------------
    id_counts = Counter(r["utterance_id"] for r in all_records)
    duplicate_ids = {k for k, c in id_counts.items() if c > 1}

    unique_records: dict[str, dict[str, str]] = {}
    conflicts: list[dict[str, str]] = []
    duplicate_complete = 0
    for record in all_records:
        utt_id = record["utterance_id"]
        existing = unique_records.get(utt_id)
        if existing is None:
            unique_records[utt_id] = record
            continue
        if (
            existing["speaker_id"] == record["speaker_id"]
            and existing["transcript"] == record["transcript"]
        ):
            duplicate_complete += 1
        else:
            conflicts.append(
                {
                    "utterance_id": utt_id,
                    "first_speaker_id": existing["speaker_id"],
                    "first_transcript": existing["transcript"],
                    "first_file": existing["source_file"],
                    "second_speaker_id": record["speaker_id"],
                    "second_transcript": record["transcript"],
                    "second_file": record["source_file"],
                }
            )

    # Cross-file overlap: ids appearing in more than one TSV.
    ids_by_file = {}
    for summary_path, path in zip(file_summaries, metadata_files):
        ids_by_file[summary_path["file"]] = set(
            r["utterance_id"] for r in all_records if r["source_file"] == path.as_posix()
        )
    file_names = list(ids_by_file)
    cross_file_overlap = set()
    for i, a in enumerate(file_names):
        for b in file_names[i + 1 :]:
            cross_file_overlap |= ids_by_file[a] & ids_by_file[b]

    identical_id_sets = (
        len({summary["id_fingerprint"] for summary in file_summaries}) == 1
        and len(file_summaries) > 1
    )

    # ------------------------------------------------------------------
    # Join metadata <-> audio.
    # ------------------------------------------------------------------
    audio_ids = set(audio_index)
    metadata_ids = set(unique_records)

    matched: list[dict[str, str]] = []
    metadata_without_audio: list[dict[str, str]] = []
    for utt_id in sorted(metadata_ids):
        record = unique_records[utt_id]
        audio = audio_index.get(utt_id)
        if audio is None:
            metadata_without_audio.append(record)
            continue
        matched.append(
            {
                "utterance_id": utt_id,
                "speaker_id": record["speaker_id"],
                "audio_path": audio["audio_path"],
                "transcript": record["transcript"],
            }
        )

    audio_without_metadata = [
        {"utterance_id": utt_id, "audio_path": audio_index[utt_id]["audio_path"],
         "filename": audio_index[utt_id].get("filename", "")}
        for utt_id in sorted(audio_ids - metadata_ids)
    ]

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utterance_id", "speaker_id", "audio_path", "transcript"]
        )
        writer.writeheader()
        writer.writerows(matched)

    def write_rows(name: str, fieldnames: list[str], rows: list[dict]) -> Path:
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    write_rows(
        "metadata_file_summary.csv",
        [
            "file",
            "rows",
            "malformed_rows",
            "unique_utterance_ids",
            "duplicate_utterance_ids",
            "duplicate_complete_records",
            "unique_speakers",
            "id_fingerprint",
        ],
        file_summaries,
    )
    if malformed_rows:
        write_rows(
            "metadata_malformed.csv",
            ["file", "line_number", "columns", "reason"],
            malformed_rows,
        )
    if conflicts:
        write_rows(
            "metadata_conflicts.csv",
            [
                "utterance_id",
                "first_speaker_id",
                "first_transcript",
                "first_file",
                "second_speaker_id",
                "second_transcript",
                "second_file",
            ],
            conflicts,
        )
    if metadata_without_audio:
        write_rows(
            "metadata_without_audio.csv",
            ["utterance_id", "speaker_id", "transcript", "source_file"],
            metadata_without_audio,
        )
    if audio_without_metadata:
        write_rows(
            "audio_without_metadata.csv",
            ["utterance_id", "audio_path", "filename"],
            audio_without_metadata,
        )

    # ------------------------------------------------------------------
    # Metrics.
    # ------------------------------------------------------------------
    unique_metadata_ids = len(metadata_ids)
    unique_audio_ids = len(audio_ids)
    n_matched = len(matched)
    match_rate = n_matched / unique_metadata_ids if unique_metadata_ids else 0.0
    audio_coverage = n_matched / unique_audio_ids if unique_audio_ids else 0.0
    metadata_coverage = match_rate

    stats = {
        "data_root": data_root.as_posix(),
        "metadata_files": len(metadata_files),
        "metadata_file_summaries": file_summaries,
        "total_metadata_rows": total_metadata_rows,
        "total_metadata_records": len(all_records),
        "unique_metadata_ids": unique_metadata_ids,
        "duplicate_metadata_ids": len(duplicate_ids),
        "duplicate_complete_records": duplicate_complete,
        "conflicting_duplicate_ids": len(conflicts),
        "malformed_metadata_rows": len(malformed_rows),
        "cross_file_overlapping_ids": len(cross_file_overlap),
        "all_files_have_identical_id_sets": identical_id_sets,
        "total_audio_files": len(audio_index),
        "unique_audio_ids": unique_audio_ids,
        "duplicate_audio_ids": 0,
        "matched_records": n_matched,
        "metadata_without_audio": len(metadata_without_audio),
        "audio_without_metadata": len(audio_without_metadata),
        "manifest_rows": len(matched),
        "match_rate": match_rate,
        "audio_coverage": audio_coverage,
        "metadata_coverage": metadata_coverage,
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    print()
    print("-" * 70)
    print("METADATA ANALYSIS")
    print("-" * 70)
    print(f"total metadata rows        : {total_metadata_rows:,}")
    print(f"parsed records             : {len(all_records):,}")
    print(f"malformed rows             : {len(malformed_rows):,}")
    print(f"unique metadata ids        : {unique_metadata_ids:,}")
    print(f"duplicate metadata ids     : {len(duplicate_ids):,}")
    print(f"duplicate complete records : {duplicate_complete:,}")
    print(f"conflicting duplicate ids  : {len(conflicts):,}")
    print(f"cross-file overlapping ids : {len(cross_file_overlap):,}")
    print(f"all TSV id-sets identical  : {identical_id_sets}")

    print()
    print("-" * 70)
    print("MANIFEST")
    print("-" * 70)
    print(f"manifest rows              : {n_matched:,}")
    print(f"metadata without audio     : {len(metadata_without_audio):,}")
    print(f"audio without metadata     : {len(audio_without_metadata):,}")
    print(f"match_rate                 : {match_rate:.6f}")
    print(f"audio_coverage             : {audio_coverage:.6f}")
    print(f"metadata_coverage          : {metadata_coverage:.6f}")

    print()
    print(f"manifest written: {args.output}")
    print(f"stats written   : {args.stats_output}")

    if malformed_rows or conflicts or len(duplicate_ids) or metadata_without_audio:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
