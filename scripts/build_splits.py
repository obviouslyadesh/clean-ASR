#!/usr/bin/env python3
"""
Week 2 prerequisite: speaker-disjoint train/val/test splits.

This project reached Week 3 without the Week 2 split files, so they are
created here from the Week 1 manifest (manifests/all.csv), which is the
single source of truth for utterances.

Design decisions
----------------
* Splits are **speaker-disjoint**: a speaker appears in exactly one split,
  so no speaker is seen in training and evaluated elsewhere. With 527
  speakers, a 90/5/5 speaker split yields ~26 speakers per eval split.
* Assignment is deterministic and re-runnable without a seed file: speakers
  are ordered by sha256(speaker_id) and sliced by percentage. Re-running
  with the same manifest produces byte-identical splits.
* Durations come from the Week 1 full-decode validation
  (manifests/audio_validation.csv); every manifest row is present there.

Outputs
-------
manifests/train.csv, manifests/val.csv, manifests/test.csv
    same columns as all.csv (utterance_id, speaker_id, audio_path, transcript)
manifests/split_summary.json
    machine-readable split statistics and overlap checks
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ["utterance_id", "speaker_id", "audio_path", "transcript"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def speaker_key(speaker_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()
    return digest, speaker_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Build speaker-disjoint splits.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests" / "all.csv")
    parser.add_argument(
        "--validation",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "audio_validation.csv",
        help="Week 1 audio validation CSV (durations).",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--train-pct", type=float, default=90.0)
    parser.add_argument("--val-pct", type=float, default=5.0)
    args = parser.parse_args()

    rows = load_csv(args.manifest)
    durations = {r["utterance_id"]: float(r["duration_seconds"]) for r in load_csv(args.validation)}

    speakers = sorted({r["speaker_id"] for r in rows}, key=speaker_key)
    n = len(speakers)
    n_val = round(n * args.val_pct / 100.0)
    n_test = round(n * args.val_pct / 100.0)
    val_speakers = set(speakers[:n_val])
    test_speakers = set(speakers[n_val : n_val + n_test])
    train_speakers = set(speakers[n_val + n_test :])

    assert len(val_speakers | test_speakers | train_speakers) == n, "speaker sets must partition"
    assert not (val_speakers & test_speakers), "val/test speaker overlap"
    assert not (val_speakers & train_speakers), "val/train speaker overlap"
    assert not (test_speakers & train_speakers), "test/train speaker overlap"

    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        if row["speaker_id"] in val_speakers:
            splits["val"].append(row)
        elif row["speaker_id"] in test_speakers:
            splits["test"].append(row)
        else:
            splits["train"].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "method": "speaker-disjoint; speakers ordered by sha256(speaker_id), sliced 90/5/5",
        "speaker_ordering": "sha256(speaker_id) hex digest, ascending; tie-broken by speaker_id",
        "durations_source": args.validation.name,
        "total_utterances": len(rows),
        "total_speakers": n,
        "total_duration_seconds": round(sum(durations.get(r["utterance_id"], 0.0) for r in rows), 1),
        "splits": {},
    }

    for name, split_rows in splits.items():
        split_rows.sort(key=lambda r: r["utterance_id"])
        out_path = args.out_dir / f"{name}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(split_rows)

        split_speakers = {r["speaker_id"] for r in split_rows}
        duration = round(sum(durations.get(r["utterance_id"], 0.0) for r in split_rows), 1)
        summary["splits"][name] = {
            "file": out_path.name,
            "utterances": len(split_rows),
            "speakers": len(split_speakers),
            "duration_seconds": duration,
            "hours": round(duration / 3600.0, 2),
            "utterance_pct": round(100.0 * len(split_rows) / len(rows), 2),
            "speaker_pct": round(100.0 * len(split_speakers) / n, 2),
        }
        print(
            f"{name:5s}: {len(split_rows):>7,} utts, {len(split_speakers):>4} speakers, "
            f"{duration / 3600.0:>7.2f} h -> {out_path}"
        )

    # Overlap checks (must all be zero).
    ids = {name: {r["utterance_id"] for r in rows_} for name, rows_ in splits.items()}
    spk = {"train": train_speakers, "val": val_speakers, "test": test_speakers}
    summary["overlap_checks"] = {
        "shared_speakers_train_val": len(train_speakers & val_speakers),
        "shared_speakers_train_test": len(train_speakers & test_speakers),
        "shared_speakers_val_test": len(val_speakers & test_speakers),
        "shared_utterances_train_val": len(ids["train"] & ids["val"]),
        "shared_utterances_train_test": len(ids["train"] & ids["test"]),
        "shared_utterances_val_test": len(ids["val"] & ids["test"]),
        "utterances_accounted": sum(len(v) for v in ids.values()),
    }

    summary_path = args.out_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {summary_path}")
    print(
        "overlap checks:",
        {k: v for k, v in summary["overlap_checks"].items() if k != "utterances_accounted"},
        "| accounted:",
        summary["overlap_checks"]["utterances_accounted"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
