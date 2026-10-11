#!/usr/bin/env python3
"""
Week 2 prerequisite: character vocabulary for CTC.

Built from the **train split only** (manifests/train.csv) so that validation
and test text never influences the model's output space.

Conventions
-----------
* id 0 is the CTC <blank>. It is not a real character and never appears in
  encoded targets.
* Characters are ordered by frequency (descending), tie-broken by code point,
  so the vocabulary is deterministic given the train split.
* Control characters (Unicode category Cc, e.g. \\n \\r \\t) are excluded from
  the vocabulary; if any occur in transcripts they are skipped at encoding
  time and reported.

Outputs
-------
manifests/vocab.json
    {"blank_id": 0, "chars": [...], "char_to_id": {...}, "counts": {...}, ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the character vocabulary.")
    parser.add_argument("--train", type=Path, default=PROJECT_ROOT / "manifests" / "train.csv")
    parser.add_argument("--val", type=Path, default=PROJECT_ROOT / "manifests" / "val.csv")
    parser.add_argument("--test", type=Path, default=PROJECT_ROOT / "manifests" / "test.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "manifests" / "vocab.json")
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Keep characters occurring at least this many times in train.",
    )
    args = parser.parse_args()

    def transcripts(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [r["transcript"] for r in csv.DictReader(f)]

    train_texts = transcripts(args.train)
    counts: Counter[str] = Counter()
    for text in train_texts:
        counts.update(text)

    control_chars = {c for c in counts if unicodedata.category(c) == "Cc"}
    eligible = {c: n for c, n in counts.items() if c not in control_chars and n >= args.min_count}
    dropped_rare = {c: n for c, n in counts.items() if c not in control_chars and n < args.min_count}

    ordered = sorted(eligible, key=lambda c: (-eligible[c], c))
    chars = ordered  # id = index + 1 (blank takes 0)
    char_to_id = {c: i + 1 for i, c in enumerate(chars)}

    # Coverage checks.
    def coverage(texts: list[str]) -> tuple[int, int, Counter[str]]:
        total = oov = 0
        oov_counts: Counter[str] = Counter()
        for text in texts:
            for ch in text:
                total += 1
                if ch not in char_to_id:
                    oov += 1
                    oov_counts[ch] += 1
        return total, oov, oov_counts

    tr_total, tr_oov, _ = coverage(train_texts)
    val_texts = transcripts(args.val) if args.val.is_file() else []
    te_texts = transcripts(args.test) if args.test.is_file() else []
    val_total, val_oov, val_oov_counts = coverage(val_texts)
    te_total, te_oov, te_oov_counts = coverage(te_texts)

    vocab = {
        "blank_id": 0,
        "n_chars": len(chars),
        "vocab_size": len(chars) + 1,  # classes the model outputs, incl. blank
        "chars": chars,
        "char_to_id": char_to_id,
        "counts": {c: counts[c] for c in chars},
        "source": args.train.name,
        "min_count": args.min_count,
        "excluded_control_chars": sorted(control_chars),
        "excluded_rare_chars": {c: n for c, n in sorted(dropped_rare.items())},
        "coverage": {
            "train_chars": tr_total,
            "train_oov": tr_oov,
            "train_coverage_pct": round(100.0 * (tr_total - tr_oov) / max(tr_total, 1), 4),
            "val_chars": val_total,
            "val_oov": val_oov,
            "val_oov_top": dict(val_oov_counts.most_common(10)),
            "test_chars": te_total,
            "test_oov": te_oov,
            "test_oov_top": dict(te_oov_counts.most_common(10)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"vocab size (incl. blank): {vocab['vocab_size']}")
    print(f"unique chars (excl. blank): {len(chars)}")
    print(f"excluded control chars: {[hex(ord(c)) for c in sorted(control_chars)]}")
    print(f"excluded rare chars (<{args.min_count}): {len(dropped_rare)}")
    print(
        f"train coverage: {vocab['coverage']['train_coverage_pct']}% | "
        f"val OOV chars: {val_oov}/{val_total} | test OOV chars: {te_oov}/{te_total}"
    )
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
