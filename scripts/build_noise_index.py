#!/usr/bin/env python3
"""
Week 3 Day 1-2: index the augmentation corpora (MUSAN + SLR28 RIRS_NOISES).

Only header metadata is read (soundfile.info); audio is never decoded into
memory here. Scanning is parallelised across processes.

Corpora on disk (inspected, not assumed):
    data/musan/{noise,music,speech}/<source>/<file>.wav
    data/rirs_noises/RIRS_NOISES/
        simulated_rirs/{smallroom,mediumroom,largeroom}/RoomNNN/RoomNNN-NNNNN.wav
        pointsource_noises/noise-free-sound-NNNN.wav
        real_rirs_isotropic_noises/*.wav

Outputs
-------
manifests/musan.csv  file_id, audio_path, category, subset, duration_seconds,
                     sample_rate, channels, num_samples, status, error
manifests/rir.csv    same columns; category is `rir` for impulse responses and
                     `noise` for point-source / isotropic noise recordings.
manifests/noise_index_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COLUMNS = [
    "file_id",
    "audio_path",
    "category",
    "subset",
    "duration_seconds",
    "sample_rate",
    "channels",
    "num_samples",
    "status",
    "error",
]

# MUSAN top-level dirs map directly to augmentation categories.
MUSAN_CATEGORY_DIRS = ("noise", "music", "speech")


def scan_one(path: Path, category: str, subset: str) -> dict[str, object]:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    file_id = path.stem
    try:
        info = sf.info(str(path))
        return {
            "file_id": file_id,
            "audio_path": rel,
            "category": category,
            "subset": subset,
            "duration_seconds": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "num_samples": info.frames,
            "status": "ok",
            "error": "",
        }
    except Exception as exc:  # unreadable / non-audio file
        return {
            "file_id": file_id,
            "audio_path": rel,
            "category": category,
            "subset": subset,
            "duration_seconds": "",
            "sample_rate": "",
            "channels": "",
            "num_samples": "",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def collect_musan(musan_root: Path) -> list[tuple[Path, str, str]]:
    jobs: list[tuple[Path, str, str]] = []
    for category in MUSAN_CATEGORY_DIRS:
        cat_dir = musan_root / category
        if not cat_dir.is_dir():
            raise SystemExit(f"expected MUSAN subdirectory missing: {cat_dir}")
        for wav in sorted(cat_dir.rglob("*.wav")):
            if not wav.is_file():
                continue
            subset = wav.parent.name if wav.parent != cat_dir else category
            jobs.append((wav, category, subset))
    return jobs


def collect_rirs(rir_root: Path) -> list[tuple[Path, str, str]]:
    base = rir_root / "RIRS_NOISES"
    if not base.is_dir():
        raise SystemExit(f"expected RIRS_NOISES directory missing: {base}")
    jobs: list[tuple[Path, str, str]] = []
    for name, category, subset in (
        ("simulated_rirs", "rir", "simulated_rirs"),
        ("pointsource_noises", "noise", "pointsource_noises"),
        ("real_rirs_isotropic_noises", "noise", "real_rirs_isotropic_noises"),
    ):
        d = base / name
        if not d.is_dir():
            raise SystemExit(f"expected SLR28 subdirectory missing: {d}")
        for wav in sorted(d.rglob("*.wav")):
            if wav.is_file():
                jobs.append((wav, category, subset))
    return jobs


def run_jobs(jobs, workers: int) -> list[dict[str, object]]:
    if workers <= 1:
        return [scan_one(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(scan_one, *zip(*jobs)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    by_cat: dict[str, dict[str, float]] = {}
    for r in rows:
        cat = str(r["category"])
        sub = f"{cat}/{r['subset']}"
        entry = by_cat.setdefault(sub, {"files": 0, "hours": 0.0, "errors": 0})
        entry["files"] += 1
        if r["status"] == "ok":
            entry["hours"] += float(r["duration_seconds"]) / 3600.0
        else:
            entry["errors"] += 1
    for entry in by_cat.values():
        entry["hours"] = round(entry["hours"], 2)
    return by_cat


def main() -> int:
    parser = argparse.ArgumentParser(description="Index MUSAN and RIRS_NOISES.")
    parser.add_argument("--musan-root", type=Path, default=PROJECT_ROOT / "data" / "musan")
    parser.add_argument(
        "--rir-root", type=Path, default=PROJECT_ROOT / "data" / "rirs_noises"
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "manifests")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    print("indexing MUSAN ...")
    musan_jobs = collect_musan(args.musan_root)
    musan_rows = run_jobs(musan_jobs, args.workers)
    write_csv(args.out_dir / "musan.csv", musan_rows)

    print("indexing RIRS_NOISES ...")
    rir_jobs = collect_rirs(args.rir_root)
    rir_rows = run_jobs(rir_jobs, args.workers)
    write_csv(args.out_dir / "rir.csv", rir_rows)

    all_rows = musan_rows + rir_rows
    errors = [r for r in all_rows if r["status"] != "ok"]
    sample_rates: dict[int, int] = {}
    channel_counts: dict[int, int] = {}
    for r in all_rows:
        if r["status"] == "ok":
            sample_rates[int(r["sample_rate"])] = sample_rates.get(int(r["sample_rate"]), 0) + 1
            channel_counts[int(r["channels"])] = channel_counts.get(int(r["channels"]), 0) + 1

    summary = {
        "musan_root": args.musan_root.as_posix(),
        "rir_root": args.rir_root.as_posix(),
        "musan_files": len(musan_rows),
        "rir_files": len(rir_rows),
        "total_files": len(all_rows),
        "errors": len(errors),
        "sample_rate_distribution": {str(k): v for k, v in sorted(sample_rates.items())},
        "channel_distribution": {str(k): v for k, v in sorted(channel_counts.items())},
        "categories": {**summarise(musan_rows), **summarise(rir_rows)},
    }
    summary_path = args.out_dir / "noise_index_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"musan.csv rows : {len(musan_rows):,} -> {args.out_dir / 'musan.csv'}")
    print(f"rir.csv rows   : {len(rir_rows):,} -> {args.out_dir / 'rir.csv'}")
    print(f"errors         : {len(errors)}")
    for cat, entry in summary["categories"].items():
        print(f"  {cat:38s} {entry['files']:>6,} files  {entry['hours']:>8.2f} h  {entry['errors']} errors")
    print(f"sample rates   : {summary['sample_rate_distribution']}")
    print(f"channels       : {summary['channel_distribution']}")
    print(f"summary        -> {summary_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
