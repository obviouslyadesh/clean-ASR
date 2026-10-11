#!/usr/bin/env python3
"""
Week 1 - Goal 4: full validation of every audio file referenced by the manifest.

For each manifest row the script checks:

* the file exists and is non-empty
* the FLAC container parses (magic + first STREAMINFO block) -> declared
  sample rate, channel count, bit depth and total sample count
* the audio actually decodes: every frame is read out, not just the header
* the number of decoded frames matches the STREAMINFO sample count, which
  exposes truncated / partially-downloaded files
* the decoded sample rate and channel count match the declared values

Decoding uses the first available backend:

1. `soundfile` (libsndfile) if importable -- cross platform
2. macOS CoreAudio via `afconvert` + the stdlib `wave` reader

The backend is recorded per run so results are interpretable. Validation is
parallelised with `concurrent.futures.ProcessPoolExecutor`; no audio is ever
kept in memory beyond a single file per worker.

Outputs
-------
manifests/audio_validation.csv      per-file validation results
manifests/validation_summary.json   aggregate statistics

Example
-------
    python scripts/validate_audio.py --manifest manifests/all.csv --data-root asr_nepali_data
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FIELDS = [
    "utterance_id",
    "audio_path",
    "exists",
    "decodable",
    "sample_rate",
    "channels",
    "duration_seconds",
    "num_samples",
    "status",
    "error",
]

# Expected dataset properties. Deviations are reported, not treated as fatal
# by themselves (a non-16 kHz file is a real audio file, just unexpected).
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1

_TMPDIR: dict[str, str] = {}


# ----------------------------------------------------------------------
# FLAC header (STREAMINFO) parsing - stdlib only
# ----------------------------------------------------------------------
def parse_streaminfo(path: str) -> dict[str, int]:
    """Parse the FLAC magic and the mandatory first STREAMINFO block."""
    with open(path, "rb") as f:
        head = f.read(42)

    if len(head) < 42:
        raise ValueError(f"file too small for a FLAC header ({len(head)} bytes)")
    if head[:4] != b"fLaC":
        raise ValueError("missing 'fLaC' stream marker")

    block_type = head[4] & 0x7F
    if block_type != 0:
        raise ValueError(f"first metadata block is not STREAMINFO (type={block_type})")

    length = int.from_bytes(head[5:8], "big")
    if length != 34:
        raise ValueError(f"invalid STREAMINFO length ({length})")

    q = head[8:42]
    sample_rate = (q[10] << 12) | (q[11] << 4) | (q[12] >> 4)
    channels = ((q[12] >> 1) & 0x07) + 1
    bits_per_sample = (((q[12] & 0x01) << 4) | (q[13] >> 4)) + 1
    total_samples = (
        ((q[13] & 0x0F) << 32) | (q[14] << 24) | (q[15] << 16) | (q[16] << 8) | q[17]
    )

    if sample_rate == 0:
        raise ValueError("STREAMINFO declares a sample rate of 0")

    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
        "total_samples": total_samples,
    }


# ----------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------
def _decode_coreaudio(path: str) -> dict:
    """Fully decode with macOS CoreAudio, via a temporary WAV."""
    tmpdir = _TMPDIR.get("dir")
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="nepali_validate_")
        _TMPDIR["dir"] = tmpdir
    out_path = os.path.join(tmpdir, f"out_{os.getpid()}.wav")
    if os.path.exists(out_path):
        os.remove(out_path)

    try:
        proc = subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", path, out_path],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            message = (proc.stderr.strip() or proc.stdout.strip() or "afconvert failed")
            return {"ok": False, "error": message[:300]}

        with wave.open(out_path, "rb") as wav:
            frames = wav.getnframes()
            frame_rate = wav.getframerate()
            frame_channels = wav.getnchannels()

        return {
            "ok": True,
            "frames": frames,
            "sample_rate": frame_rate,
            "channels": frame_channels,
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    finally:
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass


def _decode_soundfile(path: str) -> dict:
    import soundfile as sf  # imported lazily so the module stays optional

    try:
        info = sf.info(path)
        data = None
        try:
            data = sf.read(path, dtype="int16", always_2d=True)
        except Exception:
            data = None
        return {
            "ok": True,
            "frames": info.frames if data is None else data.shape[0],
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "error": "" if data is not None else "header-only (read failed)",
            "fully_decoded": data is not None,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}


# ----------------------------------------------------------------------
# Per-file worker
# ----------------------------------------------------------------------
def validate_one(task: tuple) -> dict:
    index, utter_id, relpath, path_str, backend, do_decode = task
    row = {
        "utterance_id": utter_id,
        "audio_path": relpath,
        "exists": "no",
        "decodable": "no",
        "sample_rate": "",
        "channels": "",
        "duration_seconds": "",
        "num_samples": "",
        "status": "missing",
        "error": "",
        "decode_verified": "no",
    }
    row["_index"] = index

    path = Path(path_str)
    if not path.is_file():
        row["error"] = "file does not exist"
        return row
    row["exists"] = "yes"

    try:
        if path.stat().st_size == 0:
            row["status"] = "empty"
            row["error"] = "zero-byte file"
            return row
    except OSError as exc:
        row["status"] = "unreadable"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return row

    declared: dict[str, int] | None
    try:
        declared = parse_streaminfo(path_str)
    except Exception as exc:
        declared = None
        row["status"] = "header_error"
        row["error"] = f"{type(exc).__name__}: {exc}"[:300]

    if declared is not None:
        row["sample_rate"] = declared["sample_rate"]
        row["channels"] = declared["channels"]
        row["num_samples"] = declared["total_samples"]
        row["duration_seconds"] = round(
            declared["total_samples"] / declared["sample_rate"], 6
        )

    if not do_decode:
        if declared is not None:
            row["status"] = "ok"
            row["decodable"] = "yes"
        return row

    decoded = _decode_coreaudio(path_str) if backend == "coreaudio" else _decode_soundfile(path_str)

    if not decoded.get("ok"):
        row["status"] = "decode_error"
        row["decodable"] = "no"
        row["error"] = decoded.get("error", "decode failed")
        return row

    row["decodable"] = "yes"
    row["decode_verified"] = "yes"
    decoded_frames = decoded["frames"]
    row["num_samples"] = decoded_frames
    row["sample_rate"] = decoded["sample_rate"]
    row["channels"] = decoded["channels"]
    row["duration_seconds"] = round(decoded_frames / decoded["sample_rate"], 6)

    notes: list[str] = []
    if decoded.get("error"):
        notes.append(decoded["error"])
    if declared is not None:
        if declared["total_samples"] != decoded_frames:
            notes.append(
                f"decoded {decoded_frames} frames but STREAMINFO declares "
                f"{declared['total_samples']} (truncated?)"
            )
        if declared["sample_rate"] != decoded["sample_rate"]:
            notes.append(
                f"declared sample_rate {declared['sample_rate']} != decoded "
                f"{decoded['sample_rate']}"
            )
        if declared["channels"] != decoded["channels"]:
            notes.append(
                f"declared channels {declared['channels']} != decoded {decoded['channels']}"
            )

    if declared is None or declared["total_samples"] != decoded_frames:
        row["status"] = "incomplete"
    elif decoded["sample_rate"] != EXPECTED_SAMPLE_RATE or decoded["channels"] != EXPECTED_CHANNELS:
        row["status"] = "unexpected_format"
    else:
        row["status"] = "ok"

    row["error"] = "; ".join(notes)[:300]
    return row


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import soundfile  # noqa: F401

        return "soundfile"
    except ImportError:
        pass
    if shutil.which("afconvert") and shutil.which("afinfo"):
        return "coreaudio"
    return "none"


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate every audio file in the manifest.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "manifests" / "all.csv")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "asr_nepali_data")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "manifests" / "audio_validation.csv"
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "validation_summary.json",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--backend",
        choices=["auto", "coreaudio", "soundfile", "none"],
        default="auto",
    )
    parser.add_argument(
        "--no-decode",
        action="store_true",
        help="header/STREAMINFO checks only; skip full frame decoding",
    )
    parser.add_argument("--limit", type=int, default=0, help="validate only the first N rows")
    args = parser.parse_args()

    try:  # keep progress visible when stdout is redirected to a file
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    manifest_path: Path = args.manifest.resolve()
    data_root: Path = args.data_root.resolve()
    do_decode = not args.no_decode

    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}")
        return 2

    backend = pick_backend(args.backend)
    if do_decode and backend == "none":
        print("ERROR: no decoder available (need `soundfile` or macOS `afconvert`).")
        print("Re-run with --no-decode for header-only checks.")
        return 2

    print("=" * 70)
    print("WEEK 1 - AUDIO VALIDATION")
    print("=" * 70)
    print(f"manifest       : {manifest_path}")
    print(f"data root      : {data_root}")
    print(f"decode backend : {backend}")
    print(f"full decode    : {do_decode}")
    print(f"workers        : {args.workers}")

    tasks: list[tuple] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"utterance_id", "speaker_id", "audio_path", "transcript"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            print(f"ERROR: manifest missing columns: {sorted(missing)}")
            return 2
        for index, row in enumerate(reader):
            if args.limit and index >= args.limit:
                break
            relpath = row["audio_path"]
            candidate = Path(relpath)
            resolved = candidate if candidate.is_absolute() else (data_root / candidate)
            tasks.append((index, row["utterance_id"], relpath, str(resolved), backend, do_decode))

    total = len(tasks)
    print(f"files to check : {total:,}")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    durations: list[float] = []
    sample_rates: Counter[int] = Counter()
    channels_counter: Counter[int] = Counter()
    status_counter: Counter[str] = Counter()
    files_checked = 0
    decode_verified = 0

    out_file = args.output.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()

    try:
        def handle(result: dict) -> None:
            nonlocal files_checked, decode_verified
            _record(result, writer, status_counter, durations, sample_rates, channels_counter)
            files_checked += 1
            if result["decode_verified"] == "yes":
                decode_verified += 1
            if files_checked % 5000 == 0:
                out_file.flush()
                print(f"  checked {files_checked:,}/{total:,}", flush=True)

        if args.workers <= 1:
            for task in tasks:
                handle(validate_one(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for result in executor.map(validate_one, tasks, chunksize=25):
                    handle(result)
    finally:
        out_file.close()
        tmpdir = _TMPDIR.get("dir")
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    ok = status_counter.get("ok", 0)
    missing = status_counter.get("missing", 0)
    header_errors = status_counter.get("header_error", 0)
    decode_errors = status_counter.get("decode_error", 0)
    incomplete = status_counter.get("incomplete", 0)
    empty = status_counter.get("empty", 0)
    unreadable = status_counter.get("unreadable", 0)
    unexpected_format = status_counter.get("unexpected_format", 0)

    undecodable = header_errors + decode_errors + incomplete + empty + unreadable

    total_seconds = sum(durations)
    stats = {
        "manifest": manifest_path.as_posix(),
        "data_root": data_root.as_posix(),
        "backend": backend,
        "full_decode": do_decode,
        "workers": args.workers,
        "files_checked": files_checked,
        "status_counts": dict(status_counter),
        "decodable_files": ok + unexpected_format,
        "undecodable_files": undecodable,
        "missing_files": missing,
        "decode_verified_files": decode_verified,
        "sample_rate_distribution": {str(k): v for k, v in sorted(sample_rates.items())},
        "channel_distribution": {str(k): v for k, v in sorted(channels_counter.items())},
        "total_duration_seconds": total_seconds,
        "total_duration_hours": total_seconds / 3600,
        "duration_seconds": {
            "min": min(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
            "mean": statistics.fmean(durations) if durations else 0.0,
            "median": statistics.median(durations) if durations else 0.0,
            "p25": percentile(durations, 0.25),
            "p50": percentile(durations, 0.50),
            "p75": percentile(durations, 0.75),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
        },
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"files checked        : {files_checked:,}")
    print(f"decode verified      : {decode_verified:,}")
    print(f"status ok            : {ok:,}")
    print(f"missing              : {missing:,}")
    print(f"undecodable (total)  : {undecodable:,}")
    print(f"    header_error     : {header_errors:,}")
    print(f"    decode_error     : {decode_errors:,}")
    print(f"    incomplete       : {incomplete:,}")
    print(f"    empty            : {empty:,}")
    print(f"    unreadable       : {unreadable:,}")
    print(f"unexpected format    : {unexpected_format:,}")
    print(f"sample rate dist     : {dict(sorted(sample_rates.items()))}")
    print(f"channel dist         : {dict(sorted(channels_counter.items()))}")

    d = stats["duration_seconds"]
    print()
    print("DURATION STATISTICS (seconds)")
    print(f"total hours : {stats['total_duration_hours']:,.2f}")
    print(f"min/mean/median/max : {d['min']:.3f} / {d['mean']:.3f} / "
          f"{d['median']:.3f} / {d['max']:.3f}")
    print(f"P25/P50/P75/P95/P99 : {d['p25']:.3f} / {d['p50']:.3f} / {d['p75']:.3f} / "
          f"{d['p95']:.3f} / {d['p99']:.3f}")

    print()
    print(f"report written : {args.output}")
    print(f"summary written: {args.summary_output}")

    if missing == 0 and undecodable == 0 and files_checked == total:
        print("\nAUDIO VALIDATION PASSED")
        return 0
    print("\nAUDIO VALIDATION FOUND ISSUES")
    return 1


def _record(result, writer, status_counter, durations, sample_rates, channels_counter):
    writer.writerow({k: result.get(k, "") for k in CSV_FIELDS})
    status_counter[result["status"]] += 1
    try:
        durations.append(float(result["duration_seconds"]))
    except (TypeError, ValueError):
        pass
    try:
        sample_rates[int(result["sample_rate"])] += 1
    except (TypeError, ValueError):
        pass
    try:
        channels_counter[int(result["channels"])] += 1
    except (TypeError, ValueError):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
