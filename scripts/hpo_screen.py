#!/usr/bin/env python3
"""Week 6: controlled hyperparameter screening (weight decay + LR schedule).

Screens exactly two knobs against the full-data clean baseline -- weight decay
and learning-rate scheduling -- keeping everything else fixed (architecture,
dropout, batch size, seed, grad clip, max utterance, augmentation OFF).

Design constraints (see experiments/hyperparameter_screening.md):
  * full train split (139.06 h / 142,816 utts), identical to the baseline
  * per-experiment wall-clock cap (~110 min) so three arms fit a ~6 h budget
  * early stopping as a secondary guard
  * fully resumable: every experiment records its config + status before it
    starts and its results when it finishes; completed experiments are never
    re-run or overwritten.

Usage:
    python scripts/hpo_screen.py --status            # print current state
    python scripts/hpo_screen.py                     # run all pending arms in order
    python scripts/hpo_screen.py --only A            # run a single arm
    python scripts/hpo_screen.py --max-seconds 6600  # override the per-arm cap
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_JSON = EXPERIMENTS_DIR / "hyperparameter_results.json"
MASTER_MD = EXPERIMENTS_DIR / "hyperparameter_screening.md"
REPORTS = PROJECT_ROOT / "reports"
PYTHON = sys.executable

# --- shared protocol: everything except the screened knob --------------------
COMMON = {
    "device": "mps",
    "epochs": 40,          # ceiling only; the wall-clock cap stops the run first
    "subset_hours": 0.0,   # full train split
    "batch_size": 16,
    "num_workers": 4,
    "lr": 3e-4,
    "grad_clip": 5.0,
    "seed": 42,
    "max_utt_seconds": 16.0,
    "val_max_utts": 0,     # 0 = full val.csv (7,623 utts) for tight CER/WER
    "patience": 3,         # early stopping (secondary guard)
    "max_seconds": 6600.0, # ~110 min wall-clock cap per arm
}

# Baseline is already trained on full data -- we never retrain it, we only
# re-reference its saved numbers.
BASELINE_REPORT = REPORTS / "clean_full_overnight_metrics.json"
BASELINE_FULL_EVAL = REPORTS / "val_full_reeval.json"

EXPERIMENTS: dict[str, dict] = {
    "A": {"dir": "wd_1e-4", "wd": 1e-4, "scheduler": "none"},
    "B": {"dir": "wd_1e-3", "wd": 1e-3, "scheduler": "none"},
    "C": {"dir": "best_wd_scheduler", "wd": None, "scheduler": "plateau"},  # wd filled at runtime
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def exp_dir(key: str) -> Path:
    return EXPERIMENTS_DIR / EXPERIMENTS[key]["dir"]


def status_path(key: str) -> Path:
    return exp_dir(key) / "status.json"


def load_status(key: str) -> dict:
    return read_json(status_path(key)) or {"status": "PENDING"}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def existing_selection() -> dict | None:
    prev = read_json(RESULTS_JSON)
    return prev.get("selection") if prev else None


def baseline_entry() -> dict:
    """Reference numbers for the already-complete clean baseline (never retrained)."""
    rep = read_json(BASELINE_REPORT) or {}
    full = read_json(BASELINE_FULL_EVAL) or {}
    best = next((r for r in full.get("results", []) if "best" in Path(r["checkpoint"]).name), {})
    return {
        "key": "baseline",
        "name": "baseline (clean, reference)",
        "status": "COMPLETED",
        "source": "existing run -- reports/clean_full_overnight_metrics.json",
        "git_commit": git_hash(),
        "git_commit_note": "trained in a prior session (Week 5 overnight); commit recorded is the current repo tip",
        "started_at": "2026-09-30T22:00 local (Week 5 overnight)",
        "finished_at": "2026-10-01T05:32 local (Week 5 overnight)",
        "weight_decay": 1e-5,
        "scheduler": "none",
        "lr": 3e-4,
        "seed": 42,
        "dataset": "full train split (139.06 h / 142,816 utts), val full.csv (7,623)",
        "epochs_completed": len(rep.get("history", [])),
        "best_epoch": 12,
        "best_val_loss": rep.get("best_val_loss", 0.5138),
        "final_train_loss": rep.get("final_train_loss"),
        "val_cer": best.get("cer_greedy", 0.1392),
        "val_wer": best.get("wer_greedy", 0.4983),
        "cer_250_slice": rep.get("cer_greedy", 0.1454),
        "wer_250_slice": rep.get("wer_greedy", 0.4843),
        "val_slice": "full val.csv (7,623 utts)",
        "training_seconds": rep.get("total_training_seconds"),
        "lr_history": [3e-4],
        "scheduler_reductions": 0,
        "final_lr": 3e-4,
        "checkpoint": "checkpoints/clean_baseline/clean_full_overnight_best.pt",
        "observations": (
            "Val loss best 0.5138 @ ep 12 then rose monotonically to 0.765 by ep 40 "
            "while train loss fell to 0.0781 -> strong overfitting after ep 12; "
            "~70% of the 7.4 h run was post-best and wasted."
        ),
    }


def collect_completed() -> dict[str, dict]:
    """Merge baseline + every COMPLETED experiment into one results mapping."""
    results: dict[str, dict] = {"baseline": baseline_entry()}
    for key in EXPERIMENTS:
        st = load_status(key)
        metrics_path = exp_dir(key) / "metrics.json"
        m = read_json(metrics_path)
        if st.get("status") == "COMPLETED" and m is not None:
            results[key] = {
                "key": key,
                "name": EXPERIMENTS[key]["dir"],
                "status": "COMPLETED",
                "started_at": st.get("started_at"),
                "finished_at": st.get("finished_at"),
                "git_commit": st.get("git_commit"),
                "weight_decay": m.get("weight_decay"),
                "scheduler": m.get("scheduler"),
                "lr": m.get("learning_rate"),
                "seed": 42,
                "dataset": "full train split (139.06 h / 142,816 utts), val full.csv (7,623)",
                "epochs_completed": m.get("epochs_completed"),
                "best_epoch": m.get("best_epoch"),
                "best_val_loss": m.get("best_val_loss"),
                "final_train_loss": m.get("final_train_loss"),
                "val_cer": m.get("cer_greedy"),
                "val_wer": m.get("wer_greedy"),
                "val_cer_final": m.get("cer_greedy_final"),
                "val_wer_final": m.get("wer_greedy_final"),
                "lr_history": m.get("lr_history"),
                "scheduler_reductions": m.get("scheduler_reductions"),
                "final_lr": m.get("final_lr"),
                "early_stopped": m.get("early_stopped"),
                "wall_clock_cap_hit": m.get("wall_clock_cap_hit"),
                "training_seconds": m.get("total_training_seconds"),
                "checkpoint": st.get("checkpoint"),
                "observations": st.get("observations"),
            }
    return results


def select_best_wd(results: dict[str, dict]) -> tuple[str, float, str]:
    """Pick BEST_WD from {baseline 1e-5, A 1e-4, B 1e-3}.

    Priority (per task): validation CER, then WER, then val loss, with an
    explicit note on overfitting behaviour. Returns (key, wd, rationale).
    """
    candidates = {k: results[k] for k in ("baseline", "A", "B") if k in results}
    if len(candidates) < 3:
        missing = [k for k in ("baseline", "A", "B") if k not in candidates]
        raise RuntimeError(f"cannot select BEST_WD: missing results for {missing}")

    def rank(e: dict) -> tuple:
        return (e["val_cer"], e["val_wer"], e["best_val_loss"])

    ordered = sorted(candidates.items(), key=lambda kv: rank(kv[1]))
    key, best = ordered[0]
    lines = ["Weight-decay comparison (lower is better; primary = validation CER):"]
    for k, e in sorted(candidates.items(), key=lambda kv: rank(kv[1])):
        lines.append(
            f"  {k:8s} WD={e['weight_decay']:<8g} CER={e['val_cer']:.4f} "
            f"WER={e['val_wer']:.4f} val_loss={e['best_val_loss']:.4f} "
            f"(best ep {e['best_epoch']})"
        )
    # If CER and WER disagree on the winner, say so explicitly.
    by_cer = min(candidates.items(), key=lambda kv: kv[1]["val_cer"])[0]
    by_wer = min(candidates.items(), key=lambda kv: kv[1]["val_wer"])[0]
    if by_cer != by_wer:
        lines.append(
            f"  NOTE: CER prefers {by_cer} but WER prefers {by_wer}; CER was treated "
            f"as primary and WER as tie-breaker, with val loss as the final tie-break."
        )
    rationale = "\n".join(lines) + f"\n  => BEST_WD = {best['weight_decay']:g} (from arm {key})"
    return key, float(best["weight_decay"]), rationale


def build_command(key: str, wd: float, max_seconds: float, resume: bool) -> list[str]:
    c = COMMON
    d = exp_dir(key)
    cmd = [
        PYTHON, "scripts/train.py",
        "--device", c["device"],
        "--epochs", str(c["epochs"]),
        "--subset-hours", str(c["subset_hours"]),
        "--batch-size", str(c["batch_size"]),
        "--num-workers", str(c["num_workers"]),
        "--lr", str(c["lr"]),
        "--weight-decay", str(wd),
        "--scheduler", EXPERIMENTS[key]["scheduler"],
        "--patience", str(c["patience"]),
        "--max-seconds", str(max_seconds),
        "--val-max-utts", str(c["val_max_utts"]),
        "--checkpoint-dir", str(d.relative_to(PROJECT_ROOT)),
        "--tag", EXPERIMENTS[key]["dir"],
        "--no-augment",
        "--no-spec-augment",
        "--eval-samples", "8",
    ]
    if EXPERIMENTS[key]["scheduler"] == "plateau":
        cmd += ["--sched-factor", "0.5", "--sched-patience", "2", "--min-lr", "1e-6"]
    if resume:
        cmd.append("--resume")
    return cmd


def run_experiment(key: str, max_seconds: float) -> str:
    d = exp_dir(key)
    d.mkdir(parents=True, exist_ok=True)
    st = load_status(key)
    wd = EXPERIMENTS[key]["wd"]

    # --- resume or fresh start -------------------------------------------------
    latest = d / f"{EXPERIMENTS[key]['dir']}_latest.pt"
    resuming = st.get("status") in {"STARTED", "INTERRUPTED"} and latest.is_file()
    if st.get("status") == "STARTED" and not resuming:
        # A STARTED marker without a checkpoint means the previous attempt died
        # before the first epoch; restart cleanly (no results to lose).
        pass
    cmd = build_command(key, wd, max_seconds, resume=resuming)

    write_json(d / "config.json", {
        "experiment": key,
        "name": EXPERIMENTS[key]["dir"],
        "weight_decay": wd,
        "scheduler": EXPERIMENTS[key]["scheduler"],
        "shared_protocol": COMMON,
        "max_seconds": max_seconds,
        "command": " ".join(cmd),
        "created_at": now_iso(),
    })
    (d / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    write_json(status_path(key), {
        "status": "STARTED",
        "resumed": resuming,
        "started_at": st.get("started_at") or now_iso(),
        "last_started_at": now_iso(),
        "git_commit": git_hash(),
        "command": " ".join(cmd),
        "weight_decay": wd,
        "scheduler": EXPERIMENTS[key]["scheduler"],
    })

    t0 = time.time()
    log_path = d / "train.log"
    print("=" * 78)
    print(f"[{key}] {EXPERIMENTS[key]['dir']}  WD={wd:g} scheduler={EXPERIMENTS[key]['scheduler']}")
    print(f"[{key}] {' '.join(cmd)}")
    print(f"[{key}] log -> {log_path}")
    print("=" * 78)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {'RESUME' if resuming else 'START'} {now_iso()} =====\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(f"[{key}] {line}")
            sys.stdout.flush()
        rc = proc.wait()
    elapsed = time.time() - t0

    metrics = read_json(REPORTS / f"{EXPERIMENTS[key]['dir']}_metrics.json")
    status = "COMPLETED" if rc == 0 and metrics is not None else "FAILED"
    if status == "COMPLETED":
        write_json(d / "metrics.json", metrics)
        ckpt = f"checkpoints/hpo/{EXPERIMENTS[key]['dir']}/{EXPERIMENTS[key]['dir']}_best.pt"
        obs = observations_from(metrics)
        write_json(status_path(key), {
            "status": "COMPLETED",
            "started_at": st.get("started_at") or now_iso(),
            "finished_at": now_iso(),
            "git_commit": git_hash(),
            "command": " ".join(cmd),
            "weight_decay": wd,
            "scheduler": EXPERIMENTS[key]["scheduler"],
            "wall_seconds": round(elapsed, 1),
            "checkpoint": ckpt,
            "observations": obs,
        })
        print(f"[{key}] COMPLETED in {elapsed/60:.1f} min -> {ckpt}")
    else:
        write_json(status_path(key), {
            "status": "FAILED",
            "finished_at": now_iso(),
            "command": " ".join(cmd),
            "weight_decay": wd,
            "scheduler": EXPERIMENTS[key]["scheduler"],
            "return_code": rc,
        })
        print(f"[{key}] FAILED (rc={rc}); see {log_path}")
    return status


def observations_from(m: dict) -> str:
    h = m.get("history", [])
    if not h:
        return "no epochs completed"
    gap = h[-1]["train_loss"] - h[-1]["val_loss"]
    trend = "still improving" if m.get("best_epoch") == h[-1]["epoch"] else "past best"
    return (
        f"{len(h)} epochs in {m.get('total_training_seconds', 0)/60:.1f} min; "
        f"best val {m.get('best_val_loss'):.4f} @ ep {m.get('best_epoch')}; "
        f"final train {m.get('final_train_loss'):.4f} (train-val gap {gap:+.4f}); "
        f"best-ckpt CER {m.get('cer_greedy'):.4f} / WER {m.get('wer_greedy'):.4f}; "
        f"lr {m.get('learning_rate'):g} -> {m.get('final_lr'):g} "
        f"({m.get('scheduler_reductions', 0)} reductions); {trend}; "
        f"{'wall-clock cap hit' if m.get('wall_clock_cap_hit') else 'stopped by early-stopping' if m.get('early_stopped') else 'reached epoch ceiling'}."
    )


def write_master(results: dict[str, dict], selection: dict | None) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RESULTS_JSON, {
        "generated_at": now_iso(),
        "git_commit": git_hash(),
        "protocol": {
            "dataset": "OpenSLR SLR54; full train split 139.06 h / 142,816 utts",
            "validation": "full val.csv (7,623 utts, 26 unseen speakers)",
            "architecture": "CNN-BiLSTM-CTC 3,023,259 params (unchanged)",
            "optimizer": "AdamW", "lr": 3e-4, "dropout": 0.1, "batch_size": 16,
            "seed": 42, "grad_clip": 5.0, "max_utt_seconds": 16.0,
            "waveform_augmentation": False, "spec_augment": False,
            "per_experiment_wall_clock_cap_seconds": COMMON["max_seconds"],
            "scope": "limited screening (not exhaustive HPO)",
        },
        "experiments": results,
        "selection": selection,
    })
    write_master_md(results, selection)


def write_master_md(results: dict[str, dict], selection: dict | None) -> None:
    order = ["baseline", "A", "B", "C"]
    rows = []
    for k in order:
        e = results.get(k)
        if not e:
            continue
        rows.append(
            f"| {e['name']} | {e['weight_decay']:g} | {e['scheduler']} | "
            f"{e['val_cer']:.4f} | {e['val_wer']:.4f} | {e['best_val_loss']:.4f} | "
            f"{e.get('best_epoch')} | {e.get('epochs_completed')} |"
        )
    table = "\n".join(rows) if rows else "| _none yet_ | | | | | | | |"
    sel = ""
    if selection and "final_wd" in selection:
        sel = (
            f"\n## Selection\n\n```text\n"
            f"FINAL_WD = {selection['final_wd']:g}\n"
            f"FINAL_SCHEDULER = {selection['final_scheduler']}\n"
            f"FINAL_LR = {selection['final_lr']:g}\n```\n\n"
            f"Rationale:\n\n{selection['rationale']}\n"
        )
    elif selection and "best_wd" in selection:
        sel = (
            f"\n## Selection (weight decay resolved; scheduler arm pending)\n\n"
            f"```text\nBEST_WD = {selection['best_wd']:g}\n```\n\n"
            f"Rationale:\n\n{selection['rationale']}\n"
        )
    md = f"""# Nepali ASR — Controlled Hyperparameter Screening

Full-data clean baseline (WD 1e-5, no scheduler, no augmentation) is already
trained. This log screens **only** weight decay and LR scheduling; every other
hyperparameter is frozen (see `experiments/hyperparameter_results.json` for the
machine-readable protocol and per-arm records). This is a *limited* screening
narrowed to a promising configuration under a ~6 h compute budget — **not**
exhaustive HPO.

Scope explicitly NOT tuned: architecture, CNN layers, RNN size/layers, dropout,
batch size, optimizer, features, vocabulary, dataset, seed, gradient clipping.

## Results (validation = full `val.csv`, greedy decoding)

| Arm | Weight decay | Scheduler | CER | WER | best val loss | best ep | epochs run |
|---|---:|---|---:|---:|---:|---:|---:|
{table}
{sel}
## Per-experiment directories

Each arm writes its own directory under `experiments/` containing `config.json`,
`status.json`, `command.txt`, `train.log`, and (on success) `metrics.json`.
Completed arms are never overwritten; an interrupted arm resumes from its latest
checkpoint via `scripts/hpo_screen.py`.

## Interpretation

Limited hyperparameter screening was performed to identify a promising weight
decay and learning-rate scheduling strategy under the available compute budget.
The selected configuration is the one to carry into the single final full-data
run (with waveform augmentation + SpecAugment enabled). It is **not** claimed to
be globally optimal.

_Generated {now_iso()} at commit {git_hash()}._
"""
    MASTER_MD.write_text(md, encoding="utf-8")


def print_status() -> None:
    print(f"experiments dir : {EXPERIMENTS_DIR}")
    base = baseline_entry()
    print(f"  baseline    : COMPLETED (reference) WD={base['weight_decay']:g} "
          f"CER={base['val_cer']:.4f} WER={base['val_wer']:.4f}")
    for key in EXPERIMENTS:
        st = load_status(key)
        print(f"  {key:11s} : {st.get('status','PENDING'):9s} "
              f"WD={EXPERIMENTS[key]['wd']} scheduler={EXPERIMENTS[key]['scheduler']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Week 6 hyperparameter screening runner.")
    ap.add_argument("--only", choices=list(EXPERIMENTS), default=None)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="override the per-arm wall-clock cap")
    ap.add_argument("--status", action="store_true", help="print experiment state and exit")
    args = ap.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.status:
        print_status()
        write_master(collect_completed(), existing_selection())
        return 0

    max_seconds = args.max_seconds if args.max_seconds is not None else COMMON["max_seconds"]
    keys = [args.only] if args.only else list(EXPERIMENTS)

    for key in keys:
        st = load_status(key)
        if st.get("status") == "COMPLETED":
            print(f"[{key}] already COMPLETED -- skipping (results preserved)")
            write_master(collect_completed(), existing_selection())
            continue
        if key == "C" and EXPERIMENTS["C"]["wd"] is None:
            key_sel, wd, rationale = select_best_wd(collect_completed())
            EXPERIMENTS["C"]["wd"] = wd
            print(f"[selection] BEST_WD = {wd:g} (from arm {key_sel})\n{rationale}")
            sel = {"best_wd_from": key_sel, "final_wd": wd, "rationale": rationale}
        status = run_experiment(key, max_seconds)
        res = collect_completed()
        # refresh selection block whenever we have all three reference arms
        sel = None
        if all(k in res for k in ("baseline", "A", "B")):
            _, wd, rationale = select_best_wd(res)
            sel = {"best_wd": wd, "rationale": rationale}
            if key == "C" and "C" in res:
                sel = {
                    "final_wd": res["C"]["weight_decay"],
                    "final_scheduler": res["C"]["scheduler"],
                    "final_lr": res["C"]["lr"],
                    "rationale": rationale
                    + f"\n  Experiment C: WD={res['C']['weight_decay']:g} + "
                    f"ReduceLROnPlateau -> CER {res['C']['val_cer']:.4f} / "
                    f"WER {res['C']['val_wer']:.4f} vs best no-scheduler arm.",
                }
        write_master(res, sel)
        if status == "FAILED":
            print(f"[{key}] failed -- stopping so the failure can be inspected")
            return 1

    print("[done] master log ->", MASTER_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
