#!/usr/bin/env python3
"""Single-file inference for the Nepali CNN-BiLSTM-CTC ASR model.

Usage:
    python scripts/infer.py /path/to/audio.flac
    python scripts/infer.py /path/to/audio.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from config import Config, PROJECT_ROOT
from dataset import load_vocab
from frontend import LogMelFrontend
from model import ASRModel


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "pilot" / "clean_full_best.pt"
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "checkpoints" / "pilot" / "clean_full_config.json"
)

DEFAULT_VOCAB = (
    PROJECT_ROOT / "manifests" / "vocab.json"
)


def greedy_decode(
    log_probs: torch.Tensor,
    out_lengths: torch.Tensor,
    blank: int = 0,
) -> list[list[int]]:
    """Greedy CTC decoding: (T, B, C) -> collapsed token IDs."""

    best = log_probs.argmax(dim=-1).transpose(0, 1)

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


def ids_to_text(
    ids: list[int],
    id_to_char: dict[int, str],
) -> str:
    """Convert token IDs into text."""

    return "".join(
        id_to_char.get(int(i), "")
        for i in ids
    )


def load_audio(
    path: Path,
    expected_sr: int,
) -> torch.Tensor:
    """Load audio as mono float32 tensor."""

    wav, sr = sf.read(
        str(path),
        dtype="float32",
    )

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != expected_sr:
        raise ValueError(
            f"Expected {expected_sr} Hz audio, but got {sr} Hz.\n"
            f"Please resample the audio to {expected_sr} Hz."
        )

    return torch.from_numpy(wav)


def main() -> None:

    # ---------------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Nepali ASR inference"
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Path to .wav or .flac audio file",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained .pt checkpoint",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to checkpoint config JSON",
    )

    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        choices=["mps", "cpu"],
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate paths
    # ---------------------------------------------------------------

    if not args.audio.exists():
        raise FileNotFoundError(
            f"Audio file not found: {args.audio}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    if not args.config.exists():
        raise FileNotFoundError(
            f"Config not found: {args.config}"
        )

    # ---------------------------------------------------------------
    # Device
    # ---------------------------------------------------------------

    device = torch.device(args.device)

    # ---------------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------------

    cfg = Config.load(args.config)

    # ---------------------------------------------------------------
    # Load vocabulary
    # ---------------------------------------------------------------

    vocab = load_vocab(DEFAULT_VOCAB)

    cfg.model.vocab_size = int(
        vocab["vocab_size"]
    )

    cfg.model.blank_id = int(
        vocab["blank_id"]
    )

    id_to_char = {
        int(i): char
        for char, i in vocab["char_to_id"].items()
    }

    # ---------------------------------------------------------------
    # Load audio
    # ---------------------------------------------------------------

    wav = load_audio(
        args.audio,
        expected_sr=cfg.features.sample_rate,
    )

    duration = (
        wav.shape[0]
        / cfg.features.sample_rate
    )

    # ---------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------

    frontend = LogMelFrontend(
        cfg.features
    ).to(device)

    with torch.no_grad():
        features = frontend(
            wav.to(device)
        )

    # Shape:
    # (T, 80)

    features = features.unsqueeze(0)

    # Shape:
    # (1, T, 80)

    input_lengths = torch.tensor(
        [features.shape[1]],
        dtype=torch.long,
        device=device,
    )

    # ---------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------

    model = ASRModel(
        cfg.model
    ).to(device)

    # ---------------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------------

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        payload["model_state"]
    )

    model.eval()

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    with torch.no_grad():

        log_probs, out_lengths = model(
            features,
            input_lengths,
        )

        decoded_ids = greedy_decode(
            log_probs,
            out_lengths,
            blank=cfg.model.blank_id,
        )

    transcript = ids_to_text(
        decoded_ids[0],
        id_to_char,
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    print()
    print("=" * 60)
    print("Nepali ASR Inference")
    print("=" * 60)

    print(f"Audio      : {args.audio}")
    print(f"Sample rate: {cfg.features.sample_rate} Hz")
    print(f"Duration   : {duration:.2f} sec")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    print()
    print("Transcript:")
    print(transcript)

    print("=" * 60)


if __name__ == "__main__":
    main()