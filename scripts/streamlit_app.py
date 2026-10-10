#!/usr/bin/env python3
"""Streamlit demo: transcribe Nepali speech with the trained CNN-BiLSTM-CTC model.

Run:
    streamlit run scripts/streamlit_app.py

Two input paths, both reduced to 16 kHz mono before the log-Mel frontend:
  * microphone recording  (st.audio_input -> WAV supplied by the browser)
  * file upload           (.wav / .flac / .ogg / .mp3)

The feature extraction, model and greedy decoding reuse the exact code paths
from scripts/infer.py, so a demo transcript matches
`python scripts/infer.py <file> --checkpoint ...` on the same audio.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio

# scripts/ on sys.path so `config`, `dataset`, `frontend`, `model` import the
# same way they do in the other CLIs (train.py / infer.py / eval_val.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, PROJECT_ROOT  # noqa: E402
from dataset import load_vocab  # noqa: E402
from frontend import LogMelFrontend  # noqa: E402
from model import ASRModel  # noqa: E402

TARGET_SR = 16_000
VOCAB_PATH = PROJECT_ROOT / "manifests" / "vocab.json"

# Sidebar presets: label -> checkpoint path.
# Deployment bundle ships the final augmented model (epoch 39) only. Its
# architecture/feature config is embedded in the .pt, so no config JSON is needed.
CHECKPOINTS: dict[str, Path] = {
    "Final augmented (epoch 39)": PROJECT_ROOT / "models" / "final_aug_best.pt",
}


# --------------------------------------------------------------------------- #
# Core inference (importable / testable without a running Streamlit server)
# --------------------------------------------------------------------------- #
def load_model_impl(ckpt_path: Path, device_name: str):
    """Build the model + frontend from a checkpoint. Returns (model, front, id_to_char)."""
    device = torch.device(device_name)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Prefer the architecture/feature config embedded in the checkpoint; fall
    # back to the project defaults (identical for both checkpoints here).
    if isinstance(payload.get("config"), dict):
        cfg = Config.from_dict(payload["config"])
    else:
        cfg = Config()

    # Vocab is the single source of truth for symbol set / blank id.
    vocab = load_vocab(VOCAB_PATH)
    cfg.model.vocab_size = int(vocab["vocab_size"])
    cfg.model.blank_id = int(vocab["blank_id"])
    id_to_char = {int(i): c for c, i in vocab["char_to_id"].items()}

    model = ASRModel(cfg.model).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    frontend = LogMelFrontend(cfg.features).to(device)
    return model, frontend, id_to_char


def decode_audio_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    """Decode in-memory audio to a mono float32 tensor. Returns (wav, sample_rate)."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # downmix to mono
    wav = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
    return wav, int(sr)


def resample_to_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Resample a mono waveform to 16 kHz (no-op when already 16 kHz)."""
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def greedy_decode(log_probs: torch.Tensor, out_lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    """Greedy CTC decoding: (T, B, C) -> collapsed token ID sequences."""
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


def transcribe(model, frontend, id_to_char, wav: torch.Tensor, device_name: str) -> str:
    """Waveform (16 kHz mono) -> transcript string."""
    device = torch.device(device_name)
    with torch.no_grad():
        feats = frontend(wav.to(device))          # (T, n_mels)
        feats = feats.unsqueeze(0).to(device)     # (1, T, n_mels)
        in_lengths = torch.tensor([feats.shape[1]], dtype=torch.long, device=device)
        log_probs, out_lengths = model(feats, in_lengths)
        decoded = greedy_decode(log_probs, out_lengths, blank=0)
    return "".join(id_to_char.get(int(i), "") for i in decoded[0])


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, device_name: str):
    """Cached wrapper so the checkpoint is loaded once per (checkpoint, device)."""
    return load_model_impl(Path(ckpt_path), device_name)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _run(ckpt_path: str, device_name: str, data: bytes, label: str) -> None:
    model, frontend, id_to_char = load_model(ckpt_path, device_name)

    wav, sr = decode_audio_bytes(data)
    duration = wav.shape[0] / sr
    wav16 = resample_to_16k(wav, sr)

    st.audio(data, format="audio/wav")
    st.caption(
        f"{label} · input {sr} Hz · {duration:.2f} s"
        + ("" if sr == TARGET_SR else f" · resampled to {TARGET_SR} Hz")
    )

    with st.spinner("Transcribing…"):
        text = transcribe(model, frontend, id_to_char, wav16, device_name)

    if text.strip():
        st.success(text)
    else:
        st.warning("No speech recognised (empty transcript).")
    st.text_area("Transcript (copyable)", value=text, height=90)


def main() -> None:
    st.set_page_config(page_title="Nepali ASR", page_icon="🎙️", layout="centered")
    st.title("🎙️ Nepali Speech Recognition")
    st.caption("CNN-BiLSTM + CTC · character-level · 91-symbol vocab · 16 kHz input")

    with st.sidebar:
        st.header("Model")
        ckpt_label = st.selectbox("Checkpoint", list(CHECKPOINTS))
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        device_name = st.selectbox("Device", devices, index=0)
        st.caption("Record with your microphone or upload a file; both are "
                   "resampled to 16 kHz mono before inference.")

    ckpt_path = str(CHECKPOINTS[ckpt_label])
    if not Path(ckpt_path).exists():
        st.error(f"Checkpoint not found: {ckpt_path}")
        st.stop()

    tab_rec, tab_up = st.tabs(["🎙️ Record", "📁 Upload"])

    with tab_rec:
        recording = st.audio_input("Record audio")
        if recording is not None:
            _run(ckpt_path, device_name, recording.getvalue(), "recording")

    with tab_up:
        upload = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "mp3"])
        if upload is not None:
            _run(ckpt_path, device_name, upload.getvalue(), upload.name)


if __name__ == "__main__":
    main()
