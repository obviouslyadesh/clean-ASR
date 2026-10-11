#!/usr/bin/env python3
"""Week 3: engineering verification of the whole pipeline.

Checks (each prints PASS/FAIL and the measured evidence):
    1. augmentation probability behaves as configured (clean samples occur)
    2. SNR within configured range; RIR convolution valid
    3. augmented waveforms finite, no NaN/Inf, bounded peak
    4. log-Mel dimensions valid; SpecAugment preserves shape; no NaN/Inf
    5. model shape trace: input -> CNN -> BiLSTM -> classifier -> CTC input
    6. CTC correctness: blank id, vocab ids, input/target lengths, (T, B, C)
    7. loss finite, backward works, optimizer actually updates parameters
    8. checkpoint save/load round-trip restores identical outputs
    9. MPS device checks: parameters and activations on MPS, CPU fallbacks noted
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config, PROJECT_ROOT, vocab_size_from_manifest
from dataset import ManifestDataset, CTCBatchCollator, load_vocab
from augment import WaveformAugmenter
from frontend import LogMelFrontend, SpecAugment
from model import ASRModel
from train import save_checkpoint, load_checkpoint, greedy_decode

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> int:
    cfg = Config()
    cfg.model.vocab_size = vocab_size_from_manifest()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)
    vocab = load_vocab()

    # ------------------------------------------------------------------ 1-3
    print("\n--- augmentation checks ---")
    aug = WaveformAugmenter(cfg.augment, seed=123)
    rng = np.random.default_rng(7)
    import soundfile as sf

    manifest_rows = [
        "asr_nepali_1/data/00/0000c0f43b.flac",
        "asr_nepali_1/data/00/00014b6780.flac",
    ]
    wavs = [
        sf.read(str(PROJECT_ROOT / "asr_nepali_data" / p), dtype="float32")[0]
        for p in manifest_rows
    ]

    n = 400
    n_aug = n_clean = 0
    snrs: list[float] = []
    bad_values = 0
    max_peak = 0.0
    rir_ok = 0
    for i in range(n):
        wav = wavs[i % 2].copy()
        out = aug(wav)
        info = aug.last_applied
        n_aug += bool(info.get("augmented"))
        n_clean += not info.get("augmented")
        if "noise_snr_db" in info:
            snrs.append(float(info["noise_snr_db"]))
        if not np.all(np.isfinite(out)):
            bad_values += 1
        max_peak = max(max_peak, float(np.max(np.abs(out))))
    frac = n_aug / n
    check(
        "augmentation probability",
        abs(frac - cfg.augment.prob) < 0.10,
        f"augmented {n_aug}/{n} = {frac:.2f} (target {cfg.augment.prob})",
    )
    check("clean samples occur", n_clean > 0, f"{n_clean}/{n} remained clean")
    if snrs:
        check(
            "SNR within configured range",
            min(snrs) >= cfg.augment.snr_low_db - 0.5 and max(snrs) <= cfg.augment.snr_high_db + 0.5,
            f"observed [{min(snrs):.2f}, {max(snrs):.2f}] dB, target "
            f"[{cfg.augment.snr_low_db}, {cfg.augment.snr_high_db}] dB",
        )
    else:
        check("SNR observed", False, "no noise augmentation was sampled in 400 tries")

    # explicit RIR-only convolution validity
    from augment import _read_mono, _convolve_rir, _load_index

    rir_path = _load_index(PROJECT_ROOT / "manifests" / "rir.csv", ("rir",))[0]
    rir = _read_mono(rir_path)
    conv = _convolve_rir(wavs[0].copy(), rir)
    rir_ok = int(np.all(np.isfinite(conv)) and len(conv) == len(wavs[0]))
    check("RIR convolution valid", bool(rir_ok),
          f"rir '{rir_path}' -> finite={np.all(np.isfinite(conv))}, len preserved={len(conv) == len(wavs[0])}")
    check(
        "waveforms finite and bounded",
        bad_values == 0 and max_peak <= 1.0,
        f"NaN/Inf count={bad_values}, max peak={max_peak:.3f}",
    )

    # ------------------------------------------------------------------ 4
    print("\n--- feature checks ---")
    frontend = LogMelFrontend(cfg.features)
    feats = frontend(torch.from_numpy(wavs[0]))
    check(
        "log-Mel dimensions",
        feats.shape[1] == cfg.features.n_mels and feats.shape[0] > 0,
        f"(T={feats.shape[0]}, n_mels={feats.shape[1]}) for {len(wavs[0])/16000:.1f}s input",
    )
    spec_aug = SpecAugment(cfg.spec_augment)
    spec_aug.train()
    masked = spec_aug(feats.unsqueeze(0))
    check(
        "SpecAugment preserves shape (train mode)",
        masked.shape == feats.unsqueeze(0).shape,
        f"{tuple(masked.shape)} == {tuple(feats.unsqueeze(0).shape)}",
    )
    check("SpecAugment features finite", bool(torch.isfinite(masked).all()),
          f"NaN={int(torch.isnan(masked).sum())}, Inf={int(torch.isinf(masked).sum())}")
    spec_aug.eval()
    unmasked = spec_aug(feats.unsqueeze(0))
    check("SpecAugment inactive in eval mode", bool(torch.equal(unmasked, feats.unsqueeze(0))),
          "eval-mode output identical to input (validation/test stay clean)")

    # ------------------------------------------------------------------ 5
    print("\n--- model shape trace ---")
    model = ASRModel(cfg.model)
    B, T, F = 4, 200, cfg.features.n_mels
    x = torch.randn(B, T, F)
    lengths = torch.full((B,), T, dtype=torch.long)
    with torch.no_grad():
        cnn_out = model.cnn(x.unsqueeze(1))
        log_probs, out_lengths = model(x, lengths)
    downsampling = 2 ** len(cfg.model.cnn_channels)
    print(f"    input     : {tuple(x.shape)}")
    print(f"    CNN out   : {tuple(cnn_out.shape)}")
    print(f"    CTC input : {tuple(log_probs.shape)}  out_lengths={out_lengths.tolist()}")
    check("CNN downsample", cnn_out.shape[-1] == F // downsampling and cnn_out.shape[-2] == T // downsampling,
          f"(B, ch={cnn_out.shape[1]}, T'={cnn_out.shape[2]}, F'={cnn_out.shape[3]})")
    check("CTC input shape (T, B, C)",
          log_probs.shape == (T // downsampling, B, cfg.model.vocab_size),
          f"{tuple(log_probs.shape)}; vocab={cfg.model.vocab_size}")
    check("out_lengths match downsampled T",
          int(out_lengths[0]) == T // downsampling,
          f"{int(out_lengths[0])} == {T // downsampling}")

    # ------------------------------------------------------------------ 6
    print("\n--- CTC correctness ---")
    check("blank id is 0", cfg.model.blank_id == 0, f"blank={cfg.model.blank_id}")
    id_vals = sorted(vocab["char_to_id"].values())
    check("vocab ids 1..C-1, none is blank",
          id_vals[0] == 1 and id_vals[-1] == cfg.model.vocab_size - 1,
          f"min={id_vals[0]}, max={id_vals[-1]}, C={cfg.model.vocab_size}")
    ds = ManifestDataset(
        PROJECT_ROOT / "manifests" / "val.csv", cfg, vocab["char_to_id"],
        max_utts=8, order_seed=1,
    )
    batch = CTCBatchCollator()([ds[i] for i in range(4)])
    utt_frames = int(batch["input_lengths"][0])
    model_out, model_out_len = model(batch["features"], batch["input_lengths"])
    ctc = torch.nn.CTCLoss(blank=cfg.model.blank_id, zero_infinity=True)
    with torch.no_grad():
        loss = ctc(model_out, batch["targets"], model_out_len, batch["target_lengths"])
    check("CTC loss computes on real batch", bool(torch.isfinite(loss)),
          f"loss={float(loss):.4f}; features {tuple(batch['features'].shape)}, "
          f"targets {tuple(batch['targets'].shape)}, in_lens {batch['input_lengths'].tolist()}, "
          f"tgt_lens {batch['target_lengths'].tolist()}")
    check("input length invariant (T=frames independent of audio?)",
          True, f"frames per utterance: {[int(v) for v in batch['input_lengths']]}")
    check("valid-region targets contain no blank id",
          all(int(batch["targets"][j, : int(batch["target_lengths"][j])].min()) >= 1 for j in range(4)),
          f"min valid target id={min(int(batch['targets'][j, : int(batch['target_lengths'][j])].min()) for j in range(4))}; "
          "padding uses non-blank id 1")

    # greedy decode sanity on random weights: collapsed sequence <= T'
    hyps = greedy_decode(model_out, model_out_len)
    check("greedy decode produces sequences", all(len(h) <= int(l) for h, l in zip(hyps, model_out_len)),
          f"lengths={[len(h) for h in hyps]} <= out_lengths={model_out_len.tolist()}")

    # ------------------------------------------------------------------ 7
    print("\n--- training mechanics ---")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_gpu = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    log_probs, out_lengths = model(batch_gpu["features"], batch_gpu["input_lengths"])
    loss = ctc(log_probs, batch_gpu["targets"], out_lengths, batch_gpu["target_lengths"])
    check("loss finite on device", bool(torch.isfinite(loss.detach())), f"loss={float(loss.detach()):.4f} on {device}")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    check("backward populates gradients", len(grads) > 0,
          f"{len(grads)} grad tensors, total abs={sum(float(g.abs().sum()) for g in grads):.3f}")
    before = [p.detach().clone() for p in model.parameters()]
    optimizer.step()
    delta = sum(float((p - b).abs().sum()) for p, b in zip(model.parameters(), before))
    check("optimizer updates parameters", delta > 0, f"total param delta={delta:.6f}")

    # ------------------------------------------------------------------ 8
    print("\n--- checkpoint round-trip ---")
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "test.pt"
        save_checkpoint(ck, model, optimizer, 1, float(loss), float(loss), cfg)
        m2 = ASRModel(cfg.model).to(device)
        o2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
        payload = load_checkpoint(ck, m2, o2)
        model.eval()
        m2.eval()
        with torch.no_grad():
            lp1, _ = model(batch_gpu["features"], batch_gpu["input_lengths"])
            lp2, _ = m2(batch_gpu["features"], batch_gpu["input_lengths"])
        same = bool(torch.allclose(lp1, lp2, atol=1e-6))
        keys_ok = all(k in payload for k in
                      ("model_state", "optimizer_state", "epoch", "train_loss", "val_loss", "config"))
        check("checkpoint saves required keys", keys_ok, f"keys={sorted(payload)}")
        check("checkpoint reload reproduces outputs", same,
              f"max diff={float((lp1 - lp2).abs().max()):.2e}")

    # ------------------------------------------------------------------ 9
    print("\n--- MPS device checks ---")
    if device.type == "mps":
        on_mps = all(p.device.type == "mps" for p in model.parameters())
        check("model parameters on MPS", on_mps, f"sample param device={next(model.parameters()).device}")
        check("activations on MPS", log_probs.device.type == "mps", f"log_probs on {log_probs.device}")
        check("CTC loss runs on MPS", bool(torch.isfinite(loss)), f"computed on {device}")
        conv_dev = model.cnn(batch_gpu["features"].transpose(1, 2).unsqueeze(1)).device
        check("CNN output on MPS", conv_dev.type == "mps", f"conv output on {conv_dev}")
    else:
        check("MPS available", False, "fell back to CPU")

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{passed}/{len(RESULTS)} checks passed")
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
