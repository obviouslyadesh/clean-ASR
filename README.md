# clean-ASR — Nepali ASR Streamlit demo

Minimal, self-contained bundle to **run and deploy** the final augmented
Nepali ASR model (CNN-BiLSTM + CTC, epoch 39). Everything the app needs at
inference time is here; nothing else.

- **Model:** character-level CTC, 3.02M params, 91 symbols, 16 kHz input
- **Headline quality:** test-set **CER 10.79% / WER 41.26%** on unseen speakers
- **Inference:** CPU, ~1 s per 5-s utterance (no GPU needed)

---

## Contents

| Path | Why it is here |
|---|---|
| `scripts/streamlit_app.py` | the Streamlit app (mic record + file upload) |
| `scripts/config.py` | config dataclasses (`PROJECT_ROOT`, feature/model config) |
| `scripts/frontend.py` | log-Mel feature extractor |
| `scripts/model.py` | CNN-BiLSTM + CTC network |
| `scripts/dataset.py` | provides `load_vocab` |
| `scripts/augment.py` | imported by `dataset.py` (train-only, unused at inference) |
| `manifests/vocab.json` | symbol ↔ id mapping, blank id (single source of truth) |
| `models/final_aug_best.pt` | the trained model (**embeds its own config**) |
| `requirements.txt` | pip dependencies |
| `.streamlit/config.toml` | minimal Streamlit settings |

**Deliberately not included** (not needed to run the model): training data,
training/eval scripts, other checkpoints, the noise/RIR corpora, reports, and
the separate config JSON — the checkpoint already carries its architecture and
feature config inside the `.pt`.

---

## Run locally

```bash
cd clean-ASR
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run scripts/streamlit_app.py
```

Opens `http://localhost:8501` → record with the mic or upload a file (audio is
auto-resampled to 16 kHz mono).

---

## Deploy to Streamlit Community Cloud (free)

1. **Push this folder to a GitHub repo.** The model is ~35 MB, under GitHub's
   per-file limit — no Git LFS required. Commit `models/final_aug_best.pt`.
2. Go to **share.streamlit.io** → **New app** → pick your repo and branch.
3. Set **Main file path** to:
   ```text
   scripts/streamlit_app.py
   ```
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically
   (the first build takes a few minutes because of `torch`).

> If `clean-ASR` is a subfolder of a larger repo, point the repo root at it
> (repo settings → "Root directory") or make `clean-ASR` its own repo.

## Deploy to Hugging Face Spaces (free, alternative)

1. Create a **New Space** → SDK: **Streamlit**.
2. Upload the files in this folder.
3. Tell the Space which file to run by setting the `app_file` in the Space's
   `README.md` YAML front-matter:
   ```yaml
   ---
   title: Nepali ASR
   sdk: streamlit
   app_file: scripts/streamlit_app.py
   pinned: false
   ---
   ```
4. The Space builds and starts automatically.

---

## Notes, limits, and expected behaviour

- **CPU-only in the cloud** (no Apple MPS): inference is ~1 s/utterance, fine for
  interactive use. The sidebar Device selector will show `cpu` only.
- **Cold starts:** on scale-to-zero hosts the first request after idle takes
  ~5–15 s (torch import + model load).
- **Sample rate:** the model needs 16 kHz; recordings/uploads at 44.1/48 kHz are
  resampled automatically.
- **Accuracy caveat:** the 10.79% CER is on *clean read speech*. Laptop-mic,
  far-field, or noisy audio will score noticeably worse. The model is
  character-level with no language model or punctuation.
- **Repo size:** GitHub warning threshold is 50 MB; the single 35 MB model is
  fine, but avoid committing the `venv/`.
