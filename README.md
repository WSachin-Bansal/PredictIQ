# PredictIQ

Multimodal deep learning for industrial machine fault detection and Remaining Useful Life (RUL) prediction.
Branches: sensor (LSTM/TCN), vision (transfer-learning CNN), audio (Mel-spectrogram CNN) -> fusion -> multi-task heads.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Layout (final; do not rearrange)

```
PredictIQ/
  .github/                  PR template, issue template
  configs/default.yaml      hyperparameters and paths
  data/raw/                 original datasets (NOT in git): mafaulda/, cmapss/, mvtec/
  data/processed/           preprocessed windows (NOT in git)
  checkpoints/              model weights (NOT in git)
  results/                  JSON result of every training run (in git)
  docs/                     GIT_GUIDE, PROJECT_PLAN, EXPERIMENT_LOG, DATASETS
  notebooks/                EDA and analysis notebooks
  reports/figures/          figures and the final report
  scripts/                  setup_folders.ps1, run_experiments.ps1, summarize_results.py, get_data.sh
  src/predictiq/
    data/                   MAFAULDA index/splits/preprocessing, sensor, audio, vision
    models/                 encoders, FaultNet (MAFAULDA), fusion model
    training/               losses, metrics, training scripts
    explain/                Grad-CAM / saliency (next phase)
    api/                    FastAPI backend (later phase)
  dashboard/                Streamlit dashboard (later phase)
  tests/
```

Create any missing folders with `powershell -File scripts/setup_folders.ps1`.

## Build order (do not skip ahead)

1. **Pick datasets (week 1).** Decide this BEFORE writing model code.
   - Sensor/RUL: NASA C-MAPSS (easiest, RUL labels) and/or NASA IMS or CWRU bearing vibration.
   - Vision: MVTec AD (or any defect image set).
   - Audio: ToyADMOS / MIMII.
   - These come from different machines, so there are no true aligned (sensor, image, audio) samples.
     Pick one of two honest strategies and document it in the report:
     (A) **Paired-by-label simulation**: build synthetic samples by pairing a sensor window, an image and an audio clip that share the same class label (normal / fault type) and a shared severity bucket.
     (B) **Use one aligned dataset** if you can find one (e.g. a bearing set that includes vibration + acoustic channels) and treat vision as a separate proof-of-concept.
   - Whichever you choose, never claim real industrial multimodal results.
2. **Unimodal baselines (M1, M2, M3).** Train each branch alone and get sensible numbers. Split by trajectory/engine/run, not by random window, to avoid leakage.
3. **Fusion (M4-M7).** Concatenate embeddings -> MLP -> three heads. Train the table of configurations with the same splits and seeds.
4. **Robustness.** Modality dropout during training + missing-modality evaluation.
5. **Explainability.** Grad-CAM for vision, saliency/integrated gradients for sensors, spectrogram saliency for audio.
6. **API + dashboard + replay simulation.**
7. **Testing, report, demo.**

## Run the smoke test

```bash
pytest -q
```

This builds all encoders and the fusion model on random tensors to confirm shapes and the multitask loss work before you plug in real data.

## MAFAULDA workflow (real aligned vibration + microphone data)

```bash
pip install kaggle                       # token at ~/.kaggle/kaggle.json
bash scripts/get_data.sh                 # -> data/raw/mafaulda (not committed)
python -m predictiq.data.mafaulda --raw data/raw/mafaulda --out data/processed/mafaulda
python -m predictiq.training.train_mafaulda --mode sensor
python -m predictiq.training.train_mafaulda --mode audio
python -m predictiq.training.train_mafaulda --mode fusion
```

- The preprocess step prints the fault classes it found. Check them against the folder names.
- Splits are by recording (never by window). `--split severity` holds out whole severity
  levels and is the more honest test; report both.
- `--task severity` classifies normal + each severity level (e.g. imbalance_6g ... 35g); use it when
  only a few fault types are available. Combine it with `--split file` only.
- Fusion also reports scores with sensor or audio hidden (missing-modality robustness).
- MAFAULDA has no RUL labels, and no images: the vision branch and RUL head are separate
  experiments, and any pairing with other datasets must be reported as simulated.
- Run each mode with several `--seed` values and report the mean and spread.

## Team and project tracking
- `docs/GIT_GUIDE.md` - git/GitHub for beginners; `CONTRIBUTING.md` - how we work
- `docs/PROJECT_PLAN.md` - who owns what, milestones
- `docs/EXPERIMENT_LOG.md` - every training run; `docs/DATASETS.md` - dataset notes
- `CHANGELOG.md` - notable changes
