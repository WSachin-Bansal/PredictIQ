# Datasets

## MAFAULDA (aligned vibration + microphone) - main dataset
- Location: `data/raw/mafaulda` (not in git). Processed windows: `data/processed/mafaulda`.
- Each CSV: 8 columns, 50 kHz: tachometer, 2 x triaxial accelerometer, microphone.
- Class folders found (fill in after running the preprocess step): _TODO_
- Number of recordings / windows per class: _TODO (copy the table printed by the preprocess step)_
- Splits: by recording; `file` (random files per class) and `severity` (held-out severity levels).
- Known limitations: lab test rig, clean conditions; no RUL labels; no images.

## Other datasets (separate experiments)
- NASA C-MAPSS: sensor-only RUL baseline.
- MVTec AD (or similar): vision baseline. Pairing with MAFAULDA samples is a label-based
  SIMULATION and must be reported as such.

## Honesty rule
Only sensor + audio fusion on MAFAULDA is real aligned multimodal data. Anything involving
vision or RUL together with it is simulated or unimodal, and the report must say so.
