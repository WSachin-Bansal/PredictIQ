# Project plan and work split

Every member owns real work packages and delivers them through their own branches and pull
requests. Replace `Member A-D` with names. Each package is a GitHub Issue (use the
"Task" template) so progress is tracked on a GitHub Project board (To do / In progress /
In review / Done).

| Member | Package | Branch names | Deliverables |
|---|---|---|---|
| **A: Data** | MAFAULDA data layer | `feature/data-eda`, `feature/split-checks` | Verify class/severity parsing on the real files, EDA plots (signals, spectra, class counts), `docs/DATASETS.md` filled in, extra data tests |
| **B: Sensor** | Vibration branch + RUL | `feature/sensor-baseline`, `feature/cmapss-rul` | M1 sensor-only results (3+ seeds, both splits), tuning of the 1D-CNN, C-MAPSS RUL baseline with `train_sensor.py` |
| **C: Audio + Vision** | Audio and image branches | `feature/audio-baseline`, `feature/vision-mvtec` | M2 audio-only results (3+ seeds, both splits), vision baseline on MVTec AD, label-paired simulation (clearly labelled as simulated) |
| **D: Fusion + Robustness** | Multimodal model | `feature/fusion-experiments`, `feature/missing-modality` | M4+ fusion results, modality dropout ablation, missing-modality table, results comparison script/plots |

Shared, later phases (assign when reached): explainability (Grad-CAM / saliency), FastAPI
backend, Streamlit dashboard, final report and demo. Everyone reviews at least two pull
requests written by someone else.

## Milestones
1. Repo live, everyone has cloned it and merged one small change (docs or test).
2. MAFAULDA preprocessed; splits agreed (`--split file` and `--split severity`).
3. Unimodal baselines (M1, M2) done with 3+ seeds.
4. Fusion + missing-modality results.
5. Explainability, API, dashboard.
6. Report and demo.

## Tracking
- Issues + Project board on GitHub (one issue per package/sub-task).
- `docs/EXPERIMENT_LOG.md`: one row per training run (who, command, seed, split, result).
- `CHANGELOG.md`: notable changes per milestone.
- Weekly 15-minute sync: review the board, assign next issues.
