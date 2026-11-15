# Experiment log

One row per training run. Copy the JSON from `results/` or the printed TEST line. Commit this
file with the pull request that introduces the result. Never edit old rows; add a new one.

| Date | Who | Branch / commit | Mode | Split | Seed | Test acc | Test macro-F1 | File acc | Notes (e.g. missing sensor / audio) |
| 2026-11-14 | Sachin | 77381aa | sensor | file | 42 | 0.9770 | 0.9730 | 0.9795 | 3-epoch trial, CPU, batch 64 |
| 2026-11-14 | Sachin | 77381aa | audio | file | 42 | 0.5723 | 0.4791 | 0.5729 | 3-epoch trial, CPU, batch 64 |
| 2026-11-14 | Sachin | 77381aa | fusion | file | 42 | 0.9400 | 0.9417 | 0.9463 | 3-epoch trial, CPU, batch 64; sensor hidden acc 0.4672, audio hidden acc 0.8843 |
| 2026-11-14 | Sachin | 77381aa | sensor | file | 42 | 0.9699 | 0.9667 | 0.9719 | 3-epoch trial, GPU RTX 2050, batch 64 |
| 2026-11-14 | Sachin | feature/full-runs-seed1 | sensor | file | 1 | 0.9886 | 0.9853 | 0.9872 | 20 epochs, batch 32, GPU RTX 2050 |
| 2026-11-14 | Sachin | feature/full-runs-seed1 | audio | file | 1 | 0.8255 | 0.8014 | 0.8440 | 20 epochs, batch 32, GPU RTX 2050 |
| 2026-11-14 | Sachin | feature/full-runs-seed1 | fusion | file | 1 | 0.9932 | 0.9929 | 0.9949 | 20 epochs, batch 32, GPU RTX 2050; sensor hidden acc 0.7292, audio hidden acc 0.9739 |


## Summary table for the report (mean +- std over seeds)

| Config | Split | Accuracy | Macro-F1 | Sensor hidden | Audio hidden |
|---|---|---|---|---|---|
| M1 sensor | file | | | n/a | n/a |
| M2 audio | file | | | n/a | n/a |
| M4 fusion | file | | | | |
| M1 sensor | severity | | | n/a | n/a |
| M2 audio | severity | | | n/a | n/a |
| M4 fusion | severity | | | | |
