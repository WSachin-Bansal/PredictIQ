# Contributing

Read `docs/GIT_GUIDE.md` first. Summary:

1. Pick or create an Issue for your task; assign it to yourself.
2. `git checkout main && git pull`, then `git checkout -b feature/<short-name>`.
3. Commit small, clear changes (what + why). Run `pytest -q` before pushing.
4. Open a Pull Request using the template; at least one teammate reviews it.
5. Merge only after approval; delete the branch afterwards.

Do not commit: datasets, `data/`, `checkpoints/`, `results/*.pt`, API keys, `kaggle.json`.
Add a row to `docs/EXPERIMENT_LOG.md` for every result you report, and a line to
`CHANGELOG.md` for notable changes.

Commit message style: imperative, specific. Good: `Add severity-held-out split`.
Bad: `update`, `fix stuff`.
