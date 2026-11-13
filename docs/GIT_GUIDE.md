# Git & GitHub guide for our team (4 people, beginners)

## One-time setup (everyone, on your own computer)
1. Install Git: https://git-scm.com/downloads
2. Tell Git who you are. **Use the same email as your GitHub account**, otherwise your commits
   will not be linked to your GitHub profile:
   ```bash
   git config --global user.name  "Your Real Name"
   git config --global user.email "you@example.com"
   ```
3. Log in to GitHub from the terminal the first time you push (a browser window or a
   Personal Access Token will be requested; GitHub shows the steps).

## Project owner: create the repo (once)
1. On github.com click **New repository**. Name: `PredictIQ`. Leave "Add README / .gitignore"
   **unchecked** (we already have them). Create it.
2. In your PredictIQ folder:
   ```bash
   git init -b main
   git add .
   git status                      # CHECK: data/ and .venv/ must NOT be listed
   git commit -m "Initial project scaffold"
   git remote add origin https://github.com/<owner>/PredictIQ.git
   git push -u origin main
   ```
3. Add teammates: repo **Settings -> Collaborators -> Add people** (they must accept the
   email invitation).
4. Recommended: **Settings -> Branches -> Add rule** for `main`: require a pull request
   before merging. This forces everyone to work through branches and reviews.

## Teammates: get the project (once)
```bash
git clone https://github.com/<owner>/PredictIQ.git
cd PredictIQ
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
```
The dataset is NOT in git. Each person downloads MAFAULDA into `data/raw/mafaulda`
(see README).

## Daily workflow for every task
```bash
git checkout main
git pull                                  # get everyone's latest work
git checkout -b feature/audio-baseline    # one branch per task (see docs/PROJECT_PLAN.md)

# ... edit files, run code ...

git status                                # what changed?
git add src/predictiq/training/train_audio.py tests/test_audio.py   # add specific files
git commit -m "Add audio-only baseline training script"
git push -u origin feature/audio-baseline
```
Then on GitHub click **Compare & pull request**, fill in the template, and ask a teammate
to review. After approval click **Merge**, then locally:
```bash
git checkout main && git pull
```

## Rules of thumb
- Commit small and often, with a message that says what changed and why.
- Never work directly on `main`. Never commit data, checkpoints, `kaggle.json` or passwords.
- `git pull` before you start work every day.
- Only add files you understand: prefer `git add <file>` over `git add .`
- If a merge conflict appears, do not panic: open the file, keep the right lines, delete
  the `<<<<<<<`, `=======`, `>>>>>>>` markers, then `git add <file>` and `git commit`.
  Ask the team if unsure.
- Mistake before pushing? `git restore <file>` discards edits; `git commit --amend` fixes
  the last message. Do not use `git push --force` on shared branches.

## Seeing who did what
```bash
git log --oneline --graph --decorate --all
git shortlog -sn           # commits per person
```
On GitHub: **Insights -> Contributors** and the pull request list.
