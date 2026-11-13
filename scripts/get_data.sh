#!/usr/bin/env bash
# Download MAFAULDA from Kaggle into data/raw/mafaulda (data is NOT stored in git).
# Needs: pip install kaggle, and your API token at ~/.kaggle/kaggle.json
set -euo pipefail

DEST="${1:-data/raw/mafaulda}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Run: pip install kaggle" >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d uysalserkan/fault-induction-motor-dataset -p "$DEST" --unzip
echo "Downloaded to $DEST. Folder structure (check it looks right):"
find "$DEST" -maxdepth 3 -type d | head -40
