#!/usr/bin/env bash
# Rejected first dataset. This script is not part of the main PEERS pipeline.
#
# It downloads OpenNeuro ds008496, the Gen-AI essay-writing EEG dataset that was
# dropped in stage 1 after a task confound. It is kept so that rejection stays
# reproducible. Do not use these files as inputs to the PEERS analysis.
#
# Fetch from the public S3 bucket. Downloads ~1.2 GB into data/ds008496/.
# Safe to re-run: files already on disk are skipped, so if it gets interrupted
# just run it again.
#
# Usage:  ./scripts/download_ds008496.sh

set -euo pipefail

BASE="https://s3.amazonaws.com/openneuro.org/ds008496"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${NEUROAI_DATA_ROOT:-$REPO/data}/ds008496"

echo "Downloading ds008496 -> $DEST"
echo "This takes a few minutes. Each line below is one subject finishing."
echo

mkdir -p "$DEST"

# Dataset-level metadata files.
for f in dataset_description.json README participants.tsv; do
  if [ ! -f "$DEST/$f" ]; then
    curl -sL --fail -o "$DEST/$f" "$BASE/$f"
  fi
  echo "  $f"
done
echo

# Per-subject EEG. Subjects 001-008 did the active-AI task, 009-015 the passive-AI task.
for i in $(seq -w 1 15); do
  if [ "$((10#$i))" -le 8 ]; then task=activeai; else task=passiveai; fi
  sub="sub-0$i"
  mkdir -p "$DEST/$sub/eeg"
  for ext in eeg.set eeg.fdt eeg.json events.tsv channels.tsv; do
    name="${sub}_task-${task}_${ext}"
    if [ ! -f "$DEST/$sub/eeg/$name" ]; then
      curl -sL --fail -o "$DEST/$sub/eeg/$name" "$BASE/$sub/eeg/$name"
    fi
  done
  echo "  $sub ($task)"
done

echo
echo "Done. Total size on disk:"
du -sh "$DEST"
