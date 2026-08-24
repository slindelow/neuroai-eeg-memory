#!/usr/bin/env bash
# Fetch the behavioural event files from PEERS (OpenNeuro ds004395).
#
# Events only -- no EEG. About 11 MB total, so this takes seconds, not hours.
# The 640 MB EEG recordings stay on the server until Session 2 needs them.
#
# Usage:  ./scripts/download_peers_events.sh [n_subjects] [n_sessions]
#         defaults: 15 subjects, 5 sessions each

set -euo pipefail

N_SUBJECTS="${1:-15}"
N_SESSIONS="${2:-5}"
BASE="https://s3.amazonaws.com/openneuro.org/ds004395"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${NEUROAI_DATA_ROOT:-$REPO/data}/peers"

mkdir -p "$DEST"

echo "Fetching participant list..."
curl -sL --fail -o "$DEST/participants.tsv" "$BASE/participants.tsv"

# ltpFR subjects are the LTP* ids. Take the first N.
subjects=$(tail -n +2 "$DEST/participants.tsv" | cut -f1 | grep '^sub-LTP' | head -n "$N_SUBJECTS")

echo "Downloading events for $N_SUBJECTS subjects x $N_SESSIONS sessions"
echo

for sub in $subjects; do
  got=0
  for ses in $(seq 0 $((N_SESSIONS * 3))); do          # scan ahead; not every ses is ltpFR
    [ "$got" -ge "$N_SESSIONS" ] && break
    name="${sub}_ses-${ses}_task-ltpFR_events.tsv"
    out="$DEST/$sub/ses-$ses/$name"
    if [ -f "$out" ]; then got=$((got + 1)); continue; fi
    mkdir -p "$(dirname "$out")"
    if curl -sL --fail -o "$out" "$BASE/$sub/ses-$ses/eeg/$name" 2>/dev/null; then
      got=$((got + 1))
    else
      rm -f "$out"; rmdir "$(dirname "$out")" 2>/dev/null || true
    fi
  done
  echo "  $sub  ($got sessions)"
done

echo
echo "Done. Size on disk:"
du -sh "$DEST"
