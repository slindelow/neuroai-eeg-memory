#!/usr/bin/env bash
# Fetch PEERS EEG recordings (ds004395) for selected subjects and sessions.
#
# EDFs are large (~600-700 MB per session), so this is deliberately selective.
# Safe to re-run: existing files are skipped.
#
# Usage:  ./scripts/download_peers_eeg.sh sub-LTP063 sub-LTP075
#         SESSIONS="0 1 2" ./scripts/download_peers_eeg.sh sub-LTP063

set -euo pipefail

SESSIONS="${SESSIONS:-0 1 2}"
BASE="https://s3.amazonaws.com/openneuro.org/ds004395"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${NEUROAI_DATA_ROOT:-$REPO/data}/peers"

[ "$#" -ge 1 ] || { echo "usage: $0 <subject> [subject ...]"; exit 1; }

for sub in "$@"; do
  for ses in $SESSIONS; do
    out="$DEST/$sub/ses-$ses"
    mkdir -p "$out"
    for ext in eeg.edf channels.tsv space-CapTrak_electrodes.tsv; do
      case "$ext" in
        space-CapTrak_electrodes.tsv) name="${sub}_ses-${ses}_${ext}" ;;
        *)                            name="${sub}_ses-${ses}_task-ltpFR_${ext}" ;;
      esac
      # Resume rather than skip. A previous run killed mid-transfer leaves a partial
      # file, and "skip if present" would keep that truncated file forever -- which
      # surfaces much later as a confusing EDF read error rather than a download error.
      if ! curl -sL --fail -C - -o "$out/$name" "$BASE/$sub/ses-$ses/eeg/$name"; then
        # curl exits 33 when the server refuses a range request but the file is
        # already complete; only treat a genuinely absent file as an error.
        [ -s "$out/$name" ] || { rm -f "$out/$name"; echo "  !! missing: $name"; }
      fi
    done
    echo "  $sub ses-$ses  $(du -h "$out" | cut -f1)"
  done
done

echo
echo "Total on disk:"
du -sh "$DEST"
df -h "$DEST" | tail -1
