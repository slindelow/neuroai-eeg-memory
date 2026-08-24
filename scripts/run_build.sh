#!/usr/bin/env bash
# Supervisor for build_dataset.py: restart it if it dies, until the target is reached.
#
# The build has been killed twice by memory pressure, silently -- no traceback, no error
# line, because a process killed by the OS never gets to write one. Since build_dataset.py
# is resumable and idempotent (it skips sessions already cached), the correct response to
# an unexplained death is simply to start it again.
#
# Also kills orphaned AutoReject workers between attempts. joblib's workers do not die
# with their parent; left behind, they hold ~250 MB each and make the next attempt more
# likely to fail than the last.
#
# Usage:  ./scripts/run_build.sh [n_subjects] [n_sessions]
#         nohup ./scripts/run_build.sh 40 3 >> data/build.log 2>&1 &

set -uo pipefail

N_SUBJECTS="${1:-40}"
N_SESSIONS="${2:-3}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO"

if pgrep -f "build_dataset.py" > /dev/null; then
  echo "[$(date +%H:%M:%S)] a build is already running -- refusing to start a second one"
  exit 1
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "[$(date +%H:%M:%S)] supervisor: attempt $attempt/$MAX_ATTEMPTS"

  PYTHONPATH=src uv run python scripts/build_dataset.py "$N_SUBJECTS" "$N_SESSIONS"
  status=$?

  # Reap workers orphaned by a killed parent before trying again.
  pkill -f "joblib.externals.loky" 2>/dev/null && \
    echo "[$(date +%H:%M:%S)] supervisor: cleaned up orphaned AutoReject workers"

  if [ "$status" -eq 0 ]; then
    echo "[$(date +%H:%M:%S)] supervisor: build completed normally"
    exit 0
  fi

  echo "[$(date +%H:%M:%S)] supervisor: exited with status $status -- restarting in 60s"
  sleep 60
done

echo "[$(date +%H:%M:%S)] supervisor: gave up after $MAX_ATTEMPTS attempts"
exit 1
