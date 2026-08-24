"""Build the analysis dataset: download, epoch, repair, then delete the source EDF.

A raw session EDF is ~640 MB; the epochs derived from it are ~99 MB, because the
analysis only needs 3-second windows around 256 words (768 s of a 5200 s recording)
at 250 Hz rather than 500. Deleting the EDF after epoching keeps disk flat regardless
of how many subjects are processed, and re-fetching one costs about 40 seconds.

Resumable: a session whose epoch files already exist is skipped, so an interrupted run
picks up where it stopped. Safe to re-run.

Usage:
    PYTHONPATH=src uv run python scripts/build_dataset.py            # 20 subjects
    PYTHONPATH=src uv run python scripts/build_dataset.py 12 3       # 12 subjects, 3 sessions
    KEEP_EDF=1 PYTHONPATH=src uv run python scripts/build_dataset.py # don't delete sources
"""

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import mne

from peers.eeg import epochs_path, raw_path
from peers.events import DATA_ROOT, build_trial_table, load_all_events

mne.set_log_level("ERROR")

N_SUBJECTS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
KEEP_EDF = bool(os.environ.get("KEEP_EDF"))
CONDITIONS = [True, False]  # ocular-removed, ocular-kept

REPO = Path(__file__).resolve().parent.parent
DOWNLOAD_EVENTS = REPO / "scripts" / "download_peers_events.sh"
DOWNLOAD_EEG = REPO / "scripts" / "download_peers_eeg.sh"
BUILD_ONE = REPO / "scripts" / "build_one_session.py"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    log(f"target: {N_SUBJECTS} subjects x {N_SESSIONS} sessions, both ocular conditions")

    log("fetching event files (small)")
    subprocess.run(
        [str(DOWNLOAD_EVENTS), str(N_SUBJECTS), str(N_SESSIONS)],
        check=True, capture_output=True,
    )

    trials = build_trial_table(load_all_events())
    subjects = sorted(trials["subject_id"].unique())[:N_SUBJECTS]
    log(f"trial table: {len(trials):,} studied words across {len(subjects)} subjects")

    started = time.time()
    done = skipped = failed = 0

    for subject in subjects:
        available = sorted(trials.loc[trials["subject_id"] == subject, "session"].unique())
        for session in available[:N_SESSIONS]:
            if all(epochs_path(subject, session, oc).exists() for oc in CONDITIONS):
                skipped += 1
                continue

            if not raw_path(subject, session).exists():
                subprocess.run(
                    [str(DOWNLOAD_EEG), subject],
                    env={**os.environ, "SESSIONS": str(session)},
                    check=False, capture_output=True,
                )
            if not raw_path(subject, session).exists():
                log(f"  {subject} ses-{session}: EDF unavailable, skipping")
                failed += 1
                continue

            # One subprocess per session. AutoReject's joblib pool cannot recover once a
            # worker dies, so a single bad session would otherwise fail every session
            # after it. Isolating each one confines that to the session it happened in.
            # Announce before starting. A session takes ~5 minutes and previously logged
            # nothing until it finished, so a silent log was indistinguishable from a dead
            # process -- which is exactly the state that needs to be visible.
            log(f"  {subject} ses-{session}: starting")
            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, str(BUILD_ONE), subject, str(session)],
                cwd=REPO, env={**os.environ, "PYTHONPATH": "src"},
                capture_output=True, text=True,
            )
            if proc.returncode == 0:
                for line in proc.stdout.strip().splitlines():
                    tag, kept, rec, forg = line.split(":")
                    log(
                        f"  {subject} ses-{session} {tag:>14}: {int(kept):3d} epochs kept "
                        f"({rec} rec / {forg} forg)"
                    )
                log(f"  {subject} ses-{session} done in {time.time() - t0:.0f}s")
                done += 1
            else:
                tail = (proc.stderr.strip().splitlines() or ["no stderr"])[-1]
                log(f"  {subject} ses-{session}: FAILED -- {tail}")
                failed += 1

            if not KEEP_EDF:
                raw_path(subject, session).unlink(missing_ok=True)

    mins = (time.time() - started) / 60
    log(f"finished in {mins:.0f} min -- {done} built, {skipped} already cached, {failed} failed")
    total = sum(p.stat().st_size for p in Path(DATA_ROOT).parent.rglob("*_epo.fif"))
    log(f"epoch cache: {total / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
