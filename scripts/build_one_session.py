"""Build both epoch sets for a single session, in a process of its own.

Run as a subprocess by build_dataset.py. The isolation is the point: AutoReject
parallelises through joblib, and if a worker dies (memory pressure, for instance) the
process pool is left permanently unusable -- every later call raises
"PicklingError: Could not pickle the task to send it to the workers", even for sessions
that would otherwise be fine. One long run poisons the rest of the batch.

Confining a session to its own process bounds that failure to one session: the poisoned
pool dies with the process and the next session starts clean.

Usage:  python scripts/build_one_session.py sub-LTP063 0
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import mne

from peers.eeg import build_session_epochs
from peers.events import build_trial_table, load_all_events

mne.set_log_level("ERROR")


def main() -> int:
    subject, session = sys.argv[1], int(sys.argv[2])
    trials = build_trial_table(load_all_events())
    for remove_ocular in (True, False):
        epochs = build_session_epochs(subject, session, trials, remove_ocular=remove_ocular)
        tag = "ocular-removed" if remove_ocular else "ocular-kept"
        print(
            f"{tag}:{len(epochs)}:{len(epochs['recalled'])}:{len(epochs['forgotten'])}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
