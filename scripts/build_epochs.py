"""Preprocess and epoch every downloaded PEERS session, cached to data/derivatives/.

Each session is built twice -- with and without ocular correction -- so that the
subsequent memory effect can be compared between them. An effect present only in the
uncorrected data is an eye-movement artifact.
"""

import warnings

warnings.filterwarnings("ignore")

import mne

from peers.eeg import DERIV_ROOT, build_session_epochs, raw_path
from peers.events import build_trial_table, load_all_events

mne.set_log_level("ERROR")

SUBJECTS = ["sub-LTP063", "sub-LTP075"]
SESSIONS = [0, 1, 2]

trials = build_trial_table(load_all_events())

for subject in SUBJECTS:
    for session in SESSIONS:
        if not raw_path(subject, session).exists():
            print(f"{subject} ses-{session}: no EDF, skipping")
            continue
        for remove_ocular in (True, False):
            tag = "ocular-removed" if remove_ocular else "ocular-kept"
            epochs = build_session_epochs(subject, session, trials, remove_ocular=remove_ocular)
            print(
                f"{subject} ses-{session} {tag:>14}: "
                f"{len(epochs):3d} epochs "
                f"({len(epochs['recalled'])} recalled / {len(epochs['forgotten'])} forgotten)",
                flush=True,
            )

print(f"\ncache: {DERIV_ROOT}")
