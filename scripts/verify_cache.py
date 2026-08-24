"""Check that every cached epoch file is complete and loadable.

Worth running after any crash, forced quit, or full disk. A half-written FIF is the
quiet kind of failure: build_dataset.py skips sessions whose files already exist, so a
truncated file would be reused for the rest of the project rather than rebuilt. Any file
reported here as CORRUPT should be deleted; the next build run will regenerate it.

Usage:  PYTHONPATH=src uv run python scripts/verify_cache.py
"""

import warnings

warnings.filterwarnings("ignore")

import sys
from collections import defaultdict

import mne
import numpy as np

from peers.eeg import DERIV_ROOT

mne.set_log_level("ERROR")

EXPECTED_CHANNELS = 125
EXPECTED_SFREQ = 250.0
CONDITIONS = ("ocular-removed", "ocular-kept")


def main() -> int:
    files = sorted(DERIV_ROOT.glob("*/*_epo.fif"))
    if not files:
        print("no cached epochs found")
        return 1

    corrupt = []
    have = defaultdict(set)

    # Every file is fully loaded to check its samples, so this reads the whole cache -- tens of
    # gigabytes off an external volume. Report progress, otherwise a healthy run is
    # indistinguishable from a hung one for several minutes.
    print(f"checking {len(files)} files (reads the entire epoch cache; expect several minutes)")

    for index, path in enumerate(files, start=1):
        print(f"\r  {index}/{len(files)}  {path.name[:60]:<60}", end="", flush=True)
        try:
            epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
            if len(epochs) != len(epochs.metadata):
                raise ValueError("metadata rows != epochs")
            if len(epochs.ch_names) != EXPECTED_CHANNELS:
                raise ValueError(f"{len(epochs.ch_names)} channels")
            if epochs.info["sfreq"] != EXPECTED_SFREQ:
                raise ValueError(f"sfreq {epochs.info['sfreq']}")
            if not np.isfinite(epochs.get_data(copy=False)).all():
                raise ValueError("non-finite samples")
        except Exception as exc:                                  # noqa: BLE001
            corrupt.append((path, exc))
            continue
        subject = path.parent.name
        session = int(path.name.split("_ses-")[1].split("_")[0])
        condition = next(c for c in CONDITIONS if c in path.name)
        have[(subject, session)].add(condition)

    print()                                       # close the carriage-returned progress line

    complete = {k for k, v in have.items() if len(v) == len(CONDITIONS)}
    partial = {k: v for k, v in have.items() if len(v) < len(CONDITIONS)}

    print(f"files checked      : {len(files)}")
    print(f"corrupt            : {len(corrupt)}")
    for path, exc in corrupt:
        print(f"    {path.name}: {exc}")
    print(f"complete sessions  : {len(complete)}  (both ocular conditions)")
    print(f"subjects           : {len({s for s, _ in complete})}")
    if partial:
        print(f"half-built sessions: {len(partial)}  (will be finished on the next run)")
        for (subject, session), conditions in sorted(partial.items()):
            missing = set(CONDITIONS) - conditions
            print(f"    {subject} ses-{session}: missing {', '.join(sorted(missing))}")

    if corrupt:
        print("\nDelete the corrupt files listed above, then re-run scripts/build_dataset.py.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
