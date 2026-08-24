"""Loading PEERS behavioural event files and deriving recall labels."""

import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DATA_ROOT = Path(os.environ.get("NEUROAI_DATA_ROOT", REPO_ROOT / "data"))
DATA_ROOT = PROJECT_DATA_ROOT / "peers"


def load_events(path: Path) -> pd.DataFrame:
    """Read one PEERS events.tsv into a DataFrame.

    Columns of interest (there are 20 in total):
        onset        float   seconds from the start of the recording
        trial_type   str     WORD, REC_WORD, FFR_REC_WORD, RECOG_TARGET, ...
        subject_id   str     derived here, e.g. "sub-LTP063"
        session      int     which session this file is
        trial        float   which word list (1-16). <= 0 means a practice list.
        item_name    str     the word itself, e.g. "PIE"
        item_num     float   the word's id in the word pool. -1 = intrusion/vocalisation.
    """
    df = pd.read_csv(path, sep="\t")
    df["source_file"] = path.name
    # The file's own `subject` column omits the "sub-" prefix; normalise so that one
    # session and many sessions have identical columns.
    df["subject_id"] = "sub-" + df["subject"].astype(str)
    return df


def load_all_events(root: Path = DATA_ROOT) -> pd.DataFrame:
    """Stack every events.tsv under ``root`` into one DataFrame."""
    frames = [load_events(p) for p in sorted(root.glob("sub-*/ses-*/*_events.tsv"))]
    if not frames:
        raise FileNotFoundError(
            f"No events files under {root}. Run ./scripts/download_peers_events.sh first."
        )
    return pd.concat(frames, ignore_index=True)


#: Fields that together identify one presentation of one word. The word pool is reused
#: across lists, so item_num alone is ambiguous — PIE in list 3 and PIE in list 11 are
#: different encoding events.
PRESENTATION_KEYS = ["subject_id", "session", "trial", "item_num"]


def unaligned_presentations(events: pd.DataFrame) -> pd.DataFrame:
    """Studied-word events the dataset could not place in EEG time.

    PEERS events are timed by the task computer and aligned to the amplifier using sync
    pulses. An event with no corresponding EEG sample is marked ``sample = -1``, which the
    onset column then reports as -1/sfreq seconds (-0.002 s at 500 Hz). These are real
    presentations that happened, but there is no EEG for them.

    The condition is stated as a property of the defect rather than as a list of affected
    sessions, so it applies unchanged to any data added later.
    """
    return events[
        (events.trial_type == "WORD")
        & (events.trial >= 1)
        & (events["sample"] < 0)
    ]


def build_trial_table(events: pd.DataFrame) -> pd.DataFrame:
    """One row per studied word, labelled by whether it was later recalled.

    Returns columns:
        subject_id, session, trial, serial_position, item_name, item_num, recalled

    Practice lists (trial <= 0) are excluded, as are presentations the dataset could not
    align to the EEG recording (see `unaligned_presentations`). Both exclusions happen here
    and nowhere else, so every downstream analysis sees the same set of trials.

    `recalled` is defined by **immediate free recall** (REC_WORD), not final free recall
    or recognition. Immediate recall is the standard label in the subsequent-memory-effect
    literature, which keeps results comparable. It does conflate encoding success with
    retrieval success — a word that fails here but returns during final free recall was
    stored fine and simply wasn't found on the first attempt.

    Intrusions are excluded (item_num == -1 marks a word that was never studied, or a
    vocalisation). Prior-list intrusions — a genuine word recalled during the wrong list's
    recall period — match no presentation in that list and so credit nothing, which is the
    intended behaviour.

    Recall events are deliberately *not* filtered on EEG alignment. A recall is matched to
    its presentation by subject, session, list and item, never by time, so an unaligned
    recall still correctly credits an aligned presentation. Filtering them would silently
    relabel recalled words as forgotten.
    """
    studied = events[
        (events.trial_type == "WORD")
        & (events.trial >= 1)
        & (events["sample"] >= 0)
    ].copy()

    # Serial position restarts at 1 for every list. Counting across the session instead
    # would run to 256 and no longer track rehearsal or recency — position is used as a
    # baseline predictor (B1), so it has to be measured faithfully rather than obscured.
    studied = studied.sort_values(["subject_id", "session", "trial", "onset"])
    studied["serial_position"] = (
        studied.groupby(["subject_id", "session", "trial"]).cumcount() + 1
    )

    recalls = events[
        (events.trial_type == "REC_WORD")
        & (events.trial >= 1)
        & (events.item_num > 0)
    ]
    recalled_presentations = set(map(tuple, recalls[PRESENTATION_KEYS].to_numpy()))
    studied["recalled"] = [
        tuple(row) in recalled_presentations
        for row in studied[PRESENTATION_KEYS].to_numpy()
    ]

    for col in ("session", "trial", "item_num"):
        studied[col] = studied[col].astype(int)

    return studied[
        ["subject_id", "session", "trial", "serial_position",
         "item_name", "item_num", "onset", "recalled"]
    ].rename(columns={"onset": "onset_s"}).reset_index(drop=True)
