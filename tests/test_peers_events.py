"""Tests for the PEERS trial table.

These describe what build_trial_table must produce. Each one corresponds to a decision
about what "remembered" means — read them as a specification, not as a chore.
"""

from pathlib import Path

import pandas as pd
import pytest

from peers.events import DATA_ROOT, build_trial_table, load_all_events, load_events

ONE_SESSION = (
    DATA_ROOT / "sub-LTP063" / "ses-0" / "sub-LTP063_ses-0_task-ltpFR_events.tsv"
)

pytestmark = pytest.mark.skipif(
    not ONE_SESSION.is_file(),
    reason="PEERS events are not mounted; run ./scripts/download_peers_events.sh",
)


@pytest.fixture(scope="module")
def one_session_events():
    return load_events(ONE_SESSION)


@pytest.fixture(scope="module")
def one_session_table(one_session_events):
    return build_trial_table(one_session_events)


# --- the loaders (already written for you; these just confirm the data is there) ---

def test_events_file_loads(one_session_events):
    assert len(one_session_events) == 1435
    assert "trial_type" in one_session_events.columns


def test_all_sessions_load():
    """Stacking every events file works and yields well-formed rows.

    Deliberately does not assert an exact subject count: how many subjects are on disk
    depends on how far the download has got, and a test that fails because more data
    arrived is a test that trains you to ignore failures.
    """
    df = load_all_events()
    assert df["subject_id"].nunique() >= 2
    assert df["subject_id"].str.startswith("sub-LTP").all()
    assert (df.groupby("subject_id")["session"].nunique() >= 1).all()
    assert "WORD" in set(df["trial_type"])


# --- the trial table (yours) ---

def test_one_row_per_studied_word(one_session_table):
    """16 word lists x 16 words = 256 studied words in this session."""
    assert len(one_session_table) == 256


def test_required_columns(one_session_table):
    assert set(one_session_table.columns) >= {
        "subject_id", "session", "trial", "serial_position",
        "item_name", "item_num", "recalled",
    }


def test_practice_lists_excluded(one_session_table):
    """Lists numbered 0 or below are practice and must not appear."""
    assert one_session_table["trial"].min() >= 1


def test_serial_position_runs_1_to_16_in_every_list(one_session_table):
    for trial, group in one_session_table.groupby("trial"):
        assert sorted(group["serial_position"]) == list(range(1, 17)), f"list {trial}"


def test_recalled_is_boolean(one_session_table):
    assert one_session_table["recalled"].dtype == bool


def test_recall_rate_is_plausible(one_session_table):
    """Free recall of a 16-word list lands well under 100% and well over chance.

    If you get ~0, your join isn't matching. If you get ~1, you're probably counting
    final free recall as well.
    """
    rate = one_session_table["recalled"].mean()
    assert 0.30 < rate < 0.60, f"recall rate was {rate:.3f}"


def test_intrusions_do_not_create_recalls(one_session_table):
    """item_num == -1 marks an intrusion or a cough, not a real recalled word.

    A word the participant never studied cannot be a hit.
    """
    assert one_session_table["item_num"].min() > 0


def test_final_free_recall_is_not_counted(one_session_events, one_session_table):
    """FFR_REC_WORD is a separate memory test at the end of the session.

    Counting it inflates the recall rate. This checks you excluded it, by recomputing
    immediate-recall hits independently and comparing.

    Note the `k in studied_keys` filter. Without it this test is wrong, because some
    recalls are **prior-list intrusions** — the participant, recalling list 3, says a
    word that was actually on list 2. Those are real recall events with valid item
    numbers, but they don't belong to any word studied in the list being recalled.
    In sub-LTP063 ses-0 there are exactly two. (This test shipped broken because of
    them; it counted 111 hits where the correct answer is 109.)
    """
    ev = one_session_events
    studied = ev[(ev["trial_type"] == "WORD") & (ev["trial"] >= 1)]
    studied_keys = set(zip(studied["trial"], studied["item_num"]))

    immediate = ev[
        (ev["trial_type"] == "REC_WORD")
        & (ev["trial"] >= 1)
        & (ev["item_num"] > 0)
    ]
    hits = {
        k for k in zip(immediate["trial"], immediate["item_num"])
        if k in studied_keys
    }
    assert one_session_table["recalled"].sum() == len(hits)


# --- EEG alignment (F16) -----------------------------------------------------------

BROKEN_SESSION = (
    DATA_ROOT / "sub-LTP079" / "ses-4" / "sub-LTP079_ses-4_task-ltpFR_events.tsv"
)


@pytest.fixture(scope="module")
def broken_session_events():
    if not BROKEN_SESSION.exists():
        pytest.skip("sub-LTP079 ses-4 not downloaded")
    return load_events(BROKEN_SESSION)


def test_unaligned_presentations_are_detected(broken_session_events):
    """The dataset marks events it could not place in EEG time with sample = -1.

    sub-LTP079 ses-4 began behavioural recording before the amplifier, so 192 of its 256
    presentations have no EEG. This is the only such session in PEERS.
    """
    from peers.events import unaligned_presentations

    unaligned = unaligned_presentations(broken_session_events)
    assert len(unaligned) == 192
    assert (unaligned["sample"] == -1).all()
    assert (unaligned["onset"] == -0.002).all()


def test_unaligned_presentations_are_excluded_from_the_trial_table(broken_session_events):
    table = build_trial_table(broken_session_events)
    assert len(table) == 64, "only the four aligned lists should survive"
    assert (table["onset_s"] > 0).all()


def test_surviving_lists_still_have_full_serial_positions(broken_session_events):
    """Excluding whole lists must not renumber the lists that remain."""
    table = build_trial_table(broken_session_events)
    per_list = table.groupby("trial")["serial_position"].agg(["min", "max", "size"])
    assert (per_list["min"] == 1).all()
    assert (per_list["max"] == 16).all()
    assert (per_list["size"] == 16).all()


def test_recalls_are_not_filtered_on_alignment(broken_session_events):
    """An unaligned recall must still credit an aligned presentation.

    Filtering recall events on EEG alignment would relabel recalled words as forgotten,
    which is silent and would bias every downstream model.
    """
    table = build_trial_table(broken_session_events)
    assert table["recalled"].any(), "the aligned lists should contain recalled words"


def test_only_one_session_in_peers_is_affected():
    """Stated as a measurement, so a future dataset version that fixes or adds cases fails."""
    from peers.events import unaligned_presentations

    events = load_all_events()
    unaligned = unaligned_presentations(events)
    affected = unaligned.groupby(["subject_id", "session"]).size()
    assert len(affected) == 1
    assert affected.iloc[0] == 192
