"""Leakage tests for the behavioral comparison models.

The memorability feature is allowed to use outcomes from other participants because the
word pool is shared. It must never use any outcome from the participant whose feature is
being computed—not even through the shrinkage prior.
"""

import numpy as np
import pandas as pd
import pytest

from peers.baselines import MEMORABILITY_SMOOTHING, word_memorability


def synthetic_trials() -> pd.DataFrame:
    """Three participants with two shared words and one target-only word."""
    return pd.DataFrame(
        {
            "subject_id": ["target", "target", "target", "other-a", "other-a", "other-b", "other-b"],
            "item_num": [10, 20, 30, 10, 20, 10, 20],
            "recalled": [True, False, True, False, False, False, True],
        }
    )


def test_memorability_uses_target_excluded_prior_and_word_counts():
    trials = synthetic_trials()

    estimates = word_memorability(trials)
    target = estimates[trials["subject_id"].eq("target")].to_numpy()

    # The four rows from the two other participants have a recall mean of 1/4.
    # Words 10 and 20 have other-subject recall totals of 0/2 and 1/2. Word 30
    # has no other-subject observations, so shrinkage returns the other-subject prior.
    prior = 0.25
    expected = np.array(
        [
            (0 + MEMORABILITY_SMOOTHING * prior) / (2 + MEMORABILITY_SMOOTHING),
            (1 + MEMORABILITY_SMOOTHING * prior) / (2 + MEMORABILITY_SMOOTHING),
            prior,
        ]
    )
    np.testing.assert_allclose(target, expected)


def test_target_outcomes_cannot_change_its_own_memorability_features():
    original = synthetic_trials()
    perturbed = original.copy()
    target_rows = perturbed["subject_id"].eq("target")
    perturbed.loc[target_rows, "recalled"] = ~perturbed.loc[target_rows, "recalled"]

    before = word_memorability(original)
    after = word_memorability(perturbed)

    np.testing.assert_allclose(before[target_rows], after[target_rows], rtol=0, atol=0)
    assert not np.allclose(before[~target_rows], after[~target_rows]), (
        "the fixture must remain sensitive: target outcomes are legitimate evidence for other participants"
    )


def test_memorability_fails_when_no_other_participant_exists():
    one_subject = synthetic_trials().query("subject_id == 'target'")

    with pytest.raises(ValueError, match="at least two participants"):
        word_memorability(one_subject)
