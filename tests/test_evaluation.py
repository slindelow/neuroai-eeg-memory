"""Tests for the incremental-value metrics.

These are worth testing properly: IDI and NRI are easy to implement with a sign error or a
swapped argument, and neither would crash -- they would just report the wrong conclusion about
whether EEG added anything.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from peers.evaluation import (
    delong_test,
    incremental_value,
    integrated_discrimination_improvement,
    net_reclassification_improvement,
    sign_flip_test,
)

RNG = np.random.default_rng(0)


@pytest.fixture
def scenario():
    """A model that genuinely improves on a baseline, with known direction."""
    y = np.array([1] * 100 + [0] * 100, dtype=bool)
    old = RNG.uniform(0.3, 0.7, 200)
    # The new model shifts events up and non-events down -- a real improvement.
    new = np.clip(old + np.where(y, 0.15, -0.15), 0, 1)
    return y, old, new


# --- DeLong ---------------------------------------------------------------------------

def test_delong_identical_models_gives_no_difference():
    y = np.array([1] * 50 + [0] * 50, dtype=bool)
    p = RNG.uniform(size=100)
    z, pval = delong_test(y, p, p)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert pval == pytest.approx(1.0)


def test_delong_detects_a_large_difference(scenario):
    y, old, new = scenario
    z, pval = delong_test(y, new, old)
    assert z > 0, "the better model should give a positive z"
    assert pval < 0.05


def test_delong_is_antisymmetric(scenario):
    y, old, new = scenario
    z_forward, p_forward = delong_test(y, new, old)
    z_reverse, p_reverse = delong_test(y, old, new)
    assert z_forward == pytest.approx(-z_reverse)
    assert p_forward == pytest.approx(p_reverse)


def test_delong_auc_matches_sklearn(scenario):
    """The internal AUC computation must agree with a trusted implementation."""
    y, old, new = scenario
    z, _ = delong_test(y, new, old)
    # Reconstructing the difference from sklearn should carry the same sign.
    assert np.sign(z) == np.sign(roc_auc_score(y, new) - roc_auc_score(y, old))


def test_delong_handles_single_class():
    y = np.ones(20, dtype=bool)
    z, pval = delong_test(y, RNG.uniform(size=20), RNG.uniform(size=20))
    assert np.isnan(z) and np.isnan(pval)


# --- IDI ------------------------------------------------------------------------------

def test_idi_is_zero_for_identical_models():
    y = np.array([1] * 50 + [0] * 50, dtype=bool)
    p = RNG.uniform(size=100)
    assert integrated_discrimination_improvement(y, p, p) == pytest.approx(0.0)


def test_idi_positive_when_probabilities_separate(scenario):
    y, old, new = scenario
    assert integrated_discrimination_improvement(y, old, new) > 0


def test_idi_negative_when_arguments_are_swapped(scenario):
    """Guards against the easiest possible mistake: old and new the wrong way round."""
    y, old, new = scenario
    assert integrated_discrimination_improvement(y, new, old) < 0


# --- NRI ------------------------------------------------------------------------------

def test_nri_is_zero_for_identical_models():
    y = np.array([1] * 50 + [0] * 50, dtype=bool)
    p = RNG.uniform(size=100)
    assert net_reclassification_improvement(y, p, p) == pytest.approx(0.0)


def test_nri_is_maximal_when_every_case_moves_correctly(scenario):
    """Every event up and every non-event down is a perfect reclassification: NRI = 2."""
    y, old, new = scenario
    assert net_reclassification_improvement(y, old, new) == pytest.approx(2.0)


def test_nri_is_minimal_when_every_case_moves_wrongly(scenario):
    y, old, new = scenario
    assert net_reclassification_improvement(y, new, old) == pytest.approx(-2.0)


def test_nri_ignores_magnitude_unlike_idi():
    """NRI counts direction only, so scaling the improvement leaves it unchanged."""
    y = np.array([1] * 50 + [0] * 50, dtype=bool)
    old = np.full(100, 0.5)
    small = np.clip(old + np.where(y, 0.01, -0.01), 0, 1)
    large = np.clip(old + np.where(y, 0.30, -0.30), 0, 1)
    nri_small = net_reclassification_improvement(y, old, small)
    nri_large = net_reclassification_improvement(y, old, large)
    assert nri_small == pytest.approx(nri_large)
    idi_small = integrated_discrimination_improvement(y, old, small)
    idi_large = integrated_discrimination_improvement(y, old, large)
    assert idi_large > idi_small


# --- the sensitivity claim that motivated all this ------------------------------------

def test_idi_moves_where_delta_auc_cannot():
    """The reason three metrics are reported rather than one [AJE 2012, D25].

    AUC depends only on the *ranking* of predictions, so any strictly increasing transform
    applied to all of them leaves it exactly unchanged. Spreading the probabilities away from
    0.5 is such a transform: discrimination is identical, yet the predicted probabilities for
    events and non-events separate, and IDI registers that.

    This is the insensitivity of a rank-based statistic in its purest form -- ΔAUC is not
    merely small here, it is exactly zero by construction, while IDI is large.

    (The first version of this test was wrong: it shifted events up and non-events down by a
    constant, which reorders overlapping distributions and moved AUC by 0.147. A transform has
    to be applied to *every* prediction to leave the ranking untouched.)
    """
    y = np.array([1] * 100 + [0] * 100, dtype=bool)
    old = np.concatenate([RNG.uniform(0.45, 0.60, 100), RNG.uniform(0.40, 0.55, 100)])
    new = 0.5 + 3.0 * (old - 0.5)          # strictly increasing, stays inside [0, 1]

    delta_auc = roc_auc_score(y, new) - roc_auc_score(y, old)
    idi = integrated_discrimination_improvement(y, old, new)

    assert delta_auc == pytest.approx(0.0), "a monotone transform cannot change AUC at all"
    assert idi > 0.01, "but the probabilities separated, and IDI sees it"


# --- aggregation ----------------------------------------------------------------------

def test_incremental_value_returns_all_metrics(scenario):
    y, old, new = scenario
    result = incremental_value(y, old, new)
    assert set(result) == {
        "auc_old", "auc_new", "delta_auc", "idi", "nri", "delong_z", "delong_p"
    }
    assert result["delta_auc"] == pytest.approx(result["auc_new"] - result["auc_old"])


def test_incremental_value_empty_for_single_class():
    y = np.zeros(20, dtype=bool)
    assert incremental_value(y, RNG.uniform(size=20), RNG.uniform(size=20)) == {}


def test_sign_flip_detects_consistent_effect():
    assert sign_flip_test(np.full(20, 0.05)) < 0.001


def test_sign_flip_null_for_symmetric_values():
    assert sign_flip_test(np.array([0.05, -0.05] * 10)) > 0.5
