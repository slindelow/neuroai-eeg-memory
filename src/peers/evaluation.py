"""Metrics for the incremental value of EEG over the no-EEG baseline.

The headline question is not "how well does EEG predict recall" but "what does EEG add that
the word and its position do not already provide". That is the classical incremental-value
problem from biomarker research, and it comes with a known trap: ΔAUC is insensitive when the
baseline model is strong, because AUC is rank-based [AJE 2012]. Our baseline is AUC 0.637,
stronger than any published EEG-only classifier on this dataset family [D28], so a small ΔAUC
is expected and would not by itself mean the brain contributed nothing.

Three metrics are therefore reported together [D25]:

    ΔAUC  discrimination gain -- comparable to the SME literature, but conservative
    IDI   integrated discrimination improvement -- how much the predicted probabilities
          separate, which moves when ΔAUC will not
    NRI   continuous net reclassification improvement -- depends on the effect size of the
          new marker rather than on the strength of the baseline

IDI and NRI can indicate improvement where none exists [Added value 2018], so all three are
reported and none is treated as decisive alone.

Everything here is computed per subject and aggregated across subjects, matching the primary
protocol [D21]: trials within a subject are not independent, so pooling predictions across
subjects would misstate the uncertainty [D22].
"""

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks, with ties averaged. Helper for the fast DeLong covariance."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> tuple[float, float]:
    """Compare two AUCs on the same observations (DeLong's test for correlated ROC curves).

    Nested models scored on identical data produce *correlated* ROC curves, so an ordinary
    two-sample comparison would overstate significance. Implements the fast algorithm using
    midrank structural components.

    Returns (z, two-sided p). Comparing a model with itself gives z = 0, p = 1.
    """
    y_true = np.asarray(y_true).astype(bool)
    order = np.concatenate([np.where(y_true)[0], np.where(~y_true)[0]])
    predictions = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    m = int(y_true.sum())
    n = predictions.shape[1] - m
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    tx = np.vstack([_midrank(row[:m]) for row in predictions])
    ty = np.vstack([_midrank(row[m:]) for row in predictions])
    tz = np.vstack([_midrank(row) for row in predictions])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n

    contrast = np.array([1.0, -1.0])
    variance = contrast @ covariance @ contrast
    if variance <= 0:
        return 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    return z, float(2 * stats.norm.sf(abs(z)))


def integrated_discrimination_improvement(
    y_true: np.ndarray, prob_old: np.ndarray, prob_new: np.ndarray
) -> float:
    """IDI: how much further apart the predicted probabilities move.

    The mean predicted probability rises for events and falls for non-events when a marker
    adds information. Unlike ΔAUC this is not rank-based, so it responds where AUC saturates.
    """
    y_true = np.asarray(y_true).astype(bool)
    prob_old, prob_new = np.asarray(prob_old), np.asarray(prob_new)
    events = prob_new[y_true].mean() - prob_old[y_true].mean()
    non_events = prob_new[~y_true].mean() - prob_old[~y_true].mean()
    return float(events - non_events)


def net_reclassification_improvement(
    y_true: np.ndarray, prob_old: np.ndarray, prob_new: np.ndarray
) -> float:
    """Continuous NRI: net proportion of observations moved in the right direction.

    Counts, among events, how many received a higher predicted probability under the new
    model than the old, minus how many received a lower one; and the reverse among
    non-events. Category-free, so no risk thresholds have to be chosen.

    Depends on the effect size of the new marker and its correlation with existing
    predictors, rather than on how strong the baseline already is [AJE 2012].
    """
    y_true = np.asarray(y_true).astype(bool)
    difference = np.asarray(prob_new) - np.asarray(prob_old)
    events, non_events = difference[y_true], difference[~y_true]
    if not len(events) or not len(non_events):
        return float("nan")
    event_net = np.mean(events > 0) - np.mean(events < 0)
    non_event_net = np.mean(non_events < 0) - np.mean(non_events > 0)
    return float(event_net + non_event_net)


def incremental_value(
    y_true: np.ndarray, prob_old: np.ndarray, prob_new: np.ndarray
) -> dict[str, float]:
    """All three incremental-value metrics plus both AUCs, for one subject's held-out data."""
    y_true = np.asarray(y_true).astype(bool)
    if len(np.unique(y_true)) < 2:
        return {}
    auc_old = roc_auc_score(y_true, prob_old)
    auc_new = roc_auc_score(y_true, prob_new)
    z, p = delong_test(y_true, prob_new, prob_old)
    return {
        "auc_old": auc_old,
        "auc_new": auc_new,
        "delta_auc": auc_new - auc_old,
        "idi": integrated_discrimination_improvement(y_true, prob_old, prob_new),
        "nri": net_reclassification_improvement(y_true, prob_old, prob_new),
        "delong_z": z,
        "delong_p": p,
    }


def sign_flip_test(values: np.ndarray, n_permutations: int = 10000, seed: int = 0) -> float:
    """Two-sided p for whether a per-subject effect differs from zero.

    Under the null a subject's effect is equally likely to point either way, so the null
    distribution is generated by randomly flipping signs. Treats subject as the unit of
    observation, which is what generalising to new people requires [D22].
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = values.mean()
    flips = rng.choice([-1.0, 1.0], size=(n_permutations, len(values)))
    null = (flips * values).mean(axis=1)
    return float((np.abs(null) >= abs(observed)).mean())
