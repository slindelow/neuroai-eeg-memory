"""Baselines that predict recall without using EEG at all.

These exist to answer: how much of recall is explained by *where the word appeared* and
*which word it was*? Any EEG model has to beat this to have shown that the brain
contributed something the stimulus did not.

Establishing the bar before looking at any EEG result is deliberate. A baseline built
after seeing the effect can always be suspected of having been tuned, however honestly,
by the many small choices about which features to include.

    B0  base rate only
    B1  serial position, list number, session number
    B2  the word itself
    B3  B1 + B2   <- the null hypothesis the EEG models must beat
"""

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

#: Shrinkage for the word-memorability estimate. A word seen by few subjects gets an
#: unreliable recall rate, so estimates are pulled toward the overall recall rate in
#: proportion to how little evidence supports them. 10 is worth roughly ten extra
#: observations at the population mean -- weak enough not to flatten real differences,
#: strong enough that a word seen twice does not arrive at 0.0 or 1.0.
MEMORABILITY_SMOOTHING = 10.0

N_SERIAL_POSITIONS = 16


def word_memorability(trials: pd.DataFrame, smoothing: float = MEMORABILITY_SMOOTHING) -> pd.Series:
    """Leave-one-subject-out estimate of how memorable each word is.

    For every (subject, word) pair, returns the fraction of *other* subjects who recalled
    that word. The estimate is shrunk toward that target subject's other-subject recall
    rate, so neither the word evidence nor its prior contains the target's outcomes.

    Excluding the subject is not optional. The word pool is reused, so a subject sees the
    same word in several lists; an estimate that included their own outcomes would carry
    the answer inside the feature and inflate every score computed from it.

    Computed by subtracting each subject's own contribution from the word and dataset
    totals rather than by refitting once per subject -- same result, one pass instead of
    eighty. At least two participants are required because a leave-one-subject-out prior
    does not exist otherwise.
    """
    subject_totals = trials.groupby("subject_id")["recalled"].agg(["sum", "count"])
    if len(subject_totals) < 2:
        raise ValueError("word memorability requires at least two participants")

    other_subject_sum = subject_totals["sum"].sum() - subject_totals["sum"]
    other_subject_count = subject_totals["count"].sum() - subject_totals["count"]
    other_subject_mean = other_subject_sum / other_subject_count

    totals = trials.groupby("item_num")["recalled"].agg(["sum", "count"])
    own = trials.groupby(["subject_id", "item_num"])["recalled"].agg(["sum", "count"])

    subject_index = own.index.get_level_values("subject_id")
    item_index = own.index.get_level_values("item_num")
    others_sum = totals["sum"].reindex(item_index).to_numpy() - own["sum"].to_numpy()
    others_count = totals["count"].reindex(item_index).to_numpy() - own["count"].to_numpy()
    smoothing_prior = other_subject_mean.reindex(subject_index).to_numpy()

    shrunk = pd.Series(
        (others_sum + smoothing * smoothing_prior) / (others_count + smoothing),
        index=own.index,
        name="word_memorability",
    )

    result = trials.join(shrunk, on=["subject_id", "item_num"])["word_memorability"]
    if result.isna().any():
        raise ValueError("word memorability is undefined for one or more trial rows")
    return result


def lexical_features(trials: pd.DataFrame) -> pd.DataFrame:
    """Interpretable properties of the word itself.

    Deliberately few and nameable, so a result can be stated in words: "longer words were
    recalled less often", not "component 7 of an embedding mattered". Frequency is on the
    Zipf scale (log10 occurrences per billion), so a one-unit difference is a tenfold
    difference in how often the word appears.
    """
    words = trials["item_name"].astype(str).str.lower()
    return pd.DataFrame(
        {
            "word_length": words.str.len().to_numpy(float),
            "word_frequency": [zipf_frequency(w, "en") for w in words],
        },
        index=trials.index,
    )


def position_features(trials: pd.DataFrame) -> pd.DataFrame:
    """Where the word appeared, in the list and in the session.

    Serial position is one-hot rather than linear or quadratic. The recall curve is
    U-shaped (so linear cannot fit it) and asymmetric -- recency reaches ~0.94 while
    primacy reaches ~0.73 -- so a parabola, forced to treat the two ends alike, would
    misfit it too. One-hot assumes nothing about the shape and costs 15 parameters
    against tens of thousands of trials.

    These features carry no leakage risk: position is known before anyone recalls
    anything, so unlike word memorability nothing is estimated from outcomes.
    """
    onehot = pd.get_dummies(
        pd.Categorical(trials["serial_position"], categories=range(1, N_SERIAL_POSITIONS + 1)),
        prefix="pos",
        drop_first=True,
    ).astype(float)
    onehot.index = trials.index
    onehot["list_number"] = trials["trial"].to_numpy(float)
    onehot["session_number"] = trials["session"].to_numpy(float)
    return onehot


def build_features(trials: pd.DataFrame, blocks: tuple[str, ...]) -> pd.DataFrame:
    """Assemble a design matrix from named feature blocks.

    ``blocks`` may contain "position", "lexical", "memorability". An empty tuple gives an
    intercept-only model (B0), which predicts the base rate for every trial.
    """
    parts = []
    if "position" in blocks:
        parts.append(position_features(trials))
    if "lexical" in blocks:
        parts.append(lexical_features(trials))
    if "memorability" in blocks:
        parts.append(word_memorability(trials).rename("word_memorability").to_frame())
    if not parts:
        return pd.DataFrame({"intercept": np.ones(len(trials))}, index=trials.index)
    return pd.concat(parts, axis=1)


MODELS: dict[str, tuple[str, ...]] = {
    "B0 base rate": (),
    "B1 position": ("position",),
    "B2 word": ("lexical", "memorability"),
    "B3 position + word": ("position", "lexical", "memorability"),
}
