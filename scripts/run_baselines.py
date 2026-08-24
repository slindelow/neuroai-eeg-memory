"""Evaluate the no-EEG baselines under the project's primary protocol.

Primary protocol (spec section 6): within subject, train on all sessions but the last and
test on the held-out last session. Reported as the mean AUC across subjects with a
bootstrap confidence interval over subjects -- never over trials, because trials within a
subject are not independent.

Run before any EEG result exists. B3 is the number the EEG models have to beat.
"""

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from peers.baselines import MODELS, build_features
from peers.events import build_trial_table, load_all_events

OUT = Path(__file__).resolve().parents[1] / "findings"
N_BOOTSTRAP = 2000
RNG = np.random.default_rng(0)


def evaluate(trials: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Per-subject AUC on that subject's held-out final session."""
    scores = {}
    for subject, rows in trials.groupby("subject_id"):
        sessions = sorted(rows["session"].unique())
        if len(sessions) < 2:
            continue
        test_mask = rows["session"] == sessions[-1]
        train_idx, test_idx = rows.index[~test_mask], rows.index[test_mask]

        y_train, y_test = trials.loc[train_idx, "recalled"], trials.loc[test_idx, "recalled"]
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)
        )
        model.fit(features.loc[train_idx], y_train)
        probabilities = model.predict_proba(features.loc[test_idx])[:, 1]
        scores[subject] = roc_auc_score(y_test, probabilities)
    return pd.Series(scores, name="auc")


def bootstrap_ci(values: np.ndarray, n: int = N_BOOTSTRAP) -> tuple[float, float]:
    draws = RNG.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    trials = build_trial_table(load_all_events())
    print(f"trials: {len(trials):,}   subjects: {trials['subject_id'].nunique()}   "
          f"recall rate: {trials['recalled'].mean():.3f}\n")

    results = []
    for name, blocks in MODELS.items():
        features = build_features(trials, blocks)
        auc = evaluate(trials, features)
        low, high = bootstrap_ci(auc.to_numpy())
        results.append(
            {"model": name, "n_features": features.shape[1], "n_subjects": len(auc),
             "auc": auc.mean(), "ci_low": low, "ci_high": high}
        )
        print(f"  {name:<22} {features.shape[1]:3d} features   "
              f"AUC {auc.mean():.4f}  [{low:.4f}, {high:.4f}]")

    table = pd.DataFrame(results)
    b1 = table.loc[table["model"].str.startswith("B1"), "auc"].iloc[0]
    b2 = table.loc[table["model"].str.startswith("B2"), "auc"].iloc[0]
    b3 = table.loc[table["model"].str.startswith("B3"), "auc"].iloc[0]
    print(f"\n  position alone      : {b1 - 0.5:+.4f} over chance")
    print(f"  word alone          : {b2 - 0.5:+.4f} over chance")
    print(f"  together            : {b3 - 0.5:+.4f} over chance")
    print(f"  word adds over position: {b3 - b1:+.4f}")
    print(f"\n  >>> EEG must beat AUC {b3:.4f} to have shown anything <<<")

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "baselines.csv", index=False)
    print(f"\n  saved {OUT}/baselines.csv")


if __name__ == "__main__":
    main()
