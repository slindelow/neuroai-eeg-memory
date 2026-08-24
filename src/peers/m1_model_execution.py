"""Frozen B3/M1/M3 execution without held-session performance scoring.

The public entry point deliberately returns predictions and fit provenance, but no AUC,
accuracy, effect estimate, or interpretation for the outer held session.  Inner-session
ROC-AUC is used only for the prespecified M1 ``C`` selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from peers.baselines import MEMORABILITY_SMOOTHING, lexical_features, position_features
from peers.events import build_trial_table, load_events


TrialKey = tuple[str, int, int, int]
M1MatrixFactory = Callable[[Sequence[int], Sequence[int]], tuple[Sequence[TrialKey], np.ndarray]]


class ModelExecutionError(ValueError):
    """The frozen learner or leakage boundary could not be executed exactly."""


@dataclass(frozen=True)
class ModelExecutionResult:
    """Unscored held predictions and sufficient execution provenance."""

    test_keys: tuple[TrialKey, ...]
    prob_b3: np.ndarray
    prob_m1: np.ndarray
    prob_m3: np.ndarray
    selected_m1_c: float
    inner_selection_auc_by_c: Mapping[str, tuple[float, ...]]
    cross_fit_prob_b3: np.ndarray
    cross_fit_prob_m1: np.ndarray
    model_artifact_sha256: Mapping[str, str]
    transform_chain_sha256: Mapping[str, str]
    selected_hyperparameters_sha256: Mapping[str, str]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    return sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_full_behavioral_reference(
    data_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Load and authenticate every valid local ltpFR behavioral presentation.

    This population is intentionally broader than the EEG cohort.  It is the frozen B3
    reference used to compute target-subject-excluded word memorability.
    """
    root = Path(data_root)
    paths = sorted((root / "peers").glob("sub-LTP*/ses-*/*_events.tsv"))
    if not paths:
        raise ModelExecutionError("full behavioral reference has no ltpFR event files")
    frames: list[pd.DataFrame] = []
    files: list[dict[str, Any]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ModelExecutionError(f"behavioral reference source is unsafe: {path}")
        events = load_events(path)
        trials = build_trial_table(events)
        if trials.empty:
            raise ModelExecutionError(f"behavioral reference source has no trials: {path}")
        expected_subject = path.parts[-3]
        observed_subjects = set(trials["subject_id"].astype(str))
        if observed_subjects != {expected_subject}:
            raise ModelExecutionError(f"behavioral reference identity mismatch: {path}")
        frames.append(trials)
        source_id = path.relative_to(root).as_posix()
        files.append(
            {
                "source_id": source_id,
                "subject_id": expected_subject,
                "session": int(trials["session"].iloc[0]),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    reference = pd.concat(frames, ignore_index=True)
    key_fields = ["subject_id", "session", "trial", "item_num"]
    if reference.duplicated(key_fields).any():
        raise ModelExecutionError("full behavioral reference has duplicate presentation keys")
    reference = reference.sort_values(key_fields).reset_index(drop=True)
    inventory = {
        "schema_id": "neuroai-m1-behavioral-reference-inventory-v1",
        "schema_version": "1.0",
        "population": "full_valid_local_ltpFR_behavioral_trial_table",
        "files": files,
        "trial_count": len(reference),
        "subject_count": int(reference["subject_id"].nunique()),
    }
    return reference, inventory, _canonical_sha256(inventory)


def build_b3_design(
    target_trials: pd.DataFrame,
    behavioral_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Construct B3 using a memorability value computed on the full reference once."""
    target_required = {
        "subject_id", "session", "trial", "item_num", "item_name",
        "serial_position",
    }
    reference_required = target_required | {"recalled"}
    if not target_required.issubset(target_trials) or not reference_required.issubset(
        behavioral_reference
    ):
        raise ModelExecutionError("B3 inputs lack required behavioral fields")
    subjects = target_trials["subject_id"].astype(str).unique()
    if len(subjects) != 1:
        raise ModelExecutionError("B3 design requires one target subject")
    others = behavioral_reference.loc[
        behavioral_reference["subject_id"].astype(str) != subjects[0]
    ]
    if others["subject_id"].nunique() < 1:
        raise ModelExecutionError("B3 memorability requires other participants")
    prior = float(others["recalled"].mean())
    totals = others.groupby("item_num")["recalled"].agg(["sum", "count"])
    totals["word_memorability"] = (
        totals["sum"] + MEMORABILITY_SMOOTHING * prior
    ) / (totals["count"] + MEMORABILITY_SMOOTHING)
    joined = target_trials.loc[:, ["item_num"]].merge(
        totals[["word_memorability"]],
        on="item_num",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined["word_memorability"].isna().any():
        raise ModelExecutionError("target trial is absent from behavioral reference")
    design = pd.concat(
        [
            position_features(target_trials).reset_index(drop=True),
            lexical_features(target_trials).reset_index(drop=True),
            joined[["word_memorability"]].reset_index(drop=True),
        ],
        axis=1,
    )
    values = design.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ModelExecutionError("B3 design contains a non-finite value")
    return design


def _keys(frame: pd.DataFrame) -> tuple[TrialKey, ...]:
    return tuple(
        (str(row.subject_id), int(row.session), int(row.trial), int(row.item_num))
        for row in frame.itertuples(index=False)
    )


def _logistic(contract: Mapping[str, Any], *, c: float) -> LogisticRegression:
    return LogisticRegression(
        penalty=contract["penalty"],
        C=float(c),
        solver=contract["solver"],
        max_iter=int(contract["max_iter"]),
        fit_intercept=bool(contract["fit_intercept"]),
        class_weight=contract["class_weight"],
        random_state=0,
    )


def _model_digest(model: LogisticRegression, scaler: StandardScaler | None = None) -> str:
    payload: dict[str, Any] = {
        "classes": model.classes_.tolist(),
        "coef": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "n_iter": model.n_iter_.tolist(),
    }
    if scaler is not None:
        payload["scaler"] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
        }
    return _canonical_sha256(payload)


def _fit(model: LogisticRegression, values: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit the frozen explicit L2 contract while silencing sklearn's transition warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'penalty' was deprecated in version 1.8.*",
            category=FutureWarning,
        )
        return model.fit(values, labels)


def execute_frozen_models_without_outer_scoring(
    trials: pd.DataFrame,
    behavioral_reference: pd.DataFrame,
    *,
    charter: Mapping[str, Any],
    m1_matrix: M1MatrixFactory,
) -> ModelExecutionResult:
    """Fit B3/M1/M3 and return held-session probabilities without scoring them."""
    key_fields = ["subject_id", "session", "trial", "item_num"]
    trials = trials.sort_values(key_fields).reset_index(drop=True)
    if trials.empty or trials.duplicated(key_fields).any():
        raise ModelExecutionError("model trials must be nonempty with unique keys")
    if trials["subject_id"].nunique() != 1:
        raise ModelExecutionError("model execution requires one target subject")
    sessions = sorted(int(value) for value in trials["session"].unique())
    outer_test = max(sessions)
    train_sessions = sessions[:-1]
    if len(train_sessions) < 2:
        raise ModelExecutionError("model execution requires at least two training sessions")
    session_indices = {
        session: np.flatnonzero(trials["session"].to_numpy(int) == session)
        for session in sessions
    }
    y = np.zeros(len(trials), dtype=np.int64)
    train_mask = trials["session"].isin(train_sessions).to_numpy()
    y[train_mask] = trials.loc[train_mask, "recalled"].to_numpy(dtype=np.int64)
    if any(np.unique(y[session_indices[value]]).size != 2 for value in train_sessions):
        raise ModelExecutionError("every fit and validation session needs both classes")
    b3 = build_b3_design(trials, behavioral_reference).to_numpy(dtype=np.float64)

    c_grid = [float(value) for value in charter["learners"]["M1"]["C_grid"]]
    selection_scores: dict[str, list[float]] = {format(c, ".17g"): [] for c in c_grid}
    for validation_session in train_sessions:
        fit_sessions = [value for value in train_sessions if value != validation_session]
        fit_index = np.concatenate([session_indices[value] for value in fit_sessions])
        val_index = session_indices[validation_session]
        feature_keys, values = m1_matrix(fit_sessions, [validation_session])
        if tuple(feature_keys) != _keys(trials.iloc[val_index]):
            raise ModelExecutionError("inner M1 feature keys do not match validation rows")
        fit_keys, fit_values = m1_matrix(fit_sessions, fit_sessions)
        if tuple(fit_keys) != _keys(trials.iloc[fit_index]):
            raise ModelExecutionError("inner M1 feature keys do not match fit rows")
        for c in c_grid:
            model = _fit(
                _logistic(charter["learners"]["M1"], c=c), fit_values, y[fit_index]
            )
            probability = model.predict_proba(values)[:, 1]
            selection_scores[format(c, ".17g")].append(
                float(roc_auc_score(y[val_index], probability))
            )
    mean_scores = [float(np.mean(selection_scores[format(c, ".17g")])) for c in c_grid]
    best = max(mean_scores)
    selected_c = min(c for c, score in zip(c_grid, mean_scores, strict=True) if score == best)

    cross_b3 = np.empty(sum(len(session_indices[s]) for s in train_sessions), dtype=np.float64)
    cross_m1 = np.empty_like(cross_b3)
    train_order = np.concatenate([session_indices[s] for s in train_sessions])
    order_lookup = {int(global_index): local for local, global_index in enumerate(train_order)}
    inner_m1_digests: list[str] = []
    inner_b3_digests: list[str] = []
    for validation_session in train_sessions:
        fit_sessions = [value for value in train_sessions if value != validation_session]
        fit_index = np.concatenate([session_indices[value] for value in fit_sessions])
        val_index = session_indices[validation_session]
        scaler = StandardScaler().fit(b3[fit_index])
        b3_model = _fit(
            _logistic(charter["learners"]["B3"], c=1.0),
            scaler.transform(b3[fit_index]),
            y[fit_index],
        )
        fit_keys, fit_values = m1_matrix(fit_sessions, fit_sessions)
        val_keys, val_values = m1_matrix(fit_sessions, [validation_session])
        if tuple(fit_keys) != _keys(trials.iloc[fit_index]) or tuple(val_keys) != _keys(
            trials.iloc[val_index]
        ):
            raise ModelExecutionError("qualified cross-fit M1 keys drifted")
        m1_model = _fit(
            _logistic(charter["learners"]["M1"], c=selected_c),
            fit_values,
            y[fit_index],
        )
        local = np.array([order_lookup[int(value)] for value in val_index])
        cross_b3[local] = b3_model.predict_proba(scaler.transform(b3[val_index]))[:, 1]
        cross_m1[local] = m1_model.predict_proba(val_values)[:, 1]
        inner_b3_digests.append(_model_digest(b3_model, scaler))
        inner_m1_digests.append(_model_digest(m1_model))

    meta_values = np.column_stack([cross_b3, cross_m1])
    m3_model = _fit(
        _logistic(charter["learners"]["M3"], c=1.0), meta_values, y[train_order]
    )
    scaler = StandardScaler().fit(b3[train_order])
    b3_model = _fit(
        _logistic(charter["learners"]["B3"], c=1.0),
        scaler.transform(b3[train_order]),
        y[train_order],
    )
    train_keys, train_values = m1_matrix(train_sessions, train_sessions)
    test_keys, test_values = m1_matrix(train_sessions, [outer_test])
    if tuple(train_keys) != _keys(trials.iloc[train_order]):
        raise ModelExecutionError("outer M1 fit keys drifted")
    test_index = session_indices[outer_test]
    if tuple(test_keys) != _keys(trials.iloc[test_index]):
        raise ModelExecutionError("outer M1 prediction keys drifted")
    m1_model = _fit(
        _logistic(charter["learners"]["M1"], c=selected_c),
        train_values,
        y[train_order],
    )
    prob_b3 = b3_model.predict_proba(scaler.transform(b3[test_index]))[:, 1]
    prob_m1 = m1_model.predict_proba(test_values)[:, 1]
    prob_m3 = m3_model.predict_proba(np.column_stack([prob_b3, prob_m1]))[:, 1]
    for value in (prob_b3, prob_m1, prob_m3, cross_b3, cross_m1):
        if value.dtype != np.float64 or not np.isfinite(value).all():
            raise ModelExecutionError("model execution produced invalid probabilities")

    return ModelExecutionResult(
        test_keys=tuple(test_keys),
        prob_b3=prob_b3,
        prob_m1=prob_m1,
        prob_m3=prob_m3,
        selected_m1_c=selected_c,
        inner_selection_auc_by_c={key: tuple(value) for key, value in selection_scores.items()},
        cross_fit_prob_b3=cross_b3,
        cross_fit_prob_m1=cross_m1,
        model_artifact_sha256={
            "B3": _model_digest(b3_model, scaler),
            "M1": _model_digest(m1_model),
            "M3": _model_digest(m3_model),
        },
        transform_chain_sha256={
            "B3": _canonical_sha256({"pipeline": ["StandardScaler", "LogisticRegression"]}),
            "M1": _canonical_sha256({"normalization": "D26_A", "extra_scaler": False}),
            "M3": _canonical_sha256({"inputs": ["prob_b3", "prob_m1"], "extra_scaler": False}),
        },
        selected_hyperparameters_sha256={
            "B3": _canonical_sha256({"C": 1.0}),
            "M1": _canonical_sha256({"C": selected_c}),
            "M3": _canonical_sha256({"C": 1.0}),
        },
    )
