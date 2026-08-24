"""Invented-data tests for the frozen learner execution boundary."""

from pathlib import Path

import numpy as np
import pandas as pd

from peers.m1_manifests import load_json_unique
from peers.m1_model_execution import (
    build_b3_design,
    execute_frozen_models_without_outer_scoring,
)


ROOT = Path(__file__).resolve().parents[1]
CHARTER = load_json_unique(ROOT / "docs" / "analysis-charter-m1-v1.json")


def _reference() -> pd.DataFrame:
    rows = []
    for subject_index, subject in enumerate(("sub-SYN001", "sub-SYN002", "sub-SYN003")):
        for session in range(3):
            for item in range(1, 5):
                rows.append(
                    {
                        "subject_id": subject,
                        "session": session,
                        "trial": 1,
                        "serial_position": item,
                        "item_name": ("apple", "bridge", "candle", "drum")[item - 1],
                        "item_num": item,
                        "onset_s": float(item),
                        "recalled": bool((item + session + subject_index) % 2),
                    }
                )
    return pd.DataFrame(rows)


def _target(reference: pd.DataFrame) -> pd.DataFrame:
    return reference.loc[reference.subject_id == "sub-SYN001"].reset_index(drop=True)


def _factory(target: pd.DataFrame, *, held_shift: float = 0.0):
    keys = ["subject_id", "session", "trial", "item_num"]
    raw = np.column_stack(
        [
            target["serial_position"].to_numpy(float),
            target["item_num"].to_numpy(float) ** 2,
            np.sin(target["item_num"].to_numpy(float)),
        ]
    )
    raw[target.session.to_numpy() == 2] += held_shift

    def matrix(fit_sessions, apply_sessions):
        fit_mask = target.session.isin(fit_sessions).to_numpy()
        apply_mask = target.session.isin(apply_sessions).to_numpy()
        mean = raw[fit_mask].mean(axis=0)
        scale = raw[fit_mask].std(axis=0)
        scale[scale == 0.0] = 1.0
        values = np.ascontiguousarray((raw[apply_mask] - mean) / scale, dtype=np.float64)
        trial_keys = tuple(
            tuple(row)
            for row in target.loc[apply_mask, keys].itertuples(index=False, name=None)
        )
        return trial_keys, values

    return matrix


def test_b3_memorability_excludes_target_outcomes():
    reference = _reference()
    target = _target(reference)
    original = build_b3_design(target, reference)
    changed = reference.copy()
    mask = changed.subject_id == "sub-SYN001"
    changed.loc[mask, "recalled"] = ~changed.loc[mask, "recalled"]
    perturbed = build_b3_design(target, changed)
    np.testing.assert_array_equal(original.to_numpy(), perturbed.to_numpy())


def test_frozen_models_return_probabilities_without_outer_scores():
    reference = _reference()
    target = _target(reference)
    result = execute_frozen_models_without_outer_scoring(
        target,
        reference,
        charter=CHARTER,
        m1_matrix=_factory(target),
    )
    assert result.selected_m1_c in CHARTER["learners"]["M1"]["C_grid"]
    assert len(result.test_keys) == 4
    assert len(result.cross_fit_prob_b3) == len(result.cross_fit_prob_m1) == 8
    for values in (result.prob_b3, result.prob_m1, result.prob_m3):
        assert values.dtype == np.float64
        assert np.isfinite(values).all()
        assert ((0.0 <= values) & (values <= 1.0)).all()
    assert not hasattr(result, "outer_auc")


def test_held_session_labels_cannot_affect_fitted_predictions():
    reference = _reference()
    target = _target(reference)
    first = execute_frozen_models_without_outer_scoring(
        target, reference, charter=CHARTER, m1_matrix=_factory(target)
    )
    changed_target = target.copy()
    held = changed_target.session == 2
    changed_target.loc[held, "recalled"] = ~changed_target.loc[held, "recalled"]
    changed_reference = reference.copy()
    reference_held = (changed_reference.subject_id == "sub-SYN001") & (
        changed_reference.session == 2
    )
    changed_reference.loc[reference_held, "recalled"] = ~changed_reference.loc[
        reference_held, "recalled"
    ]
    second = execute_frozen_models_without_outer_scoring(
        changed_target,
        changed_reference,
        charter=CHARTER,
        m1_matrix=_factory(changed_target),
    )
    assert first.selected_m1_c == second.selected_m1_c
    np.testing.assert_array_equal(first.prob_b3, second.prob_b3)
    np.testing.assert_array_equal(first.prob_m1, second.prob_m1)
    np.testing.assert_array_equal(first.prob_m3, second.prob_m3)
    assert first.model_artifact_sha256 == second.model_artifact_sha256


def test_held_session_eeg_distribution_never_changes_training_fits():
    reference = _reference()
    target = _target(reference)
    first = execute_frozen_models_without_outer_scoring(
        target, reference, charter=CHARTER, m1_matrix=_factory(target)
    )
    shifted = execute_frozen_models_without_outer_scoring(
        target,
        reference,
        charter=CHARTER,
        m1_matrix=_factory(target, held_shift=1000.0),
    )
    assert first.selected_m1_c == shifted.selected_m1_c
    np.testing.assert_array_equal(first.cross_fit_prob_m1, shifted.cross_fit_prob_m1)
    assert first.model_artifact_sha256 == shifted.model_artifact_sha256
    _, unshifted_values = _factory(target)([0, 1], [2])
    _, shifted_values = _factory(target, held_shift=1000.0)([0, 1], [2])
    assert not np.array_equal(unshifted_values, shifted_values)
