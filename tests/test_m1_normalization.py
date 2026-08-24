"""Synthetic numerical tests for Sofia's resolved strict-inductive D26-A transform.

All arrays are invented. The tests do not read an epoch, choose D29, fit a model, use a real
label, or inspect any held-out outcome.
"""

from copy import deepcopy

import numpy as np
import pytest

from peers.m1_features import (
    FeatureTableValidationError,
    apply_strict_inductive_normalizer,
    fit_strict_inductive_normalizer,
    log10_power,
    normalization_array_sha256,
    normalization_statistics_evidence,
)


def invented_log_power(*, trials=3, times=5):
    values = np.arange(trials * 40 * 8 * times, dtype=np.float64)
    values = values.reshape(trials, 40, 8, times) / 1000.0
    values[:, 0, 0, :] = 7.0
    return values


def test_log10_power_is_exact_float64_and_rejects_invalid_power():
    power = np.array([0.0, 1.0, 9.0], dtype=np.float64)
    transformed = log10_power(power)

    np.testing.assert_array_equal(transformed, np.log10(power + 1e-30))
    assert transformed.dtype == np.float64

    with pytest.raises(FeatureTableValidationError, match="nonnegative"):
        log10_power(np.array([-1.0], dtype=np.float64))
    with pytest.raises(FeatureTableValidationError, match="finite"):
        log10_power(np.array([np.nan], dtype=np.float64))
    with pytest.raises(FeatureTableValidationError, match="float64"):
        log10_power(np.array([1.0], dtype=np.float32))


def test_fit_uses_population_statistics_over_training_trial_and_time_only():
    training = invented_log_power()
    statistics = fit_strict_inductive_normalizer(training)

    expected_mean = training.mean(axis=(0, 3), dtype=np.float64)
    expected_raw_std = training.std(axis=(0, 3), ddof=0, dtype=np.float64)
    expected_scale = np.where(expected_raw_std == 0.0, 1.0, expected_raw_std)
    np.testing.assert_array_equal(statistics.mean, expected_mean)
    np.testing.assert_array_equal(statistics.raw_std, expected_raw_std)
    np.testing.assert_array_equal(statistics.scale, expected_scale)
    np.testing.assert_array_equal(
        statistics.zero_standard_deviation_mask, expected_raw_std == 0.0
    )
    assert statistics.n_fit_trials == 3
    assert statistics.n_fit_timepoints == 5
    assert statistics.ddof == 0
    assert statistics.mean.shape == (40, 8)
    assert statistics.mean.flags.writeable is False


def test_apply_uses_fitted_statistics_unchanged_and_is_target_batch_invariant():
    training = invented_log_power()
    statistics = fit_strict_inductive_normalizer(training)
    held = invented_log_power(trials=2, times=3) + 50.0

    together = apply_strict_inductive_normalizer(held, statistics)
    separately = np.concatenate(
        [
            apply_strict_inductive_normalizer(held[index : index + 1], statistics)
            for index in range(held.shape[0])
        ],
        axis=0,
    )
    expected = (held - statistics.mean[None, :, :, None]) / statistics.scale[
        None, :, :, None
    ]
    np.testing.assert_array_equal(together, expected)
    np.testing.assert_array_equal(together, separately)

    changed_held = held * 100.0 - 3.0
    unchanged_statistics = fit_strict_inductive_normalizer(training)
    np.testing.assert_array_equal(unchanged_statistics.mean, statistics.mean)
    np.testing.assert_array_equal(unchanged_statistics.scale, statistics.scale)
    assert not np.array_equal(
        fit_strict_inductive_normalizer(
            np.concatenate(
                [
                    training,
                    invented_log_power(trials=2, times=5) * 100.0 - 3.0,
                ],
                axis=0,
            )
        ).mean,
        statistics.mean,
    )


def test_zero_training_standard_deviation_centers_without_division_failure():
    training = invented_log_power()
    statistics = fit_strict_inductive_normalizer(training)
    normalized_training = apply_strict_inductive_normalizer(training, statistics)

    assert statistics.raw_std[0, 0] == 0.0
    assert statistics.scale[0, 0] == 1.0
    assert statistics.zero_standard_deviation_mask[0, 0]
    np.testing.assert_array_equal(normalized_training[:, 0, 0, :], 0.0)
    assert np.isfinite(normalized_training).all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.astype(np.float32), "float64"),
        (lambda value: value[:, :39], "40 channels"),
        (lambda value: value[:, :, :7], "8 frequencies"),
        (lambda value: value[:0], "nonempty trial"),
        (lambda value: value[:, :, :, :0], "nonempty time"),
    ],
)
def test_fit_rejects_dtype_shape_and_empty_axes(mutation, message):
    with pytest.raises(FeatureTableValidationError, match=message):
        fit_strict_inductive_normalizer(mutation(invented_log_power()))


def test_fit_and_apply_reject_nonfinite_or_incompatible_arrays():
    training = invented_log_power()
    nonfinite = training.copy()
    nonfinite[0, 0, 0, 0] = np.inf
    with pytest.raises(FeatureTableValidationError, match="finite"):
        fit_strict_inductive_normalizer(nonfinite)

    statistics = fit_strict_inductive_normalizer(training)
    with pytest.raises(FeatureTableValidationError, match="40 channels"):
        apply_strict_inductive_normalizer(training[:, :39], statistics)
    with pytest.raises(FeatureTableValidationError, match="finite"):
        apply_strict_inductive_normalizer(nonfinite, statistics)


def test_statistic_hashes_are_deterministic_typed_and_bridge_to_provenance_fields():
    statistics = fit_strict_inductive_normalizer(invented_log_power())
    evidence = normalization_statistics_evidence(statistics)

    assert evidence == normalization_statistics_evidence(statistics)
    assert evidence["mean_array_sha256"] == normalization_array_sha256(statistics.mean)
    assert evidence["raw_std_array_sha256"] == normalization_array_sha256(
        statistics.raw_std
    )
    assert evidence["std_array_sha256"] == normalization_array_sha256(statistics.scale)
    assert evidence["zero_std_mask_sha256"] == normalization_array_sha256(
        statistics.zero_standard_deviation_mask
    )
    assert evidence["n_zero_std_cells"] == 1
    assert evidence["n_fit_timepoints"] == 5

    changed = statistics.mean.copy()
    changed[0, 1] += np.finfo(np.float64).eps
    assert normalization_array_sha256(changed) != evidence["mean_array_sha256"]
    with pytest.raises(FeatureTableValidationError, match="supported dtype"):
        normalization_array_sha256(changed.astype(np.float32))


def test_statistics_objects_defensively_copy_the_fit_result():
    training = invented_log_power()
    statistics = fit_strict_inductive_normalizer(training)
    original_mean = statistics.mean.copy()
    training[:] = -99.0

    np.testing.assert_array_equal(statistics.mean, original_mean)
    with pytest.raises(ValueError, match="read-only"):
        statistics.mean[0, 0] = 0.0
