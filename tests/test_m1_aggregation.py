"""Synthetic tests for frozen D29-A and generic time/ROI aggregation.

Sofia's approved D29-A is authoritative. Both documented candidates remain invented regression
fixtures; no epoch, recall label, real feature, model, prediction, metric, or outcome is read.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from peers.m1_features import (
    FeatureTableValidationError,
    aggregation_contract_sha256,
    aggregate_normalized_log_power,
    resolved_time_bin_evidence,
    resolved_time_bin_masks,
)


ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
CONTRACT_PATH = ROOT / "docs" / "m1-feature-table-contract.json"


@pytest.fixture(scope="module")
def frozen_charter():
    return json.loads(CHARTER_PATH.read_text())


@pytest.fixture()
def pending_charter(frozen_charter):
    charter = deepcopy(frozen_charter)
    charter["features"]["analysis_window_s"] = "PENDING_D29"
    charter["features"]["time_bins_s"] = "PENDING_D29"
    charter["pending_human_gates"].insert(0, "D29_window_and_midpoint")
    return charter


@pytest.fixture(scope="module")
def feature_contract():
    return json.loads(CONTRACT_PATH.read_text())


def resolved_charter(pending_charter, option):
    charter = deepcopy(pending_charter)
    if option == "A":
        charter["features"]["analysis_window_s"] = [0.3, 1.6]
        charter["features"]["time_bins_s"] = [[0.3, 0.95], [0.95, 1.6]]
    elif option == "B":
        charter["features"]["analysis_window_s"] = [0.0, 2.0]
        charter["features"]["time_bins_s"] = [[0.0, 1.0], [1.0, 2.0]]
    else:
        raise AssertionError(f"unknown invented option {option}")
    return charter


def test_actual_charter_is_frozen_d29_a(frozen_charter, feature_contract):
    assert frozen_charter["features"]["analysis_window_s"] == [0.3, 1.6]
    assert frozen_charter["features"]["time_bins_s"] == [
        [0.3, 0.95],
        [0.95, 1.6],
    ]
    time_s = np.array([0.3, 0.6, 0.95, 1.2, 1.6], dtype=np.float64)
    t0, t1 = resolved_time_bin_masks(time_s, frozen_charter, feature_contract)
    np.testing.assert_array_equal(t0, [True, True, False, False, False])
    np.testing.assert_array_equal(t1, [False, False, True, True, True])


def declared_channels(charter):
    return [
        channel
        for roi_id in charter["features"]["roi_order"]
        for channel in charter["features"]["roi_channels"][roi_id]
    ]


def invented_normalized_power(*, trials=2, times=8):
    trial = np.arange(trials, dtype=np.float64)[:, None, None, None] * 10000.0
    channel = np.arange(40, dtype=np.float64)[None, :, None, None] * 100.0
    frequency = np.arange(8, dtype=np.float64)[None, None, :, None] * 10.0
    time = np.arange(times, dtype=np.float64)[None, None, None, :]
    return np.ascontiguousarray(trial + channel + frequency + time)


@pytest.mark.parametrize(
    ("option", "expected_t0", "expected_t1"),
    [
        ("A", [False, True, True, False, False, False, False, False],
         [False, False, False, True, True, True, True, False]),
        ("B", [True, True, True, True, False, False, False, False],
         [False, False, False, False, True, True, True, True]),
    ],
)
def test_masks_implement_both_candidate_boundaries_without_selecting_one(
    pending_charter, feature_contract, option, expected_t0, expected_t1
):
    time_s = np.array([0.0, 0.3, 0.6, 0.95, 1.0, 1.2, 1.6, 2.0], dtype=np.float64)
    masks = resolved_time_bin_masks(
        time_s,
        resolved_charter(pending_charter, option),
        feature_contract,
    )

    np.testing.assert_array_equal(masks[0], expected_t0)
    np.testing.assert_array_equal(masks[1], expected_t1)
    assert masks[0].flags.writeable is False
    assert masks[1].flags.writeable is False
    assert not np.any(masks[0] & masks[1])


def test_continuous_boundary_does_not_require_an_exact_sample(
    pending_charter, feature_contract
):
    time_s = np.array([0.3, 0.949, 0.951, 1.6], dtype=np.float64)
    t0, t1 = resolved_time_bin_masks(
        time_s,
        resolved_charter(pending_charter, "A"),
        feature_contract,
    )

    np.testing.assert_array_equal(t0, [True, True, False, False])
    np.testing.assert_array_equal(t1, [False, False, True, True])


@pytest.mark.parametrize("option", ["A", "B"])
def test_aggregation_is_exact_time_then_roi_then_frequency(
    pending_charter, feature_contract, option
):
    charter = resolved_charter(pending_charter, option)
    time_s = np.array([0.0, 0.3, 0.6, 0.95, 1.0, 1.2, 1.6, 2.0], dtype=np.float64)
    values = invented_normalized_power()
    channels = declared_channels(charter)

    features = aggregate_normalized_log_power(
        values, time_s, channels, charter, feature_contract
    )
    masks = resolved_time_bin_masks(time_s, charter, feature_contract)
    expected_blocks = []
    for mask in masks:
        for roi_index in range(8):
            expected_blocks.append(
                values[:, roi_index * 5 : (roi_index + 1) * 5, :, :][
                    :, :, :, mask
                ].mean(axis=(1, 3), dtype=np.float64)
            )
    expected = np.concatenate(expected_blocks, axis=1)

    np.testing.assert_array_equal(features, expected)
    assert features.shape == (2, 128)
    assert features.dtype == np.float64
    assert features.flags.c_contiguous
    assert np.isfinite(features).all()


def test_aggregation_reorders_channels_to_frozen_roi_membership(
    pending_charter, feature_contract
):
    charter = resolved_charter(pending_charter, "A")
    time_s = np.array([0.0, 0.3, 0.6, 0.95, 1.0, 1.2, 1.6, 2.0], dtype=np.float64)
    values = invented_normalized_power()
    channels = declared_channels(charter)
    expected = aggregate_normalized_log_power(
        values, time_s, channels, charter, feature_contract
    )

    permutation = np.arange(39, -1, -1)
    actual = aggregate_normalized_log_power(
        values[:, permutation, :, :],
        time_s,
        [channels[index] for index in permutation],
        charter,
        feature_contract,
    )
    np.testing.assert_array_equal(actual, expected)


def test_values_outside_resolved_window_cannot_change_option_a_features(
    pending_charter, feature_contract
):
    charter = resolved_charter(pending_charter, "A")
    time_s = np.array([0.0, 0.3, 0.6, 0.95, 1.0, 1.2, 1.6, 2.0], dtype=np.float64)
    values = invented_normalized_power()
    changed = values.copy()
    changed[:, :, :, [0, 7]] = 1e12
    channels = declared_channels(charter)

    np.testing.assert_array_equal(
        aggregate_normalized_log_power(
            values, time_s, channels, charter, feature_contract
        ),
        aggregate_normalized_log_power(
            changed, time_s, channels, charter, feature_contract
        ),
    )


def test_time_bin_evidence_binds_axis_masks_counts_and_charter(
    pending_charter, feature_contract
):
    charter = resolved_charter(pending_charter, "A")
    time_s = np.array([0.0, 0.3, 0.6, 0.95, 1.0, 1.2, 1.6, 2.0], dtype=np.float64)
    evidence = resolved_time_bin_evidence(time_s, charter, feature_contract)

    assert evidence == resolved_time_bin_evidence(time_s, charter, feature_contract)
    assert evidence["aggregation_contract_sha256"] == aggregation_contract_sha256(
        feature_contract
    )
    assert evidence["time_bin_ids"] == ["t0", "t1"]
    assert evidence["time_bin_sample_counts"] == [2, 4]
    assert len(evidence["time_axis_sha256"]) == 64
    assert len(evidence["time_bin_mask_sha256"]) == 2
    assert all(len(value) == 64 for value in evidence["time_bin_mask_sha256"])
    assert len(evidence["feature_schema_sha256"]) == 64


def test_aggregation_contract_drift_fails_before_sample_selection(
    pending_charter, feature_contract
):
    charter = resolved_charter(pending_charter, "A")
    time_s = np.array([0.3, 0.6, 0.95, 1.2, 1.6], dtype=np.float64)
    drifted = deepcopy(feature_contract)
    drifted["aggregation_numerical_contract"]["boundary_tolerance"] = "isclose"

    with pytest.raises(FeatureTableValidationError, match="numerical contract drifted"):
        resolved_time_bin_masks(time_s, charter, drifted)


def test_pending_d29_fails_before_sample_selection_or_aggregation(
    pending_charter, feature_contract
):
    time_s = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(FeatureTableValidationError, match="PENDING_D29"):
        resolved_time_bin_masks(time_s, pending_charter, feature_contract)
    with pytest.raises(FeatureTableValidationError, match="PENDING_D29"):
        aggregate_normalized_log_power(
            invented_normalized_power(times=3),
            time_s,
            declared_channels(pending_charter),
            pending_charter,
            feature_contract,
        )


@pytest.mark.parametrize(
    ("time_s", "message"),
    [
        (np.array([0.0, 0.3, 0.95, 1.6], dtype=np.float32), "float64"),
        (np.array([0.0, 0.95, 0.3, 1.6], dtype=np.float64), "strictly increasing"),
        (np.array([0.4, 0.95, 1.6], dtype=np.float64), "cover"),
        (np.array([0.0, 0.95, 1.6], dtype=np.float64), "nonempty t0"),
        (np.array([0.0, 0.3, np.nan, 1.6], dtype=np.float64), "finite"),
    ],
)
def test_time_axis_rejects_ambiguous_or_incomplete_sampling(
    pending_charter, feature_contract, time_s, message
):
    with pytest.raises(FeatureTableValidationError, match=message):
        resolved_time_bin_masks(
            time_s,
            resolved_charter(pending_charter, "A"),
            feature_contract,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values, times, channels: (values[:, :, :, :-1], times, channels),
         "time-axis length"),
        (lambda values, times, channels: (values.astype(np.float32), times, channels),
         "float64"),
        (lambda values, times, channels: (values[:, :39], times, channels[:-1]),
         "exactly 40 channels"),
        (lambda values, times, channels: (values, times, channels[:-1] + ["E999"]),
         "channel set"),
        (lambda values, times, channels: (values, times, channels[:-1] + [channels[0]]),
         "unique"),
    ],
)
def test_aggregation_rejects_axis_channel_and_dtype_drift(
    pending_charter, feature_contract, mutation, message
):
    charter = resolved_charter(pending_charter, "A")
    time_s = np.array([0.0, 0.3, 0.6, 0.95, 1.0, 1.2, 1.6, 2.0], dtype=np.float64)
    values, mutated_time, channels = mutation(
        invented_normalized_power(), time_s, declared_channels(charter)
    )
    with pytest.raises(FeatureTableValidationError, match=message):
        aggregate_normalized_log_power(
            values, mutated_time, channels, charter, feature_contract
        )
