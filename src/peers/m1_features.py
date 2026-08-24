"""Outcome-blind logical contracts for future M1 feature tables.

This module validates synthetic structure, identity, numeric shape, and provenance only. It
enforces Sofia's resolved D26-A and D29-A contracts but does not read EEG, build real features,
fit a model, or inspect performance.
"""

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from peers.m1_manifests import (
    ManifestValidationError,
    _fsync_directory,
    _rename_directory_noreplace,
    _write_fsynced,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    load_json_unique,
    manifest_sha256,
    validate_exact_key_alignment,
    validate_manifest,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PENDING_RE = re.compile(r"^PENDING_[A-Z0-9_]+$")

STRICT_INDUCTIVE_NORMALIZATION = {
    "decision": "D26_A_strict_inductive",
    "input": "log10_morlet_power_before_time_bin_and_roi_aggregation",
    "fit_scope": "permitted_model_training_sessions_only",
    "inner_fit_scope": "inner_training_sessions_only",
    "outer_refit_scope": "all_outer_training_sessions_only",
    "statistic_axes": ["trial", "time"],
    "preserved_axes": ["channel", "frequency"],
    "ddof": 0,
    "zero_standard_deviation_policy": "replace_with_1_then_center",
    "validation_application": "apply_inner_training_mean_and_std_unchanged",
    "outer_test_application": "apply_outer_training_mean_and_std_unchanged",
    "label_access": False,
    "held_session_distribution_access": False,
    "optional_sensitivity_if_compute_permits": (
        "D26_B_transductive_whole_held_session_batch_zscore_prespecified_not_primary"
    ),
}

M1_AGGREGATION_CONTRACT = {
    "decision_dependency": "resolved_D29_boundaries_from_frozen_charter",
    "input": "D26_A_normalized_log_power",
    "input_shape": "trial_x_40_selected_channels_x_8_frequencies_x_time",
    "time_axis_dtype": "float64_seconds_strictly_increasing",
    "first_bin_predicate": "time_greater_than_or_equal_start_and_less_than_midpoint",
    "second_bin_predicate": "time_greater_than_or_equal_midpoint_and_less_than_or_equal_end",
    "boundary_tolerance": "none_exact_float64_comparisons",
    "coverage": "time_axis_covers_complete_window_and_each_bin_has_samples",
    "channel_set": "exact_40_frozen_roi_channels",
    "channel_order": "reorder_to_roi_order_then_declared_channel_order",
    "roi_aggregation": "arithmetic_mean_over_5_channels_and_selected_time_samples",
    "output_order": "time_bin_then_roi_then_frequency_ascending",
    "output_shape": "trial_x_128",
    "output_dtype": "finite_float64",
    "evidence_fields": [
        "aggregation_contract_sha256",
        "feature_schema_sha256",
        "time_axis_sha256",
        "time_bin_ids",
        "time_bin_mask_sha256",
        "time_bin_sample_counts",
    ],
}

FEATURE_TABLE_BUNDLE_CONTRACT = {
    "schema_id": "neuroai-m1-feature-bundle-v1",
    "schema_version": "1.0",
    "atomic_publication": "temporary_directory_fsync_then_atomic_noreplace_rename",
    "exact_files": [
        "metadata.json",
        "trial-keys.json",
        "values.npy",
        "bundle.json",
        "bundle.sha256",
    ],
    "metadata": "logical_table_exact_fields_except_rows_in_canonical_JSON",
    "trial_keys": {
        "schema_id": "neuroai-m1-feature-bundle-trial-keys-v1",
        "schema_version": "1.0",
        "row_key": ["subject_id", "session", "trial", "item_num"],
        "canonical_order": "logical_table_row_order",
    },
    "values": {
        "format": "npy",
        "dtype": "little_endian_float64",
        "shape": "n_rows_by_128",
        "order": "C_contiguous",
        "allow_pickle": False,
        "trailing_bytes_allowed": False,
    },
    "bundle_manifest_exact_fields": [
        "schema_id",
        "schema_version",
        "bundle_contract_sha256",
        "logical_table_sha256",
        "metadata_sha256",
        "trial_keys_sha256",
        "values_npy_sha256",
        "row_count",
        "feature_count",
        "values_dtype",
        "values_shape",
        "values_c_contiguous",
        "allow_pickle",
    ],
    "outcome_fields_allowed": False,
    "real_publication_allowed_before_final_readiness": False,
    "external_bundle_sha256_required_on_readback": True,
    "failure_staging_policy": "retain_for_evidence_no_automatic_deletion",
}

FEATURE_TABLE_BUNDLE_FILES = frozenset(FEATURE_TABLE_BUNDLE_CONTRACT["exact_files"])


class FeatureTableValidationError(ManifestValidationError):
    """A logical feature table violates the pre-result contract."""


@dataclass(frozen=True, slots=True)
class StrictInductiveNormalizationStatistics:
    """Immutable D26-A statistics fitted on one permitted training partition."""

    mean: np.ndarray
    raw_std: np.ndarray
    scale: np.ndarray
    zero_standard_deviation_mask: np.ndarray
    n_fit_trials: int
    n_fit_timepoints: int
    ddof: int = 0


def _canonical_object_bytes(value: Any) -> bytes:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FeatureTableValidationError(f"value is not canonical finite JSON: {exc}") from exc
    return serialized.encode("utf-8") + b"\n"


def _require_float64_array(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FeatureTableValidationError(f"{label} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise FeatureTableValidationError(f"{label} must have exact float64 dtype")
    if value.size == 0:
        raise FeatureTableValidationError(f"{label} must not be empty")
    if not np.isfinite(value).all():
        raise FeatureTableValidationError(f"{label} must contain only finite values")
    return value


def _require_log_power_shape(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FeatureTableValidationError(f"{label} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise FeatureTableValidationError(f"{label} must have exact float64 dtype")
    array = value
    if array.ndim != 4:
        raise FeatureTableValidationError(
            f"{label} must have shape trial x 40 channels x 8 frequencies x time"
        )
    if array.shape[0] == 0:
        raise FeatureTableValidationError(f"{label} requires a nonempty trial axis")
    if array.shape[1] != 40:
        raise FeatureTableValidationError(f"{label} must contain exactly 40 channels")
    if array.shape[2] != 8:
        raise FeatureTableValidationError(f"{label} must contain exactly 8 frequencies")
    if array.shape[3] == 0:
        raise FeatureTableValidationError(f"{label} requires a nonempty time axis")
    if not np.isfinite(array).all():
        raise FeatureTableValidationError(f"{label} must contain only finite values")
    return array


def _readonly_copy(value: np.ndarray, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    copied = np.array(value, dtype=dtype, order="C", copy=True)
    copied.setflags(write=False)
    return copied


def _validate_strict_inductive_statistics(
    statistics: Any,
) -> StrictInductiveNormalizationStatistics:
    if not isinstance(statistics, StrictInductiveNormalizationStatistics):
        raise FeatureTableValidationError(
            "normalization statistics must use StrictInductiveNormalizationStatistics"
        )
    float_arrays = {
        "mean": statistics.mean,
        "raw_std": statistics.raw_std,
        "scale": statistics.scale,
    }
    for name, array in float_arrays.items():
        _require_float64_array(array, f"normalization statistics {name}")
        if array.shape != (40, 8):
            raise FeatureTableValidationError(
                f"normalization statistics {name} must have shape 40 x 8"
            )
        if array.flags.writeable:
            raise FeatureTableValidationError(
                f"normalization statistics {name} must be read-only"
            )
    mask = statistics.zero_standard_deviation_mask
    if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(np.bool_):
        raise FeatureTableValidationError(
            "normalization zero-standard-deviation mask must have bool dtype"
        )
    if mask.shape != (40, 8) or mask.flags.writeable:
        raise FeatureTableValidationError(
            "normalization zero-standard-deviation mask must be read-only shape 40 x 8"
        )
    if np.any(statistics.raw_std < 0.0):
        raise FeatureTableValidationError("normalization raw standard deviation is negative")
    expected_mask = statistics.raw_std == 0.0
    if not np.array_equal(mask, expected_mask):
        raise FeatureTableValidationError(
            "normalization zero-standard-deviation mask does not match raw std"
        )
    expected_scale = np.where(expected_mask, 1.0, statistics.raw_std)
    if not np.array_equal(statistics.scale, expected_scale):
        raise FeatureTableValidationError(
            "normalization scale does not implement zero-std replacement"
        )
    if not _is_integer(statistics.n_fit_trials) or statistics.n_fit_trials <= 0:
        raise FeatureTableValidationError(
            "normalization statistics require a positive fit-trial count"
        )
    if not _is_integer(statistics.n_fit_timepoints) or statistics.n_fit_timepoints <= 0:
        raise FeatureTableValidationError(
            "normalization statistics require a positive fit-timepoint count"
        )
    if statistics.ddof != 0:
        raise FeatureTableValidationError("normalization statistics ddof must equal 0")
    return statistics


def log10_power(power: Any) -> np.ndarray:
    """Apply the frozen `log10(power + 1e-30)` transform to finite float64 power."""
    array = _require_float64_array(power, "power")
    if np.any(array < 0.0):
        raise FeatureTableValidationError("power must be nonnegative")
    transformed = np.log10(array + np.float64(1e-30))
    if transformed.dtype != np.dtype(np.float64) or not np.isfinite(transformed).all():
        raise FeatureTableValidationError("log10 power transform produced a non-finite value")
    return np.ascontiguousarray(transformed, dtype=np.float64)


def fit_strict_inductive_normalizer(
    training_log_power: Any,
) -> StrictInductiveNormalizationStatistics:
    """Fit D26-A over training trial/time axes, preserving 40 channel x 8 frequency cells."""
    training = _require_log_power_shape(training_log_power, "training log power")
    mean = training.mean(axis=(0, 3), dtype=np.float64)
    raw_std = training.std(axis=(0, 3), ddof=0, dtype=np.float64)
    zero_mask = raw_std == 0.0
    scale = np.where(zero_mask, 1.0, raw_std)
    statistics = StrictInductiveNormalizationStatistics(
        mean=_readonly_copy(mean, dtype=np.float64),
        raw_std=_readonly_copy(raw_std, dtype=np.float64),
        scale=_readonly_copy(scale, dtype=np.float64),
        zero_standard_deviation_mask=_readonly_copy(zero_mask, dtype=np.bool_),
        n_fit_trials=training.shape[0],
        n_fit_timepoints=training.shape[3],
        ddof=0,
    )
    return _validate_strict_inductive_statistics(statistics)


def apply_strict_inductive_normalizer(
    log_power: Any,
    statistics: StrictInductiveNormalizationStatistics,
) -> np.ndarray:
    """Apply already-fitted D26-A statistics unchanged to any permitted partition."""
    values = _require_log_power_shape(log_power, "log power to normalize")
    fitted = _validate_strict_inductive_statistics(statistics)
    normalized = (values - fitted.mean[None, :, :, None]) / fitted.scale[
        None, :, :, None
    ]
    if normalized.dtype != np.dtype(np.float64) or not np.isfinite(normalized).all():
        raise FeatureTableValidationError("strict-inductive normalization produced nonfinite data")
    return np.ascontiguousarray(normalized, dtype=np.float64)


def normalization_array_sha256(value: Any) -> str:
    """Hash one typed numerical D26 artifact using fixed little-endian C-order bytes."""
    if not isinstance(value, np.ndarray):
        raise FeatureTableValidationError("normalization array must be a NumPy array")
    if value.size == 0:
        raise FeatureTableValidationError("normalization array must not be empty")
    if value.dtype == np.dtype(np.float64):
        if not np.isfinite(value).all():
            raise FeatureTableValidationError("normalization array must be finite")
        canonical = np.ascontiguousarray(value.astype("<f8", copy=False))
        dtype_id = "little_endian_float64"
    elif value.dtype == np.dtype(np.bool_):
        canonical = np.ascontiguousarray(value.astype(np.uint8, copy=False))
        dtype_id = "boolean_uint8"
    else:
        raise FeatureTableValidationError(
            "normalization array does not have a supported dtype (float64 or bool)"
        )
    header = _canonical_object_bytes(
        {"dtype": dtype_id, "shape": [int(size) for size in value.shape]}
    )
    return sha256(header + canonical.tobytes(order="C")).hexdigest()


def normalization_statistics_evidence(
    statistics: StrictInductiveNormalizationStatistics,
) -> dict[str, Any]:
    """Return exact provenance fields for one fitted numerical D26-A transform."""
    fitted = _validate_strict_inductive_statistics(statistics)
    return {
        "mean_array_sha256": normalization_array_sha256(fitted.mean),
        "raw_std_array_sha256": normalization_array_sha256(fitted.raw_std),
        "std_array_sha256": normalization_array_sha256(fitted.scale),
        "zero_std_mask_sha256": normalization_array_sha256(
            fitted.zero_standard_deviation_mask
        ),
        "n_zero_std_cells": int(np.count_nonzero(fitted.zero_standard_deviation_mask)),
        "n_fit_timepoints": fitted.n_fit_timepoints,
    }


def _require_time_axis(time_s: Any) -> np.ndarray:
    time = _require_float64_array(time_s, "time axis")
    if time.ndim != 1:
        raise FeatureTableValidationError("time axis must be one-dimensional")
    if time.size < 2:
        raise FeatureTableValidationError("time axis must contain at least two samples")
    if not np.all(np.diff(time) > 0.0):
        raise FeatureTableValidationError("time axis must be strictly increasing")
    return time


def _aggregation_contract_payload(
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is None:
        return deepcopy(M1_AGGREGATION_CONTRACT)
    try:
        declared = contract["aggregation_numerical_contract"]
    except (KeyError, TypeError) as exc:
        raise FeatureTableValidationError(
            "feature contract is missing the aggregation numerical contract"
        ) from exc
    if declared != M1_AGGREGATION_CONTRACT:
        raise FeatureTableValidationError(
            "feature aggregation numerical contract drifted from the implementation"
        )
    return deepcopy(declared)


def aggregation_contract_sha256(
    contract: Mapping[str, Any] | None = None,
) -> str:
    """Digest the numerical mapping applied to the charter's resolved D29 bounds."""
    return sha256(_canonical_object_bytes(_aggregation_contract_payload(contract))).hexdigest()


def resolved_time_bin_masks(
    time_s: Any,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a resolved D29 charter to exact, disjoint sample masks without tolerance."""
    _aggregation_contract_payload(contract)
    _feature_schema_payload(charter, contract)
    time = _require_time_axis(time_s)
    window_start, window_end = charter["features"]["analysis_window_s"]
    first_bin, second_bin = charter["features"]["time_bins_s"]
    if time[0] > window_start or time[-1] < window_end:
        raise FeatureTableValidationError(
            "time axis does not cover the complete resolved analysis window"
        )

    masks = (
        (time >= first_bin[0]) & (time < first_bin[1]),
        (time >= second_bin[0]) & (time <= second_bin[1]),
    )
    for index, mask in enumerate(masks):
        if not np.any(mask):
            raise FeatureTableValidationError(
                f"resolved time axis requires a nonempty t{index} sample set"
            )
    if np.any(masks[0] & masks[1]):
        raise FeatureTableValidationError("resolved time-bin sample masks overlap")
    window_mask = (time >= window_start) & (time <= window_end)
    if not np.array_equal(masks[0] | masks[1], window_mask):
        raise FeatureTableValidationError(
            "resolved time-bin masks do not exactly partition the analysis window"
        )
    return (
        _readonly_copy(masks[0], dtype=np.bool_),
        _readonly_copy(masks[1], dtype=np.bool_),
    )


def resolved_time_bin_evidence(
    time_s: Any,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind one resolved D29 sample mapping without storing real feature values."""
    time = _require_time_axis(time_s)
    masks = resolved_time_bin_masks(time, charter, contract)
    return {
        "aggregation_contract_sha256": aggregation_contract_sha256(contract),
        "feature_schema_sha256": feature_schema_sha256(charter, contract),
        "time_axis_sha256": normalization_array_sha256(time),
        "time_bin_ids": list(charter["features"]["time_bin_ids"]),
        "time_bin_mask_sha256": [normalization_array_sha256(mask) for mask in masks],
        "time_bin_sample_counts": [int(np.count_nonzero(mask)) for mask in masks],
    }


def aggregate_normalized_log_power(
    normalized_log_power: Any,
    time_s: Any,
    channel_names: Sequence[str],
    charter: Mapping[str, Any],
    contract: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Aggregate a resolved normalized array in frozen time, ROI, frequency order."""
    _aggregation_contract_payload(contract)
    _feature_schema_payload(charter, contract)
    values = _require_log_power_shape(
        normalized_log_power, "normalized log power to aggregate"
    )
    time = _require_time_axis(time_s)
    if values.shape[3] != time.size:
        raise FeatureTableValidationError(
            "normalized log-power time-axis length does not match the time vector"
        )
    if isinstance(channel_names, (str, bytes)) or not isinstance(channel_names, Sequence):
        raise FeatureTableValidationError("channel names must be a sequence")
    names = list(channel_names)
    if len(names) != 40 or not all(isinstance(name, str) and name for name in names):
        raise FeatureTableValidationError("aggregation requires exactly 40 channel names")
    if len(set(names)) != len(names):
        raise FeatureTableValidationError("aggregation channel names must be unique")

    features = charter["features"]
    if features["aggregation"] != (
        "arithmetic_mean_over_declared_time_samples_and_roi_channels"
    ):
        raise FeatureTableValidationError("aggregation rule does not match the frozen contract")
    expected_channels = [
        channel
        for roi_id in features["roi_order"]
        for channel in features["roi_channels"][roi_id]
    ]
    if set(names) != set(expected_channels):
        raise FeatureTableValidationError(
            "aggregation channel set does not match the frozen ROI map"
        )
    source_index = {name: index for index, name in enumerate(names)}
    ordered = values[:, [source_index[name] for name in expected_channels], :, :]
    masks = resolved_time_bin_masks(time, charter, contract)

    blocks = []
    for mask in masks:
        for roi_index in range(len(features["roi_order"])):
            roi_values = ordered[:, roi_index * 5 : (roi_index + 1) * 5, :, :]
            blocks.append(
                roi_values[:, :, :, mask].mean(axis=(1, 3), dtype=np.float64)
            )
    aggregated = np.concatenate(blocks, axis=1)
    if aggregated.shape != (values.shape[0], features["expected_feature_count"]):
        raise FeatureTableValidationError(
            "aggregated feature shape does not match the frozen feature count"
        )
    if aggregated.dtype != np.dtype(np.float64) or not np.isfinite(aggregated).all():
        raise FeatureTableValidationError("aggregated features must be finite float64")
    return np.ascontiguousarray(aggregated, dtype=np.float64)


def _require_exact_fields(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise FeatureTableValidationError(
            f"{label} fields mismatch; missing={missing}; extra={extra}"
        )


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise FeatureTableValidationError(f"{label} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise FeatureTableValidationError(f"{label} must be a valid UTC timestamp") from exc


def _validate_type(value: Any, expected: str, label: str) -> None:
    valid = False
    if expected == "string":
        valid = isinstance(value, str) and bool(value.strip())
    elif expected == "timestamp_utc":
        _validate_timestamp(value, label)
        return
    elif expected == "sha256":
        valid = isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
    elif expected == "sha256_or_null":
        valid = value is None or (
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
        )
    elif expected == "git_commit":
        valid = isinstance(value, str) and GIT_COMMIT_RE.fullmatch(value) is not None
    elif expected == "array[string]":
        valid = isinstance(value, list) and all(
            isinstance(item, str) and bool(item) for item in value
        )
    elif expected == "array[object]":
        valid = isinstance(value, list) and all(isinstance(item, dict) for item in value)
    elif expected == "array[integer]":
        valid = isinstance(value, list) and all(_is_integer(item) for item in value)
    elif expected == "integer":
        valid = _is_integer(value)
    elif expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "string_or_null":
        valid = value is None or (isinstance(value, str) and bool(value.strip()))
    elif expected == "array[float64]":
        valid = isinstance(value, list) and all(type(item) is float for item in value)
    else:
        raise FeatureTableValidationError(f"unknown contract type {expected!r} at {label}")
    if not valid:
        raise FeatureTableValidationError(f"{label} must have type {expected}")


def _find_pending(value: Any) -> str | None:
    if isinstance(value, str) and PENDING_RE.fullmatch(value):
        return value
    if isinstance(value, dict):
        for child in value.values():
            pending = _find_pending(child)
            if pending:
                return pending
    if isinstance(value, list):
        for child in value:
            pending = _find_pending(child)
            if pending:
                return pending
    return None


def _validate_resolved_time_axis(features: Mapping[str, Any]) -> None:
    window = features["analysis_window_s"]
    bins = features["time_bins_s"]
    if not (
        isinstance(window, list)
        and len(window) == 2
        and all(type(value) is float for value in window)
        and window[0] < window[1]
    ):
        raise FeatureTableValidationError("analysis_window_s must be two increasing floats")
    if not (
        isinstance(bins, list)
        and len(bins) == 2
        and all(
            isinstance(time_bin, list)
            and len(time_bin) == 2
            and all(type(value) is float for value in time_bin)
            and time_bin[0] < time_bin[1]
            for time_bin in bins
        )
    ):
        raise FeatureTableValidationError("time_bins_s must contain two increasing float pairs")
    if bins[0][0] != window[0] or bins[1][1] != window[1]:
        raise FeatureTableValidationError("time bins must span the analysis window exactly")
    if bins[0][1] != bins[1][0]:
        raise FeatureTableValidationError("time bins must share one exact boundary")
    if features["time_bin_interval_convention"] != [
        "left_closed_right_open",
        "left_closed_right_closed",
    ]:
        raise FeatureTableValidationError(
            "time-bin interval convention must assign the midpoint only to t1"
        )


def _feature_schema_payload(
    charter: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    features = charter.get("features")
    if not isinstance(features, dict):
        raise FeatureTableValidationError("charter features must be an object")
    if contract is None:
        fields = (
            "source",
            "method",
            "frequencies_hz",
            "frequency_ids",
            "frequency_formula",
            "n_cycles_formula",
            "decim",
            "log_transform",
            "normalization",
            "analysis_window_s",
            "time_bins_s",
            "time_bin_interval_convention",
            "time_bin_ids",
            "roi_map_id",
            "roi_order",
            "roi_channels",
            "aggregation",
            "feature_order",
            "feature_name_template",
            "feature_axis_id_policy",
            "dtype",
            "finite_required",
            "missing_channel_policy",
            "expected_feature_count",
        )
    else:
        fields = tuple(contract["feature_schema_sha256"]["fields"])
    missing = [field for field in fields if field not in features]
    if missing:
        raise FeatureTableValidationError(f"charter feature schema fields missing: {missing}")
    payload = {field: deepcopy(features[field]) for field in fields}
    pending = _find_pending(payload)
    if pending:
        raise FeatureTableValidationError(
            f"feature schema cannot be frozen while {pending} is unresolved"
        )

    _normalization_contract_payload(charter, contract)
    _validate_resolved_time_axis(features)
    time_ids = features["time_bin_ids"]
    frequency_ids = features["frequency_ids"]
    frequencies = features["frequencies_hz"]
    roi_order = features["roi_order"]
    if time_ids != ["t0", "t1"]:
        raise FeatureTableValidationError("time_bin_ids must equal ['t0', 't1']")
    if frequency_ids != [f"f{index:02d}" for index in range(8)]:
        raise FeatureTableValidationError("frequency_ids must be the frozen f00..f07 sequence")
    if not (
        isinstance(frequencies, list)
        and len(frequencies) == len(frequency_ids)
        and all(type(value) is float and math.isfinite(value) for value in frequencies)
        and frequencies == sorted(frequencies)
        and len(set(frequencies)) == len(frequencies)
    ):
        raise FeatureTableValidationError("frequencies_hz must be eight finite ascending floats")
    if not isinstance(roi_order, list) or len(roi_order) != 8 or len(set(roi_order)) != 8:
        raise FeatureTableValidationError("roi_order must contain eight unique identifiers")
    roi_channels = features["roi_channels"]
    if not isinstance(roi_channels, dict) or list(roi_channels) != roi_order:
        raise FeatureTableValidationError("roi_channels must preserve the declared ROI order")
    if features["feature_order"] != ["time_bin", "roi", "frequency_ascending"]:
        raise FeatureTableValidationError("feature_order does not match the frozen contract")
    template = "eeg__time-{time_bin_id}__roi-{roi_id}__freq-{frequency_id}"
    if features["feature_name_template"] != template:
        raise FeatureTableValidationError(
            "feature_name_template does not match the frozen contract"
        )
    expected_count = len(time_ids) * len(roi_order) * len(frequency_ids)
    if features["expected_feature_count"] != expected_count or expected_count != 128:
        raise FeatureTableValidationError("expected feature count does not equal 2 x 8 x 8")
    if features["dtype"] != "float64" or features["finite_required"] is not True:
        raise FeatureTableValidationError("feature values must be finite float64")
    return payload


def feature_schema_sha256(
    charter: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> str:
    """Digest the resolved computational feature schema; pending D26/D29 fail closed."""
    return sha256(_canonical_object_bytes(_feature_schema_payload(charter, contract))).hexdigest()


def _normalization_contract_payload(
    charter: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    expected = STRICT_INDUCTIVE_NORMALIZATION
    expected_extra_scaler = False
    if contract is not None:
        try:
            declared = contract["normalization_contract"]
            declared_normalization = declared["required_charter_normalization"]
            declared_extra_scaler = declared["required_m1_extra_scaler"]
        except (KeyError, TypeError) as exc:
            raise FeatureTableValidationError(
                "feature contract is missing the D26 normalization contract"
            ) from exc
        if (
            declared_normalization != STRICT_INDUCTIVE_NORMALIZATION
            or declared_extra_scaler is not False
        ):
            raise FeatureTableValidationError(
                "feature contract drifted from strict-inductive D26-A"
            )
        expected = declared_normalization
        expected_extra_scaler = declared_extra_scaler
    try:
        normalization = charter["features"]["normalization"]
        extra_scaler = charter["learners"]["M1"]["extra_scaler"]
    except (KeyError, TypeError) as exc:
        raise FeatureTableValidationError(
            "charter is missing the D26 normalization or M1 scaler contract"
        ) from exc
    pending = _find_pending(normalization)
    if pending:
        raise FeatureTableValidationError(
            f"normalization contract cannot be frozen while {pending} is unresolved"
        )
    if not isinstance(normalization, dict):
        raise FeatureTableValidationError(
            "normalization must be the exact strict-inductive D26-A object"
        )
    if normalization.get("held_session_distribution_access") is not False:
        raise FeatureTableValidationError(
            "strict-inductive D26-A forbids held-session distribution access"
        )
    if normalization.get("label_access") is not False:
        raise FeatureTableValidationError("strict-inductive normalization forbids label access")
    if normalization != expected:
        raise FeatureTableValidationError(
            "normalization does not equal the exact strict-inductive D26-A contract"
        )
    if extra_scaler is not expected_extra_scaler:
        raise FeatureTableValidationError(
            "M1 extra scaler must be false under the strict-inductive D26-A contract"
        )
    return {
        "features.normalization": deepcopy(normalization),
        "learners.M1.extra_scaler": extra_scaler,
    }


def normalization_contract_sha256(
    charter: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> str:
    """Digest Sofia's exact D26-A transform contract without requiring D29."""
    return sha256(
        _canonical_object_bytes(_normalization_contract_payload(charter, contract))
    ).hexdigest()


def _normalization_record_sort_key(record: Mapping[str, Any]) -> tuple[int, str]:
    if record.get("stage") == "inner_model_selection":
        return (0, str(record.get("inner_fold_id", "")))
    return (1, str(record.get("transform_id", "")))


def _validate_normalization_transform_provenance(
    provenance: Any,
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
    require_canonical_order: bool,
) -> None:
    if not isinstance(provenance, dict):
        raise FeatureTableValidationError(
            "normalization transform provenance must be an object"
        )
    try:
        provenance_contract = feature_contract["normalization_transform_provenance"]
    except (KeyError, TypeError) as exc:
        raise FeatureTableValidationError(
            "feature contract is missing normalization transform provenance"
        ) from exc
    _require_exact_fields(
        provenance, provenance_contract["exact_fields"], "normalization provenance"
    )
    for field, expected_type in provenance_contract["field_types"].items():
        _validate_type(
            provenance[field], expected_type, f"normalization provenance {field}"
        )
    if provenance["schema_id"] != provenance_contract["schema_id"]:
        raise FeatureTableValidationError("normalization provenance schema_id mismatch")
    if provenance["schema_version"] != provenance_contract["schema_version"]:
        raise FeatureTableValidationError("normalization provenance schema_version mismatch")
    if provenance["condition"] not in provenance_contract["allowed_conditions"]:
        raise FeatureTableValidationError("normalization provenance condition is not allowed")
    if provenance["outer_test_session"] < 0:
        raise FeatureTableValidationError("outer_test_session must be nonnegative")

    required_bindings = set(provenance_contract["external_binding_fields"])
    if set(expected_bindings) != required_bindings:
        raise FeatureTableValidationError(
            "expected normalization bindings must exactly match the contract"
        )
    for field, expected in expected_bindings.items():
        if provenance[field] != expected:
            raise FeatureTableValidationError(
                f"normalization provenance binding mismatch for {field}"
            )

    try:
        validate_manifest(trial_manifest, manifest_contract)
        validate_manifest(inner_fold_manifest, manifest_contract)
        trial_digest = manifest_sha256(trial_manifest, manifest_contract)
        fold_digest = manifest_sha256(inner_fold_manifest, manifest_contract)
    except ManifestValidationError as exc:
        raise FeatureTableValidationError(str(exc)) from exc
    recomputed_bindings = {
        "trial_manifest_sha256": trial_digest,
        "inner_fold_manifest_sha256": fold_digest,
        "feature_schema_sha256": feature_schema_sha256(charter, feature_contract),
        "normalization_contract_sha256": normalization_contract_sha256(
            charter, feature_contract
        ),
    }
    for field, recomputed in recomputed_bindings.items():
        if provenance[field] != recomputed:
            raise FeatureTableValidationError(
                f"normalization provenance {field} does not match its source"
            )

    subject_id = provenance["subject_id"]
    condition = provenance["condition"]
    trial_rows = trial_manifest["rows"]
    trial_groups = {(row["subject_id"], row["condition"]) for row in trial_rows}
    if trial_groups != {(subject_id, condition)}:
        raise FeatureTableValidationError(
            "trial manifest must contain exactly the normalization subject-condition"
        )
    outer_train_rows = [
        row
        for row in trial_rows
        if row["condition_survivor"] and row["outer_role"] == "train"
    ]
    outer_test_rows = [
        row
        for row in trial_rows
        if row["condition_survivor"] and row["outer_role"] == "test"
    ]
    if not outer_train_rows or not outer_test_rows:
        raise FeatureTableValidationError(
            "normalization requires nonempty outer train and test survivor rows"
        )
    outer_test_sessions = {row["session"] for row in outer_test_rows}
    if outer_test_sessions != {provenance["outer_test_session"]}:
        raise FeatureTableValidationError(
            "normalization outer test session does not match trial survivors"
        )
    if any(row["session"] == provenance["outer_test_session"] for row in outer_train_rows):
        raise FeatureTableValidationError(
            "normalization outer training rows contain the held session"
        )

    fold_rows = [row for row in inner_fold_manifest["rows"] if row["eligible"]]
    fold_groups = {(row["subject_id"], row["condition"]) for row in fold_rows}
    if not fold_rows or fold_groups != {(subject_id, condition)}:
        raise FeatureTableValidationError(
            "eligible inner folds must match exactly the normalization subject-condition"
        )
    if any(row["outer_test_session"] != provenance["outer_test_session"] for row in fold_rows):
        raise FeatureTableValidationError(
            "inner folds and normalization provenance disagree on outer test session"
        )

    channel_order = [
        channel
        for roi_id in charter["features"]["roi_order"]
        for channel in charter["features"]["roi_channels"][roi_id]
    ]
    if len(channel_order) != 40 or len(set(channel_order)) != 40:
        raise FeatureTableValidationError(
            "charter must provide 40 unique normalization statistic channels"
        )
    if provenance["statistic_dtype"] != provenance_contract["statistic_dtype"]:
        raise FeatureTableValidationError("normalization statistic dtype mismatch")
    if provenance["statistic_channel_order"] != channel_order:
        raise FeatureTableValidationError("normalization statistic channel order mismatch")
    if provenance["statistic_frequency_ids"] != charter["features"]["frequency_ids"]:
        raise FeatureTableValidationError("normalization statistic frequency order mismatch")
    if provenance["statistic_shape"] != provenance_contract["statistic_shape"]:
        raise FeatureTableValidationError("normalization statistic shape mismatch")

    records = provenance["records"]
    expected_count = len(fold_rows) + 1
    if len(records) != expected_count:
        raise FeatureTableValidationError(
            "normalization provenance requires one record per inner fold plus outer refit"
        )
    for index, record in enumerate(records):
        _require_exact_fields(
            record,
            provenance_contract["record_exact_fields"],
            f"normalization record {index}",
        )
        for field, expected_type in provenance_contract["record_field_types"].items():
            _validate_type(
                record[field], expected_type, f"normalization record {index} {field}"
            )
        if record["n_fit_trials"] <= 0 or record["n_apply_trials"] <= 0:
            raise FeatureTableValidationError(
                "normalization fit and apply trial counts must be positive"
            )
        if record["n_fit_timepoints"] <= 0:
            raise FeatureTableValidationError(
                "normalization fit-timepoint count must be positive"
            )
        if not 0 <= record["n_zero_std_cells"] <= 40 * 8:
            raise FeatureTableValidationError(
                "normalization zero-standard-deviation count must be between 0 and 320"
            )
        if record["statistic_axes"] != STRICT_INDUCTIVE_NORMALIZATION["statistic_axes"]:
            raise FeatureTableValidationError("normalization statistic axes mismatch")
        if record["preserved_axes"] != STRICT_INDUCTIVE_NORMALIZATION["preserved_axes"]:
            raise FeatureTableValidationError("normalization preserved axes mismatch")
        if record["ddof"] != STRICT_INDUCTIVE_NORMALIZATION["ddof"]:
            raise FeatureTableValidationError("normalization ddof mismatch")
        if record["zero_standard_deviation_policy"] != STRICT_INDUCTIVE_NORMALIZATION[
            "zero_standard_deviation_policy"
        ]:
            raise FeatureTableValidationError(
                "normalization zero-standard-deviation policy mismatch"
            )
        if record["label_access"] is not False:
            raise FeatureTableValidationError("normalization label access must be false")
        if record["held_session_distribution_access"] is not False:
            raise FeatureTableValidationError(
                "normalization held-session distribution access must be false"
            )

    record_by_id = {record["transform_id"]: record for record in records}
    if len(record_by_id) != len(records):
        raise FeatureTableValidationError("duplicate normalization transform_id")
    for fold in sorted(fold_rows, key=lambda row: row["inner_fold_id"]):
        fold_id = fold["inner_fold_id"]
        transform_id = provenance_contract["inner_transform_id_template"].format(
            inner_fold_id=fold_id
        )
        try:
            record = record_by_id.pop(transform_id)
        except KeyError as exc:
            raise FeatureTableValidationError(
                f"normalization provenance is missing inner fold {fold_id}"
            ) from exc
        expected_fit_sessions = fold["fit_sessions"]
        expected_apply_sessions = [fold["validation_session"]]
        expected_fit_keys = [
            canonical_trial_key(row)
            for row in outer_train_rows
            if row["session"] in expected_fit_sessions
        ]
        expected_apply_keys = [
            canonical_trial_key(row)
            for row in outer_train_rows
            if row["session"] == fold["validation_session"]
        ]
        if record["stage"] != "inner_model_selection" or record["inner_fold_id"] != fold_id:
            raise FeatureTableValidationError(
                f"normalization inner record identity mismatch for {fold_id}"
            )
        if record["fit_sessions"] != expected_fit_sessions:
            raise FeatureTableValidationError(
                f"normalization inner {fold_id} fit sessions mismatch"
            )
        if record["apply_sessions"] != expected_apply_sessions:
            raise FeatureTableValidationError(
                f"normalization inner {fold_id} apply sessions mismatch"
            )
        if record["fit_keys_sha256"] != canonical_trial_key_set_sha256(expected_fit_keys):
            raise FeatureTableValidationError(
                f"normalization inner {fold_id} fit keys mismatch"
            )
        if record["apply_keys_sha256"] != canonical_trial_key_set_sha256(expected_apply_keys):
            raise FeatureTableValidationError(
                f"normalization inner {fold_id} apply keys mismatch"
            )
        if (
            record["fit_keys_sha256"] != fold["fit_keys_sha256"]
            or record["apply_keys_sha256"] != fold["validation_keys_sha256"]
        ):
            raise FeatureTableValidationError(
                f"normalization inner {fold_id} key hashes disagree with fold manifest"
            )
        if (
            record["n_fit_trials"] != len(expected_fit_keys)
            or record["n_apply_trials"] != len(expected_apply_keys)
        ):
            raise FeatureTableValidationError(
                f"normalization inner {fold_id} trial counts mismatch"
            )

    outer_id = provenance_contract["outer_transform_id"]
    try:
        outer_record = record_by_id.pop(outer_id)
    except KeyError as exc:
        raise FeatureTableValidationError(
            "normalization provenance is missing the outer refit"
        ) from exc
    if record_by_id:
        raise FeatureTableValidationError(
            f"normalization provenance has unknown transforms: {sorted(record_by_id)}"
        )
    expected_outer_fit_sessions = sorted({row["session"] for row in outer_train_rows})
    expected_outer_fit_keys = [canonical_trial_key(row) for row in outer_train_rows]
    expected_outer_apply_keys = [canonical_trial_key(row) for row in outer_test_rows]
    if outer_record["stage"] != "outer_refit" or outer_record["inner_fold_id"] is not None:
        raise FeatureTableValidationError("normalization outer refit identity mismatch")
    if outer_record["fit_sessions"] != expected_outer_fit_sessions:
        raise FeatureTableValidationError("normalization outer refit fit sessions mismatch")
    if outer_record["apply_sessions"] != [provenance["outer_test_session"]]:
        raise FeatureTableValidationError("normalization outer refit apply sessions mismatch")
    if outer_record["fit_keys_sha256"] != canonical_trial_key_set_sha256(
        expected_outer_fit_keys
    ):
        raise FeatureTableValidationError("normalization outer refit fit keys mismatch")
    if outer_record["apply_keys_sha256"] != canonical_trial_key_set_sha256(
        expected_outer_apply_keys
    ):
        raise FeatureTableValidationError("normalization outer refit apply keys mismatch")
    if (
        outer_record["n_fit_trials"] != len(expected_outer_fit_keys)
        or outer_record["n_apply_trials"] != len(expected_outer_apply_keys)
    ):
        raise FeatureTableValidationError("normalization outer refit trial counts mismatch")

    canonical_records = sorted(records, key=_normalization_record_sort_key)
    if require_canonical_order and records != canonical_records:
        raise FeatureTableValidationError(
            "normalization provenance records are not in canonical record order"
        )


def validate_normalization_transform_provenance(
    provenance: Any,
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> None:
    """Validate exact outcome-blind inner/outer D26 fit and apply provenance."""
    _validate_normalization_transform_provenance(
        provenance,
        charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        expected_bindings=expected_bindings,
        require_canonical_order=True,
    )


def canonical_normalization_transform_provenance_bytes(
    provenance: Any,
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> bytes:
    """Validate and serialize D26 provenance in deterministic transform order."""
    _validate_normalization_transform_provenance(
        provenance,
        charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        expected_bindings=expected_bindings,
        require_canonical_order=False,
    )
    canonical = deepcopy(provenance)
    canonical["records"] = sorted(
        canonical["records"], key=_normalization_record_sort_key
    )
    return _canonical_object_bytes(canonical)


def normalization_transform_provenance_sha256(
    provenance: Any,
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
) -> str:
    """SHA-256 of canonical, leakage-checked D26 transform provenance."""
    return sha256(
        canonical_normalization_transform_provenance_bytes(
            provenance,
            charter,
            feature_contract,
            manifest_contract,
            trial_manifest=trial_manifest,
            inner_fold_manifest=inner_fold_manifest,
            expected_bindings=expected_bindings,
        )
    ).hexdigest()


def derive_feature_names(charter: Mapping[str, Any]) -> list[str]:
    """Derive the exact 128 byte-stable names without encoding numeric D26/D29 choices."""
    _feature_schema_payload(charter)
    features = charter["features"]
    return [
        features["feature_name_template"].format(
            time_bin_id=time_bin_id,
            roi_id=roi_id,
            frequency_id=frequency_id,
        )
        for time_bin_id in features["time_bin_ids"]
        for roi_id in features["roi_order"]
        for frequency_id in features["frequency_ids"]
    ]


def _validate_feature_table(
    table: Any,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_keys: Iterable[Sequence[Any]],
    expected_bindings: Mapping[str, str],
    require_canonical_order: bool,
) -> None:
    if not isinstance(table, dict):
        raise FeatureTableValidationError("feature table must be an object")
    table_contract = contract["logical_table"]
    _require_exact_fields(table, table_contract["exact_fields"], "feature table")
    for field, expected_type in table_contract["field_types"].items():
        _validate_type(table[field], expected_type, f"feature table {field}")
    if table["schema_id"] != table_contract["schema_id"]:
        raise FeatureTableValidationError("feature table schema_id mismatch")
    if table["schema_version"] != table_contract["schema_version"]:
        raise FeatureTableValidationError("feature table schema_version mismatch")
    if table["condition"] not in table_contract["allowed_conditions"]:
        raise FeatureTableValidationError("feature table condition is not allowed")

    required_bindings = set(table_contract["external_binding_fields"])
    if set(expected_bindings) != required_bindings:
        raise FeatureTableValidationError(
            "expected feature-table bindings must exactly match the contract"
        )
    for field, expected in expected_bindings.items():
        if table[field] != expected:
            raise FeatureTableValidationError(f"feature table binding mismatch for {field}")

    actual_schema_sha256 = feature_schema_sha256(charter, contract)
    if table["feature_schema_sha256"] != actual_schema_sha256:
        raise FeatureTableValidationError("feature_schema_sha256 does not match the charter")
    expected_names = derive_feature_names(charter)
    if table["feature_names"] != expected_names:
        raise FeatureTableValidationError("feature_names do not match exact frozen axis order")

    rows = table["rows"]
    if not rows:
        raise FeatureTableValidationError("feature table must contain at least one row")
    row_fields = table_contract["row_exact_fields"]
    row_types = table_contract["row_field_types"]
    observed_keys = []
    for index, row in enumerate(rows):
        _require_exact_fields(row, row_fields, f"feature row {index}")
        for field, expected_type in row_types.items():
            _validate_type(row[field], expected_type, f"feature row {index} {field}")
        if row["subject_id"] != table["subject_id"]:
            raise FeatureTableValidationError(
                "feature row subject_id differs from table subject_id"
            )
        if len(row["values"]) != len(expected_names):
            raise FeatureTableValidationError(
                "feature row value count does not match feature_names"
            )
        if not all(math.isfinite(value) for value in row["values"]):
            raise FeatureTableValidationError("feature row contains a non-finite value")
        observed_keys.append(tuple(row[field] for field in table_contract["row_key"]))

    sorted_rows = sorted(
        rows, key=lambda row: tuple(row[field] for field in table_contract["row_key"])
    )
    if require_canonical_order and rows != sorted_rows:
        raise FeatureTableValidationError("feature table rows are not in canonical row order")

    expected_key_list = list(expected_keys)
    if not expected_key_list:
        raise FeatureTableValidationError("expected feature-table keys must not be empty")
    try:
        validate_exact_key_alignment(
            expected_key_list,
            observed_keys,
            expected_name="feature partition expected keys",
            observed_name="feature table rows",
        )
        expected_key_sha256 = canonical_trial_key_set_sha256(expected_key_list)
    except ManifestValidationError as exc:
        raise FeatureTableValidationError(str(exc)) from exc
    if table["expected_key_set_sha256"] != expected_key_sha256:
        raise FeatureTableValidationError("expected_key_set_sha256 does not match expected keys")


def validate_feature_table(
    table: Any,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_keys: Iterable[Sequence[Any]],
    expected_bindings: Mapping[str, str],
) -> None:
    """Validate one stored logical table in canonical row order."""
    _validate_feature_table(
        table,
        charter,
        contract,
        expected_keys=expected_keys,
        expected_bindings=expected_bindings,
        require_canonical_order=True,
    )


def canonical_feature_table_bytes(
    table: Any,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_keys: Iterable[Sequence[Any]],
    expected_bindings: Mapping[str, str],
) -> bytes:
    """Validate and serialize a logical table with deterministic row/key order."""
    expected_key_list = list(expected_keys)
    _validate_feature_table(
        table,
        charter,
        contract,
        expected_keys=expected_key_list,
        expected_bindings=expected_bindings,
        require_canonical_order=False,
    )
    canonical = deepcopy(table)
    row_key = contract["logical_table"]["row_key"]
    canonical["rows"] = sorted(
        canonical["rows"], key=lambda row: tuple(row[field] for field in row_key)
    )
    return _canonical_object_bytes(canonical)


def feature_table_sha256(
    table: Any,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_keys: Iterable[Sequence[Any]],
    expected_bindings: Mapping[str, str],
) -> str:
    """SHA-256 of the canonical logical feature table."""
    return sha256(
        canonical_feature_table_bytes(
            table,
            charter,
            contract,
            expected_keys=expected_keys,
            expected_bindings=expected_bindings,
        )
    ).hexdigest()


def _feature_bundle_contract_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        declared = contract["storage_bundle"]
    except (KeyError, TypeError) as exc:
        raise FeatureTableValidationError(
            "feature contract is missing the storage bundle contract"
        ) from exc
    if declared != FEATURE_TABLE_BUNDLE_CONTRACT:
        raise FeatureTableValidationError(
            "feature storage bundle contract drifted from the implementation"
        )
    return deepcopy(declared)


def feature_table_bundle_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Digest the exact technical feature-bundle layout without scientific outcomes."""
    return sha256(_canonical_object_bytes(_feature_bundle_contract_payload(contract))).hexdigest()


def _feature_bundle_parts(
    table: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    row_key = contract["logical_table"]["row_key"]
    metadata = {field: deepcopy(value) for field, value in table.items() if field != "rows"}
    trial_keys = {
        "schema_id": FEATURE_TABLE_BUNDLE_CONTRACT["trial_keys"]["schema_id"],
        "schema_version": FEATURE_TABLE_BUNDLE_CONTRACT["trial_keys"]["schema_version"],
        "row_key": list(row_key),
        "rows": [
            {field: deepcopy(row[field]) for field in row_key} for row in table["rows"]
        ],
    }
    values = np.ascontiguousarray(
        np.asarray([row["values"] for row in table["rows"]], dtype="<f8")
    )
    return metadata, trial_keys, values


def _write_npy_fsynced(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def write_feature_table_bundle_atomic(
    bundle_path: Path,
    table: Mapping[str, Any],
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_keys: Iterable[Sequence[Any]],
    expected_bindings: Mapping[str, str],
) -> str:
    """Publish one immutable typed feature bundle from an already validated logical table."""
    _feature_bundle_contract_payload(contract)
    expected_key_list = list(expected_keys)
    validate_feature_table(
        table,
        charter,
        contract,
        expected_keys=expected_key_list,
        expected_bindings=expected_bindings,
    )
    destination = Path(bundle_path)
    parent = destination.parent
    if not parent.is_dir():
        raise FeatureTableValidationError(
            f"feature bundle parent is not a directory: {parent}"
        )
    if os.path.lexists(destination):
        raise FeatureTableValidationError(
            f"feature table bundle already exists: {destination}"
        )

    metadata, trial_keys, values = _feature_bundle_parts(table, contract)
    metadata_bytes = _canonical_object_bytes(metadata)
    trial_keys_bytes = _canonical_object_bytes(trial_keys)
    logical_digest = feature_table_sha256(
        table,
        charter,
        contract,
        expected_keys=expected_key_list,
        expected_bindings=expected_bindings,
    )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent)
    )
    try:
        _write_fsynced(temporary / "metadata.json", metadata_bytes)
        _write_fsynced(temporary / "trial-keys.json", trial_keys_bytes)
        _write_npy_fsynced(temporary / "values.npy", values)
        values_bytes = (temporary / "values.npy").read_bytes()
        manifest = {
            "schema_id": FEATURE_TABLE_BUNDLE_CONTRACT["schema_id"],
            "schema_version": FEATURE_TABLE_BUNDLE_CONTRACT["schema_version"],
            "bundle_contract_sha256": feature_table_bundle_contract_sha256(contract),
            "logical_table_sha256": logical_digest,
            "metadata_sha256": sha256(metadata_bytes).hexdigest(),
            "trial_keys_sha256": sha256(trial_keys_bytes).hexdigest(),
            "values_npy_sha256": sha256(values_bytes).hexdigest(),
            "row_count": int(values.shape[0]),
            "feature_count": int(values.shape[1]),
            "values_dtype": "little_endian_float64",
            "values_shape": [int(size) for size in values.shape],
            "values_c_contiguous": bool(values.flags.c_contiguous),
            "allow_pickle": False,
        }
        manifest_bytes = _canonical_object_bytes(manifest)
        bundle_digest = sha256(manifest_bytes).hexdigest()
        _write_fsynced(temporary / "bundle.json", manifest_bytes)
        _write_fsynced(
            temporary / "bundle.sha256", f"{bundle_digest}\n".encode("ascii")
        )
        _fsync_directory(temporary)
        if os.path.lexists(destination):
            raise FeatureTableValidationError(
                f"feature table bundle already exists: {destination}"
            )
        _rename_directory_noreplace(temporary, destination)
        _fsync_directory(parent)
    except Exception as exc:
        raise FeatureTableValidationError(
            "feature table bundle publication failed; staged evidence retained at "
            f"{temporary}: {exc}"
        ) from exc
    return bundle_digest


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FeatureTableValidationError(f"{label} must be a regular file")
    try:
        value = load_json_unique(path)
    except ManifestValidationError as exc:
        raise FeatureTableValidationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise FeatureTableValidationError(f"{label} must contain one JSON object")
    if path.read_bytes() != _canonical_object_bytes(value):
        raise FeatureTableValidationError(f"{label} is not canonical JSON bytes")
    return value


def load_feature_table_bundle(
    bundle_path: Path,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_keys: Iterable[Sequence[Any]],
    expected_bindings: Mapping[str, str],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Read, authenticate, reconstruct, and revalidate one immutable feature bundle."""
    bundle_contract = _feature_bundle_contract_payload(contract)
    if not isinstance(expected_bundle_sha256, str) or not SHA256_RE.fullmatch(
        expected_bundle_sha256
    ):
        raise FeatureTableValidationError(
            "expected external bundle digest must be a lowercase SHA-256"
        )
    bundle = Path(bundle_path)
    if not bundle.is_dir() or bundle.is_symlink():
        raise FeatureTableValidationError(
            f"feature table bundle is not a regular directory: {bundle}"
        )
    actual_files = {path.name for path in bundle.iterdir()}
    if actual_files != FEATURE_TABLE_BUNDLE_FILES:
        raise FeatureTableValidationError(
            "feature table bundle does not contain the exact files required by the contract"
        )
    for name in FEATURE_TABLE_BUNDLE_FILES:
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise FeatureTableValidationError(
                f"feature table bundle entry must be a regular file: {name}"
            )

    sidecar = (bundle / "bundle.sha256").read_bytes()
    if len(sidecar) != 65 or not sidecar.endswith(b"\n"):
        raise FeatureTableValidationError(
            "feature bundle sidecar must be one lowercase SHA-256 plus newline"
        )
    try:
        sidecar_digest = sidecar[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise FeatureTableValidationError("feature bundle sidecar is not ASCII") from exc
    if not SHA256_RE.fullmatch(sidecar_digest):
        raise FeatureTableValidationError(
            "feature bundle sidecar is not a lowercase SHA-256"
        )

    manifest_path = bundle / "bundle.json"
    manifest = _read_canonical_json(manifest_path, "feature bundle manifest")
    manifest_bytes = manifest_path.read_bytes()
    bundle_digest = sha256(manifest_bytes).hexdigest()
    if sidecar_digest != bundle_digest:
        raise FeatureTableValidationError(
            "feature bundle sidecar does not match the canonical bundle manifest"
        )
    if expected_bundle_sha256 != bundle_digest:
        raise FeatureTableValidationError(
            "external bundle digest does not match the canonical bundle manifest"
        )
    _require_exact_fields(
        manifest,
        bundle_contract["bundle_manifest_exact_fields"],
        "feature bundle manifest",
    )
    if (
        manifest["schema_id"] != bundle_contract["schema_id"]
        or manifest["schema_version"] != bundle_contract["schema_version"]
    ):
        raise FeatureTableValidationError("feature bundle schema identity mismatch")
    if manifest["bundle_contract_sha256"] != feature_table_bundle_contract_sha256(
        contract
    ):
        raise FeatureTableValidationError("feature bundle contract digest mismatch")
    if manifest["values_dtype"] != "little_endian_float64":
        raise FeatureTableValidationError("feature bundle values dtype declaration mismatch")
    if manifest["values_c_contiguous"] is not True or manifest["allow_pickle"] is not False:
        raise FeatureTableValidationError(
            "feature bundle must declare C-order values with pickle disabled"
        )
    if not (
        _is_integer(manifest["row_count"])
        and manifest["row_count"] > 0
        and _is_integer(manifest["feature_count"])
        and manifest["feature_count"] == 128
        and manifest["values_shape"] == [
            manifest["row_count"],
            manifest["feature_count"],
        ]
    ):
        raise FeatureTableValidationError("feature bundle shape metadata is invalid")
    for field in (
        "logical_table_sha256",
        "metadata_sha256",
        "trial_keys_sha256",
        "values_npy_sha256",
    ):
        if not isinstance(manifest[field], str) or not SHA256_RE.fullmatch(manifest[field]):
            raise FeatureTableValidationError(f"feature bundle {field} is not a SHA-256")

    metadata_path = bundle / "metadata.json"
    keys_path = bundle / "trial-keys.json"
    values_path = bundle / "values.npy"
    metadata = _read_canonical_json(metadata_path, "feature bundle metadata")
    trial_keys = _read_canonical_json(keys_path, "feature bundle trial keys")
    if sha256(metadata_path.read_bytes()).hexdigest() != manifest["metadata_sha256"]:
        raise FeatureTableValidationError("feature bundle metadata digest mismatch")
    if sha256(keys_path.read_bytes()).hexdigest() != manifest["trial_keys_sha256"]:
        raise FeatureTableValidationError("feature bundle trial-keys digest mismatch")
    if sha256(values_path.read_bytes()).hexdigest() != manifest["values_npy_sha256"]:
        raise FeatureTableValidationError("feature bundle values.npy digest mismatch")

    expected_metadata_fields = set(contract["logical_table"]["exact_fields"]) - {"rows"}
    _require_exact_fields(metadata, expected_metadata_fields, "feature bundle metadata")
    _require_exact_fields(
        trial_keys,
        ("schema_id", "schema_version", "row_key", "rows"),
        "feature bundle trial keys",
    )
    if (
        trial_keys["schema_id"] != bundle_contract["trial_keys"]["schema_id"]
        or trial_keys["schema_version"]
        != bundle_contract["trial_keys"]["schema_version"]
        or trial_keys["row_key"] != contract["logical_table"]["row_key"]
    ):
        raise FeatureTableValidationError("feature bundle trial-key schema mismatch")
    if not isinstance(trial_keys["rows"], list):
        raise FeatureTableValidationError("feature bundle trial-key rows must be an array")
    for index, row in enumerate(trial_keys["rows"]):
        if not isinstance(row, dict):
            raise FeatureTableValidationError(
                f"feature bundle trial-key row {index} must be an object"
            )
        _require_exact_fields(
            row,
            contract["logical_table"]["row_key"],
            f"feature bundle trial-key row {index}",
        )

    try:
        with values_path.open("rb") as handle:
            values = np.load(handle, allow_pickle=False)
            trailing = handle.read(1)
    except (OSError, ValueError) as exc:
        raise FeatureTableValidationError(
            f"feature bundle values.npy is not a safe NumPy array: {exc}"
        ) from exc
    if trailing:
        raise FeatureTableValidationError("feature bundle values.npy has trailing bytes")
    if not isinstance(values, np.ndarray) or values.dtype.str != "<f8":
        raise FeatureTableValidationError(
            "feature bundle values.npy must be little-endian float64"
        )
    if not values.flags.c_contiguous:
        raise FeatureTableValidationError("feature bundle values.npy must be C-contiguous")
    if list(values.shape) != manifest["values_shape"]:
        raise FeatureTableValidationError("feature bundle values.npy shape mismatch")
    if not np.isfinite(values).all():
        raise FeatureTableValidationError("feature bundle values.npy contains non-finite values")
    if len(trial_keys["rows"]) != manifest["row_count"]:
        raise FeatureTableValidationError("feature bundle trial-key row count mismatch")

    rows = []
    for index, key in enumerate(trial_keys["rows"]):
        rows.append({**deepcopy(key), "values": [float(value) for value in values[index]]})
    table = {**metadata, "rows": rows}
    expected_key_list = list(expected_keys)
    validate_feature_table(
        table,
        charter,
        contract,
        expected_keys=expected_key_list,
        expected_bindings=expected_bindings,
    )
    logical_digest = feature_table_sha256(
        table,
        charter,
        contract,
        expected_keys=expected_key_list,
        expected_bindings=expected_bindings,
    )
    if logical_digest != manifest["logical_table_sha256"]:
        raise FeatureTableValidationError(
            "feature bundle reconstructed logical-table digest mismatch"
        )
    return table
