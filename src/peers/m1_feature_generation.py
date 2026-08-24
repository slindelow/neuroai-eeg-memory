"""Outcome-free real-source bridge for frozen M1 spectral features.

The cache contains no recall label or model output. It preserves enough raw log-power evidence to
fit D26-A on permitted training sessions and to derive the frozen D29/ROI features without
recomputing Morlet power for every fold.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from peers.m1_features import (
    FeatureTableValidationError,
    StrictInductiveNormalizationStatistics,
    _canonical_object_bytes,
    _readonly_copy,
    _write_npy_fsynced,
    normalization_statistics_evidence,
    resolved_time_bin_masks,
)
from peers.m1_manifests import (
    _fsync_directory,
    _rename_directory_noreplace,
    _write_fsynced,
    canonical_trial_key,
    load_json_unique,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CONDITIONS = ("ocular-removed", "ocular-kept")

RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT = {
    "schema_id": "neuroai-m1-raw-spectral-session-bundle-v1",
    "schema_version": "1.0",
    "purpose": (
        "outcome_free_single_session_bridge_from_locked_epoch_rows_to_"
        "strict_inductive_fold_features"
    ),
    "exact_files": [
        "metadata.json",
        "trial-keys.json",
        "bin-means.npy",
        "statistics-mean.npy",
        "statistics-m2.npy",
        "time-axis.npy",
        "bundle.json",
        "bundle.sha256",
    ],
    "trial_keys": {
        "schema_id": "neuroai-m1-raw-spectral-trial-keys-v1",
        "schema_version": "1.0",
        "row_key": ["subject_id", "session", "trial", "item_num"],
        "canonical_order": "manifest_trial_key_order",
        "outcome_fields_allowed": False,
    },
    "metadata_exact_fields": [
        "schema_id",
        "schema_version",
        "created_at_utc",
        "subject_id",
        "session",
        "condition",
        "derivative_id",
        "derivative_sha256",
        "trial_manifest_sha256",
        "source_inventory_sha256",
        "charter_sha256",
        "roi_map_sha256",
        "feature_schema_sha256",
        "uv_lock_sha256",
        "package_versions_sha256",
        "code_commit",
        "dirty_diff_sha256",
        "generator_command",
        "python_version",
        "mne_version",
        "numpy_version",
        "frequencies_hz",
        "n_cycles",
        "decim",
        "parallel_worker_count",
        "channel_order",
        "time_bin_ids",
        "trial_count",
        "full_time_sample_count",
        "statistics_observation_count",
        "time_bin_sample_counts",
        "outcome_access_attestation",
    ],
    "external_binding_fields": [
        "derivative_id",
        "derivative_sha256",
        "trial_manifest_sha256",
        "source_inventory_sha256",
        "charter_sha256",
        "roi_map_sha256",
        "feature_schema_sha256",
        "uv_lock_sha256",
        "package_versions_sha256",
        "code_commit",
        "dirty_diff_sha256",
    ],
    "arrays": {
        "dtype": "little_endian_float64",
        "allow_pickle": False,
        "trailing_bytes_allowed": False,
        "bin_means_shape": "n_trials_by_2_time_bins_by_40_channels_by_8_frequencies",
        "statistics_mean_shape": "40_channels_by_8_frequencies",
        "statistics_m2_shape": "40_channels_by_8_frequencies",
        "time_axis_shape": "n_decimated_full_epoch_time_samples",
        "bin_means_semantics": (
            "per_trial_log10_morlet_power_arithmetic_mean_within_each_D29_time_bin_"
            "before_D26_normalization"
        ),
        "statistics_semantics": (
            "per_session_float64_mean_and_sum_squared_deviation_over_trial_and_complete_"
            "decimated_epoch_time_axes"
        ),
    },
    "combination": {
        "session_order": "ascending_session_id",
        "algorithm": "pairwise_Chan_mean_M2_merge",
        "population_variance": "combined_M2_divided_by_combined_trial_time_count",
        "zero_standard_deviation_policy": "replace_with_1_then_center",
        "normalization_fit_scope": (
            "only_sessions_named_by_the_inner_or_outer_training_partition"
        ),
        "held_session_statistics_allowed": False,
    },
    "feature_reduction_identity": (
        "mean_over_roi_channels_of_bin_mean_minus_training_mean_divided_by_training_"
        "scale_equals_full_time_then_roi_mean_of_normalized_log_power"
    ),
    "bundle_manifest_exact_fields": [
        "schema_id",
        "schema_version",
        "bundle_contract_sha256",
        "metadata_sha256",
        "trial_keys_sha256",
        "bin_means_npy_sha256",
        "statistics_mean_npy_sha256",
        "statistics_m2_npy_sha256",
        "time_axis_npy_sha256",
        "trial_count",
        "statistics_observation_count",
        "bin_means_shape",
        "statistics_shape",
        "time_axis_shape",
        "values_dtype",
        "allow_pickle",
    ],
    "atomic_publication": "temporary_directory_fsync_then_atomic_noreplace_rename",
    "external_bundle_sha256_required_on_readback": True,
    "source_derivative_sha256_revalidated_before_extraction": True,
    "exact_manifest_epoch_rows_required": True,
    "label_probability_metric_or_coefficient_fields_allowed": False,
    "real_publication_requires_locked_N2_run": True,
    "failure_staging_policy": "retain_for_evidence_no_automatic_deletion",
}

RAW_FILES = frozenset(RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["exact_files"])
RAW_KEY_FIELDS = tuple(
    RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["trial_keys"]["row_key"]
)


@dataclass(frozen=True, slots=True)
class SessionSpectralSummary:
    """Outcome-free compact evidence for one source epoch session."""

    subject_id: str
    session: int
    condition: str
    derivative_id: str
    derivative_sha256: str
    trial_keys: tuple[tuple[str, int, int, int], ...]
    bin_means: np.ndarray
    statistics_mean: np.ndarray
    statistics_m2: np.ndarray
    time_axis: np.ndarray
    statistics_observation_count: int
    time_bin_sample_counts: tuple[int, int]


def _require_exact_fields(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise FeatureTableValidationError(
            f"{label} fields mismatch; missing={sorted(required - actual)}, "
            f"extra={sorted(actual - required)}"
        )


def _raw_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    try:
        declared = contract["raw_spectral_session_bundle"]
    except (KeyError, TypeError) as exc:
        raise FeatureTableValidationError(
            "feature contract lacks the raw spectral session bundle"
        ) from exc
    if declared != RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT:
        raise FeatureTableValidationError(
            "raw spectral session bundle contract drifted from the implementation"
        )
    return deepcopy(declared)


def raw_spectral_bundle_contract_sha256(contract: Mapping[str, Any]) -> str:
    return sha256(_canonical_object_bytes(_raw_contract(contract))).hexdigest()


def _readonly_float64(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float64):
        raise FeatureTableValidationError(f"{label} must have exact float64 dtype")
    if array.size == 0 or not np.isfinite(array).all():
        raise FeatureTableValidationError(f"{label} must be nonempty and finite")
    return _readonly_copy(array, dtype=np.float64)


def _validated_trial_keys(
    trial_keys: Iterable[Sequence[Any]], *, subject_id: str, session: int
) -> tuple[tuple[str, int, int, int], ...]:
    keys = tuple(
        canonical_trial_key(dict(zip(RAW_KEY_FIELDS, key, strict=True)))
        for key in trial_keys
    )
    if not keys:
        raise FeatureTableValidationError("raw spectral summary requires trial keys")
    if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
        raise FeatureTableValidationError(
            "raw spectral trial keys must be unique and canonically ordered"
        )
    if any(key[0] != subject_id or key[1] != session for key in keys):
        raise FeatureTableValidationError(
            "raw spectral trial keys differ from the summary subject/session"
        )
    return keys


def validate_session_spectral_summary(summary: Any, charter: Mapping[str, Any]) -> None:
    if not isinstance(summary, SessionSpectralSummary):
        raise FeatureTableValidationError(
            "raw spectral evidence must use SessionSpectralSummary"
        )
    if not summary.subject_id or type(summary.session) is not int or summary.session < 0:
        raise FeatureTableValidationError("raw spectral subject/session identity is invalid")
    if summary.condition not in CONDITIONS:
        raise FeatureTableValidationError("raw spectral condition is invalid")
    if not summary.derivative_id or not SHA256_RE.fullmatch(summary.derivative_sha256):
        raise FeatureTableValidationError("raw spectral derivative identity is invalid")
    keys = _validated_trial_keys(
        summary.trial_keys, subject_id=summary.subject_id, session=summary.session
    )
    arrays = {
        "bin_means": summary.bin_means,
        "statistics_mean": summary.statistics_mean,
        "statistics_m2": summary.statistics_m2,
        "time_axis": summary.time_axis,
    }
    for label, array in arrays.items():
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype(np.float64):
            raise FeatureTableValidationError(f"raw spectral {label} must be float64")
        if array.flags.writeable or not np.isfinite(array).all():
            raise FeatureTableValidationError(
                f"raw spectral {label} must be finite and read-only"
            )
    if summary.bin_means.shape != (len(keys), 2, 40, 8):
        raise FeatureTableValidationError("raw spectral bin_means shape mismatch")
    if summary.statistics_mean.shape != (40, 8) or summary.statistics_m2.shape != (40, 8):
        raise FeatureTableValidationError("raw spectral statistic shape mismatch")
    if np.any(summary.statistics_m2 < 0.0):
        raise FeatureTableValidationError("raw spectral M2 must be nonnegative")
    if summary.time_axis.ndim != 1 or summary.time_axis.size < 2:
        raise FeatureTableValidationError("raw spectral time axis shape is invalid")
    if not np.all(np.diff(summary.time_axis) > 0.0):
        raise FeatureTableValidationError("raw spectral time axis must increase strictly")
    masks = resolved_time_bin_masks(summary.time_axis, charter)
    expected_bin_counts = tuple(int(np.count_nonzero(mask)) for mask in masks)
    if summary.time_bin_sample_counts != expected_bin_counts:
        raise FeatureTableValidationError("raw spectral time-bin sample counts mismatch")
    expected_count = len(keys) * int(summary.time_axis.size)
    if summary.statistics_observation_count != expected_count:
        raise FeatureTableValidationError(
            "raw spectral statistics observation count mismatch"
        )


def summarize_log_power_session(
    log_power: np.ndarray,
    time_axis: np.ndarray,
    trial_keys: Iterable[Sequence[Any]],
    *,
    subject_id: str,
    session: int,
    condition: str,
    derivative_id: str,
    derivative_sha256: str,
    charter: Mapping[str, Any],
) -> SessionSpectralSummary:
    """Reduce one full log-power array without fitting or applying normalization."""
    values = np.asarray(log_power)
    if values.dtype != np.dtype(np.float64) or values.ndim != 4:
        raise FeatureTableValidationError(
            "session log power must be float64 trial x 40 x 8 x time"
        )
    if values.shape[1:3] != (40, 8) or values.shape[0] == 0 or values.shape[3] < 2:
        raise FeatureTableValidationError("session log power shape is invalid")
    if not np.isfinite(values).all():
        raise FeatureTableValidationError("session log power contains non-finite values")
    time = np.asarray(time_axis)
    if time.dtype != np.dtype(np.float64) or time.shape != (values.shape[3],):
        raise FeatureTableValidationError("session log-power time axis mismatch")
    keys = _validated_trial_keys(trial_keys, subject_id=subject_id, session=session)
    if len(keys) != values.shape[0]:
        raise FeatureTableValidationError("session log-power trial-key count mismatch")
    masks = resolved_time_bin_masks(time, charter)
    bin_means = np.stack(
        [values[..., mask].mean(axis=3, dtype=np.float64) for mask in masks], axis=1
    )
    mean = values.mean(axis=(0, 3), dtype=np.float64)
    centered = values - mean[None, :, :, None]
    m2 = np.sum(centered * centered, axis=(0, 3), dtype=np.float64)
    summary = SessionSpectralSummary(
        subject_id=subject_id,
        session=session,
        condition=condition,
        derivative_id=derivative_id,
        derivative_sha256=derivative_sha256,
        trial_keys=keys,
        bin_means=_readonly_float64(bin_means, "raw spectral bin means"),
        statistics_mean=_readonly_float64(mean, "raw spectral mean"),
        statistics_m2=_readonly_float64(m2, "raw spectral M2"),
        time_axis=_readonly_float64(time, "raw spectral time axis"),
        statistics_observation_count=int(values.shape[0] * values.shape[3]),
        time_bin_sample_counts=tuple(int(np.count_nonzero(mask)) for mask in masks),
    )
    validate_session_spectral_summary(summary, charter)
    return summary


def combine_training_session_statistics(
    summaries: Sequence[SessionSpectralSummary],
    *,
    fit_sessions: Sequence[int],
    charter: Mapping[str, Any],
) -> StrictInductiveNormalizationStatistics:
    """Combine only explicitly named training sessions in canonical order."""
    if not summaries:
        raise FeatureTableValidationError("normalization requires session summaries")
    for summary in summaries:
        validate_session_spectral_summary(summary, charter)
    requested = list(fit_sessions)
    if not requested or requested != sorted(requested) or len(set(requested)) != len(requested):
        raise FeatureTableValidationError(
            "fit_sessions must be a nonempty unique ascending sequence"
        )
    by_session = {summary.session: summary for summary in summaries}
    if len(by_session) != len(summaries) or set(requested) - set(by_session):
        raise FeatureTableValidationError("fit_sessions do not match supplied session summaries")
    selected = [by_session[session] for session in requested]
    identities = {(item.subject_id, item.condition) for item in selected}
    time_axes = {item.time_axis.tobytes() for item in selected}
    if len(identities) != 1 or len(time_axes) != 1:
        raise FeatureTableValidationError(
            "training session summaries must share subject, condition, and time axis"
        )

    mean = np.array(selected[0].statistics_mean, dtype=np.float64, copy=True)
    m2 = np.array(selected[0].statistics_m2, dtype=np.float64, copy=True)
    count = selected[0].statistics_observation_count
    for summary in selected[1:]:
        next_count = summary.statistics_observation_count
        total = count + next_count
        delta = summary.statistics_mean - mean
        mean = mean + delta * (np.float64(next_count) / np.float64(total))
        m2 = (
            m2
            + summary.statistics_m2
            + delta * delta * (np.float64(count * next_count) / np.float64(total))
        )
        count = total
    variance = np.maximum(m2 / np.float64(count), 0.0)
    raw_std = np.sqrt(variance, dtype=np.float64)
    zero_mask = raw_std == 0.0
    scale = np.where(zero_mask, 1.0, raw_std)
    statistics = StrictInductiveNormalizationStatistics(
        mean=_readonly_copy(mean, dtype=np.float64),
        raw_std=_readonly_copy(raw_std, dtype=np.float64),
        scale=_readonly_copy(scale, dtype=np.float64),
        zero_standard_deviation_mask=_readonly_copy(zero_mask, dtype=np.bool_),
        n_fit_trials=sum(len(item.trial_keys) for item in selected),
        n_fit_timepoints=int(selected[0].time_axis.size),
        ddof=0,
    )
    normalization_statistics_evidence(statistics)
    return statistics


def compact_feature_values(
    summary: SessionSpectralSummary,
    statistics: StrictInductiveNormalizationStatistics,
    *,
    charter: Mapping[str, Any],
) -> np.ndarray:
    """Apply D26-A to raw bin means, then perform the frozen five-channel ROI mean."""
    validate_session_spectral_summary(summary, charter)
    normalization_statistics_evidence(statistics)
    normalized = (
        summary.bin_means - statistics.mean[None, None, :, :]
    ) / statistics.scale[None, None, :, :]
    values = normalized.reshape(len(summary.trial_keys), 2, 8, 5, 8).mean(
        axis=3, dtype=np.float64
    )
    flattened = np.ascontiguousarray(values.reshape(len(summary.trial_keys), 128))
    if flattened.dtype != np.dtype(np.float64) or not np.isfinite(flattened).all():
        raise FeatureTableValidationError("compact feature reduction produced invalid values")
    return flattened


def compact_features_for_sessions(
    summaries: Sequence[SessionSpectralSummary],
    statistics: StrictInductiveNormalizationStatistics,
    *,
    apply_sessions: Sequence[int],
    charter: Mapping[str, Any],
) -> tuple[tuple[tuple[str, int, int, int], ...], np.ndarray]:
    requested = list(apply_sessions)
    if not requested or requested != sorted(requested) or len(set(requested)) != len(requested):
        raise FeatureTableValidationError(
            "apply_sessions must be a nonempty unique ascending sequence"
        )
    by_session = {summary.session: summary for summary in summaries}
    if set(requested) - set(by_session):
        raise FeatureTableValidationError("apply_sessions lack raw session summaries")
    keys: list[tuple[str, int, int, int]] = []
    matrices: list[np.ndarray] = []
    for session in requested:
        summary = by_session[session]
        keys.extend(summary.trial_keys)
        matrices.append(compact_feature_values(summary, statistics, charter=charter))
    return tuple(keys), np.ascontiguousarray(np.concatenate(matrices, axis=0))


def build_raw_cache_metadata(
    summary: SessionSpectralSummary,
    *,
    created_at_utc: str,
    bindings: Mapping[str, Any],
    generator_command: str,
    python_version: str,
    mne_version: str,
    numpy_version: str,
    frequencies_hz: Sequence[float],
    n_cycles: Sequence[float],
    decim: int,
    parallel_worker_count: int,
    channel_order: Sequence[str],
    time_bin_ids: Sequence[str],
) -> dict[str, Any]:
    required_bindings = (
        "trial_manifest_sha256",
        "source_inventory_sha256",
        "charter_sha256",
        "roi_map_sha256",
        "feature_schema_sha256",
        "uv_lock_sha256",
        "package_versions_sha256",
        "code_commit",
        "dirty_diff_sha256",
    )
    _require_exact_fields(bindings, required_bindings, "raw cache bindings")
    return {
        "schema_id": RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["schema_id"],
        "schema_version": RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["schema_version"],
        "created_at_utc": created_at_utc,
        "subject_id": summary.subject_id,
        "session": summary.session,
        "condition": summary.condition,
        "derivative_id": summary.derivative_id,
        "derivative_sha256": summary.derivative_sha256,
        **dict(bindings),
        "generator_command": generator_command,
        "python_version": python_version,
        "mne_version": mne_version,
        "numpy_version": numpy_version,
        "frequencies_hz": [float(value) for value in frequencies_hz],
        "n_cycles": [float(value) for value in n_cycles],
        "decim": decim,
        "parallel_worker_count": parallel_worker_count,
        "channel_order": list(channel_order),
        "time_bin_ids": list(time_bin_ids),
        "trial_count": len(summary.trial_keys),
        "full_time_sample_count": int(summary.time_axis.size),
        "statistics_observation_count": summary.statistics_observation_count,
        "time_bin_sample_counts": list(summary.time_bin_sample_counts),
        "outcome_access_attestation": False,
    }


def _validate_metadata(
    metadata: Mapping[str, Any],
    summary: SessionSpectralSummary,
    charter: Mapping[str, Any],
) -> None:
    validate_session_spectral_summary(summary, charter)
    _require_exact_fields(
        metadata,
        RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["metadata_exact_fields"],
        "raw spectral metadata",
    )
    if (
        metadata["schema_id"] != RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["schema_id"]
        or metadata["schema_version"]
        != RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["schema_version"]
    ):
        raise FeatureTableValidationError("raw spectral metadata schema mismatch")
    for field in (
        "derivative_sha256",
        "trial_manifest_sha256",
        "source_inventory_sha256",
        "charter_sha256",
        "roi_map_sha256",
        "feature_schema_sha256",
        "uv_lock_sha256",
        "package_versions_sha256",
    ):
        if not isinstance(metadata[field], str) or not SHA256_RE.fullmatch(metadata[field]):
            raise FeatureTableValidationError(f"raw spectral metadata {field} is invalid")
    if not isinstance(metadata["code_commit"], str) or not GIT_COMMIT_RE.fullmatch(
        metadata["code_commit"]
    ):
        raise FeatureTableValidationError("raw spectral metadata code_commit is invalid")
    dirty = metadata["dirty_diff_sha256"]
    if dirty is not None and (not isinstance(dirty, str) or not SHA256_RE.fullmatch(dirty)):
        raise FeatureTableValidationError("raw spectral dirty-diff identity is invalid")
    if not (
        isinstance(metadata["created_at_utc"], str)
        and metadata["created_at_utc"].endswith("Z")
        and metadata["subject_id"] == summary.subject_id
        and metadata["session"] == summary.session
        and metadata["condition"] == summary.condition
        and metadata["derivative_id"] == summary.derivative_id
        and metadata["derivative_sha256"] == summary.derivative_sha256
        and metadata["trial_count"] == len(summary.trial_keys)
        and metadata["full_time_sample_count"] == summary.time_axis.size
        and metadata["statistics_observation_count"]
        == summary.statistics_observation_count
        and metadata["time_bin_sample_counts"] == list(summary.time_bin_sample_counts)
        and metadata["outcome_access_attestation"] is False
    ):
        raise FeatureTableValidationError("raw spectral metadata/summary mismatch")
    frequencies = metadata["frequencies_hz"]
    cycles = metadata["n_cycles"]
    expected_frequencies = charter["features"]["frequencies_hz"]
    if frequencies != expected_frequencies or cycles != [value / 2.0 for value in frequencies]:
        raise FeatureTableValidationError("raw spectral Morlet frequency/cycle contract drift")
    expected_channels = [
        channel
        for roi in charter["features"]["roi_order"]
        for channel in charter["features"]["roi_channels"][roi]
    ]
    if (
        metadata["decim"] != charter["features"]["decim"]
        or metadata["channel_order"] != expected_channels
        or metadata["time_bin_ids"] != charter["features"]["time_bin_ids"]
        or type(metadata["parallel_worker_count"]) is not int
        or metadata["parallel_worker_count"] <= 0
    ):
        raise FeatureTableValidationError("raw spectral numerical execution metadata drift")


def _trial_keys_document(summary: SessionSpectralSummary) -> dict[str, Any]:
    return {
        "schema_id": RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["trial_keys"]["schema_id"],
        "schema_version": RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["trial_keys"][
            "schema_version"
        ],
        "row_key": list(RAW_KEY_FIELDS),
        "rows": [dict(zip(RAW_KEY_FIELDS, key, strict=True)) for key in summary.trial_keys],
    }


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_raw_spectral_session_bundle_atomic(
    bundle_path: Path,
    summary: SessionSpectralSummary,
    metadata: Mapping[str, Any],
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> str:
    """Publish one authenticated immutable outcome-free raw session cache."""
    _raw_contract(contract)
    _validate_metadata(metadata, summary, charter)
    destination = Path(bundle_path)
    if not destination.parent.is_dir() or os.path.lexists(destination):
        raise FeatureTableValidationError(
            "raw spectral bundle parent must exist and destination must not exist"
        )
    metadata_bytes = _canonical_object_bytes(dict(metadata))
    keys_document = _trial_keys_document(summary)
    keys_bytes = _canonical_object_bytes(keys_document)
    arrays = {
        "bin-means.npy": np.ascontiguousarray(summary.bin_means, dtype="<f8"),
        "statistics-mean.npy": np.ascontiguousarray(summary.statistics_mean, dtype="<f8"),
        "statistics-m2.npy": np.ascontiguousarray(summary.statistics_m2, dtype="<f8"),
        "time-axis.npy": np.ascontiguousarray(summary.time_axis, dtype="<f8"),
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        _write_fsynced(temporary / "metadata.json", metadata_bytes)
        _write_fsynced(temporary / "trial-keys.json", keys_bytes)
        for name, values in arrays.items():
            _write_npy_fsynced(temporary / name, values)
        manifest = {
            "schema_id": RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["schema_id"],
            "schema_version": RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT["schema_version"],
            "bundle_contract_sha256": raw_spectral_bundle_contract_sha256(contract),
            "metadata_sha256": sha256(metadata_bytes).hexdigest(),
            "trial_keys_sha256": sha256(keys_bytes).hexdigest(),
            "bin_means_npy_sha256": _sha256_file(temporary / "bin-means.npy"),
            "statistics_mean_npy_sha256": _sha256_file(
                temporary / "statistics-mean.npy"
            ),
            "statistics_m2_npy_sha256": _sha256_file(temporary / "statistics-m2.npy"),
            "time_axis_npy_sha256": _sha256_file(temporary / "time-axis.npy"),
            "trial_count": len(summary.trial_keys),
            "statistics_observation_count": summary.statistics_observation_count,
            "bin_means_shape": list(summary.bin_means.shape),
            "statistics_shape": [40, 8],
            "time_axis_shape": list(summary.time_axis.shape),
            "values_dtype": "little_endian_float64",
            "allow_pickle": False,
        }
        manifest_bytes = _canonical_object_bytes(manifest)
        bundle_digest = sha256(manifest_bytes).hexdigest()
        _write_fsynced(temporary / "bundle.json", manifest_bytes)
        _write_fsynced(temporary / "bundle.sha256", f"{bundle_digest}\n".encode("ascii"))
        _fsync_directory(temporary)
        if os.path.lexists(destination):
            raise FeatureTableValidationError(f"raw spectral bundle already exists: {destination}")
        _rename_directory_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception as exc:
        raise FeatureTableValidationError(
            f"raw spectral bundle publication failed; staged evidence retained at {temporary}: {exc}"
        ) from exc
    return bundle_digest


def _read_canonical_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FeatureTableValidationError(f"{label} must be a regular file")
    value = load_json_unique(path)
    if not isinstance(value, dict) or path.read_bytes() != _canonical_object_bytes(value):
        raise FeatureTableValidationError(f"{label} must be canonical JSON")
    return value


def _load_safe_float64_npy(path: Path, expected_sha256: str, label: str) -> np.ndarray:
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_sha256:
        raise FeatureTableValidationError(f"{label} file identity mismatch")
    try:
        with path.open("rb") as handle:
            value = np.load(handle, allow_pickle=False)
            trailing = handle.read(1)
    except (OSError, ValueError) as exc:
        raise FeatureTableValidationError(f"{label} is not a safe NPY array: {exc}") from exc
    if trailing or not isinstance(value, np.ndarray) or value.dtype.str != "<f8":
        raise FeatureTableValidationError(f"{label} must be exact little-endian float64 NPY")
    if not value.flags.c_contiguous or not np.isfinite(value).all():
        raise FeatureTableValidationError(f"{label} must be C-contiguous and finite")
    return value


def load_raw_spectral_session_bundle(
    bundle_path: Path,
    charter: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_bundle_sha256: str,
    expected_metadata_bindings: Mapping[str, Any],
) -> tuple[SessionSpectralSummary, dict[str, Any]]:
    """Authenticate, reconstruct, and fully validate one raw session cache."""
    raw_contract = _raw_contract(contract)
    if not isinstance(expected_bundle_sha256, str) or not SHA256_RE.fullmatch(
        expected_bundle_sha256
    ):
        raise FeatureTableValidationError("external raw bundle digest is invalid")
    bundle = Path(bundle_path)
    if not bundle.is_dir() or bundle.is_symlink() or {p.name for p in bundle.iterdir()} != RAW_FILES:
        raise FeatureTableValidationError("raw spectral bundle file set is invalid")
    for name in RAW_FILES:
        if (bundle / name).is_symlink() or not (bundle / name).is_file():
            raise FeatureTableValidationError("raw spectral bundle entries must be regular files")
    sidecar = (bundle / "bundle.sha256").read_bytes()
    if len(sidecar) != 65 or not sidecar.endswith(b"\n"):
        raise FeatureTableValidationError("raw spectral bundle sidecar is invalid")
    sidecar_digest = sidecar[:-1].decode("ascii")
    manifest = _read_canonical_object(bundle / "bundle.json", "raw spectral manifest")
    bundle_digest = sha256((bundle / "bundle.json").read_bytes()).hexdigest()
    if sidecar_digest != bundle_digest or expected_bundle_sha256 != bundle_digest:
        raise FeatureTableValidationError("raw spectral bundle external/sidecar digest mismatch")
    _require_exact_fields(
        manifest, raw_contract["bundle_manifest_exact_fields"], "raw spectral manifest"
    )
    if (
        manifest["schema_id"] != raw_contract["schema_id"]
        or manifest["schema_version"] != raw_contract["schema_version"]
        or manifest["bundle_contract_sha256"]
        != raw_spectral_bundle_contract_sha256(contract)
        or manifest["values_dtype"] != "little_endian_float64"
        or manifest["allow_pickle"] is not False
    ):
        raise FeatureTableValidationError("raw spectral bundle manifest contract mismatch")
    metadata_path = bundle / "metadata.json"
    keys_path = bundle / "trial-keys.json"
    metadata = _read_canonical_object(metadata_path, "raw spectral metadata")
    keys_document = _read_canonical_object(keys_path, "raw spectral trial keys")
    if (
        sha256(metadata_path.read_bytes()).hexdigest() != manifest["metadata_sha256"]
        or sha256(keys_path.read_bytes()).hexdigest() != manifest["trial_keys_sha256"]
    ):
        raise FeatureTableValidationError("raw spectral JSON member digest mismatch")
    _require_exact_fields(
        expected_metadata_bindings,
        raw_contract["external_binding_fields"],
        "raw spectral external bindings",
    )
    for field, expected in expected_metadata_bindings.items():
        if field not in metadata or metadata[field] != expected:
            raise FeatureTableValidationError(f"raw spectral external binding mismatch: {field}")
    _require_exact_fields(
        keys_document,
        ("schema_id", "schema_version", "row_key", "rows"),
        "raw spectral trial keys",
    )
    if (
        keys_document["schema_id"] != raw_contract["trial_keys"]["schema_id"]
        or keys_document["schema_version"] != raw_contract["trial_keys"]["schema_version"]
        or keys_document["row_key"] != list(RAW_KEY_FIELDS)
        or not isinstance(keys_document["rows"], list)
    ):
        raise FeatureTableValidationError("raw spectral trial-key schema mismatch")
    keys = []
    for index, row in enumerate(keys_document["rows"]):
        if not isinstance(row, dict):
            raise FeatureTableValidationError(f"raw spectral trial key {index} is invalid")
        _require_exact_fields(row, RAW_KEY_FIELDS, f"raw spectral trial key {index}")
        keys.append(tuple(row[field] for field in RAW_KEY_FIELDS))
    bin_means = _load_safe_float64_npy(
        bundle / "bin-means.npy", manifest["bin_means_npy_sha256"], "bin means"
    )
    statistics_mean = _load_safe_float64_npy(
        bundle / "statistics-mean.npy",
        manifest["statistics_mean_npy_sha256"],
        "statistics mean",
    )
    statistics_m2 = _load_safe_float64_npy(
        bundle / "statistics-m2.npy",
        manifest["statistics_m2_npy_sha256"],
        "statistics M2",
    )
    time_axis = _load_safe_float64_npy(
        bundle / "time-axis.npy", manifest["time_axis_npy_sha256"], "time axis"
    )
    if (
        list(bin_means.shape) != manifest["bin_means_shape"]
        or list(statistics_mean.shape) != manifest["statistics_shape"]
        or list(statistics_m2.shape) != manifest["statistics_shape"]
        or list(time_axis.shape) != manifest["time_axis_shape"]
        or manifest["trial_count"] != len(keys)
    ):
        raise FeatureTableValidationError("raw spectral manifest array shape mismatch")
    summary = SessionSpectralSummary(
        subject_id=metadata["subject_id"],
        session=metadata["session"],
        condition=metadata["condition"],
        derivative_id=metadata["derivative_id"],
        derivative_sha256=metadata["derivative_sha256"],
        trial_keys=tuple(keys),
        bin_means=_readonly_float64(bin_means, "loaded bin means"),
        statistics_mean=_readonly_float64(statistics_mean, "loaded statistics mean"),
        statistics_m2=_readonly_float64(statistics_m2, "loaded statistics M2"),
        time_axis=_readonly_float64(time_axis, "loaded time axis"),
        statistics_observation_count=metadata["statistics_observation_count"],
        time_bin_sample_counts=tuple(metadata["time_bin_sample_counts"]),
    )
    _validate_metadata(metadata, summary, charter)
    if manifest["statistics_observation_count"] != summary.statistics_observation_count:
        raise FeatureTableValidationError("raw spectral manifest observation-count mismatch")
    return summary, metadata


def extract_locked_epoch_session(
    epoch_path: Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
    charter: Mapping[str, Any],
    n_jobs: int,
) -> SessionSpectralSummary:
    """Read exact locked epoch rows and compute one outcome-free Morlet session summary."""
    if not manifest_rows:
        raise FeatureTableValidationError("real extraction requires locked manifest rows")
    if any(
        row.get("included") is not True
        or row.get("condition_survivor") is not True
        or row.get("exclusion_reason_code") != "INCLUDED"
        for row in manifest_rows
    ):
        raise FeatureTableValidationError(
            "real extraction accepts only locked condition-survivor rows"
        )
    if type(n_jobs) is not int or n_jobs <= 0:
        raise FeatureTableValidationError("real extraction n_jobs must be a positive integer")
    rows = sorted(manifest_rows, key=canonical_trial_key)
    keys = tuple(canonical_trial_key(row) for row in rows)
    if keys != tuple(canonical_trial_key(row) for row in manifest_rows):
        raise FeatureTableValidationError("real extraction rows must be canonically ordered")
    subject_ids = {row["subject_id"] for row in rows}
    sessions = {row["session"] for row in rows}
    conditions = {row["condition"] for row in rows}
    derivative_ids = {row["derivative_id"] for row in rows}
    derivative_hashes = {row["derivative_sha256"] for row in rows}
    if not (
        len(subject_ids)
        == len(sessions)
        == len(conditions)
        == len(derivative_ids)
        == len(derivative_hashes)
        == 1
    ):
        raise FeatureTableValidationError("real extraction manifest identity is not singular")
    subject_id = next(iter(subject_ids))
    session = next(iter(sessions))
    condition = next(iter(conditions))
    derivative_id = next(iter(derivative_ids))
    derivative_sha256 = next(iter(derivative_hashes))
    resolved = Path(data_root) / derivative_id
    if resolved.resolve() != Path(epoch_path).resolve() or _sha256_file(resolved) != derivative_sha256:
        raise FeatureTableValidationError("real extraction derivative path/digest mismatch")
    epoch_rows = [row["epoch_row"] for row in rows]
    if any(type(index) is not int or index < 0 for index in epoch_rows) or len(set(epoch_rows)) != len(
        epoch_rows
    ):
        raise FeatureTableValidationError("real extraction epoch rows are invalid")

    import mne

    epochs = mne.read_epochs(resolved, preload=True, verbose="ERROR")
    if max(epoch_rows) >= len(epochs) or epochs.metadata is None:
        raise FeatureTableValidationError("real extraction epoch row is unavailable")
    for key, epoch_row in zip(keys, epoch_rows, strict=True):
        metadata_row = epochs.metadata.iloc[epoch_row]
        observed = (
            str(metadata_row["subject_id"]),
            int(metadata_row["session"]),
            int(metadata_row["trial"]),
            int(metadata_row["item_num"]),
        )
        if observed != key:
            raise FeatureTableValidationError("real extraction manifest/epoch key mismatch")
    channel_order = [
        channel
        for roi in charter["features"]["roi_order"]
        for channel in charter["features"]["roi_channels"][roi]
    ]
    if len(channel_order) != 40 or any(epochs.ch_names.count(name) != 1 for name in channel_order):
        raise FeatureTableValidationError("real extraction frozen channel set is unavailable")
    selected = epochs[epoch_rows].copy().pick(channel_order)
    frequencies = np.asarray(charter["features"]["frequencies_hz"], dtype=np.float64)
    tfr = selected.compute_tfr(
        method="morlet",
        freqs=frequencies,
        n_cycles=frequencies / 2.0,
        return_itc=False,
        average=False,
        decim=charter["features"]["decim"],
        n_jobs=n_jobs,
        verbose="ERROR",
    )
    if list(tfr.ch_names) != channel_order:
        raise FeatureTableValidationError("real Morlet channel order drifted")
    power = np.asarray(tfr.get_data(), dtype=np.float64)
    if np.any(power < 0.0) or not np.isfinite(power).all():
        raise FeatureTableValidationError("real Morlet power is negative or non-finite")
    log_power = np.log10(power + np.float64(1e-30))
    return summarize_log_power_session(
        np.ascontiguousarray(log_power, dtype=np.float64),
        np.asarray(tfr.times, dtype=np.float64),
        keys,
        subject_id=subject_id,
        session=session,
        condition=condition,
        derivative_id=derivative_id,
        derivative_sha256=derivative_sha256,
        charter=charter,
    )
