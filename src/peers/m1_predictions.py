"""Outcome-blind logical contracts for future paired M1 prediction tables.

This module validates synthetic identity, labels, probability shape, and provenance, including
the resolved D44-C qualified meta-prediction boundary, and provides authenticated typed storage
bundles. It does not fit a model, generate a real prediction, calculate a metric, or inspect
performance.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
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

from peers.m1_features import FeatureTableValidationError, feature_schema_sha256
from peers.m1_manifests import (
    ManifestValidationError,
    _fsync_directory,
    _rename_directory_noreplace,
    _write_fsynced,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    load_json_unique,
    manifest_sha256,
    validate_cross_condition_intersection,
    validate_exact_key_alignment,
    validate_manifest,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PENDING_RE = re.compile(r"^PENDING_[A-Z0-9_]+$")
D44_C_PREDICTION_PROVENANCE_CONTRACT = {
    "schema_id": "neuroai-m1-d44-c-cross-fit-provenance-v1",
    "decision": "D44_C_outer_training_selected_C_qualified_meta_predictions",
    "meta_prediction_claim": (
        "session_out_coefficients_and_transforms_conditional_on_outer_training_selected_C"
    ),
    "fully_session_cross_fitted_claim_allowed": False,
    "predicted_session_labels_may_influence_selected_C": True,
    "outer_test_session_used_for_selection": False,
    "table_binding_field": "cross_fit_provenance_sha256",
    "M3_upstream_binding_required": True,
}

PREDICTION_TABLE_BUNDLE_CONTRACT = {
    "schema_id": "neuroai-m1-prediction-bundle-v1",
    "schema_version": "1.0",
    "atomic_publication": "temporary_directory_fsync_then_atomic_noreplace_rename",
    "exact_files": [
        "metadata.json",
        "row-records.json",
        "probabilities.npy",
        "bundle.json",
        "bundle.sha256",
    ],
    "metadata": "logical_table_exact_fields_except_rows_in_canonical_JSON",
    "row_records": {
        "schema_id": "neuroai-m1-prediction-bundle-row-records-v1",
        "schema_version": "1.0",
        "exact_fields": ["schema_id", "schema_version", "row_fields", "rows"],
        "content": "logical_row_exact_fields_except_probability_fields",
        "canonical_order": "logical_table_row_order",
    },
    "probabilities": {
        "format": "npy",
        "field_order": ["prob_b3", "prob_m1", "prob_m3"],
        "dtype": "little_endian_float64",
        "shape": "n_rows_by_3",
        "order": "C_contiguous",
        "allow_pickle": False,
        "trailing_bytes_allowed": False,
        "finite_closed_interval_0_1": True,
    },
    "bundle_manifest_exact_fields": [
        "schema_id",
        "schema_version",
        "bundle_contract_sha256",
        "logical_table_sha256",
        "metadata_sha256",
        "row_records_sha256",
        "probabilities_npy_sha256",
        "row_count",
        "probability_count",
        "probability_fields",
        "probabilities_dtype",
        "probabilities_shape",
        "probabilities_c_contiguous",
        "allow_pickle",
    ],
    "stored_outcome_fields": ["recalled", "prob_b3", "prob_m1", "prob_m3"],
    "metric_or_result_fields_allowed": False,
    "real_publication_allowed_before_final_readiness": False,
    "external_bundle_sha256_required_on_readback": True,
    "failure_staging_policy": "retain_for_evidence_no_automatic_deletion",
}

PREDICTION_TABLE_BUNDLE_FILES = frozenset(
    PREDICTION_TABLE_BUNDLE_CONTRACT["exact_files"]
)

CONDITION_COMPARISON_BUNDLE_CONTRACT = {
    "schema_id": "neuroai-m1-condition-comparison-bundle-v1",
    "schema_version": "1.0",
    "atomic_publication": "temporary_directory_fsync_then_atomic_noreplace_rename",
    "exact_files": [
        "metadata.json",
        "row-records.json",
        "probabilities.npy",
        "bundle.json",
        "bundle.sha256",
    ],
    "metadata": "condition_comparison_exact_fields_except_rows_in_canonical_JSON",
    "row_records": {
        "schema_id": "neuroai-m1-condition-comparison-bundle-row-records-v1",
        "schema_version": "1.0",
        "exact_fields": ["schema_id", "schema_version", "row_fields", "rows"],
        "content": "condition_comparison_row_exact_fields_except_probability_fields",
        "canonical_order": "condition_comparison_row_order",
    },
    "probabilities": {
        "format": "npy",
        "field_order": [
            "ocular_removed_prob_b3",
            "ocular_removed_prob_m1",
            "ocular_removed_prob_m3",
            "ocular_kept_prob_b3",
            "ocular_kept_prob_m1",
            "ocular_kept_prob_m3",
        ],
        "dtype": "little_endian_float64",
        "shape": "n_rows_by_6",
        "order": "C_contiguous",
        "allow_pickle": False,
        "trailing_bytes_allowed": False,
        "finite_closed_interval_0_1": True,
    },
    "bundle_manifest_exact_fields": [
        "schema_id",
        "schema_version",
        "bundle_contract_sha256",
        "logical_table_sha256",
        "metadata_sha256",
        "row_records_sha256",
        "probabilities_npy_sha256",
        "row_count",
        "probability_count",
        "probability_fields",
        "probabilities_dtype",
        "probabilities_shape",
        "probabilities_c_contiguous",
        "allow_pickle",
    ],
    "stored_outcome_fields": [
        "recalled",
        "ocular_removed_prob_b3",
        "ocular_removed_prob_m1",
        "ocular_removed_prob_m3",
        "ocular_kept_prob_b3",
        "ocular_kept_prob_m1",
        "ocular_kept_prob_m3",
    ],
    "metric_or_result_fields_allowed": False,
    "real_publication_allowed_before_final_readiness": False,
    "external_bundle_sha256_required_on_readback": True,
    "failure_staging_policy": "retain_for_evidence_no_automatic_deletion",
}

CONDITION_COMPARISON_BUNDLE_FILES = frozenset(
    CONDITION_COMPARISON_BUNDLE_CONTRACT["exact_files"]
)


class PredictionTableValidationError(ManifestValidationError):
    """A logical paired prediction table violates the pre-result contract."""


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
        raise PredictionTableValidationError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc
    return serialized.encode("utf-8") + b"\n"


def _require_exact_fields(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise PredictionTableValidationError(
            f"{label} fields mismatch; missing={missing}; extra={extra}"
        )


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PredictionTableValidationError(f"{label} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PredictionTableValidationError(f"{label} must be a valid UTC timestamp") from exc


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
    elif expected == "integer":
        valid = _is_integer(value)
    elif expected == "array[integer]":
        valid = isinstance(value, list) and all(_is_integer(item) for item in value)
    elif expected == "object":
        valid = isinstance(value, dict)
    elif expected == "array[object]":
        valid = isinstance(value, list) and all(isinstance(item, dict) for item in value)
    else:
        raise PredictionTableValidationError(f"unknown contract type {expected!r} at {label}")
    if not valid:
        raise PredictionTableValidationError(f"{label} must have type {expected}")


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


def learner_contract_sha256(charter: Mapping[str, Any], model_id: str) -> str:
    """Digest the exact resolved learner/split contract for B3, M1, or M3."""
    if model_id not in {"B3", "M1", "M3"}:
        raise PredictionTableValidationError(f"unknown learner model_id: {model_id}")
    try:
        payload = {
            "model_id": model_id,
            "learner": deepcopy(charter["learners"][model_id]),
            "splits": deepcopy(charter["splits"]),
            "cross_fitting": deepcopy(charter["cross_fitting"]),
            "global_seed": charter["determinism"]["global_seed"],
        }
        if model_id == "M1":
            payload["resolved_features"] = deepcopy(charter["features"])
    except (KeyError, TypeError) as exc:
        raise PredictionTableValidationError(
            f"charter is missing the {model_id} learner contract"
        ) from exc
    pending = _find_pending(payload)
    if pending:
        raise PredictionTableValidationError(
            f"learner contract cannot be frozen while {pending} is unresolved"
        )
    return sha256(_canonical_object_bytes(payload)).hexdigest()


def _validate_model_provenance(
    provenance: Any,
    contract: Mapping[str, Any],
    *,
    require_all_models: bool,
) -> None:
    if (
        contract.get("D44_C_cross_fit_provenance")
        != D44_C_PREDICTION_PROVENANCE_CONTRACT
    ):
        raise PredictionTableValidationError(
            "D44-C qualified cross-fit provenance contract drifted"
        )
    if not isinstance(provenance, dict) or not provenance:
        raise PredictionTableValidationError("model_provenance must be a nonempty object")
    provenance_contract = contract["model_provenance"]
    required_models = set(provenance_contract["required_models"])
    actual_models = set(provenance)
    if require_all_models and actual_models != required_models:
        raise PredictionTableValidationError(
            f"model provenance set mismatch; expected={sorted(required_models)}; "
            f"actual={sorted(actual_models)}"
        )
    if not require_all_models and not actual_models <= required_models:
        raise PredictionTableValidationError("model provenance contains an unknown model")

    for model_id, model in provenance.items():
        if not isinstance(model, dict):
            raise PredictionTableValidationError(f"{model_id} provenance must be an object")
        _require_exact_fields(
            model,
            provenance_contract["exact_fields"],
            f"{model_id} provenance",
        )
        for field, expected_type in provenance_contract["field_types"].items():
            _validate_type(model[field], expected_type, f"{model_id} provenance {field}")
        if model["model_id"] != model_id:
            raise PredictionTableValidationError(f"{model_id} model_id mismatch")
        expected_role = provenance_contract["fit_roles"][model_id]
        if model["fit_role"] != expected_role:
            raise PredictionTableValidationError(f"{model_id} fit_role mismatch")
        sessions = model["fit_sessions"]
        if sessions != sorted(set(sessions)) or any(session < 0 for session in sessions):
            raise PredictionTableValidationError(
                f"{model_id} fit_sessions must be sorted unique nonnegative integers"
            )
        if model_id in {"B3", "M1"} and model["upstream_cross_fit_sha256"] is not None:
            raise PredictionTableValidationError(
                f"{model_id} upstream_cross_fit_sha256 must be null"
            )
        if model_id == "M3" and model["upstream_cross_fit_sha256"] is None:
            raise PredictionTableValidationError(
                "M3 upstream_cross_fit_sha256 must be a SHA-256"
            )


def model_provenance_sha256(
    provenance: Any,
    contract: Mapping[str, Any],
    *,
    require_all_models: bool = True,
) -> str:
    """Digest a closed model-provenance object or one row-linked model entry."""
    _validate_model_provenance(
        provenance,
        contract,
        require_all_models=require_all_models,
    )
    return sha256(_canonical_object_bytes(provenance)).hexdigest()


def _validate_prediction_table(
    table: Any,
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    require_canonical_order: bool,
) -> None:
    if not isinstance(table, dict):
        raise PredictionTableValidationError("prediction table must be an object")
    table_contract = prediction_contract["logical_table"]
    _require_exact_fields(table, table_contract["exact_fields"], "prediction table")
    for field, expected_type in table_contract["field_types"].items():
        _validate_type(table[field], expected_type, f"prediction table {field}")
    if table["schema_id"] != table_contract["schema_id"]:
        raise PredictionTableValidationError("prediction table schema_id mismatch")
    if table["schema_version"] != table_contract["schema_version"]:
        raise PredictionTableValidationError("prediction table schema_version mismatch")
    if table["condition"] not in table_contract["allowed_conditions"]:
        raise PredictionTableValidationError("prediction table condition is not allowed")
    if table["outer_test_session"] < 0:
        raise PredictionTableValidationError("outer_test_session must be nonnegative")
    expected_fold_id = table_contract["outer_fold_id_template"].format(
        subject_id=table["subject_id"],
        condition=table["condition"],
        outer_test_session=table["outer_test_session"],
    )
    if table["outer_fold_id"] != expected_fold_id:
        raise PredictionTableValidationError("outer_fold_id does not match the frozen template")

    required_bindings = set(table_contract["external_binding_fields"])
    if set(expected_bindings) != required_bindings:
        raise PredictionTableValidationError(
            "expected prediction-table bindings must exactly match the contract"
        )
    for field, expected in expected_bindings.items():
        if table[field] != expected:
            raise PredictionTableValidationError(
                f"prediction table binding mismatch for {field}"
            )

    try:
        validate_manifest(trial_manifest, manifest_contract)
        actual_trial_manifest_sha256 = manifest_sha256(trial_manifest, manifest_contract)
        actual_feature_schema_sha256 = feature_schema_sha256(charter, feature_contract)
    except (ManifestValidationError, FeatureTableValidationError) as exc:
        raise PredictionTableValidationError(str(exc)) from exc
    if table["trial_manifest_sha256"] != actual_trial_manifest_sha256:
        raise PredictionTableValidationError(
            "trial_manifest_sha256 does not match the supplied manifest"
        )
    if table["feature_schema_sha256"] != actual_feature_schema_sha256:
        raise PredictionTableValidationError(
            "feature_schema_sha256 does not match the resolved charter"
        )

    model_provenance = table["model_provenance"]
    actual_model_provenance_sha256 = model_provenance_sha256(
        model_provenance,
        prediction_contract,
    )
    if table["model_provenance_sha256"] != actual_model_provenance_sha256:
        raise PredictionTableValidationError("model_provenance_sha256 does not recompute")

    subject_rows = [
        row
        for row in trial_manifest["rows"]
        if row["subject_id"] == table["subject_id"]
        and row["condition"] == table["condition"]
        and row["condition_survivor"]
    ]
    train_rows = [row for row in subject_rows if row["outer_role"] == "train"]
    test_rows = [row for row in subject_rows if row["outer_role"] == "test"]
    if not train_rows or not test_rows:
        raise PredictionTableValidationError(
            "prediction table requires nonempty outer train and test survivor rows"
        )
    if {row["session"] for row in test_rows} != {table["outer_test_session"]}:
        raise PredictionTableValidationError(
            "trial-manifest outer test rows do not match outer_test_session"
        )
    train_keys = [canonical_trial_key(row) for row in train_rows]
    test_keys = [canonical_trial_key(row) for row in test_rows]
    expected_fit_keys_sha256 = canonical_trial_key_set_sha256(train_keys)
    expected_prediction_keys_sha256 = canonical_trial_key_set_sha256(test_keys)
    expected_fit_sessions = sorted({row["session"] for row in train_rows})

    for model_id, model in model_provenance.items():
        if model["fit_keys_sha256"] != expected_fit_keys_sha256:
            raise PredictionTableValidationError(f"{model_id} fit_keys_sha256 mismatch")
        if model["prediction_keys_sha256"] != expected_prediction_keys_sha256:
            raise PredictionTableValidationError(
                f"{model_id} prediction_keys_sha256 mismatch"
            )
        if model["fit_sessions"] != expected_fit_sessions:
            raise PredictionTableValidationError(f"{model_id} fit_sessions mismatch")
        if model["learner_contract_sha256"] != learner_contract_sha256(charter, model_id):
            raise PredictionTableValidationError(
                f"{model_id} learner_contract_sha256 mismatch"
            )
        if model["seed"] != charter["determinism"]["global_seed"]:
            raise PredictionTableValidationError(f"{model_id} seed mismatch")
        if model_id == "M3" and (
            model["upstream_cross_fit_sha256"]
            != table["cross_fit_provenance_sha256"]
        ):
            raise PredictionTableValidationError(
                "M3 upstream cross-fit provenance does not match the table"
            )

    rows = table["rows"]
    if not rows:
        raise PredictionTableValidationError("prediction table must contain rows")
    trial_by_key = {canonical_trial_key(row): row for row in test_rows}
    observed_keys = []
    row_fields = table_contract["row_exact_fields"]
    probability_fields = table_contract["probability_fields"]
    provenance_links = prediction_contract["model_provenance"]["row_link_fields"]
    expected_model_links = {
        model_id: model_provenance_sha256(
            {model_id: model_provenance[model_id]},
            prediction_contract,
            require_all_models=False,
        )
        for model_id in prediction_contract["model_provenance"]["required_models"]
    }
    for index, row in enumerate(rows):
        _require_exact_fields(row, row_fields, f"prediction row {index}")
        key_values = [row[field] for field in table_contract["row_key"]]
        try:
            key = canonical_trial_key(dict(zip(table_contract["row_key"], key_values)))
        except ManifestValidationError as exc:
            raise PredictionTableValidationError(str(exc)) from exc
        observed_keys.append(key)
        if row["subject_id"] != table["subject_id"]:
            raise PredictionTableValidationError("prediction row subject_id mismatch")
        if row["condition"] != table["condition"]:
            raise PredictionTableValidationError("prediction row condition mismatch")
        if row["outer_fold_id"] != table["outer_fold_id"]:
            raise PredictionTableValidationError("prediction row outer_fold_id mismatch")
        if not isinstance(row["recalled"], bool):
            raise PredictionTableValidationError("prediction row recalled must be boolean")
        if key in trial_by_key and row["recalled"] != trial_by_key[key]["recalled"]:
            raise PredictionTableValidationError("prediction row recall label mismatch")
        for field in probability_fields:
            probability = row[field]
            if type(probability) is not float:
                raise PredictionTableValidationError(f"{field} must be float64")
            if not math.isfinite(probability):
                raise PredictionTableValidationError(f"{field} must be finite")
            if not 0.0 <= probability <= 1.0:
                raise PredictionTableValidationError(
                    f"{field} must be in the closed interval [0, 1]"
                )
        for model_id, link_field in provenance_links.items():
            link = row[link_field]
            if not isinstance(link, str) or SHA256_RE.fullmatch(link) is None:
                raise PredictionTableValidationError(f"{link_field} must be a SHA-256")
            if link != expected_model_links[model_id]:
                raise PredictionTableValidationError(f"{link_field} does not match provenance")

    try:
        validate_exact_key_alignment(
            test_keys,
            observed_keys,
            expected_name="outer-test survivor rows",
            observed_name="prediction table rows",
        )
    except ManifestValidationError as exc:
        raise PredictionTableValidationError(str(exc)) from exc
    row_key = table_contract["row_key"]
    sorted_rows = sorted(rows, key=lambda row: tuple(row[field] for field in row_key))
    if require_canonical_order and rows != sorted_rows:
        raise PredictionTableValidationError(
            "prediction table rows are not in canonical row order"
        )


def validate_prediction_table(
    table: Any,
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
) -> None:
    """Validate one stored paired outer-test prediction table."""
    _validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
        require_canonical_order=True,
    )


def canonical_prediction_table_bytes(
    table: Any,
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
) -> bytes:
    """Validate and serialize a logical prediction table in canonical row order."""
    _validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
        require_canonical_order=False,
    )
    canonical = deepcopy(table)
    row_key = prediction_contract["logical_table"]["row_key"]
    canonical["rows"] = sorted(
        canonical["rows"], key=lambda row: tuple(row[field] for field in row_key)
    )
    return _canonical_object_bytes(canonical)


def prediction_table_sha256(
    table: Any,
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
) -> str:
    """SHA-256 of the canonical logical paired prediction table."""
    return sha256(
        canonical_prediction_table_bytes(
            table,
            trial_manifest,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=expected_bindings,
        )
    ).hexdigest()


def _prediction_bundle_contract_payload(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        declared = contract["storage_bundle"]
    except (KeyError, TypeError) as exc:
        raise PredictionTableValidationError(
            "prediction contract is missing the storage bundle contract"
        ) from exc
    if declared != PREDICTION_TABLE_BUNDLE_CONTRACT:
        raise PredictionTableValidationError(
            "prediction storage bundle contract drifted from the implementation"
        )
    return deepcopy(declared)


def prediction_table_bundle_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Digest the exact technical prediction-bundle layout without calculating metrics."""
    return sha256(
        _canonical_object_bytes(_prediction_bundle_contract_payload(contract))
    ).hexdigest()


def _prediction_bundle_parts(
    table: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    probability_fields = contract["logical_table"]["probability_fields"]
    row_fields = [
        field
        for field in contract["logical_table"]["row_exact_fields"]
        if field not in probability_fields
    ]
    metadata = {field: deepcopy(value) for field, value in table.items() if field != "rows"}
    row_records = {
        "schema_id": PREDICTION_TABLE_BUNDLE_CONTRACT["row_records"]["schema_id"],
        "schema_version": PREDICTION_TABLE_BUNDLE_CONTRACT["row_records"][
            "schema_version"
        ],
        "row_fields": row_fields,
        "rows": [
            {field: deepcopy(row[field]) for field in row_fields}
            for row in table["rows"]
        ],
    }
    probabilities = np.ascontiguousarray(
        np.asarray(
            [
                [row[field] for field in probability_fields]
                for row in table["rows"]
            ],
            dtype="<f8",
        )
    )
    return metadata, row_records, probabilities


def _write_npy_fsynced(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def write_prediction_table_bundle_atomic(
    bundle_path: Path,
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
) -> str:
    """Publish one immutable typed bundle from an already validated prediction table."""
    _prediction_bundle_contract_payload(prediction_contract)
    validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
    )
    destination = Path(bundle_path)
    parent = destination.parent
    if not parent.is_dir():
        raise PredictionTableValidationError(
            f"prediction bundle parent is not a directory: {parent}"
        )
    if os.path.lexists(destination):
        raise PredictionTableValidationError(
            f"prediction table bundle already exists: {destination}"
        )

    metadata, row_records, probabilities = _prediction_bundle_parts(
        table, prediction_contract
    )
    metadata_bytes = _canonical_object_bytes(metadata)
    row_records_bytes = _canonical_object_bytes(row_records)
    logical_digest = prediction_table_sha256(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        _write_fsynced(temporary / "metadata.json", metadata_bytes)
        _write_fsynced(temporary / "row-records.json", row_records_bytes)
        _write_npy_fsynced(temporary / "probabilities.npy", probabilities)
        probabilities_bytes = (temporary / "probabilities.npy").read_bytes()
        probability_fields = prediction_contract["logical_table"]["probability_fields"]
        manifest = {
            "schema_id": PREDICTION_TABLE_BUNDLE_CONTRACT["schema_id"],
            "schema_version": PREDICTION_TABLE_BUNDLE_CONTRACT["schema_version"],
            "bundle_contract_sha256": prediction_table_bundle_contract_sha256(
                prediction_contract
            ),
            "logical_table_sha256": logical_digest,
            "metadata_sha256": sha256(metadata_bytes).hexdigest(),
            "row_records_sha256": sha256(row_records_bytes).hexdigest(),
            "probabilities_npy_sha256": sha256(probabilities_bytes).hexdigest(),
            "row_count": int(probabilities.shape[0]),
            "probability_count": int(probabilities.shape[1]),
            "probability_fields": list(probability_fields),
            "probabilities_dtype": "little_endian_float64",
            "probabilities_shape": [int(size) for size in probabilities.shape],
            "probabilities_c_contiguous": bool(probabilities.flags.c_contiguous),
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
            raise PredictionTableValidationError(
                f"prediction table bundle already exists: {destination}"
            )
        _rename_directory_noreplace(temporary, destination)
        _fsync_directory(parent)
    except Exception as exc:
        raise PredictionTableValidationError(
            "prediction table bundle publication failed; staged evidence retained at "
            f"{temporary}: {exc}"
        ) from exc
    return bundle_digest


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PredictionTableValidationError(f"{label} must be a regular file")
    try:
        value = load_json_unique(path)
    except ManifestValidationError as exc:
        raise PredictionTableValidationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise PredictionTableValidationError(f"{label} must contain one JSON object")
    if path.read_bytes() != _canonical_object_bytes(value):
        raise PredictionTableValidationError(f"{label} is not canonical JSON bytes")
    return value


def load_prediction_table_bundle(
    bundle_path: Path,
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Read, authenticate, reconstruct, and revalidate one prediction bundle."""
    bundle_contract = _prediction_bundle_contract_payload(prediction_contract)
    if not isinstance(expected_bundle_sha256, str) or not SHA256_RE.fullmatch(
        expected_bundle_sha256
    ):
        raise PredictionTableValidationError(
            "expected external bundle digest must be a lowercase SHA-256"
        )
    bundle = Path(bundle_path)
    if not bundle.is_dir() or bundle.is_symlink():
        raise PredictionTableValidationError(
            f"prediction table bundle is not a regular directory: {bundle}"
        )
    actual_files = {path.name for path in bundle.iterdir()}
    if actual_files != PREDICTION_TABLE_BUNDLE_FILES:
        raise PredictionTableValidationError(
            "prediction table bundle does not contain the exact files required by the contract"
        )
    for name in PREDICTION_TABLE_BUNDLE_FILES:
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise PredictionTableValidationError(
                f"prediction table bundle entry must be a regular file: {name}"
            )

    sidecar = (bundle / "bundle.sha256").read_bytes()
    if len(sidecar) != 65 or not sidecar.endswith(b"\n"):
        raise PredictionTableValidationError(
            "prediction bundle sidecar must be one lowercase SHA-256 plus newline"
        )
    try:
        sidecar_digest = sidecar[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise PredictionTableValidationError(
            "prediction bundle sidecar is not ASCII"
        ) from exc
    if not SHA256_RE.fullmatch(sidecar_digest):
        raise PredictionTableValidationError(
            "prediction bundle sidecar is not a lowercase SHA-256"
        )

    manifest_path = bundle / "bundle.json"
    manifest = _read_canonical_json(manifest_path, "prediction bundle manifest")
    manifest_bytes = manifest_path.read_bytes()
    bundle_digest = sha256(manifest_bytes).hexdigest()
    if sidecar_digest != bundle_digest:
        raise PredictionTableValidationError(
            "prediction bundle sidecar does not match the canonical bundle manifest"
        )
    if expected_bundle_sha256 != bundle_digest:
        raise PredictionTableValidationError(
            "external bundle digest does not match the canonical bundle manifest"
        )
    _require_exact_fields(
        manifest,
        bundle_contract["bundle_manifest_exact_fields"],
        "prediction bundle manifest",
    )
    if (
        manifest["schema_id"] != bundle_contract["schema_id"]
        or manifest["schema_version"] != bundle_contract["schema_version"]
    ):
        raise PredictionTableValidationError("prediction bundle schema identity mismatch")
    if manifest["bundle_contract_sha256"] != prediction_table_bundle_contract_sha256(
        prediction_contract
    ):
        raise PredictionTableValidationError("prediction bundle contract digest mismatch")
    probability_fields = prediction_contract["logical_table"]["probability_fields"]
    if (
        manifest["probability_fields"] != probability_fields
        or manifest["probabilities_dtype"] != "little_endian_float64"
        or manifest["probabilities_c_contiguous"] is not True
        or manifest["allow_pickle"] is not False
    ):
        raise PredictionTableValidationError(
            "prediction bundle probability declaration mismatch"
        )
    if not (
        _is_integer(manifest["row_count"])
        and manifest["row_count"] > 0
        and _is_integer(manifest["probability_count"])
        and manifest["probability_count"] == len(probability_fields) == 3
        and manifest["probabilities_shape"]
        == [manifest["row_count"], manifest["probability_count"]]
    ):
        raise PredictionTableValidationError(
            "prediction bundle probability shape metadata is invalid"
        )
    for field in (
        "logical_table_sha256",
        "metadata_sha256",
        "row_records_sha256",
        "probabilities_npy_sha256",
    ):
        if not isinstance(manifest[field], str) or not SHA256_RE.fullmatch(manifest[field]):
            raise PredictionTableValidationError(
                f"prediction bundle {field} is not a SHA-256"
            )

    metadata_path = bundle / "metadata.json"
    records_path = bundle / "row-records.json"
    probabilities_path = bundle / "probabilities.npy"
    metadata = _read_canonical_json(metadata_path, "prediction bundle metadata")
    row_records = _read_canonical_json(
        records_path, "prediction bundle row records"
    )
    if sha256(metadata_path.read_bytes()).hexdigest() != manifest["metadata_sha256"]:
        raise PredictionTableValidationError("prediction bundle metadata digest mismatch")
    if sha256(records_path.read_bytes()).hexdigest() != manifest["row_records_sha256"]:
        raise PredictionTableValidationError(
            "prediction bundle row-records digest mismatch"
        )
    if (
        sha256(probabilities_path.read_bytes()).hexdigest()
        != manifest["probabilities_npy_sha256"]
    ):
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy digest mismatch"
        )

    expected_metadata_fields = set(
        prediction_contract["logical_table"]["exact_fields"]
    ) - {"rows"}
    _require_exact_fields(
        metadata, expected_metadata_fields, "prediction bundle metadata"
    )
    _require_exact_fields(
        row_records,
        bundle_contract["row_records"]["exact_fields"],
        "prediction bundle row records",
    )
    expected_row_fields = [
        field
        for field in prediction_contract["logical_table"]["row_exact_fields"]
        if field not in probability_fields
    ]
    if (
        row_records["schema_id"] != bundle_contract["row_records"]["schema_id"]
        or row_records["schema_version"]
        != bundle_contract["row_records"]["schema_version"]
        or row_records["row_fields"] != expected_row_fields
    ):
        raise PredictionTableValidationError(
            "prediction bundle row-record schema mismatch"
        )
    if not isinstance(row_records["rows"], list):
        raise PredictionTableValidationError(
            "prediction bundle row records must be an array"
        )
    for index, row in enumerate(row_records["rows"]):
        if not isinstance(row, dict):
            raise PredictionTableValidationError(
                f"prediction bundle row record {index} must be an object"
            )
        _require_exact_fields(
            row, expected_row_fields, f"prediction bundle row record {index}"
        )

    try:
        with probabilities_path.open("rb") as handle:
            probabilities = np.load(handle, allow_pickle=False)
            trailing = handle.read(1)
    except (OSError, ValueError) as exc:
        raise PredictionTableValidationError(
            f"prediction bundle probabilities.npy is not a safe NumPy array: {exc}"
        ) from exc
    if trailing:
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy has trailing bytes"
        )
    if not isinstance(probabilities, np.ndarray) or probabilities.dtype.str != "<f8":
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy must be little-endian float64"
        )
    if not probabilities.flags.c_contiguous:
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy must be C-contiguous"
        )
    if list(probabilities.shape) != manifest["probabilities_shape"]:
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy shape mismatch"
        )
    if not np.isfinite(probabilities).all():
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy contains non-finite values"
        )
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise PredictionTableValidationError(
            "prediction bundle probabilities.npy must remain in [0, 1]"
        )
    if len(row_records["rows"]) != manifest["row_count"]:
        raise PredictionTableValidationError(
            "prediction bundle row-record count mismatch"
        )

    rows = []
    for index, record in enumerate(row_records["rows"]):
        probabilities_by_field = {
            field: float(probabilities[index, column])
            for column, field in enumerate(probability_fields)
        }
        rows.append({**deepcopy(record), **probabilities_by_field})
    table = {**metadata, "rows": rows}
    validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
    )
    logical_digest = prediction_table_sha256(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
    )
    if logical_digest != manifest["logical_table_sha256"]:
        raise PredictionTableValidationError(
            "prediction bundle reconstructed logical-table digest mismatch"
        )
    return table


def _require_condition_comparison_contract(
    prediction_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract = prediction_contract.get("condition_comparison_table")
    if not isinstance(contract, Mapping):
        raise PredictionTableValidationError(
            "condition_comparison_table contract must be an object"
        )
    if contract.get("schema_id") != "neuroai-m1-condition-comparison-table-v1":
        raise PredictionTableValidationError(
            "unexpected condition comparison schema_id"
        )
    if contract.get("metric_fields_allowed") is not False:
        raise PredictionTableValidationError(
            "condition comparison table must remain metric-free"
        )
    return contract


def _require_comparison_provenance(
    provenance: Mapping[str, Any],
    comparison_contract: Mapping[str, Any],
) -> None:
    expected_fields = set(comparison_contract["provenance_exact_fields"])
    if not isinstance(provenance, Mapping) or set(provenance) != expected_fields:
        raise PredictionTableValidationError(
            "comparison provenance fields do not match the closed contract"
        )
    field_types = comparison_contract["field_types"]
    for field in comparison_contract["provenance_exact_fields"]:
        _validate_type(provenance[field], field_types[field], f"comparison {field}")


def _require_expected_prediction_table_digests(
    expected_by_condition: Mapping[str, str],
) -> None:
    expected_conditions = {"ocular-removed", "ocular-kept"}
    if not isinstance(expected_by_condition, Mapping) or (
        set(expected_by_condition) != expected_conditions
    ):
        raise PredictionTableValidationError(
            "expected source prediction-table digests must contain both ocular conditions"
        )
    for condition, digest in expected_by_condition.items():
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise PredictionTableValidationError(
                f"{condition} source prediction-table digest must be a SHA-256"
            )


def _prediction_row_sha256(row: Mapping[str, Any]) -> str:
    return sha256(_canonical_object_bytes(row)).hexdigest()


def _validated_condition_comparison_rows(
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    learner_contract_sha256(charter, "M1")
    _require_condition_comparison_contract(prediction_contract)
    _require_expected_prediction_table_digests(
        expected_prediction_table_sha256_by_condition
    )
    if (
        not isinstance(expected_intersection_manifest_sha256, str)
        or SHA256_RE.fullmatch(expected_intersection_manifest_sha256) is None
    ):
        raise PredictionTableValidationError(
            "expected intersection manifest digest must be a SHA-256"
        )
    if ocular_removed_table.get("condition") != "ocular-removed":
        raise PredictionTableValidationError(
            "first source table must be ocular-removed"
        )
    if ocular_kept_table.get("condition") != "ocular-kept":
        raise PredictionTableValidationError("second source table must be ocular-kept")

    validate_prediction_table(
        ocular_removed_table,
        ocular_removed_trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_removed_bindings,
    )
    validate_prediction_table(
        ocular_kept_table,
        ocular_kept_trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_kept_bindings,
    )
    try:
        validate_cross_condition_intersection(
            intersection_manifest,
            ocular_removed_trial_manifest,
            ocular_kept_trial_manifest,
            manifest_contract,
        )
        actual_intersection_sha256 = manifest_sha256(
            intersection_manifest, manifest_contract
        )
    except ManifestValidationError as exc:
        raise PredictionTableValidationError(str(exc)) from exc
    if actual_intersection_sha256 != expected_intersection_manifest_sha256:
        raise PredictionTableValidationError("intersection manifest digest mismatch")

    source_digests = {
        "ocular-removed": prediction_table_sha256(
            ocular_removed_table,
            ocular_removed_trial_manifest,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=expected_removed_bindings,
        ),
        "ocular-kept": prediction_table_sha256(
            ocular_kept_table,
            ocular_kept_trial_manifest,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=expected_kept_bindings,
        ),
    }
    if source_digests != dict(expected_prediction_table_sha256_by_condition):
        raise PredictionTableValidationError("source prediction-table digest mismatch")

    subject_id = ocular_removed_table["subject_id"]
    if ocular_kept_table["subject_id"] != subject_id:
        raise PredictionTableValidationError(
            "condition comparison source subjects do not match"
        )
    outer_test_session = ocular_removed_table["outer_test_session"]
    if ocular_kept_table["outer_test_session"] != outer_test_session:
        raise PredictionTableValidationError(
            "condition comparison requires one common outer-test session"
        )
    if ocular_kept_table["charter_sha256"] != ocular_removed_table["charter_sha256"]:
        raise PredictionTableValidationError(
            "condition comparison source charter bindings do not match"
        )

    removed_by_key = {
        canonical_trial_key(row): row for row in ocular_removed_table["rows"]
    }
    kept_by_key = {
        canonical_trial_key(row): row for row in ocular_kept_table["rows"]
    }
    intersection_by_key = {
        canonical_trial_key(row): row
        for row in intersection_manifest["rows"]
        if row["subject_id"] == subject_id and row["session"] == outer_test_session
    }
    source_common_keys = set(removed_by_key) & set(kept_by_key)
    if not source_common_keys:
        raise PredictionTableValidationError(
            "condition comparison has no common outer-test rows"
        )
    try:
        validate_exact_key_alignment(
            intersection_by_key,
            source_common_keys,
            expected_name="outer-test intersection manifest rows",
            observed_name="common source prediction rows",
        )
    except ManifestValidationError as exc:
        raise PredictionTableValidationError(str(exc)) from exc

    rows = []
    for key in sorted(source_common_keys):
        removed = removed_by_key[key]
        kept = kept_by_key[key]
        intersection = intersection_by_key[key]
        if removed["recalled"] != kept["recalled"]:
            raise PredictionTableValidationError(
                "source prediction rows disagree on recall label"
            )
        if intersection["recalled"] != removed["recalled"]:
            raise PredictionTableValidationError(
                "intersection and prediction rows disagree on recall label"
            )
        rows.append(
            {
                "subject_id": key[0],
                "session": key[1],
                "trial": key[2],
                "item_num": key[3],
                "recalled": removed["recalled"],
                "event_provenance_sha256": intersection[
                    "event_provenance_sha256"
                ],
                "ocular_removed_prediction_row_sha256": _prediction_row_sha256(
                    removed
                ),
                "ocular_kept_prediction_row_sha256": _prediction_row_sha256(kept),
                "ocular_removed_prob_b3": removed["prob_b3"],
                "ocular_removed_prob_m1": removed["prob_m1"],
                "ocular_removed_prob_m3": removed["prob_m3"],
                "ocular_kept_prob_b3": kept["prob_b3"],
                "ocular_kept_prob_m1": kept["prob_m1"],
                "ocular_kept_prob_m3": kept["prob_m3"],
            }
        )
    return rows, source_digests


def _require_condition_comparison_shape(
    table: Any,
    prediction_contract: Mapping[str, Any],
) -> None:
    contract = _require_condition_comparison_contract(prediction_contract)
    if not isinstance(table, Mapping):
        raise PredictionTableValidationError(
            "condition comparison table must be an object"
        )
    _require_exact_fields(table, contract["exact_fields"], "condition comparison table")
    for field, expected_type in contract["field_types"].items():
        _validate_type(table[field], expected_type, f"condition comparison {field}")
    if table["schema_id"] != contract["schema_id"]:
        raise PredictionTableValidationError(
            "condition comparison schema_id mismatch"
        )
    if table["schema_version"] != contract["schema_version"]:
        raise PredictionTableValidationError(
            "condition comparison schema_version mismatch"
        )
    if table["outer_test_session"] < 0:
        raise PredictionTableValidationError(
            "condition comparison outer_test_session must be nonnegative"
        )
    if table["n_rows"] <= 0 or table["n_rows"] != len(table["rows"]):
        raise PredictionTableValidationError(
            "condition comparison n_rows must match a non-empty row set"
        )

    observed_keys = []
    for index, row in enumerate(table["rows"]):
        _require_exact_fields(
            row,
            contract["row_exact_fields"],
            f"condition comparison row {index}",
        )
        try:
            key = canonical_trial_key(row)
        except ManifestValidationError as exc:
            raise PredictionTableValidationError(str(exc)) from exc
        observed_keys.append(key)
        if row["subject_id"] != table["subject_id"]:
            raise PredictionTableValidationError(
                "condition comparison row subject mismatch"
            )
        if row["session"] != table["outer_test_session"]:
            raise PredictionTableValidationError(
                "condition comparison row is outside the common outer-test session"
            )
        if not isinstance(row["recalled"], bool):
            raise PredictionTableValidationError(
                "condition comparison recalled must be boolean"
            )
        for field in contract["row_sha256_fields"]:
            value = row[field]
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise PredictionTableValidationError(f"{field} must be a SHA-256")
        for field in contract["probability_fields"]:
            value = row[field]
            if type(value) is not float or not math.isfinite(value):
                raise PredictionTableValidationError(f"{field} must be a finite float64")
            if not 0.0 <= value <= 1.0:
                raise PredictionTableValidationError(
                    f"{field} must be in the closed interval [0, 1]"
                )
    if observed_keys != sorted(observed_keys) or len(observed_keys) != len(
        set(observed_keys)
    ):
        raise PredictionTableValidationError(
            "condition comparison rows must be unique and canonically ordered"
        )


def build_condition_comparison_table(
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    comparison_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Pair both conditions on their exact common outer-test rows without metrics."""
    contract = _require_condition_comparison_contract(prediction_contract)
    _require_comparison_provenance(comparison_provenance, contract)
    rows, source_digests = _validated_condition_comparison_rows(
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=expected_intersection_manifest_sha256,
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
    )
    table = {
        "schema_id": contract["schema_id"],
        "schema_version": contract["schema_version"],
        "subject_id": ocular_removed_table["subject_id"],
        "outer_test_session": ocular_removed_table["outer_test_session"],
        "charter_sha256": ocular_removed_table["charter_sha256"],
        "cross_condition_intersection_manifest_sha256": (
            expected_intersection_manifest_sha256
        ),
        "ocular_removed_prediction_table_sha256": source_digests[
            "ocular-removed"
        ],
        "ocular_kept_prediction_table_sha256": source_digests["ocular-kept"],
        "ocular_removed_outer_fold_id": ocular_removed_table["outer_fold_id"],
        "ocular_kept_outer_fold_id": ocular_kept_table["outer_fold_id"],
        **dict(comparison_provenance),
        "n_rows": len(rows),
        "rows": rows,
    }
    _require_condition_comparison_shape(table, prediction_contract)
    return table


def validate_condition_comparison_table(
    table: Any,
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
) -> None:
    """Recompute a complete metric-free condition comparison table."""
    _require_condition_comparison_shape(table, prediction_contract)
    expected = build_condition_comparison_table(
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=expected_intersection_manifest_sha256,
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        comparison_provenance=expected_comparison_provenance,
    )
    if _canonical_object_bytes(table) != _canonical_object_bytes(expected):
        raise PredictionTableValidationError(
            "condition comparison table does not match complete recomputation"
        )


def canonical_condition_comparison_table_bytes(
    table: Any,
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
) -> bytes:
    """Validate and serialize a common-intersection comparison table."""
    validate_condition_comparison_table(
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=expected_intersection_manifest_sha256,
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
    )
    return _canonical_object_bytes(table)


def condition_comparison_table_sha256(
    table: Any,
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
) -> str:
    """SHA-256 of one fully validated condition comparison table."""
    return sha256(
        canonical_condition_comparison_table_bytes(
            table,
            ocular_removed_table,
            ocular_kept_table,
            ocular_removed_trial_manifest,
            ocular_kept_trial_manifest,
            intersection_manifest,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_removed_bindings=expected_removed_bindings,
            expected_kept_bindings=expected_kept_bindings,
            expected_intersection_manifest_sha256=(
                expected_intersection_manifest_sha256
            ),
            expected_prediction_table_sha256_by_condition=(
                expected_prediction_table_sha256_by_condition
            ),
            expected_comparison_provenance=expected_comparison_provenance,
        )
    ).hexdigest()


def _condition_comparison_bundle_contract_payload(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        declared = contract["condition_comparison_storage_bundle"]
    except (KeyError, TypeError) as exc:
        raise PredictionTableValidationError(
            "prediction contract is missing the condition comparison storage bundle"
        ) from exc
    if declared != CONDITION_COMPARISON_BUNDLE_CONTRACT:
        raise PredictionTableValidationError(
            "condition comparison storage bundle contract drifted from the implementation"
        )
    return deepcopy(declared)


def condition_comparison_bundle_contract_sha256(
    contract: Mapping[str, Any],
) -> str:
    """Digest the exact metric-free same-trial comparison bundle layout."""
    return sha256(
        _canonical_object_bytes(
            _condition_comparison_bundle_contract_payload(contract)
        )
    ).hexdigest()


def _condition_comparison_bundle_parts(
    table: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    comparison_contract = contract["condition_comparison_table"]
    probability_fields = comparison_contract["probability_fields"]
    row_fields = [
        field
        for field in comparison_contract["row_exact_fields"]
        if field not in probability_fields
    ]
    metadata = {field: deepcopy(value) for field, value in table.items() if field != "rows"}
    row_records = {
        "schema_id": CONDITION_COMPARISON_BUNDLE_CONTRACT["row_records"][
            "schema_id"
        ],
        "schema_version": CONDITION_COMPARISON_BUNDLE_CONTRACT["row_records"][
            "schema_version"
        ],
        "row_fields": row_fields,
        "rows": [
            {field: deepcopy(row[field]) for field in row_fields}
            for row in table["rows"]
        ],
    }
    probabilities = np.ascontiguousarray(
        np.asarray(
            [[row[field] for field in probability_fields] for row in table["rows"]],
            dtype="<f8",
        )
    )
    return metadata, row_records, probabilities


def write_condition_comparison_bundle_atomic(
    bundle_path: Path,
    table: Mapping[str, Any],
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
) -> str:
    """Publish one immutable typed common-intersection comparison bundle."""
    _condition_comparison_bundle_contract_payload(prediction_contract)
    validate_condition_comparison_table(
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
    )
    destination = Path(bundle_path)
    parent = destination.parent
    if not parent.is_dir():
        raise PredictionTableValidationError(
            f"condition comparison bundle parent is not a directory: {parent}"
        )
    if os.path.lexists(destination):
        raise PredictionTableValidationError(
            f"condition comparison bundle already exists: {destination}"
        )

    metadata, row_records, probabilities = _condition_comparison_bundle_parts(
        table, prediction_contract
    )
    metadata_bytes = _canonical_object_bytes(metadata)
    row_records_bytes = _canonical_object_bytes(row_records)
    logical_digest = condition_comparison_table_sha256(
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
    )
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        _write_fsynced(temporary / "metadata.json", metadata_bytes)
        _write_fsynced(temporary / "row-records.json", row_records_bytes)
        _write_npy_fsynced(temporary / "probabilities.npy", probabilities)
        probabilities_bytes = (temporary / "probabilities.npy").read_bytes()
        probability_fields = prediction_contract["condition_comparison_table"][
            "probability_fields"
        ]
        manifest = {
            "schema_id": CONDITION_COMPARISON_BUNDLE_CONTRACT["schema_id"],
            "schema_version": CONDITION_COMPARISON_BUNDLE_CONTRACT[
                "schema_version"
            ],
            "bundle_contract_sha256": condition_comparison_bundle_contract_sha256(
                prediction_contract
            ),
            "logical_table_sha256": logical_digest,
            "metadata_sha256": sha256(metadata_bytes).hexdigest(),
            "row_records_sha256": sha256(row_records_bytes).hexdigest(),
            "probabilities_npy_sha256": sha256(probabilities_bytes).hexdigest(),
            "row_count": int(probabilities.shape[0]),
            "probability_count": int(probabilities.shape[1]),
            "probability_fields": list(probability_fields),
            "probabilities_dtype": "little_endian_float64",
            "probabilities_shape": [int(size) for size in probabilities.shape],
            "probabilities_c_contiguous": bool(probabilities.flags.c_contiguous),
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
            raise PredictionTableValidationError(
                f"condition comparison bundle already exists: {destination}"
            )
        _rename_directory_noreplace(temporary, destination)
        _fsync_directory(parent)
    except Exception as exc:
        raise PredictionTableValidationError(
            "condition comparison bundle publication failed; staged evidence retained at "
            f"{temporary}: {exc}"
        ) from exc
    return bundle_digest


def load_condition_comparison_bundle(
    bundle_path: Path,
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Authenticate, reconstruct, and recompute one comparison bundle."""
    bundle_contract = _condition_comparison_bundle_contract_payload(
        prediction_contract
    )
    if not isinstance(expected_bundle_sha256, str) or not SHA256_RE.fullmatch(
        expected_bundle_sha256
    ):
        raise PredictionTableValidationError(
            "expected external bundle digest must be a lowercase SHA-256"
        )
    bundle = Path(bundle_path)
    if not bundle.is_dir() or bundle.is_symlink():
        raise PredictionTableValidationError(
            f"condition comparison bundle is not a regular directory: {bundle}"
        )
    actual_files = {path.name for path in bundle.iterdir()}
    if actual_files != CONDITION_COMPARISON_BUNDLE_FILES:
        raise PredictionTableValidationError(
            "condition comparison bundle does not contain the exact files required by the contract"
        )
    for name in CONDITION_COMPARISON_BUNDLE_FILES:
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise PredictionTableValidationError(
                f"condition comparison bundle entry must be a regular file: {name}"
            )

    sidecar = (bundle / "bundle.sha256").read_bytes()
    if len(sidecar) != 65 or not sidecar.endswith(b"\n"):
        raise PredictionTableValidationError(
            "condition comparison bundle sidecar must be one lowercase SHA-256 plus newline"
        )
    try:
        sidecar_digest = sidecar[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise PredictionTableValidationError(
            "condition comparison bundle sidecar is not ASCII"
        ) from exc
    if not SHA256_RE.fullmatch(sidecar_digest):
        raise PredictionTableValidationError(
            "condition comparison bundle sidecar is not a lowercase SHA-256"
        )

    manifest_path = bundle / "bundle.json"
    manifest = _read_canonical_json(
        manifest_path, "condition comparison bundle manifest"
    )
    manifest_bytes = manifest_path.read_bytes()
    bundle_digest = sha256(manifest_bytes).hexdigest()
    if sidecar_digest != bundle_digest:
        raise PredictionTableValidationError(
            "condition comparison bundle sidecar does not match the canonical manifest"
        )
    if expected_bundle_sha256 != bundle_digest:
        raise PredictionTableValidationError(
            "external bundle digest does not match the canonical bundle manifest"
        )
    _require_exact_fields(
        manifest,
        bundle_contract["bundle_manifest_exact_fields"],
        "condition comparison bundle manifest",
    )
    if (
        manifest["schema_id"] != bundle_contract["schema_id"]
        or manifest["schema_version"] != bundle_contract["schema_version"]
    ):
        raise PredictionTableValidationError(
            "condition comparison bundle schema identity mismatch"
        )
    if (
        manifest["bundle_contract_sha256"]
        != condition_comparison_bundle_contract_sha256(prediction_contract)
    ):
        raise PredictionTableValidationError(
            "condition comparison bundle contract digest mismatch"
        )
    comparison_contract = prediction_contract["condition_comparison_table"]
    probability_fields = comparison_contract["probability_fields"]
    if (
        manifest["probability_fields"] != probability_fields
        or manifest["probabilities_dtype"] != "little_endian_float64"
        or manifest["probabilities_c_contiguous"] is not True
        or manifest["allow_pickle"] is not False
    ):
        raise PredictionTableValidationError(
            "condition comparison bundle probability declaration mismatch"
        )
    if not (
        _is_integer(manifest["row_count"])
        and manifest["row_count"] > 0
        and _is_integer(manifest["probability_count"])
        and manifest["probability_count"] == len(probability_fields) == 6
        and manifest["probabilities_shape"]
        == [manifest["row_count"], manifest["probability_count"]]
    ):
        raise PredictionTableValidationError(
            "condition comparison bundle probability shape metadata is invalid"
        )
    for field in (
        "logical_table_sha256",
        "metadata_sha256",
        "row_records_sha256",
        "probabilities_npy_sha256",
    ):
        if not isinstance(manifest[field], str) or not SHA256_RE.fullmatch(manifest[field]):
            raise PredictionTableValidationError(
                f"condition comparison bundle {field} is not a SHA-256"
            )

    metadata_path = bundle / "metadata.json"
    records_path = bundle / "row-records.json"
    probabilities_path = bundle / "probabilities.npy"
    metadata = _read_canonical_json(
        metadata_path, "condition comparison bundle metadata"
    )
    row_records = _read_canonical_json(
        records_path, "condition comparison bundle row records"
    )
    if sha256(metadata_path.read_bytes()).hexdigest() != manifest["metadata_sha256"]:
        raise PredictionTableValidationError(
            "condition comparison bundle metadata digest mismatch"
        )
    if sha256(records_path.read_bytes()).hexdigest() != manifest["row_records_sha256"]:
        raise PredictionTableValidationError(
            "condition comparison bundle row-records digest mismatch"
        )
    if (
        sha256(probabilities_path.read_bytes()).hexdigest()
        != manifest["probabilities_npy_sha256"]
    ):
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy digest mismatch"
        )

    expected_metadata_fields = set(comparison_contract["exact_fields"]) - {"rows"}
    _require_exact_fields(
        metadata,
        expected_metadata_fields,
        "condition comparison bundle metadata",
    )
    _require_exact_fields(
        row_records,
        bundle_contract["row_records"]["exact_fields"],
        "condition comparison bundle row records",
    )
    expected_row_fields = [
        field
        for field in comparison_contract["row_exact_fields"]
        if field not in probability_fields
    ]
    if (
        row_records["schema_id"] != bundle_contract["row_records"]["schema_id"]
        or row_records["schema_version"]
        != bundle_contract["row_records"]["schema_version"]
        or row_records["row_fields"] != expected_row_fields
    ):
        raise PredictionTableValidationError(
            "condition comparison bundle row-record schema mismatch"
        )
    if not isinstance(row_records["rows"], list):
        raise PredictionTableValidationError(
            "condition comparison bundle row records must be an array"
        )
    for index, row in enumerate(row_records["rows"]):
        if not isinstance(row, dict):
            raise PredictionTableValidationError(
                f"condition comparison bundle row record {index} must be an object"
            )
        _require_exact_fields(
            row,
            expected_row_fields,
            f"condition comparison bundle row record {index}",
        )

    try:
        with probabilities_path.open("rb") as handle:
            probabilities = np.load(handle, allow_pickle=False)
            trailing = handle.read(1)
    except (OSError, ValueError) as exc:
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy is not a safe NumPy array: "
            f"{exc}"
        ) from exc
    if trailing:
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy has trailing bytes"
        )
    if not isinstance(probabilities, np.ndarray) or probabilities.dtype.str != "<f8":
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy must be little-endian float64"
        )
    if not probabilities.flags.c_contiguous:
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy must be C-contiguous"
        )
    if list(probabilities.shape) != manifest["probabilities_shape"]:
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy shape mismatch"
        )
    if not np.isfinite(probabilities).all():
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy contains non-finite values"
        )
    if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
        raise PredictionTableValidationError(
            "condition comparison bundle probabilities.npy must remain in [0, 1]"
        )
    if len(row_records["rows"]) != manifest["row_count"]:
        raise PredictionTableValidationError(
            "condition comparison bundle row-record count mismatch"
        )

    rows = []
    for index, record in enumerate(row_records["rows"]):
        probabilities_by_field = {
            field: float(probabilities[index, column])
            for column, field in enumerate(probability_fields)
        }
        rows.append({**deepcopy(record), **probabilities_by_field})
    table = {**metadata, "rows": rows}
    validate_condition_comparison_table(
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
    )
    logical_digest = condition_comparison_table_sha256(
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
    )
    if logical_digest != manifest["logical_table_sha256"]:
        raise PredictionTableValidationError(
            "condition comparison bundle reconstructed logical-table digest mismatch"
        )
    return table
