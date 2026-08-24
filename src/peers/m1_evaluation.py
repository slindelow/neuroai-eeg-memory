"""Outcome-blind bridge from one validated prediction table to subject-level metrics.

This module does not accept free-floating label/probability arrays. Its public evaluator first
validates the complete paired B3/M1/M3 table and then passes every canonical row to the frozen
threshold-free and probability metrics. D42-A deliberately omits balanced accuracy and D44-C
requires a qualified meta-prediction claim. Real evaluation remains gated by Sofia's protocol
teach-back and real artifact evidence.
"""

from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import roc_auc_score

from peers.evaluation import incremental_value
from peers.m1_predictions import (
    condition_comparison_table_sha256,
    learner_contract_sha256,
    prediction_table_sha256,
    validate_condition_comparison_table,
    validate_prediction_table,
)


SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class EvaluationConsumerValidationError(ValueError):
    """Raised when a subject evaluation record violates its frozen logical contract."""


def _canonical_object_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationConsumerValidationError(
            "evaluation object is not canonical finite JSON"
        ) from exc
    return (text + "\n").encode("utf-8")


def evaluation_contract_sha256(evaluation_contract: Mapping[str, Any]) -> str:
    """Bind a subject record to the complete machine-readable evaluation contract."""
    return sha256(_canonical_object_bytes(evaluation_contract)).hexdigest()


def _require_contract(evaluation_contract: Mapping[str, Any]) -> None:
    if evaluation_contract.get("contract_id") != "neuroai-m1-evaluation-consumer-v1":
        raise EvaluationConsumerValidationError("unexpected evaluation contract_id")
    if evaluation_contract.get("depends_on_resolved_scientific_choices") is not True:
        raise EvaluationConsumerValidationError(
            "evaluation contract must bind the resolved D42-A and D44-C choices"
        )
    if evaluation_contract.get("resolved_scientific_inputs") != [
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
    ]:
        raise EvaluationConsumerValidationError(
            "evaluation contract D42-A or D44-C scientific input drifted"
        )
    if evaluation_contract.get("scientific_choices_not_selected_by_this_contract") != []:
        raise EvaluationConsumerValidationError(
            "evaluation contract must not retain an unresolved scientific-choice gate"
        )
    subject_record = evaluation_contract.get("subject_record")
    if not isinstance(subject_record, Mapping):
        raise EvaluationConsumerValidationError("subject_record contract must be an object")
    if subject_record.get("schema_id") != "neuroai-m1-subject-evaluation-v1":
        raise EvaluationConsumerValidationError("unexpected subject evaluation schema_id")
    common_record = evaluation_contract.get("common_intersection_subject_record")
    if not isinstance(common_record, Mapping):
        raise EvaluationConsumerValidationError(
            "common_intersection_subject_record contract must be an object"
        )
    if common_record.get("schema_id") != (
        "neuroai-m1-common-intersection-subject-evaluation-v1"
    ):
        raise EvaluationConsumerValidationError(
            "unexpected common-intersection subject schema_id"
        )
    if common_record.get("condition_order") != ["ocular-removed", "ocular-kept"]:
        raise EvaluationConsumerValidationError(
            "common-intersection condition order must remain fixed"
        )
    if common_record.get("source_scope") != (
        "complete_validated_common_intersection_table"
    ):
        raise EvaluationConsumerValidationError(
            "common-intersection source scope must remain fixed"
        )
    metric_groups = common_record.get("metric_fields_by_condition")
    if not isinstance(metric_groups, Mapping) or set(metric_groups) != {
        "ocular-removed",
        "ocular-kept",
    }:
        raise EvaluationConsumerValidationError(
            "common-intersection metric groups must contain both conditions"
        )
    grouped_fields = []
    for condition in common_record["condition_order"]:
        fields = metric_groups[condition]
        if (
            not isinstance(fields, list)
            or len(fields) != 11
            or len(fields) != len(set(fields))
            or not all(isinstance(field, str) and field for field in fields)
        ):
            raise EvaluationConsumerValidationError(
                f"{condition} common-intersection metric group is invalid"
            )
        grouped_fields.extend(fields)
    if common_record.get("metric_fields") != grouped_fields:
        raise EvaluationConsumerValidationError(
            "common-intersection metric groups do not match flat fields"
        )
    for field in (
        "balanced_accuracy_allowed",
        "result_category_allowed",
        "condition_discordance_allowed",
    ):
        if common_record.get(field) is not False:
            raise EvaluationConsumerValidationError(
                f"common-intersection {field} must remain false"
            )
    metric_scope = evaluation_contract.get("metric_scope", {})
    if metric_scope.get("balanced_accuracy") != (
        "OMITTED_BY_D42_A_NOT_COMPUTED"
    ):
        raise EvaluationConsumerValidationError(
            "balanced accuracy must remain deliberately omitted by D42-A"
        )
    for record_name in ("subject_record", "common_intersection_subject_record"):
        record_contract = evaluation_contract.get(record_name)
        for field_group in ("exact_fields", "metric_fields"):
            fields = record_contract.get(field_group, [])
            if any("balanced_accuracy" in field for field in fields):
                raise EvaluationConsumerValidationError(
                    f"{record_name} must not contain balanced-accuracy fields under D42-A"
                )
    if metric_scope.get("ocular_common_intersection_subject_metrics") != (
        "IMPLEMENTED_SYNTHETIC_ONLY"
    ):
        raise EvaluationConsumerValidationError(
            "common-intersection subject metrics must remain synthetic-only"
        )
    if metric_scope.get("ocular_condition_discordance") != "IMPLEMENTED_SYNTHETIC_ONLY":
        raise EvaluationConsumerValidationError(
            "ocular discordance must remain synthetic-only"
        )
    primary_aggregate = evaluation_contract.get("primary_aggregate")
    if not isinstance(primary_aggregate, Mapping):
        raise EvaluationConsumerValidationError(
            "primary_aggregate contract must be an object"
        )
    if primary_aggregate.get("schema_id") != "neuroai-m1-primary-aggregate-v1":
        raise EvaluationConsumerValidationError("unexpected primary aggregate schema_id")
    common_aggregate = evaluation_contract.get("common_intersection_aggregate")
    if not isinstance(common_aggregate, Mapping):
        raise EvaluationConsumerValidationError(
            "common_intersection_aggregate contract must be an object"
        )
    if common_aggregate.get("schema_id") != (
        "neuroai-m1-common-intersection-aggregate-v1"
    ):
        raise EvaluationConsumerValidationError(
            "unexpected common-intersection aggregate schema_id"
        )
    if common_aggregate.get("condition_order") != [
        "ocular-removed",
        "ocular-kept",
    ]:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate condition order must remain fixed"
        )
    if common_aggregate.get("source_scope") != (
        "complete_exact_common_intersection_subject_record_set"
    ):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate source scope must remain fixed"
        )
    if common_aggregate.get("balanced_accuracy_included") is not False:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate must not include balanced accuracy"
        )


def _require_evaluation_provenance(provenance: Mapping[str, Any]) -> None:
    expected_fields = {
        "evaluator_code_commit",
        "evaluator_dirty_diff_sha256",
        "evaluator_package_versions_sha256",
        "evaluation_command",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_fields:
        raise EvaluationConsumerValidationError(
            "evaluation provenance fields do not match the closed contract"
        )
    code_commit = provenance["evaluator_code_commit"]
    if not isinstance(code_commit, str) or GIT_COMMIT_RE.fullmatch(code_commit) is None:
        raise EvaluationConsumerValidationError("evaluator_code_commit must be a git commit")
    dirty = provenance["evaluator_dirty_diff_sha256"]
    if dirty is not None and (
        not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None
    ):
        raise EvaluationConsumerValidationError(
            "evaluator_dirty_diff_sha256 must be a SHA-256 or null"
        )
    package_hash = provenance["evaluator_package_versions_sha256"]
    if not isinstance(package_hash, str) or SHA256_RE.fullmatch(package_hash) is None:
        raise EvaluationConsumerValidationError(
            "evaluator_package_versions_sha256 must be a SHA-256"
        )
    command = provenance["evaluation_command"]
    if not isinstance(command, str) or not command.strip():
        raise EvaluationConsumerValidationError("evaluation_command must be non-empty")


def _require_aggregate_provenance(provenance: Mapping[str, Any]) -> None:
    expected_fields = {
        "aggregator_code_commit",
        "aggregator_dirty_diff_sha256",
        "aggregator_package_versions_sha256",
        "aggregation_command",
    }
    if not isinstance(provenance, Mapping) or set(provenance) != expected_fields:
        raise EvaluationConsumerValidationError(
            "aggregate provenance fields do not match the closed contract"
        )
    code_commit = provenance["aggregator_code_commit"]
    if not isinstance(code_commit, str) or GIT_COMMIT_RE.fullmatch(code_commit) is None:
        raise EvaluationConsumerValidationError("aggregator_code_commit must be a git commit")
    dirty = provenance["aggregator_dirty_diff_sha256"]
    if dirty is not None and (
        not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None
    ):
        raise EvaluationConsumerValidationError(
            "aggregator_dirty_diff_sha256 must be a SHA-256 or null"
        )
    package_hash = provenance["aggregator_package_versions_sha256"]
    if not isinstance(package_hash, str) or SHA256_RE.fullmatch(package_hash) is None:
        raise EvaluationConsumerValidationError(
            "aggregator_package_versions_sha256 must be a SHA-256"
        )
    command = provenance["aggregation_command"]
    if not isinstance(command, str) or not command.strip():
        raise EvaluationConsumerValidationError("aggregation_command must be non-empty")


def _brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels.astype(float)) ** 2))


def _require_record_shape(
    record: Any,
    evaluation_contract: Mapping[str, Any],
) -> None:
    _require_contract(evaluation_contract)
    if not isinstance(record, Mapping):
        raise EvaluationConsumerValidationError("subject evaluation record must be an object")
    record_contract = evaluation_contract["subject_record"]
    expected_fields = set(record_contract["exact_fields"])
    if set(record) != expected_fields:
        missing = sorted(expected_fields - set(record))
        extra = sorted(set(record) - expected_fields)
        raise EvaluationConsumerValidationError(
            f"subject evaluation record fields mismatch; missing={missing}, extra={extra}"
        )
    if record["schema_id"] != record_contract["schema_id"]:
        raise EvaluationConsumerValidationError("subject evaluation schema_id mismatch")
    if record["schema_version"] != record_contract["schema_version"]:
        raise EvaluationConsumerValidationError("subject evaluation schema_version mismatch")

    string_fields = record_contract["string_fields"]
    for field in string_fields:
        if not isinstance(record[field], str) or not record[field]:
            raise EvaluationConsumerValidationError(f"{field} must be a non-empty string")
    for field in record_contract["sha256_fields"]:
        if SHA256_RE.fullmatch(record[field]) is None:
            raise EvaluationConsumerValidationError(f"{field} must be a SHA-256")
    code_commit = record["evaluator_code_commit"]
    if not isinstance(code_commit, str) or GIT_COMMIT_RE.fullmatch(code_commit) is None:
        raise EvaluationConsumerValidationError("evaluator_code_commit must be a git commit")
    dirty = record["evaluator_dirty_diff_sha256"]
    if dirty is not None and (
        not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None
    ):
        raise EvaluationConsumerValidationError(
            "evaluator_dirty_diff_sha256 must be a SHA-256 or null"
        )
    for field in record_contract["integer_fields"]:
        if type(record[field]) is not int or record[field] < 0:
            raise EvaluationConsumerValidationError(
                f"{field} must be a non-negative integer"
            )
    if record["n_trials"] != record["n_recalled"] + record["n_not_recalled"]:
        raise EvaluationConsumerValidationError("subject evaluation counts do not reconcile")
    for field in record_contract["metric_fields"]:
        if type(record[field]) is not float or not math.isfinite(record[field]):
            raise EvaluationConsumerValidationError(f"{field} must be a finite float64")


def evaluate_subject_prediction_table(
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    evaluation_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one paired table and calculate its complete subject-level metric record."""
    _require_contract(evaluation_contract)
    _require_evaluation_provenance(evaluation_provenance)
    validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
    )
    table_digest = prediction_table_sha256(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=expected_bindings,
    )

    rows = table["rows"]
    labels = np.asarray([row["recalled"] for row in rows], dtype=bool)
    prob_b3 = np.asarray([row["prob_b3"] for row in rows], dtype=float)
    prob_m1 = np.asarray([row["prob_m1"] for row in rows], dtype=float)
    prob_m3 = np.asarray([row["prob_m3"] for row in rows], dtype=float)
    metrics = incremental_value(labels, prob_b3, prob_m3)
    if not metrics:
        raise EvaluationConsumerValidationError(
            "subject evaluation requires both recall classes"
        )

    record = {
        "schema_id": evaluation_contract["subject_record"]["schema_id"],
        "schema_version": evaluation_contract["subject_record"]["schema_version"],
        "subject_id": table["subject_id"],
        "condition": table["condition"],
        "outer_fold_id": table["outer_fold_id"],
        "outer_test_session": table["outer_test_session"],
        "charter_sha256": table["charter_sha256"],
        "prediction_table_sha256": table_digest,
        "evaluation_contract_sha256": evaluation_contract_sha256(
            evaluation_contract
        ),
        **dict(evaluation_provenance),
        "n_trials": len(rows),
        "n_recalled": int(labels.sum()),
        "n_not_recalled": int((~labels).sum()),
        "auc_b3": float(metrics["auc_old"]),
        "auc_m1": float(roc_auc_score(labels, prob_m1)),
        "auc_m3": float(metrics["auc_new"]),
        "delta_auc_m3_minus_b3": float(metrics["delta_auc"]),
        "idi_m3_vs_b3": float(metrics["idi"]),
        "continuous_nri_m3_vs_b3": float(metrics["nri"]),
        "paired_delong_z_m3_vs_b3": float(metrics["delong_z"]),
        "paired_delong_p_m3_vs_b3": float(metrics["delong_p"]),
        "brier_b3": _brier_score(labels, prob_b3),
        "brier_m1": _brier_score(labels, prob_m1),
        "brier_m3": _brier_score(labels, prob_m3),
    }
    _require_record_shape(record, evaluation_contract)
    return record


def validate_subject_evaluation_record(
    record: Any,
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    expected_evaluation_provenance: Mapping[str, Any],
) -> None:
    """Recompute and compare a stored subject record against its complete paired table."""
    _require_record_shape(record, evaluation_contract)
    expected = evaluate_subject_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_bindings=expected_bindings,
        evaluation_provenance=expected_evaluation_provenance,
    )
    if _canonical_object_bytes(record) != _canonical_object_bytes(expected):
        raise EvaluationConsumerValidationError(
            "subject evaluation record does not match complete recomputation"
        )


def canonical_subject_evaluation_bytes(
    record: Any,
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    expected_evaluation_provenance: Mapping[str, Any],
) -> bytes:
    """Validate and serialize one subject evaluation record canonically."""
    validate_subject_evaluation_record(
        record,
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_bindings=expected_bindings,
        expected_evaluation_provenance=expected_evaluation_provenance,
    )
    return _canonical_object_bytes(record)


def subject_evaluation_sha256(
    record: Any,
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    expected_evaluation_provenance: Mapping[str, Any],
) -> str:
    """SHA-256 of a validated canonical subject evaluation record."""
    return sha256(
        canonical_subject_evaluation_bytes(
            record,
            table,
            trial_manifest,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            evaluation_contract,
            expected_bindings=expected_bindings,
            expected_evaluation_provenance=expected_evaluation_provenance,
        )
    ).hexdigest()


def subject_record_content_sha256(
    record: Any,
    evaluation_contract: Mapping[str, Any],
) -> str:
    """Hash a shaped subject record for comparison with an independent validated digest."""
    _require_record_shape(record, evaluation_contract)
    return sha256(_canonical_object_bytes(record)).hexdigest()


def _require_common_intersection_subject_shape(
    record: Any,
    evaluation_contract: Mapping[str, Any],
) -> None:
    _require_contract(evaluation_contract)
    if not isinstance(record, Mapping):
        raise EvaluationConsumerValidationError(
            "common-intersection subject record must be an object"
        )
    contract = evaluation_contract["common_intersection_subject_record"]
    expected_fields = set(contract["exact_fields"])
    if set(record) != expected_fields:
        missing = sorted(expected_fields - set(record))
        extra = sorted(set(record) - expected_fields)
        raise EvaluationConsumerValidationError(
            "common-intersection subject fields mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if record["schema_id"] != contract["schema_id"]:
        raise EvaluationConsumerValidationError(
            "common-intersection subject schema_id mismatch"
        )
    if record["schema_version"] != contract["schema_version"]:
        raise EvaluationConsumerValidationError(
            "common-intersection subject schema_version mismatch"
        )
    for field in contract["string_fields"]:
        if not isinstance(record[field], str) or not record[field]:
            raise EvaluationConsumerValidationError(
                f"{field} must be a non-empty string"
            )
    for field in contract["sha256_fields"]:
        value = record[field]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise EvaluationConsumerValidationError(f"{field} must be a SHA-256")
    code_commit = record["evaluator_code_commit"]
    if not isinstance(code_commit, str) or GIT_COMMIT_RE.fullmatch(code_commit) is None:
        raise EvaluationConsumerValidationError(
            "evaluator_code_commit must be a git commit"
        )
    dirty = record["evaluator_dirty_diff_sha256"]
    if dirty is not None and (
        not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None
    ):
        raise EvaluationConsumerValidationError(
            "evaluator_dirty_diff_sha256 must be a SHA-256 or null"
        )
    for field in contract["integer_fields"]:
        if type(record[field]) is not int or record[field] < 0:
            raise EvaluationConsumerValidationError(
                f"{field} must be a non-negative integer"
            )
    if record["n_trials"] == 0:
        raise EvaluationConsumerValidationError(
            "common-intersection subject record must contain trials"
        )
    if record["n_trials"] != record["n_recalled"] + record["n_not_recalled"]:
        raise EvaluationConsumerValidationError(
            "common-intersection subject counts do not reconcile"
        )
    metric_fields = []
    for condition in contract["condition_order"]:
        metric_fields.extend(contract["metric_fields_by_condition"][condition])
    if metric_fields != contract["metric_fields"]:
        raise EvaluationConsumerValidationError(
            "common-intersection metric field groups do not match the flat contract"
        )
    for field in contract["metric_fields"]:
        if type(record[field]) is not float or not math.isfinite(record[field]):
            raise EvaluationConsumerValidationError(
                f"{field} must be a finite float64"
            )


def _common_intersection_condition_metrics(
    rows: list[Mapping[str, Any]],
    prefix: str,
) -> dict[str, float]:
    labels = np.asarray([row["recalled"] for row in rows], dtype=bool)
    prob_b3 = np.asarray([row[f"{prefix}_prob_b3"] for row in rows], dtype=float)
    prob_m1 = np.asarray([row[f"{prefix}_prob_m1"] for row in rows], dtype=float)
    prob_m3 = np.asarray([row[f"{prefix}_prob_m3"] for row in rows], dtype=float)
    metrics = incremental_value(labels, prob_b3, prob_m3)
    if not metrics:
        raise EvaluationConsumerValidationError(
            "common-intersection subject evaluation requires both recall classes"
        )
    return {
        f"{prefix}_auc_b3": float(metrics["auc_old"]),
        f"{prefix}_auc_m1": float(roc_auc_score(labels, prob_m1)),
        f"{prefix}_auc_m3": float(metrics["auc_new"]),
        f"{prefix}_delta_auc_m3_minus_b3": float(metrics["delta_auc"]),
        f"{prefix}_idi_m3_vs_b3": float(metrics["idi"]),
        f"{prefix}_continuous_nri_m3_vs_b3": float(metrics["nri"]),
        f"{prefix}_paired_delong_z_m3_vs_b3": float(metrics["delong_z"]),
        f"{prefix}_paired_delong_p_m3_vs_b3": float(metrics["delong_p"]),
        f"{prefix}_brier_b3": _brier_score(labels, prob_b3),
        f"{prefix}_brier_m1": _brier_score(labels, prob_m1),
        f"{prefix}_brier_m3": _brier_score(labels, prob_m3),
    }


def evaluate_common_intersection_subject(
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
    evaluation_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    evaluation_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate both conditions on every validated common-intersection row."""
    _require_contract(evaluation_contract)
    _require_evaluation_provenance(evaluation_provenance)
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
    table_digest = condition_comparison_table_sha256(
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
    rows = table["rows"]
    labels = np.asarray([row["recalled"] for row in rows], dtype=bool)
    contract = evaluation_contract["common_intersection_subject_record"]
    record = {
        "schema_id": contract["schema_id"],
        "schema_version": contract["schema_version"],
        "subject_id": table["subject_id"],
        "outer_test_session": table["outer_test_session"],
        "ocular_removed_outer_fold_id": table["ocular_removed_outer_fold_id"],
        "ocular_kept_outer_fold_id": table["ocular_kept_outer_fold_id"],
        "charter_sha256": table["charter_sha256"],
        "condition_comparison_table_sha256": table_digest,
        "evaluation_contract_sha256": evaluation_contract_sha256(
            evaluation_contract
        ),
        **dict(evaluation_provenance),
        "n_trials": len(rows),
        "n_recalled": int(labels.sum()),
        "n_not_recalled": int((~labels).sum()),
        **_common_intersection_condition_metrics(rows, "ocular_removed"),
        **_common_intersection_condition_metrics(rows, "ocular_kept"),
    }
    _require_common_intersection_subject_shape(record, evaluation_contract)
    return record


def validate_common_intersection_subject_record(
    record: Any,
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
    evaluation_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    expected_evaluation_provenance: Mapping[str, Any],
) -> None:
    """Recompute a paired subject record from its complete D37 evidence."""
    _require_common_intersection_subject_shape(record, evaluation_contract)
    expected = evaluate_common_intersection_subject(
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
        evaluation_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
        evaluation_provenance=expected_evaluation_provenance,
    )
    if _canonical_object_bytes(record) != _canonical_object_bytes(expected):
        raise EvaluationConsumerValidationError(
            "common-intersection subject record does not match complete recomputation"
        )


def canonical_common_intersection_subject_bytes(
    record: Any,
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
    evaluation_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    expected_evaluation_provenance: Mapping[str, Any],
) -> bytes:
    """Validate and serialize one paired common-intersection subject record."""
    validate_common_intersection_subject_record(
        record,
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
        evaluation_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
        expected_evaluation_provenance=expected_evaluation_provenance,
    )
    return _canonical_object_bytes(record)


def common_intersection_subject_sha256(
    record: Any,
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
    evaluation_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    expected_evaluation_provenance: Mapping[str, Any],
) -> str:
    """SHA-256 of a fully validated paired common-intersection subject record."""
    return sha256(
        canonical_common_intersection_subject_bytes(
            record,
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
            evaluation_contract,
            expected_removed_bindings=expected_removed_bindings,
            expected_kept_bindings=expected_kept_bindings,
            expected_intersection_manifest_sha256=(
                expected_intersection_manifest_sha256
            ),
            expected_prediction_table_sha256_by_condition=(
                expected_prediction_table_sha256_by_condition
            ),
            expected_comparison_provenance=expected_comparison_provenance,
            expected_evaluation_provenance=expected_evaluation_provenance,
        )
    ).hexdigest()


def common_intersection_subject_content_sha256(
    record: Any,
    evaluation_contract: Mapping[str, Any],
) -> str:
    """Hash a shaped paired record for later independent aggregate binding."""
    _require_common_intersection_subject_shape(record, evaluation_contract)
    return sha256(_canonical_object_bytes(record)).hexdigest()


def _require_primary_aggregate_contract(
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_contract(evaluation_contract)
    learner_contract_sha256(charter, "M1")
    aggregate = evaluation_contract["primary_aggregate"]
    expected = {
        "condition": charter["estimands"]["primary_condition"],
        "estimand": charter["estimands"]["primary"],
        "uncertainty": charter["estimands"]["uncertainty"],
    }
    observed = {field: aggregate[field] for field in expected}
    if observed != expected:
        raise EvaluationConsumerValidationError(
            "primary aggregate contract does not match the charter estimand"
        )
    bootstrap = aggregate["bootstrap"]
    if bootstrap != {
        "unit": "subject",
        "resamples": charter["determinism"]["bootstrap_subject_resamples"],
        "seed": charter["determinism"]["bootstrap_seed"],
        "interval_percentiles": [2.5, 97.5],
        "numpy_percentile_method": "linear",
    }:
        raise EvaluationConsumerValidationError(
            "primary aggregate bootstrap does not match the charter"
        )
    if aggregate["secondary_can_create_positive_path"] is not (
        charter["estimands"]["secondary_can_create_positive_path"]
    ):
        raise EvaluationConsumerValidationError(
            "secondary positive-path rule does not match the charter"
        )
    if aggregate["equivalence_claim_allowed"] is not (
        charter["result_categories"]["equivalence_claim_allowed"]
    ):
        raise EvaluationConsumerValidationError(
            "equivalence rule does not match the charter"
        )
    return aggregate


def _require_expected_subject_digests(
    expected_record_sha256_by_subject: Mapping[str, str],
) -> None:
    if not isinstance(expected_record_sha256_by_subject, Mapping):
        raise EvaluationConsumerValidationError(
            "expected subject-record digests must be an object"
        )
    for subject_id, digest in expected_record_sha256_by_subject.items():
        if not isinstance(subject_id, str) or not subject_id:
            raise EvaluationConsumerValidationError(
                "expected subject identifier must be non-empty"
            )
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise EvaluationConsumerValidationError(
                "expected subject-record digest must be a SHA-256"
            )


def _validate_primary_inputs(
    records: Any,
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    if (
        not isinstance(expected_charter_sha256, str)
        or SHA256_RE.fullmatch(expected_charter_sha256) is None
    ):
        raise EvaluationConsumerValidationError(
            "expected_charter_sha256 must be a SHA-256"
        )
    _require_expected_subject_digests(expected_record_sha256_by_subject)
    if not isinstance(records, (list, tuple)) or not records:
        raise EvaluationConsumerValidationError(
            "primary aggregate requires a non-empty subject-record sequence"
        )

    by_subject: dict[str, Mapping[str, Any]] = {}
    evaluator_binding = None
    primary_condition = evaluation_contract["primary_aggregate"]["condition"]
    contract_digest = evaluation_contract_sha256(evaluation_contract)
    for record in records:
        _require_record_shape(record, evaluation_contract)
        subject_id = record["subject_id"]
        if subject_id in by_subject:
            raise EvaluationConsumerValidationError(
                "primary aggregate contains a duplicate subject"
            )
        if record["condition"] != primary_condition:
            raise EvaluationConsumerValidationError(
                "primary aggregate accepts ocular-removed records only"
            )
        if record["charter_sha256"] != expected_charter_sha256:
            raise EvaluationConsumerValidationError(
                "subject record charter_sha256 mismatch"
            )
        if record["evaluation_contract_sha256"] != contract_digest:
            raise EvaluationConsumerValidationError(
                "subject record evaluation_contract_sha256 mismatch"
            )
        binding = (
            record["evaluator_code_commit"],
            record["evaluator_dirty_diff_sha256"],
            record["evaluator_package_versions_sha256"],
            record["evaluation_command"],
        )
        if evaluator_binding is None:
            evaluator_binding = binding
        elif binding != evaluator_binding:
            raise EvaluationConsumerValidationError(
                "subject records use inconsistent evaluator provenance"
            )
        by_subject[subject_id] = record

    if set(by_subject) != set(expected_record_sha256_by_subject):
        raise EvaluationConsumerValidationError(
            "primary aggregate subject set does not match expected record digests"
        )
    bindings = []
    for subject_id in sorted(by_subject):
        observed_digest = subject_record_content_sha256(
            by_subject[subject_id], evaluation_contract
        )
        if observed_digest != expected_record_sha256_by_subject[subject_id]:
            raise EvaluationConsumerValidationError(
                f"{subject_id} subject evaluation digest mismatch"
            )
        bindings.append(
            {
                "subject_id": subject_id,
                "subject_evaluation_sha256": observed_digest,
            }
        )
    return [by_subject[subject_id] for subject_id in sorted(by_subject)], bindings


def _primary_category(low: float, high: float) -> str:
    if low > 0.0:
        return "positive_increment"
    if high < 0.0:
        return "adverse_increment"
    return "no_detectable_increment__imprecise"


def _require_primary_aggregate_shape(
    record: Any,
    evaluation_contract: Mapping[str, Any],
) -> None:
    _require_contract(evaluation_contract)
    if not isinstance(record, Mapping):
        raise EvaluationConsumerValidationError("primary aggregate record must be an object")
    contract = evaluation_contract["primary_aggregate"]
    expected_fields = set(contract["exact_fields"])
    if set(record) != expected_fields:
        missing = sorted(expected_fields - set(record))
        extra = sorted(set(record) - expected_fields)
        raise EvaluationConsumerValidationError(
            f"primary aggregate fields mismatch; missing={missing}, extra={extra}"
        )
    if record["schema_id"] != contract["schema_id"]:
        raise EvaluationConsumerValidationError("primary aggregate schema_id mismatch")
    if record["schema_version"] != contract["schema_version"]:
        raise EvaluationConsumerValidationError("primary aggregate schema_version mismatch")
    for field in contract["string_fields"]:
        if not isinstance(record[field], str) or not record[field]:
            raise EvaluationConsumerValidationError(f"{field} must be a non-empty string")
    for field in contract["sha256_fields"]:
        value = record[field]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise EvaluationConsumerValidationError(f"{field} must be a SHA-256")
    code_commit = record["aggregator_code_commit"]
    if not isinstance(code_commit, str) or GIT_COMMIT_RE.fullmatch(code_commit) is None:
        raise EvaluationConsumerValidationError("aggregator_code_commit must be a git commit")
    dirty = record["aggregator_dirty_diff_sha256"]
    if dirty is not None and (
        not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None
    ):
        raise EvaluationConsumerValidationError(
            "aggregator_dirty_diff_sha256 must be a SHA-256 or null"
        )
    for field in contract["integer_fields"]:
        if type(record[field]) is not int or record[field] < 0:
            raise EvaluationConsumerValidationError(
                f"{field} must be a non-negative integer"
            )
    for field in contract["metric_fields"]:
        if type(record[field]) is not float or not math.isfinite(record[field]):
            raise EvaluationConsumerValidationError(f"{field} must be a finite float64")
    if record["ci_low"] > record["ci_high"]:
        raise EvaluationConsumerValidationError("primary aggregate interval is reversed")
    bindings = record["subject_record_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise EvaluationConsumerValidationError(
            "subject_record_bindings must be a non-empty list"
        )
    binding_fields = set(contract["subject_binding_exact_fields"])
    subject_ids = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != binding_fields:
            raise EvaluationConsumerValidationError(
                "subject record binding fields do not match the contract"
            )
        subject_id = binding["subject_id"]
        digest = binding["subject_evaluation_sha256"]
        if not isinstance(subject_id, str) or not subject_id:
            raise EvaluationConsumerValidationError(
                "subject record binding subject_id must be non-empty"
            )
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise EvaluationConsumerValidationError(
                "subject record binding digest must be a SHA-256"
            )
        subject_ids.append(subject_id)
    if subject_ids != sorted(subject_ids) or len(subject_ids) != len(set(subject_ids)):
        raise EvaluationConsumerValidationError(
            "subject record bindings must be unique and canonically ordered"
        )
    if record["n_subjects"] != len(bindings):
        raise EvaluationConsumerValidationError(
            "primary aggregate n_subjects does not match bindings"
        )
    if record["primary_category"] not in contract["result_categories"]:
        raise EvaluationConsumerValidationError("unexpected primary result category")
    if record["secondary_can_create_positive_path"] is not False:
        raise EvaluationConsumerValidationError(
            "secondary metrics cannot create a positive path"
        )
    if record["equivalence_claim_allowed"] is not False:
        raise EvaluationConsumerValidationError("equivalence claims are forbidden")


def aggregate_primary_subject_records(
    records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    aggregate_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate the exact ocular-removed subject set under the charter's primary estimand."""
    contract = _require_primary_aggregate_contract(charter, evaluation_contract)
    _require_aggregate_provenance(aggregate_provenance)
    ordered_records, bindings = _validate_primary_inputs(
        records,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
    )
    values = np.asarray(
        [record["delta_auc_m3_minus_b3"] for record in ordered_records],
        dtype=float,
    )
    bootstrap = contract["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    draws = rng.choice(
        values,
        size=(bootstrap["resamples"], len(values)),
        replace=True,
    ).mean(axis=1)
    low, high = np.percentile(
        draws,
        bootstrap["interval_percentiles"],
        method=bootstrap["numpy_percentile_method"],
    )
    record = {
        "schema_id": contract["schema_id"],
        "schema_version": contract["schema_version"],
        "condition": contract["condition"],
        "charter_sha256": expected_charter_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256(
            evaluation_contract
        ),
        "subject_record_bindings": bindings,
        **dict(aggregate_provenance),
        "n_subjects": len(ordered_records),
        "bootstrap_unit": bootstrap["unit"],
        "bootstrap_resamples": bootstrap["resamples"],
        "bootstrap_seed": bootstrap["seed"],
        "bootstrap_interval_percentiles": bootstrap["interval_percentiles"],
        "bootstrap_percentile_method": bootstrap["numpy_percentile_method"],
        "mean_delta_auc_m3_minus_b3": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "primary_category": _primary_category(float(low), float(high)),
        "secondary_can_create_positive_path": False,
        "equivalence_claim_allowed": False,
    }
    _require_primary_aggregate_shape(record, evaluation_contract)
    return record


def validate_primary_aggregate_record(
    record: Any,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> None:
    """Recompute the complete primary aggregate from its expected subject evidence."""
    _require_primary_aggregate_shape(record, evaluation_contract)
    expected = aggregate_primary_subject_records(
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        aggregate_provenance=expected_aggregate_provenance,
    )
    if _canonical_object_bytes(record) != _canonical_object_bytes(expected):
        raise EvaluationConsumerValidationError(
            "primary aggregate record does not match complete recomputation"
        )


def canonical_primary_aggregate_bytes(
    record: Any,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> bytes:
    """Validate and serialize one primary aggregate record canonically."""
    validate_primary_aggregate_record(
        record,
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        expected_aggregate_provenance=expected_aggregate_provenance,
    )
    return _canonical_object_bytes(record)


def primary_aggregate_sha256(
    record: Any,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> str:
    """SHA-256 of a validated canonical primary aggregate record."""
    return sha256(
        canonical_primary_aggregate_bytes(
            record,
            subject_records,
            charter,
            evaluation_contract,
            expected_charter_sha256=expected_charter_sha256,
            expected_record_sha256_by_subject=expected_record_sha256_by_subject,
            expected_aggregate_provenance=expected_aggregate_provenance,
        )
    ).hexdigest()


def _require_common_intersection_aggregate_contract(
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_contract(evaluation_contract)
    learner_contract_sha256(charter, "M1")
    aggregate = evaluation_contract["common_intersection_aggregate"]
    condition_order = [
        charter["estimands"]["primary_condition"],
        charter["conditions"]["sensitivity"],
    ]
    if aggregate["condition_order"] != condition_order:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate conditions do not match the charter"
        )
    if aggregate["estimand"] != (
        "unweighted_mean_subject_auc_M3_minus_refit_B3_"
        "per_condition_on_exact_common_intersection"
    ):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate estimand must remain fixed"
        )
    if aggregate["uncertainty"] != (
        "95_percentile_paired_subject_bootstrap_interval_per_condition"
    ):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate uncertainty must remain fixed"
        )
    if aggregate["result_categories"] != (
        evaluation_contract["primary_aggregate"]["result_categories"]
    ):
        raise EvaluationConsumerValidationError(
            "common-intersection result categories must match the primary contract"
        )
    if aggregate["condition_discordance_rule"] != (
        charter["result_categories"]["condition_discordant"]
    ):
        raise EvaluationConsumerValidationError(
            "condition-discordance rule does not match the charter"
        )
    bootstrap = aggregate["bootstrap"]
    if bootstrap != {
        "unit": "subject",
        "paired_condition_resampling": True,
        "resamples": charter["determinism"]["bootstrap_subject_resamples"],
        "seed": charter["determinism"]["bootstrap_seed"],
        "interval_percentiles": [2.5, 97.5],
        "numpy_percentile_method": "linear",
    }:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate bootstrap does not match the charter"
        )
    fixed_flags = {
        "primary_category_unchanged_by_sensitivity": True,
        "secondary_can_create_positive_path": False,
        "equivalence_claim_allowed": False,
        "balanced_accuracy_included": False,
    }
    if any(aggregate.get(field) is not value for field, value in fixed_flags.items()):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate fixed flags do not match the charter"
        )
    return aggregate


def _validate_common_intersection_aggregate_inputs(
    records: Any,
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    if (
        not isinstance(expected_charter_sha256, str)
        or SHA256_RE.fullmatch(expected_charter_sha256) is None
    ):
        raise EvaluationConsumerValidationError(
            "expected_charter_sha256 must be a SHA-256"
        )
    _require_expected_subject_digests(expected_record_sha256_by_subject)
    if not isinstance(records, (list, tuple)) or not records:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate requires a non-empty subject-record sequence"
        )

    by_subject: dict[str, Mapping[str, Any]] = {}
    evaluator_binding = None
    contract_digest = evaluation_contract_sha256(evaluation_contract)
    for record in records:
        _require_common_intersection_subject_shape(record, evaluation_contract)
        subject_id = record["subject_id"]
        if subject_id in by_subject:
            raise EvaluationConsumerValidationError(
                "common-intersection aggregate contains a duplicate subject"
            )
        if record["charter_sha256"] != expected_charter_sha256:
            raise EvaluationConsumerValidationError(
                "common-intersection subject record charter_sha256 mismatch"
            )
        if record["evaluation_contract_sha256"] != contract_digest:
            raise EvaluationConsumerValidationError(
                "common-intersection subject record evaluation_contract_sha256 mismatch"
            )
        binding = (
            record["evaluator_code_commit"],
            record["evaluator_dirty_diff_sha256"],
            record["evaluator_package_versions_sha256"],
            record["evaluation_command"],
        )
        if evaluator_binding is None:
            evaluator_binding = binding
        elif binding != evaluator_binding:
            raise EvaluationConsumerValidationError(
                "common-intersection subject records use inconsistent evaluator provenance"
            )
        by_subject[subject_id] = record

    if set(by_subject) != set(expected_record_sha256_by_subject):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate subject set does not match expected record digests"
        )
    bindings = []
    for subject_id in sorted(by_subject):
        observed_digest = common_intersection_subject_content_sha256(
            by_subject[subject_id], evaluation_contract
        )
        if observed_digest != expected_record_sha256_by_subject[subject_id]:
            raise EvaluationConsumerValidationError(
                f"{subject_id} common-intersection subject evaluation digest mismatch"
            )
        bindings.append(
            {
                "subject_id": subject_id,
                "common_intersection_subject_evaluation_sha256": observed_digest,
            }
        )
    return [by_subject[subject_id] for subject_id in sorted(by_subject)], bindings


def _require_common_intersection_aggregate_shape(
    record: Any,
    evaluation_contract: Mapping[str, Any],
) -> None:
    _require_contract(evaluation_contract)
    if not isinstance(record, Mapping):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate record must be an object"
        )
    contract = evaluation_contract["common_intersection_aggregate"]
    expected_fields = set(contract["exact_fields"])
    if set(record) != expected_fields:
        missing = sorted(expected_fields - set(record))
        extra = sorted(set(record) - expected_fields)
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate fields mismatch; "
            f"missing={missing}, extra={extra}"
        )
    if record["schema_id"] != contract["schema_id"]:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate schema_id mismatch"
        )
    if record["schema_version"] != contract["schema_version"]:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate schema_version mismatch"
        )
    for field in contract["string_fields"]:
        if not isinstance(record[field], str) or not record[field]:
            raise EvaluationConsumerValidationError(f"{field} must be a non-empty string")
    for field in contract["sha256_fields"]:
        value = record[field]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise EvaluationConsumerValidationError(f"{field} must be a SHA-256")
    code_commit = record["aggregator_code_commit"]
    if not isinstance(code_commit, str) or GIT_COMMIT_RE.fullmatch(code_commit) is None:
        raise EvaluationConsumerValidationError(
            "aggregator_code_commit must be a git commit"
        )
    dirty = record["aggregator_dirty_diff_sha256"]
    if dirty is not None and (
        not isinstance(dirty, str) or SHA256_RE.fullmatch(dirty) is None
    ):
        raise EvaluationConsumerValidationError(
            "aggregator_dirty_diff_sha256 must be a SHA-256 or null"
        )
    for field in contract["integer_fields"]:
        if type(record[field]) is not int or record[field] < 0:
            raise EvaluationConsumerValidationError(
                f"{field} must be a non-negative integer"
            )
    if record["n_subjects"] == 0:
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate must contain subjects"
        )
    for field in contract["metric_fields"]:
        if type(record[field]) is not float or not math.isfinite(record[field]):
            raise EvaluationConsumerValidationError(f"{field} must be a finite float64")
    for prefix in ("ocular_removed", "ocular_kept"):
        if record[f"{prefix}_ci_low"] > record[f"{prefix}_ci_high"]:
            raise EvaluationConsumerValidationError(
                f"{prefix} common-intersection aggregate interval is reversed"
            )
        if record[f"{prefix}_category"] not in contract["result_categories"]:
            raise EvaluationConsumerValidationError(
                f"unexpected {prefix} result category"
            )
    for field in contract["boolean_fields"]:
        if type(record[field]) is not bool:
            raise EvaluationConsumerValidationError(f"{field} must be boolean")
    bindings = record["subject_record_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise EvaluationConsumerValidationError(
            "subject_record_bindings must be a non-empty list"
        )
    binding_fields = set(contract["subject_binding_exact_fields"])
    subject_ids = []
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != binding_fields:
            raise EvaluationConsumerValidationError(
                "common-intersection subject binding fields do not match the contract"
            )
        subject_id = binding["subject_id"]
        digest = binding["common_intersection_subject_evaluation_sha256"]
        if not isinstance(subject_id, str) or not subject_id:
            raise EvaluationConsumerValidationError(
                "common-intersection subject binding subject_id must be non-empty"
            )
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise EvaluationConsumerValidationError(
                "common-intersection subject binding digest must be a SHA-256"
            )
        subject_ids.append(subject_id)
    if subject_ids != sorted(subject_ids) or len(subject_ids) != len(set(subject_ids)):
        raise EvaluationConsumerValidationError(
            "common-intersection subject bindings must be unique and canonically ordered"
        )
    if record["n_subjects"] != len(bindings):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate n_subjects does not match bindings"
        )
    if record["condition_discordance_rule"] != contract["condition_discordance_rule"]:
        raise EvaluationConsumerValidationError(
            "condition-discordance rule does not match the contract"
        )
    required_flags = {
        "bootstrap_paired_condition_resampling": True,
        "primary_category_unchanged_by_sensitivity": True,
        "secondary_can_create_positive_path": False,
        "equivalence_claim_allowed": False,
        "balanced_accuracy_included": False,
    }
    if any(record[field] is not value for field, value in required_flags.items()):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate fixed flags do not match the contract"
        )


def aggregate_common_intersection_subject_records(
    records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    aggregate_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate both ocular conditions over one exact paired subject set."""
    contract = _require_common_intersection_aggregate_contract(
        charter, evaluation_contract
    )
    _require_aggregate_provenance(aggregate_provenance)
    ordered_records, bindings = _validate_common_intersection_aggregate_inputs(
        records,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
    )
    values = np.asarray(
        [
            [
                record["ocular_removed_delta_auc_m3_minus_b3"],
                record["ocular_kept_delta_auc_m3_minus_b3"],
            ]
            for record in ordered_records
        ],
        dtype=float,
    )
    bootstrap = contract["bootstrap"]
    rng = np.random.default_rng(bootstrap["seed"])
    indices = rng.integers(
        0,
        len(values),
        size=(bootstrap["resamples"], len(values)),
    )
    draws = values[indices].mean(axis=1)
    bounds = np.percentile(
        draws,
        bootstrap["interval_percentiles"],
        axis=0,
        method=bootstrap["numpy_percentile_method"],
    )
    means = values.mean(axis=0)
    removed_category = _primary_category(float(bounds[0, 0]), float(bounds[1, 0]))
    kept_category = _primary_category(float(bounds[0, 1]), float(bounds[1, 1]))
    opposite_signs = (means[0] > 0.0 and means[1] < 0.0) or (
        means[0] < 0.0 and means[1] > 0.0
    )
    record = {
        "schema_id": contract["schema_id"],
        "schema_version": contract["schema_version"],
        "charter_sha256": expected_charter_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256(
            evaluation_contract
        ),
        "subject_record_bindings": bindings,
        **dict(aggregate_provenance),
        "n_subjects": len(ordered_records),
        "bootstrap_unit": bootstrap["unit"],
        "bootstrap_paired_condition_resampling": bootstrap[
            "paired_condition_resampling"
        ],
        "bootstrap_resamples": bootstrap["resamples"],
        "bootstrap_seed": bootstrap["seed"],
        "bootstrap_interval_percentiles": bootstrap["interval_percentiles"],
        "bootstrap_percentile_method": bootstrap["numpy_percentile_method"],
        "ocular_removed_mean_delta_auc_m3_minus_b3": float(means[0]),
        "ocular_removed_ci_low": float(bounds[0, 0]),
        "ocular_removed_ci_high": float(bounds[1, 0]),
        "ocular_removed_category": removed_category,
        "ocular_kept_mean_delta_auc_m3_minus_b3": float(means[1]),
        "ocular_kept_ci_low": float(bounds[0, 1]),
        "ocular_kept_ci_high": float(bounds[1, 1]),
        "ocular_kept_category": kept_category,
        "condition_discordance_rule": contract["condition_discordance_rule"],
        "condition_discordant": bool(
            opposite_signs or removed_category != kept_category
        ),
        "primary_category_unchanged_by_sensitivity": True,
        "secondary_can_create_positive_path": False,
        "equivalence_claim_allowed": False,
        "balanced_accuracy_included": False,
    }
    _require_common_intersection_aggregate_shape(record, evaluation_contract)
    return record


def validate_common_intersection_aggregate_record(
    record: Any,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> None:
    """Recompute one paired ocular aggregate from its exact subject evidence."""
    _require_common_intersection_aggregate_shape(record, evaluation_contract)
    expected = aggregate_common_intersection_subject_records(
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        aggregate_provenance=expected_aggregate_provenance,
    )
    if _canonical_object_bytes(record) != _canonical_object_bytes(expected):
        raise EvaluationConsumerValidationError(
            "common-intersection aggregate record does not match complete recomputation"
        )


def canonical_common_intersection_aggregate_bytes(
    record: Any,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> bytes:
    """Validate and serialize one paired ocular aggregate canonically."""
    validate_common_intersection_aggregate_record(
        record,
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        expected_aggregate_provenance=expected_aggregate_provenance,
    )
    return _canonical_object_bytes(record)


def common_intersection_aggregate_sha256(
    record: Any,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> str:
    """SHA-256 of a validated canonical paired ocular aggregate."""
    return sha256(
        canonical_common_intersection_aggregate_bytes(
            record,
            subject_records,
            charter,
            evaluation_contract,
            expected_charter_sha256=expected_charter_sha256,
            expected_record_sha256_by_subject=expected_record_sha256_by_subject,
            expected_aggregate_provenance=expected_aggregate_provenance,
        )
    ).hexdigest()
