"""Outcome-blind structural contracts for future M1 manifests.

This module validates deterministic structure, provenance, and frozen split rules, including
the qualified D44-C session-out meta-prediction contract. It does not generate real manifests,
build EEG features, fit models, or inspect performance. Data-dependent reconciliation is added
only after the full charter is approved and hash-bound.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import ctypes
from datetime import datetime
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TRIAL_KEY_FIELDS = ("subject_id", "session", "trial", "item_num")
REQUIRED_PREDICTION_MODELS = ("B3", "M1", "M3")
CROSS_FIT_FIRST_LEVEL_MODELS = ("B3", "M1")
MANIFEST_BUNDLE_FILES = frozenset({"manifest.json", "manifest.sha256"})
DARWIN_RENAME_EXCL = 0x00000004
LINUX_AT_FDCWD = -100
LINUX_RENAME_NOREPLACE = 0x00000001
PROTOCOL_TEACH_BACK_TOPICS = (
    "inputs_and_outcome",
    "D26_generalization_target",
    "D29_window_midpoint_and_800ms_exclusion",
    "nested_session_cv_and_cross_fitting",
    "paired_trial_set_and_primary_estimand_condition",
    "leakage_controls",
    "exploratory_claim_ceiling_and_R4_disclosure",
    "failure_mode",
    "rejected_alternative_and_tradeoff",
)
PROTOCOL_TRANSCRIPT_FIELDS = (
    "schema_id",
    "schema_version",
    "transcript_id",
    "source_thread_id",
    "response_owner",
    "authorship_attestation",
    "capture_approval_message_id",
    "messages",
)
PROTOCOL_TRANSCRIPT_MESSAGE_FIELDS = (
    "message_id",
    "role",
    "topic",
    "kind",
    "text",
    "text_sha256",
    "in_reply_to_message_id",
    "disposition",
)
PROTOCOL_RECEIPT_FIELDS = (
    "schema_id",
    "schema_version",
    "receipt_id",
    "human_gate_id",
    "charter_id",
    "charter_version",
    "charter_sha256",
    "decision_log_sha256",
    "answered_by",
    "authorship_attestation",
    "source_thread_id",
    "source_transcript_sha256",
    "capture_approval_message_id",
    "response_message_ids",
    "recorded_at_utc",
    "responses",
    "responses_sha256",
    "advisor_review",
)
PROTOCOL_REVIEW_FIELDS = (
    "reviewed_at_utc",
    "reviewed_by_role",
    "topic_verdicts",
    "overall_verdict",
    "open_revision_requests",
)
D44_C_DECISION = "D44_C_outer_training_selected_C_qualified_meta_predictions"
D44_C_META_PREDICTION_CLAIM = (
    "session_out_coefficients_and_transforms_conditional_on_outer_training_selected_C"
)
D44_C_SELECTION_SCOPE = (
    "outer_training_wide_inner_session_C_selection_including_meta_prediction_session"
)
D44_C_PROVENANCE_SCHEMA_ID = "neuroai-m1-d44-c-cross-fit-provenance-v1"
D44_C_CHARTER_CROSS_FITTING = {
    "unit": "session",
    "in_sample_first_level_predictions_allowed": False,
    "record_row_fit_key_hashes": True,
    "outer_test_first_level_fit_scope": "all_outer_training_rows_only",
    "meta_fit_scope": D44_C_META_PREDICTION_CLAIM,
    "M1_C_selection_scope": D44_C_SELECTION_SCOPE,
    "predicted_session_labels_may_influence_selected_C": True,
    "outer_test_session_used_for_selection": False,
    "fully_session_cross_fitted_claim_allowed": False,
    "provenance_contract": D44_C_PROVENANCE_SCHEMA_ID,
}
D44_C_PROVENANCE_CONTRACT = {
    "schema_id": D44_C_PROVENANCE_SCHEMA_ID,
    "schema_version": "1.0",
    "decision": D44_C_DECISION,
    "meta_prediction_claim": D44_C_META_PREDICTION_CLAIM,
    "selection_scope": D44_C_SELECTION_SCOPE,
    "fully_session_cross_fitted_claim_allowed": False,
    "predicted_session_labels_may_influence_selected_C": True,
    "outer_test_session_used_for_selection": False,
    "one_selected_C_shared_by_all_meta_folds": True,
    "canonical_digest_required": True,
}
FROZEN_EXECUTION_GATES = {
    "preflight": {
        "candidate_scope": (
            "subjects_technically_eligible_in_both_conditions_before_outcome_access"
        ),
        "workload_sort": [
            "total_eligible_trials_across_both_conditions_ascending",
            "total_input_bytes_across_both_conditions_ascending",
            "subject_id_ascending",
        ],
        "selected_roles": {
            "median": "index_floor_n_minus_1_over_2",
            "maximum": "last_index",
        },
        "distinct_if_possible": True,
        "conditions": ["ocular-removed", "ocular-kept"],
        "outcome_metrics_or_interpretation_allowed": False,
        "required_measurements": [
            "stage_wall_seconds",
            "total_wall_seconds",
            "peak_resident_memory_bytes",
            "input_bytes",
            "cache_bytes_written",
            "disk_available_bytes_before_and_after",
            "parallel_worker_count",
            "longest_stage_wall_seconds",
            "longest_no_progress_seconds",
            "transient_retry_count",
            "scaled_full_run_forecast_wall_seconds",
            "projected_remaining_cache_bytes",
            "host_safe_memory_floor_bytes",
        ],
    },
    "full_locked_run": {
        "required_gates": [
            "validated_preflight_budget_receipt",
            "Sofia_capacity_go",
        ],
        "cutoff_source": (
            "Sofia_selected_current_run_cutoff_recorded_after_preflight"
        ),
        "per_subject_timeout_formula": (
            "2.0_times_maximum_subject_preflight_wall_seconds"
        ),
        "full_run_timeout_formula": (
            "1.3_times_scaled_preflight_forecast_wall_seconds"
        ),
        "effective_deadline": (
            "earlier_of_run_start_plus_full_run_timeout_and_Sofia_cutoff"
        ),
        "transient_retry_limit": 1,
        "no_progress_abort_formula": (
            "2.0_times_longest_preflight_stage_wall_seconds"
        ),
        "disk_headroom_floor_formula": (
            "1.2_times_projected_remaining_cache_bytes_plus_5_GiB"
        ),
        "memory_floor_source": (
            "host_safe_floor_recorded_in_preflight_receipt_before_full_run"
        ),
        "resume_policy": "atomically_completed_hash_valid_cache_entries_only",
        "partial_outcome_inspection_allowed": False,
    },
    "failure": {
        "on_compute_or_capacity_failure": "remain_IN_PROGRESS",
        "protocol_weakening_allowed": False,
        "record_failure_before_replan": True,
    },
}
TrialKey = tuple[str, int, int, int]


class ManifestValidationError(ValueError):
    """A manifest or pre-generation binding violates the frozen contract."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_unique(path: Path) -> dict[str, Any]:
    """Load JSON while rejecting duplicate object keys and non-finite numbers."""
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_raise(f"non-finite JSON value: {value}")),
        )
    except ManifestValidationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot load JSON contract {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ManifestValidationError(f"JSON document must be an object: {path}")
    return document


def _raise(message: str) -> None:
    raise ManifestValidationError(message)


def _validate_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ManifestValidationError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite(child, f"{path}[{index}]")


def _canonical_object_bytes(value: Any) -> bytes:
    _validate_finite(value)
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return serialized.encode("utf-8") + b"\n"


def canonical_document_sha256(document: Mapping[str, Any]) -> str:
    """Digest a duplicate-key-safe JSON object using the contract's canonical rules."""
    if not isinstance(document, Mapping):
        raise ManifestValidationError("canonical document must be an object")
    return sha256(_canonical_object_bytes(dict(document))).hexdigest()


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    """Digest one complete manifest row using the contract's canonical JSON rules."""
    return sha256(_canonical_object_bytes(dict(row))).hexdigest()


def canonical_trial_key(row: Mapping[str, Any]) -> TrialKey:
    """Return and validate the scientific identity of one studied-word presentation."""
    missing = [field for field in TRIAL_KEY_FIELDS if field not in row]
    if missing:
        raise ManifestValidationError(f"trial row is missing canonical key fields: {missing}")
    return _coerce_trial_key(tuple(row[field] for field in TRIAL_KEY_FIELDS), "trial row")


def _coerce_trial_key(value: Sequence[Any], label: str) -> TrialKey:
    if isinstance(value, (str, bytes)) or len(value) != len(TRIAL_KEY_FIELDS):
        raise ManifestValidationError(f"{label} key must contain {TRIAL_KEY_FIELDS}")
    subject_id, session, trial, item_num = value
    if not isinstance(subject_id, str) or not subject_id:
        raise ManifestValidationError(f"{label} key subject_id must be a nonempty string")
    if not _is_integer(session) or session < 0:
        raise ManifestValidationError(f"{label} key session must be a nonnegative integer")
    if not _is_integer(trial) or trial < 1:
        raise ManifestValidationError(f"{label} key trial must be a positive integer")
    if not _is_integer(item_num) or item_num < 1:
        raise ManifestValidationError(f"{label} key item_num must be a positive integer")
    return subject_id, session, trial, item_num


def _format_trial_key(key: TrialKey) -> str:
    return ",".join(f"{field}={value}" for field, value in zip(TRIAL_KEY_FIELDS, key, strict=True))


def _unique_trial_keys(values: Iterable[Sequence[Any]], name: str) -> set[TrialKey]:
    keys: set[TrialKey] = set()
    for raw_key in values:
        key = _coerce_trial_key(raw_key, name)
        if key in keys:
            raise ManifestValidationError(f"duplicate {name} key: {_format_trial_key(key)}")
        keys.add(key)
    return keys


def validate_exact_key_alignment(
    expected_keys: Iterable[Sequence[Any]],
    observed_keys: Iterable[Sequence[Any]],
    *,
    expected_name: str,
    observed_name: str,
) -> None:
    """Require two canonical trial-key collections to match exactly, ignoring order."""
    expected = _unique_trial_keys(expected_keys, expected_name)
    observed = _unique_trial_keys(observed_keys, observed_name)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        details = []
        if missing:
            formatted = "; ".join(_format_trial_key(key) for key in missing)
            details.append(f"missing from {observed_name}: [{formatted}]")
        if extra:
            formatted = "; ".join(_format_trial_key(key) for key in extra)
            details.append(f"extra in {observed_name}: [{formatted}]")
        raise ManifestValidationError(f"{observed_name} key-set mismatch; " + "; ".join(details))


def canonical_trial_key_set_sha256(keys: Iterable[Sequence[Any]]) -> str:
    """Digest a duplicate-free canonical trial-key set independently of input order."""
    canonical_keys = sorted(_unique_trial_keys(keys, "trial-key digest"))
    payload = [dict(zip(TRIAL_KEY_FIELDS, key, strict=True)) for key in canonical_keys]
    return sha256(_canonical_object_bytes(payload)).hexdigest()


def event_provenance_sha256(row: Mapping[str, Any]) -> str:
    """Bind a trial key to the behavioral event source shared by both EEG conditions."""
    source_id = row.get("source_events_id")
    source_sha = row.get("source_events_sha256")
    if not isinstance(source_id, str) or not source_id:
        raise ManifestValidationError("event provenance requires source_events_id")
    if not isinstance(source_sha, str) or not SHA256_RE.fullmatch(source_sha):
        raise ManifestValidationError("event provenance requires source_events_sha256")
    key = canonical_trial_key(row)
    payload = {
        "canonical_trial_key": dict(zip(TRIAL_KEY_FIELDS, key, strict=True)),
        "source_events_id": source_id,
        "source_events_sha256": source_sha,
    }
    return sha256(_canonical_object_bytes(payload)).hexdigest()


def _validate_forbidden_fields(value: Any, fragments: list[str], path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(fragment in lowered for fragment in fragments):
                raise ManifestValidationError(f"forbidden field at {path}.{key}")
            _validate_forbidden_fields(child, fragments, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_forbidden_fields(child, fragments, f"{path}[{index}]")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _matches_type(value: Any, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str) and bool(value)
    if declared == "integer":
        return _is_integer(value)
    if declared == "integer_or_null":
        return value is None or _is_integer(value)
    if declared == "number":
        return (isinstance(value, (int, float)) and not isinstance(value, bool))
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "object":
        return isinstance(value, dict)
    if declared == "object[integer]":
        return (
            isinstance(value, dict)
            and all(
                isinstance(key, str) and bool(key) and _is_integer(item)
                for key, item in value.items()
            )
        )
    if declared == "array[object]":
        return isinstance(value, list) and all(isinstance(item, dict) for item in value)
    if declared == "array[integer]":
        return isinstance(value, list) and all(_is_integer(item) for item in value)
    if declared == "sha256":
        return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))
    if declared == "sha256_or_null":
        return value is None or (isinstance(value, str) and bool(SHA256_RE.fullmatch(value)))
    if declared == "object[sha256]":
        return (
            isinstance(value, dict)
            and bool(value)
            and all(
                isinstance(item, str) and bool(SHA256_RE.fullmatch(item))
                for item in value.values()
            )
        )
    if declared == "git_commit":
        return isinstance(value, str) and bool(GIT_COMMIT_RE.fullmatch(value))
    if declared == "timestamp_utc":
        if not isinstance(value, str) or not value.endswith("Z"):
            return False
        try:
            parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
            return parsed.utcoffset() is not None
        except ValueError:
            return False
    raise ManifestValidationError(f"unknown declared field type: {declared}")


def _validate_types(values: dict[str, Any], declarations: dict[str, str], label: str) -> None:
    for field, declared in declarations.items():
        if not _matches_type(values[field], declared):
            actual = type(values[field]).__name__
            raise ManifestValidationError(
                f"{label} field {field} must have type {declared}; got {actual}"
            )


def _require_exact_fields(values: dict[str, Any], expected: list[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(values)
    missing = sorted(expected_set - actual_set)
    unknown = sorted(actual_set - expected_set)
    if missing:
        raise ManifestValidationError(f"missing {label} fields: {missing}")
    if unknown:
        raise ManifestValidationError(f"unknown {label} fields: {unknown}")


def _sorted_unique_integers(values: list[int], label: str) -> None:
    if values != sorted(set(values)):
        raise ManifestValidationError(f"{label} must be sorted and unique")


def _validate_conditions(rows: list[dict[str, Any]], contract: dict[str, Any]) -> None:
    allowed = set(contract["allowed_conditions"])
    for index, row in enumerate(rows):
        condition = row.get("condition")
        if condition is not None and condition not in allowed:
            raise ManifestValidationError(f"row {index} has invalid condition: {condition}")


def _validate_cohort(rows: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    allowed_reasons = set(schema["eligibility_reason_codes"])
    for row in rows:
        for field in ("sessions_present", "sessions_technically_eligible", "outer_train_sessions"):
            _sorted_unique_integers(row[field], field)
        present = set(row["sessions_present"])
        eligible_sessions = row["sessions_technically_eligible"]
        if not set(eligible_sessions).issubset(present):
            raise ManifestValidationError("technically eligible sessions must be present")
        reason = row["eligibility_reason_code"]
        if reason not in allowed_reasons:
            raise ManifestValidationError(f"unknown eligibility reason code: {reason}")
        if row["eligible"]:
            if reason != "ELIGIBLE":
                raise ManifestValidationError("eligible cohort row must use ELIGIBLE reason code")
            if len(eligible_sessions) < 3:
                raise ManifestValidationError(
                    "eligible cohort row requires at least three sessions"
                )
            expected_test = max(eligible_sessions)
            if row["outer_test_session"] != expected_test:
                raise ManifestValidationError(
                    "outer test must be the highest technically eligible session"
                )
            expected_train = [session for session in eligible_sessions if session != expected_test]
            if row["outer_train_sessions"] != expected_train:
                raise ManifestValidationError(
                    "outer train sessions must be every eligible non-test session"
                )
        elif reason == "ELIGIBLE":
            raise ManifestValidationError("ineligible cohort row cannot use ELIGIBLE reason code")


def _validate_trial(rows: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    allowed_roles = set(schema["outer_role_values"])
    epoch_identities: set[tuple[str, int]] = set()
    for row in rows:
        if row["session"] < 0 or row["trial"] < 1 or row["item_num"] < 1:
            raise ManifestValidationError(
                "included trial identity values must be nonnegative/positive"
            )
        if not 1 <= row["serial_position"] <= 16:
            raise ManifestValidationError("serial_position must be in 1..16")
        if row["list_number"] != row["trial"]:
            raise ManifestValidationError("list_number must equal trial")
        if row["outer_role"] not in allowed_roles:
            raise ManifestValidationError(f"invalid outer_role: {row['outer_role']}")
        if row["condition_survivor"] and not row["included"]:
            raise ManifestValidationError("condition survivor must be an included trial")
        if row["included"]:
            if row["epoch_row"] < 0:
                raise ManifestValidationError("included trial requires a nonnegative epoch_row")
            identity = (row["derivative_id"], row["epoch_row"])
            if identity in epoch_identities:
                raise ManifestValidationError(f"duplicate epoch identity: {identity}")
            epoch_identities.add(identity)


def _validate_class_counts(counts: dict[str, Any], label: str, require_both: bool) -> None:
    if set(counts) != {"0", "1"}:
        raise ManifestValidationError(f"{label} must contain exactly class keys 0 and 1")
    if not all(_is_integer(value) and value >= 0 for value in counts.values()):
        raise ManifestValidationError(f"{label} values must be nonnegative integers")
    if require_both and not all(value > 0 for value in counts.values()):
        raise ManifestValidationError(f"{label} must contain both classes")


def _validate_inner_folds(rows: list[dict[str, Any]], schema: dict[str, Any]) -> None:
    allowed_reasons = set(schema["feasibility_reason_codes"])
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        _sorted_unique_integers(row["fit_sessions"], "fit_sessions")
        if row["validation_session"] in row["fit_sessions"]:
            raise ManifestValidationError("validation session cannot appear in fit sessions")
        if row["outer_test_session"] == row["validation_session"]:
            raise ManifestValidationError("outer test session cannot be the validation session")
        if row["outer_test_session"] in row["fit_sessions"]:
            raise ManifestValidationError("outer test session cannot appear in fit sessions")
        reason = row["feasibility_reason_code"]
        if reason not in allowed_reasons:
            raise ManifestValidationError(f"unknown inner-fold feasibility reason: {reason}")
        if row["eligible"] and reason != "ELIGIBLE":
            raise ManifestValidationError("eligible inner fold must use ELIGIBLE reason code")
        if not row["eligible"] and reason == "ELIGIBLE":
            raise ManifestValidationError("ineligible inner fold cannot use ELIGIBLE reason code")
        _validate_class_counts(row["fit_class_counts"], "fit_class_counts", row["eligible"])
        _validate_class_counts(
            row["validation_class_counts"], "validation_class_counts", row["eligible"]
        )
        if row["eligible"]:
            groups[(row["subject_id"], row["condition"], row["outer_test_session"])].append(row)

    for key, group in groups.items():
        validation_sessions = [row["validation_session"] for row in group]
        if len(validation_sessions) < 2:
            raise ManifestValidationError(
                f"eligible inner-fold group requires at least two folds: {key}"
            )
        if len(set(validation_sessions)) != len(validation_sessions):
            raise ManifestValidationError(
                f"each training session must be validation exactly once: {key}"
            )
        full_training_set = set(validation_sessions)
        for row in group:
            expected_fit = sorted(full_training_set - {row["validation_session"]})
            if row["fit_sessions"] != expected_fit:
                raise ManifestValidationError(f"inner fold fit sessions do not reconcile: {key}")


def _validate_attrition(
    rows: list[dict[str, Any]], schema: dict[str, Any], contract: dict[str, Any]
) -> None:
    flows = schema.get("flows")
    if not isinstance(flows, dict) or not flows:
        raise ManifestValidationError("attrition contract requires declared flows")
    allowed_conditions = set(contract["allowed_conditions"])
    actual_conditions = {row["condition"] for row in rows}
    if schema.get("require_all_conditions") and actual_conditions != allowed_conditions:
        raise ManifestValidationError(
            "attrition stage set mismatch: all declared conditions are required"
        )

    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        reason_counts = row["excluded_by_reason"]
        counts = (row["input_count"], row["survivor_count"], *reason_counts.values())
        if any(value < 0 for value in counts):
            raise ManifestValidationError("attrition counts must be nonnegative")
        if any(value == 0 for value in reason_counts.values()):
            raise ManifestValidationError("zero-count attrition reasons must be omitted")
        grouped[(row["condition"], row["flow_id"])][row["stage_index"]] = row

    expected_flow_ids = set(flows)
    for condition in sorted(allowed_conditions):
        condition_flow_ids = {
            flow_id for row_condition, flow_id in grouped if row_condition == condition
        }
        if condition_flow_ids != expected_flow_ids:
            raise ManifestValidationError(
                f"attrition stage set mismatch for {condition}; "
                f"expected flows={sorted(expected_flow_ids)}, got={sorted(condition_flow_ids)}"
            )

        for flow_id, flow_spec in flows.items():
            stage_specs = flow_spec.get("stages")
            if not isinstance(stage_specs, list) or not stage_specs:
                raise ManifestValidationError(f"attrition flow has no stages: {flow_id}")
            expected_indices = [stage["stage_index"] for stage in stage_specs]
            if expected_indices != list(range(len(stage_specs))):
                raise ManifestValidationError(
                    f"attrition contract stage indices are invalid for {flow_id}"
                )
            rows_by_index = grouped[(condition, flow_id)]
            if set(rows_by_index) != set(expected_indices):
                raise ManifestValidationError(
                    f"attrition stage set mismatch for {condition}/{flow_id}"
                )

            prior_survivor: int | None = None
            for stage_spec in stage_specs:
                stage_index = stage_spec["stage_index"]
                row = rows_by_index[stage_index]
                if row["count_unit"] != flow_spec["count_unit"]:
                    raise ManifestValidationError(
                        f"attrition count_unit mismatch for {condition}/{flow_id}/{stage_index}"
                    )
                if row["stage"] != stage_spec["stage"]:
                    raise ManifestValidationError(
                        f"attrition stage name mismatch for {condition}/{flow_id}/{stage_index}"
                    )
                allowed_reasons = set(stage_spec["reason_code_order"])
                unknown_reasons = sorted(set(row["excluded_by_reason"]) - allowed_reasons)
                if unknown_reasons:
                    raise ManifestValidationError(
                        f"unknown attrition reason for {condition}/{flow_id}/{stage_index}: "
                        f"{unknown_reasons}"
                    )
                total_excluded = sum(row["excluded_by_reason"].values())
                if row["input_count"] - total_excluded != row["survivor_count"]:
                    raise ManifestValidationError(
                        f"attrition stage counts do not reconcile for "
                        f"{condition}/{flow_id}/{stage_index}"
                    )
                if prior_survivor is not None and row["input_count"] != prior_survivor:
                    raise ManifestValidationError(
                        f"attrition stage continuity mismatch for "
                        f"{condition}/{flow_id}/{stage_index}"
                    )
                prior_survivor = row["survivor_count"]

            input_binding = flow_spec.get("input_binding")
            if input_binding is not None:
                source_key = (condition, input_binding["flow_id"])
                source_stage = input_binding["stage_index"]
                try:
                    expected_input = grouped[source_key][source_stage]["survivor_count"]
                except KeyError as exc:
                    raise ManifestValidationError(
                        f"invalid attrition input binding for {condition}/{flow_id}"
                    ) from exc
                if rows_by_index[0]["input_count"] != expected_input:
                    raise ManifestValidationError(
                        f"attrition cross-flow input binding mismatch for {condition}/{flow_id}"
                    )


def _validate_structure(
    manifest: dict[str, Any], contract: dict[str, Any], *, require_canonical_order: bool
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ManifestValidationError("manifest must be an object")
    _validate_finite(manifest)
    _validate_forbidden_fields(manifest, contract["forbidden_field_fragments"])
    _require_exact_fields(manifest, contract["common_envelope_fields"], "envelope")
    _validate_types(manifest, contract["common_envelope_field_types"], "envelope")

    kind = manifest["manifest_kind"]
    try:
        schema = contract["manifest_schemas"][kind]
    except KeyError as exc:
        raise ManifestValidationError(f"unknown manifest kind: {kind}") from exc
    if manifest["schema_id"] != schema["schema_id"]:
        raise ManifestValidationError(
            f"schema_id mismatch for {kind}: expected {schema['schema_id']}"
        )
    if manifest["schema_version"] != schema["schema_version"]:
        raise ManifestValidationError(
            f"schema_version mismatch for {kind}: expected {schema['schema_version']}"
        )
    rows = manifest["rows"]
    _validate_conditions(rows, contract)
    for index, row in enumerate(rows):
        _require_exact_fields(row, schema["required_fields"], f"{kind} row {index}")
        _validate_types(row, schema["field_types"], f"{kind} row {index}")

    key_fields = schema["row_key"]
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ManifestValidationError(f"duplicate {kind} row key: {key}")
        seen.add(key)
    sorted_rows = sorted(rows, key=lambda row: tuple(row[field] for field in key_fields))
    if require_canonical_order and rows != sorted_rows:
        raise ManifestValidationError(f"{kind} rows are not in canonical row order")

    if kind == "cohort":
        _validate_cohort(rows, schema)
    elif kind == "trial":
        _validate_trial(rows, schema)
    elif kind == "inner_fold":
        _validate_inner_folds(rows, schema)
    elif kind == "attrition":
        _validate_attrition(rows, schema, contract)
    return schema


def _canonical_bytes_unchecked(manifest: dict[str, Any], schema: dict[str, Any]) -> bytes:
    canonical = deepcopy(manifest)
    key_fields = schema["row_key"]
    canonical["rows"] = sorted(
        canonical["rows"], key=lambda row: tuple(row[field] for field in key_fields)
    )
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return serialized.encode("utf-8") + b"\n"


def canonical_manifest_bytes(manifest: dict[str, Any], contract: dict[str, Any]) -> bytes:
    """Validate and serialize a manifest in deterministic row/key order."""
    schema = _validate_structure(manifest, contract, require_canonical_order=False)
    return _canonical_bytes_unchecked(manifest, schema)


def manifest_sha256(manifest: dict[str, Any], contract: dict[str, Any]) -> str:
    """SHA-256 of canonical manifest bytes (the future sidecar value)."""
    return sha256(canonical_manifest_bytes(manifest, contract)).hexdigest()


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination does not exist."""
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)

    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_noreplace = libc.renamex_np
        except AttributeError as exc:
            raise ManifestValidationError(
                "atomic no-replace rename is unavailable on this macOS runtime"
            ) from exc
        rename_noreplace.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_noreplace.restype = ctypes.c_int
        arguments = (source_bytes, destination_bytes, DARWIN_RENAME_EXCL)
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_noreplace = libc.renameat2
        except AttributeError as exc:
            raise ManifestValidationError(
                "atomic no-replace rename is unavailable on this Linux runtime"
            ) from exc
        rename_noreplace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_noreplace.restype = ctypes.c_int
        arguments = (
            LINUX_AT_FDCWD,
            source_bytes,
            LINUX_AT_FDCWD,
            destination_bytes,
            LINUX_RENAME_NOREPLACE,
        )
    else:
        raise ManifestValidationError(
            f"atomic no-replace rename is unsupported on platform {sys.platform!r}"
        )

    ctypes.set_errno(0)
    if rename_noreplace(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    error = OSError(error_number, os.strerror(error_number), os.fspath(destination))
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ManifestValidationError(
            f"manifest bundle already exists: {destination}"
        ) from error
    if error_number in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
        raise ManifestValidationError(
            f"filesystem does not support atomic no-replace rename: {destination}"
        ) from error
    raise ManifestValidationError(
        f"atomic no-replace manifest publication failed: {destination}: {error.strerror}"
    ) from error


def write_manifest_bundle_atomic(
    bundle_path: Path, manifest: dict[str, Any], contract: dict[str, Any]
) -> str:
    """Publish one immutable canonical manifest+digest directory with a single rename."""
    bundle_path = Path(bundle_path)
    parent = bundle_path.parent
    if not parent.is_dir():
        raise ManifestValidationError(f"manifest bundle parent is not a directory: {parent}")
    if os.path.lexists(bundle_path):
        raise ManifestValidationError(f"manifest bundle already exists: {bundle_path}")

    canonical = canonical_manifest_bytes(manifest, contract)
    digest = sha256(canonical).hexdigest()
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_path.name}.tmp-", dir=parent))
    try:
        _write_fsynced(temporary / "manifest.json", canonical)
        _write_fsynced(temporary / "manifest.sha256", f"{digest}\n".encode("ascii"))
        _fsync_directory(temporary)
        if os.path.lexists(bundle_path):
            raise ManifestValidationError(f"manifest bundle already exists: {bundle_path}")
        _rename_directory_noreplace(temporary, bundle_path)
        _fsync_directory(parent)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return digest


def load_manifest_bundle(
    bundle_path: Path,
    contract: dict[str, Any],
    *,
    expected_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read an immutable bundle and reject partial, extra, noncanonical, or tampered content."""
    bundle_path = Path(bundle_path)
    if not bundle_path.is_dir() or bundle_path.is_symlink():
        raise ManifestValidationError(f"manifest bundle is not a directory: {bundle_path}")
    actual_files = {path.name for path in bundle_path.iterdir()}
    if actual_files != MANIFEST_BUNDLE_FILES:
        raise ManifestValidationError(
            "manifest bundle must contain exactly manifest.json and manifest.sha256"
        )
    manifest_path = bundle_path / "manifest.json"
    sidecar_path = bundle_path / "manifest.sha256"
    if (
        manifest_path.is_symlink()
        or sidecar_path.is_symlink()
        or not manifest_path.is_file()
        or not sidecar_path.is_file()
    ):
        raise ManifestValidationError("manifest bundle entries must be regular files")

    sidecar = sidecar_path.read_bytes()
    if len(sidecar) != 65 or not sidecar.endswith(b"\n"):
        raise ManifestValidationError("manifest sidecar must be one lowercase SHA-256 plus newline")
    try:
        digest = sidecar[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError("manifest sidecar is not ASCII") from exc
    if not SHA256_RE.fullmatch(digest):
        raise ManifestValidationError("manifest sidecar is not a lowercase SHA-256")

    raw_manifest = manifest_path.read_bytes()
    manifest = load_json_unique(manifest_path)
    canonical = canonical_manifest_bytes(manifest, contract)
    if raw_manifest != canonical:
        raise ManifestValidationError("stored manifest is not canonical bytes")
    validate_manifest(
        manifest,
        contract,
        expected_sha256=digest,
        expected_bindings=expected_bindings,
    )
    return manifest


def validate_manifest(
    manifest: dict[str, Any],
    contract: dict[str, Any],
    *,
    expected_sha256: str | None = None,
    expected_bindings: dict[str, Any] | None = None,
) -> None:
    """Validate a stored manifest, including its canonical order and optional sidecar digest."""
    schema = _validate_structure(manifest, contract, require_canonical_order=True)
    if expected_bindings is not None:
        allowed = set(contract["binding_fields"])
        unknown = sorted(set(expected_bindings) - allowed)
        if unknown:
            raise ManifestValidationError(f"unknown expected binding fields: {unknown}")
        for field, expected in expected_bindings.items():
            if manifest[field] != expected:
                raise ManifestValidationError(
                    f"binding mismatch for {field}: expected {expected}, got {manifest[field]}"
                )
    if expected_sha256 is not None:
        if not SHA256_RE.fullmatch(expected_sha256):
            raise ManifestValidationError("expected digest is not a lowercase SHA-256")
        actual = sha256(_canonical_bytes_unchecked(manifest, schema)).hexdigest()
        if actual != expected_sha256:
            raise ManifestValidationError(
                f"digest mismatch: expected {expected_sha256}, computed {actual}"
            )


def validate_cross_condition_intersection(
    intersection_manifest: dict[str, Any],
    ocular_removed_manifest: dict[str, Any],
    ocular_kept_manifest: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Recompute the exact shared-survivor manifest from two trial manifests."""
    validate_manifest(intersection_manifest, contract)
    validate_manifest(ocular_removed_manifest, contract)
    validate_manifest(ocular_kept_manifest, contract)

    removed_rows = ocular_removed_manifest["rows"]
    kept_rows = ocular_kept_manifest["rows"]
    if any(row["condition"] != "ocular-removed" for row in removed_rows):
        raise ManifestValidationError(
            "ocular-removed trial manifest contains a different condition"
        )
    if any(row["condition"] != "ocular-kept" for row in kept_rows):
        raise ManifestValidationError("ocular-kept trial manifest contains a different condition")

    removed_by_key = {
        canonical_trial_key(row): row for row in removed_rows if row["condition_survivor"]
    }
    kept_by_key = {
        canonical_trial_key(row): row for row in kept_rows if row["condition_survivor"]
    }
    common_keys = set(removed_by_key) & set(kept_by_key)
    intersection_by_key = {
        canonical_trial_key(row): row for row in intersection_manifest["rows"]
    }
    validate_exact_key_alignment(
        common_keys,
        intersection_by_key,
        expected_name="cross-condition survivor intersection",
        observed_name="stored intersection",
    )

    for key in sorted(common_keys):
        removed = removed_by_key[key]
        kept = kept_by_key[key]
        stored = intersection_by_key[key]
        formatted_key = _format_trial_key(key)

        if removed["recalled"] != kept["recalled"]:
            raise ManifestValidationError(f"recall label disagreement for {formatted_key}")
        if stored["recalled"] != removed["recalled"]:
            raise ManifestValidationError(
                f"stored intersection recall label mismatch for {formatted_key}"
            )

        removed_event_hash = event_provenance_sha256(removed)
        kept_event_hash = event_provenance_sha256(kept)
        if removed_event_hash != kept_event_hash:
            raise ManifestValidationError(
                f"event provenance disagreement between conditions for {formatted_key}"
            )
        if stored["event_provenance_sha256"] != removed_event_hash:
            raise ManifestValidationError(
                f"event provenance hash mismatch for {formatted_key}"
            )
        if stored["ocular_removed_row_sha256"] != canonical_row_sha256(removed):
            raise ManifestValidationError(f"ocular-removed row hash mismatch for {formatted_key}")
        if stored["ocular_kept_row_sha256"] != canonical_row_sha256(kept):
            raise ManifestValidationError(f"ocular-kept row hash mismatch for {formatted_key}")
        if stored["ocular_removed_epoch_row"] != removed["epoch_row"]:
            raise ManifestValidationError(
                f"ocular-removed epoch row mismatch for {formatted_key}"
            )
        if stored["ocular_kept_epoch_row"] != kept["epoch_row"]:
            raise ManifestValidationError(f"ocular-kept epoch row mismatch for {formatted_key}")


def validate_prediction_key_sets(
    trial_manifest: dict[str, Any],
    prediction_keys_by_model: Mapping[str, Iterable[Sequence[Any]]],
    contract: dict[str, Any],
) -> None:
    """Require paired B3/M1/M3 predictions for every outer-test survivor, exactly once."""
    validate_manifest(trial_manifest, contract)
    supplied_models = set(prediction_keys_by_model)
    required_models = set(REQUIRED_PREDICTION_MODELS)
    if supplied_models != required_models:
        missing = sorted(required_models - supplied_models)
        extra = sorted(supplied_models - required_models)
        raise ManifestValidationError(
            f"prediction model set mismatch; missing={missing}; extra={extra}"
        )

    expected_keys = [
        canonical_trial_key(row)
        for row in trial_manifest["rows"]
        if row["condition_survivor"] and row["outer_role"] == "test"
    ]
    for model in REQUIRED_PREDICTION_MODELS:
        validate_exact_key_alignment(
            expected_keys,
            prediction_keys_by_model[model],
            expected_name="outer-test survivors",
            observed_name=f"{model} predictions",
        )


def validate_cross_fit_key_provenance(
    trial_manifest: dict[str, Any],
    inner_fold_manifest: dict[str, Any],
    fold_keys_by_model: Mapping[
        str, Mapping[str, Mapping[str, Iterable[Sequence[Any]]]]
    ],
    contract: dict[str, Any],
) -> None:
    """Prove first-level model fit/prediction key separation for one subject-condition."""
    validate_manifest(trial_manifest, contract)
    validate_manifest(inner_fold_manifest, contract)

    trial_rows = trial_manifest["rows"]
    fold_rows = [row for row in inner_fold_manifest["rows"] if row["eligible"]]
    trial_groups = {(row["subject_id"], row["condition"]) for row in trial_rows}
    fold_groups = {(row["subject_id"], row["condition"]) for row in fold_rows}
    if len(trial_groups) != 1 or fold_groups != trial_groups:
        raise ManifestValidationError(
            "cross-fit provenance requires one matching subject-condition group"
        )
    if not fold_rows:
        raise ManifestValidationError("cross-fit provenance requires eligible inner folds")

    outer_test_sessions = {row["outer_test_session"] for row in fold_rows}
    if len(outer_test_sessions) != 1:
        raise ManifestValidationError("cross-fit folds must share one outer test session")
    outer_test_session = next(iter(outer_test_sessions))
    outer_train_rows = [
        row
        for row in trial_rows
        if row["condition_survivor"] and row["outer_role"] == "train"
    ]
    if any(row["session"] == outer_test_session for row in outer_train_rows):
        raise ManifestValidationError("outer test key appears in outer training survivors")

    fold_by_id = {row["inner_fold_id"]: row for row in fold_rows}
    validation_sessions = {row["validation_session"] for row in fold_rows}
    outer_training_sessions = {row["session"] for row in outer_train_rows}
    if validation_sessions != outer_training_sessions:
        raise ManifestValidationError(
            "inner-fold validation sessions do not equal outer-training survivor sessions"
        )

    supplied_models = set(fold_keys_by_model)
    required_models = set(CROSS_FIT_FIRST_LEVEL_MODELS)
    if supplied_models != required_models:
        missing = sorted(required_models - supplied_models)
        extra = sorted(supplied_models - required_models)
        raise ManifestValidationError(
            f"cross-fit model set mismatch; missing={missing}; extra={extra}"
        )

    expected_fold_ids = set(fold_by_id)
    all_training_keys = [canonical_trial_key(row) for row in outer_train_rows]
    for model in CROSS_FIT_FIRST_LEVEL_MODELS:
        model_folds = fold_keys_by_model[model]
        if set(model_folds) != expected_fold_ids:
            missing = sorted(expected_fold_ids - set(model_folds))
            extra = sorted(set(model_folds) - expected_fold_ids)
            raise ManifestValidationError(
                f"{model} cross-fit fold set mismatch; missing={missing}; extra={extra}"
            )

        all_predicted_keys: list[Sequence[Any]] = []
        for fold_id in sorted(expected_fold_ids):
            fold = fold_by_id[fold_id]
            payload = model_folds[fold_id]
            if set(payload) != {"fit_keys", "predicted_keys"}:
                raise ManifestValidationError(
                    f"{model} {fold_id} provenance requires exactly fit_keys and predicted_keys"
                )
            supplied_fit_keys = list(payload["fit_keys"])
            supplied_predicted_keys = list(payload["predicted_keys"])
            expected_fit_keys = [
                canonical_trial_key(row)
                for row in outer_train_rows
                if row["session"] in fold["fit_sessions"]
            ]
            expected_predicted_keys = [
                canonical_trial_key(row)
                for row in outer_train_rows
                if row["session"] == fold["validation_session"]
            ]
            validate_exact_key_alignment(
                expected_fit_keys,
                supplied_fit_keys,
                expected_name=f"{model} {fold_id} expected fit keys",
                observed_name=f"{model} {fold_id} fit keys",
            )
            validate_exact_key_alignment(
                expected_predicted_keys,
                supplied_predicted_keys,
                expected_name=f"{model} {fold_id} expected predicted keys",
                observed_name=f"{model} {fold_id} predicted keys",
            )
            if fold["fit_keys_sha256"] != canonical_trial_key_set_sha256(expected_fit_keys):
                raise ManifestValidationError(f"fit_keys_sha256 mismatch for {fold_id}")
            if fold["validation_keys_sha256"] != canonical_trial_key_set_sha256(
                expected_predicted_keys
            ):
                raise ManifestValidationError(f"validation_keys_sha256 mismatch for {fold_id}")
            all_predicted_keys.extend(supplied_predicted_keys)

        validate_exact_key_alignment(
            all_training_keys,
            all_predicted_keys,
            expected_name="outer-training survivors",
            observed_name=f"{model} cross-fit predictions",
        )


def _require_d44_c_charter_contract(
    charter: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Require Option C's exact qualified claim and forbid a fully cross-fitted claim."""
    if charter.get("scientific_decisions", {}).get("D44") != D44_C_DECISION:
        raise ManifestValidationError("D44-C decision is not fixed in the charter")
    if charter.get("cross_fitting") != D44_C_CHARTER_CROSS_FITTING:
        raise ManifestValidationError("D44-C charter cross-fitting semantics drifted")
    if (
        charter.get("learners", {}).get("M3", {}).get("meta_training_predictions")
        != D44_C_META_PREDICTION_CLAIM
    ):
        raise ManifestValidationError("D44-C qualified M3 claim drifted")
    if contract.get("D44_C_cross_fit_provenance") != D44_C_PROVENANCE_CONTRACT:
        raise ManifestValidationError("D44-C provenance contract drifted")


def validate_d44_c_cross_fit_provenance(
    trial_manifest: dict[str, Any],
    inner_fold_manifest: dict[str, Any],
    provenance: Any,
    charter: Mapping[str, Any],
    contract: dict[str, Any],
) -> None:
    """Validate Option C without misrepresenting its meta-predictions as fully cross-fitted."""
    _require_d44_c_charter_contract(charter, contract)
    if not isinstance(provenance, dict):
        raise ManifestValidationError("D44-C provenance must be an object")
    _validate_finite(provenance)
    _require_exact_fields(
        provenance,
        [
            "schema_id",
            "schema_version",
            "subject_id",
            "condition",
            "outer_test_session",
            "decision",
            "meta_prediction_claim",
            "fully_session_cross_fitted",
            "fold_keys_by_model",
            "m1_hyperparameter_selection",
        ],
        "D44-C provenance",
    )
    fixed_values = {
        "schema_id": D44_C_PROVENANCE_SCHEMA_ID,
        "schema_version": "1.0",
        "decision": D44_C_DECISION,
        "meta_prediction_claim": D44_C_META_PREDICTION_CLAIM,
        "fully_session_cross_fitted": False,
    }
    for field, expected in fixed_values.items():
        if provenance[field] != expected:
            raise ManifestValidationError(f"D44-C provenance {field} drifted")

    validate_cross_fit_key_provenance(
        trial_manifest,
        inner_fold_manifest,
        provenance["fold_keys_by_model"],
        contract,
    )
    trial_rows = trial_manifest["rows"]
    trial_groups = {(row["subject_id"], row["condition"]) for row in trial_rows}
    if len(trial_groups) != 1:
        raise ManifestValidationError(
            "D44-C provenance requires one subject-condition trial group"
        )
    subject_id, condition = next(iter(trial_groups))
    if provenance["subject_id"] != subject_id or provenance["condition"] != condition:
        raise ManifestValidationError("D44-C subject-condition binding mismatch")

    eligible_folds = [row for row in inner_fold_manifest["rows"] if row["eligible"]]
    outer_test_sessions = {row["outer_test_session"] for row in eligible_folds}
    if outer_test_sessions != {provenance["outer_test_session"]}:
        raise ManifestValidationError("D44-C outer-test session binding mismatch")
    outer_test_session = provenance["outer_test_session"]
    outer_train_rows = [
        row
        for row in trial_rows
        if row["condition_survivor"] and row["outer_role"] == "train"
    ]
    outer_train_sessions = sorted({row["session"] for row in outer_train_rows})
    outer_train_keys = [canonical_trial_key(row) for row in outer_train_rows]
    if outer_test_session in outer_train_sessions:
        raise ManifestValidationError("D44-C outer test entered outer training")

    selection = provenance["m1_hyperparameter_selection"]
    if not isinstance(selection, dict):
        raise ManifestValidationError("D44-C M1 selection provenance must be an object")
    _require_exact_fields(
        selection,
        [
            "scope",
            "candidate_C_grid",
            "selection_metric",
            "tie_break",
            "selected_C",
            "fit_sessions",
            "validation_sessions",
            "fit_keys_sha256",
            "validation_keys_sha256",
            "inner_fold_manifest_sha256",
            "predicted_sessions_in_selection_validation",
            "predicted_session_labels_may_influence_selected_C",
            "outer_test_session_used_for_selection",
            "one_selected_C_shared_by_all_meta_folds",
        ],
        "D44-C M1 selection provenance",
    )
    try:
        candidate_grid = charter["learners"]["M1"]["C_grid"]
        selection_metric = charter["splits"]["selection_metric"]
        tie_break = charter["splits"]["tie_break"]
    except (KeyError, TypeError) as exc:
        raise ManifestValidationError("D44-C charter selection contract is incomplete") from exc
    if selection["scope"] != D44_C_SELECTION_SCOPE:
        raise ManifestValidationError("D44-C selection scope drifted")
    if selection["candidate_C_grid"] != candidate_grid:
        raise ManifestValidationError("D44-C candidate C grid drifted")
    selected_c = selection["selected_C"]
    if (
        not isinstance(selected_c, (int, float))
        or isinstance(selected_c, bool)
        or not math.isfinite(float(selected_c))
        or selected_c not in candidate_grid
    ):
        raise ManifestValidationError("D44-C selected C is outside the frozen grid")
    if selection["selection_metric"] != selection_metric:
        raise ManifestValidationError("D44-C selection metric drifted")
    if selection["tie_break"] != tie_break:
        raise ManifestValidationError("D44-C tie rule drifted")
    for field in (
        "fit_sessions",
        "validation_sessions",
        "predicted_sessions_in_selection_validation",
    ):
        if selection[field] != outer_train_sessions:
            raise ManifestValidationError(
                f"D44-C {field} must equal every outer-training session"
            )
    outer_train_key_digest = canonical_trial_key_set_sha256(outer_train_keys)
    if selection["fit_keys_sha256"] != outer_train_key_digest:
        raise ManifestValidationError("D44-C selection fit-key digest mismatch")
    if selection["validation_keys_sha256"] != outer_train_key_digest:
        raise ManifestValidationError("D44-C selection validation-key digest mismatch")
    if selection["inner_fold_manifest_sha256"] != manifest_sha256(
        inner_fold_manifest, contract
    ):
        raise ManifestValidationError("D44-C inner-fold manifest digest mismatch")
    required_flags = {
        "predicted_session_labels_may_influence_selected_C": True,
        "outer_test_session_used_for_selection": False,
        "one_selected_C_shared_by_all_meta_folds": True,
    }
    for field, expected in required_flags.items():
        if selection[field] is not expected:
            raise ManifestValidationError(f"D44-C {field} drifted")


def canonical_d44_c_cross_fit_provenance_bytes(
    trial_manifest: dict[str, Any],
    inner_fold_manifest: dict[str, Any],
    provenance: Any,
    charter: Mapping[str, Any],
    contract: dict[str, Any],
) -> bytes:
    """Return canonical bytes only after validating the complete D44-C record."""
    validate_d44_c_cross_fit_provenance(
        trial_manifest, inner_fold_manifest, provenance, charter, contract
    )
    return _canonical_object_bytes(dict(provenance))


def d44_c_cross_fit_provenance_sha256(
    trial_manifest: dict[str, Any],
    inner_fold_manifest: dict[str, Any],
    provenance: Any,
    charter: Mapping[str, Any],
    contract: dict[str, Any],
) -> str:
    """Digest a validated D44-C record for downstream prediction-table binding."""
    payload = canonical_d44_c_cross_fit_provenance_bytes(
        trial_manifest, inner_fold_manifest, provenance, charter, contract
    )
    return sha256(payload).hexdigest()


def validate_roi_binding(charter: dict[str, Any], roi_map: dict[str, Any]) -> None:
    """Prove that the charter and frozen ROI companion name the same non-overlapping map."""
    features = charter.get("features", {})
    if features.get("roi_map_id") != roi_map.get("map_id"):
        raise ManifestValidationError("ROI map identifier does not match charter")
    order = roi_map.get("roi_order")
    if order != features.get("roi_order"):
        raise ManifestValidationError("ROI order does not match charter")
    rois = roi_map.get("rois")
    if not isinstance(rois, dict) or set(rois) != set(order or []):
        raise ManifestValidationError("ROI objects do not match declared ROI order")
    constraints = roi_map.get("constraints", {})
    if len(order) != constraints.get("roi_count"):
        raise ManifestValidationError("ROI count does not match frozen constraint")

    all_channels: list[str] = []
    for roi_id in order:
        channels = rois[roi_id].get("channels")
        if channels != features.get("roi_channels", {}).get(roi_id):
            raise ManifestValidationError(f"ROI channels do not match charter: {roi_id}")
        wrong_count = len(channels) != constraints.get("channels_per_roi")
        repeated_within_roi = len(set(channels)) != len(channels)
        if wrong_count or repeated_within_roi:
            raise ManifestValidationError(f"ROI channel cardinality is invalid: {roi_id}")
        all_channels.extend(channels)
    if len(set(all_channels)) != len(all_channels):
        raise ManifestValidationError("ROI channel overlap is forbidden")
    if len(all_channels) != constraints.get("required_unique_channel_count"):
        raise ManifestValidationError("unique ROI channel count does not match frozen constraint")
    if constraints.get("channel_overlap_allowed") is not False:
        raise ManifestValidationError("ROI map must explicitly forbid channel overlap")
    if features.get("missing_channel_policy") != constraints.get("missing_channel_policy"):
        raise ManifestValidationError("ROI missing-channel policy does not match charter")


def validate_protocol_teach_back_transcript(transcript: Any) -> None:
    """Validate exact multi-message provenance without judging scientific comprehension."""
    if not isinstance(transcript, dict):
        raise ManifestValidationError("protocol teach-back transcript must be an object")
    _validate_finite(transcript)
    _require_exact_fields(
        transcript, list(PROTOCOL_TRANSCRIPT_FIELDS), "protocol transcript"
    )
    fixed_values = {
        "schema_id": "neuroai-m1-protocol-teach-back-transcript-v1",
        "schema_version": "1.0",
        "response_owner": "Sofia Lindelow",
        "authorship_attestation": "verbatim_Sofia_owned_words",
    }
    for field, expected in fixed_values.items():
        if transcript[field] != expected:
            raise ManifestValidationError(
                f"protocol transcript {field} must equal {expected!r}"
            )
    for field in (
        "transcript_id",
        "source_thread_id",
        "capture_approval_message_id",
    ):
        if not isinstance(transcript[field], str) or not transcript[field].strip():
            raise ManifestValidationError(f"protocol transcript {field} must be nonempty")

    messages = transcript["messages"]
    if not isinstance(messages, list) or not messages:
        raise ManifestValidationError("protocol transcript messages must be a nonempty array")
    message_ids: set[str] = set()
    messages_by_id: dict[str, dict[str, Any]] = {}
    allowed_roles = {"Sofia Lindelow", "scientific_advisor"}
    allowed_topics = set(PROTOCOL_TEACH_BACK_TOPICS) | {"capture_approval"}
    allowed_kinds = {
        "answer",
        "revision",
        "confirmation",
        "clarification_prompt",
        "capture_approval_prompt",
        "capture_approval",
    }
    allowed_dispositions = {"accepted", "superseded", "context", "approval"}
    sofia_kinds = {"answer", "revision", "confirmation", "capture_approval"}
    advisor_kinds = {"clarification_prompt", "capture_approval_prompt"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ManifestValidationError(
                f"protocol transcript message {index} must be an object"
            )
        _require_exact_fields(
            message,
            list(PROTOCOL_TRANSCRIPT_MESSAGE_FIELDS),
            f"protocol transcript message {index}",
        )
        message_id = message["message_id"]
        if not isinstance(message_id, str) or not message_id.strip():
            raise ManifestValidationError(
                f"protocol transcript message {index} message_id must be nonempty"
            )
        if message_id in message_ids:
            raise ManifestValidationError(
                f"duplicate protocol transcript message_id: {message_id}"
            )
        if message["role"] not in allowed_roles:
            raise ManifestValidationError(
                f"protocol transcript message {message_id} role is invalid"
            )
        if message["topic"] not in allowed_topics:
            raise ManifestValidationError(
                f"protocol transcript message {message_id} topic is invalid"
            )
        if message["kind"] not in allowed_kinds:
            raise ManifestValidationError(
                f"protocol transcript message {message_id} kind is invalid"
            )
        if (
            message["role"] == "Sofia Lindelow"
            and message["kind"] not in sofia_kinds
        ) or (
            message["role"] == "scientific_advisor"
            and message["kind"] not in advisor_kinds
        ):
            raise ManifestValidationError(
                f"protocol transcript message {message_id} role/kind pairing is invalid"
            )
        if message["disposition"] not in allowed_dispositions:
            raise ManifestValidationError(
                f"protocol transcript message {message_id} disposition is invalid"
            )
        text = message["text"]
        if not isinstance(text, str) or not text.strip():
            raise ManifestValidationError(
                f"protocol transcript message {message_id} text must be nonempty"
            )
        expected_text_sha256 = sha256(text.encode("utf-8")).hexdigest()
        if message["text_sha256"] != expected_text_sha256:
            raise ManifestValidationError(
                f"protocol transcript message {message_id} text_sha256 mismatch"
            )
        reply_to = message["in_reply_to_message_id"]
        if reply_to is not None and (
            not isinstance(reply_to, str) or reply_to not in message_ids
        ):
            raise ManifestValidationError(
                f"protocol transcript message {message_id} must reply to a prior message"
            )
        message_ids.add(message_id)
        messages_by_id[message_id] = message

    approval_id = transcript["capture_approval_message_id"]
    approval = messages_by_id.get(approval_id)
    if (
        approval is None
        or approval["role"] != "Sofia Lindelow"
        or approval["topic"] != "capture_approval"
        or approval["kind"] != "capture_approval"
        or approval["disposition"] != "approval"
        or approval["text"].strip().casefold() != "i approve"
    ):
        raise ManifestValidationError(
            "protocol transcript approval must be Sofia's explicit 'I approve' message"
        )
    approval_prompt = messages_by_id.get(approval["in_reply_to_message_id"])
    if (
        approval_prompt is None
        or approval_prompt["role"] != "scientific_advisor"
        or approval_prompt["kind"] != "capture_approval_prompt"
    ):
        raise ManifestValidationError(
            "protocol transcript approval must reply to the advisor's capture approval prompt"
        )


def validate_protocol_teach_back_receipt(
    receipt: Any,
    *,
    final_charter_sha256: str,
    expected_decision_log_sha256: str,
    expected_source_transcript_sha256: str,
    transcript: Mapping[str, Any],
    charter: Mapping[str, Any] | None = None,
) -> None:
    """Validate completeness and provenance, not the scientific quality of Sofia's answers."""
    if not isinstance(receipt, dict):
        raise ManifestValidationError("protocol teach-back receipt must be an object")
    _validate_finite(receipt)
    _require_exact_fields(receipt, list(PROTOCOL_RECEIPT_FIELDS), "protocol receipt")

    fixed_values = {
        "schema_id": "neuroai-m1-protocol-teach-back-v1",
        "schema_version": "1.1",
        "receipt_id": "neuroai-m1-v1-protocol-teach-back",
        "human_gate_id": "Sofia_protocol_teach_back",
        "answered_by": "Sofia Lindelow",
        "authorship_attestation": "verbatim_Sofia_owned_words",
    }
    for field, expected in fixed_values.items():
        if receipt[field] != expected:
            raise ManifestValidationError(
                f"protocol receipt {field} must equal {expected!r}"
            )

    for field in ("charter_id", "charter_version", "source_thread_id"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise ManifestValidationError(f"protocol receipt {field} must be nonempty")
    for field in (
        "charter_sha256",
        "decision_log_sha256",
        "source_transcript_sha256",
        "responses_sha256",
    ):
        if not isinstance(receipt[field], str) or not SHA256_RE.fullmatch(receipt[field]):
            raise ManifestValidationError(f"protocol receipt {field} must be a SHA-256")
    if not isinstance(final_charter_sha256, str) or not SHA256_RE.fullmatch(
        final_charter_sha256
    ):
        raise ManifestValidationError("final_charter_sha256 must be a SHA-256")
    if receipt["charter_sha256"] != final_charter_sha256:
        raise ManifestValidationError("protocol receipt charter_sha256 binding mismatch")
    external_bindings = {
        "decision_log_sha256": expected_decision_log_sha256,
        "source_transcript_sha256": expected_source_transcript_sha256,
    }
    for field, expected in external_bindings.items():
        if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
            raise ManifestValidationError(f"expected {field} must be a SHA-256")
        if receipt[field] != expected:
            raise ManifestValidationError(f"protocol receipt {field} binding mismatch")
    validate_protocol_teach_back_transcript(transcript)
    computed_transcript_sha256 = canonical_document_sha256(transcript)
    if receipt["source_transcript_sha256"] != computed_transcript_sha256:
        raise ManifestValidationError(
            "protocol receipt source_transcript_sha256 binding mismatch"
        )
    if receipt["source_thread_id"] != transcript["source_thread_id"]:
        raise ManifestValidationError("protocol receipt source_thread_id binding mismatch")
    if (
        receipt["capture_approval_message_id"]
        != transcript["capture_approval_message_id"]
    ):
        raise ManifestValidationError(
            "protocol receipt capture_approval_message_id binding mismatch"
        )
    if charter is not None:
        if receipt["charter_id"] != charter.get("charter_id"):
            raise ManifestValidationError("protocol receipt charter_id binding mismatch")
        if receipt["charter_version"] != charter.get("version"):
            raise ManifestValidationError("protocol receipt charter_version binding mismatch")

    if not _matches_type(receipt["recorded_at_utc"], "timestamp_utc"):
        raise ManifestValidationError("protocol receipt recorded_at_utc must be a UTC timestamp")

    responses = receipt["responses"]
    if not isinstance(responses, dict) or set(responses) != set(PROTOCOL_TEACH_BACK_TOPICS):
        raise ManifestValidationError("protocol receipt response topic set mismatch")
    placeholders = {"", "tbd", "todo", "pending", "placeholder", "[answer]"}
    for topic in PROTOCOL_TEACH_BACK_TOPICS:
        response = responses[topic]
        if not isinstance(response, str):
            raise ManifestValidationError(f"protocol response {topic} must be text")
        normalized = response.strip().casefold()
        if normalized in placeholders or normalized.startswith("pending_"):
            raise ManifestValidationError(f"protocol response {topic} is a placeholder")
        if len(response.strip()) < 20:
            raise ManifestValidationError(
                f"protocol response {topic} is too short for a reviewable receipt"
            )
    response_message_ids = receipt["response_message_ids"]
    if not isinstance(response_message_ids, dict) or set(response_message_ids) != set(
        PROTOCOL_TEACH_BACK_TOPICS
    ):
        raise ManifestValidationError("protocol receipt response_message_ids topic set mismatch")
    transcript_by_id = {
        message["message_id"]: message for message in transcript["messages"]
    }
    used_message_ids: set[str] = set()
    for topic in PROTOCOL_TEACH_BACK_TOPICS:
        message_ids = response_message_ids[topic]
        if (
            not isinstance(message_ids, list)
            or not message_ids
            or any(not isinstance(message_id, str) or not message_id for message_id in message_ids)
            or len(set(message_ids)) != len(message_ids)
        ):
            raise ManifestValidationError(
                f"protocol response_message_ids {topic} must be a nonempty unique text array"
            )
        if used_message_ids.intersection(message_ids):
            raise ManifestValidationError("protocol response message IDs cannot be reused")
        used_message_ids.update(message_ids)
        selected_messages = []
        for message_id in message_ids:
            message = transcript_by_id.get(message_id)
            if (
                message is None
                or message["role"] != "Sofia Lindelow"
                or message["topic"] != topic
                or message["disposition"] != "accepted"
                or message["kind"] not in {"answer", "revision", "confirmation"}
            ):
                raise ManifestValidationError(
                    f"protocol response {topic} must select a Sofia-owned accepted message"
                )
            selected_messages.append(message)
        derived_response = "\n\n".join(message["text"] for message in selected_messages)
        if responses[topic] != derived_response:
            raise ManifestValidationError(
                f"protocol response {topic} is not derived from selected transcript messages"
            )
    expected_responses_sha256 = sha256(_canonical_object_bytes(responses)).hexdigest()
    if receipt["responses_sha256"] != expected_responses_sha256:
        raise ManifestValidationError("protocol receipt responses_sha256 mismatch")

    review = receipt["advisor_review"]
    if not isinstance(review, dict):
        raise ManifestValidationError("protocol receipt advisor_review must be an object")
    _require_exact_fields(review, list(PROTOCOL_REVIEW_FIELDS), "protocol review")
    if not _matches_type(review["reviewed_at_utc"], "timestamp_utc"):
        raise ManifestValidationError("protocol review reviewed_at_utc must be a UTC timestamp")
    if review["reviewed_by_role"] != "scientific_advisor":
        raise ManifestValidationError(
            "protocol review reviewed_by_role must equal 'scientific_advisor'"
        )
    recorded_at = datetime.fromisoformat(receipt["recorded_at_utc"].removesuffix("Z") + "+00:00")
    reviewed_at = datetime.fromisoformat(review["reviewed_at_utc"].removesuffix("Z") + "+00:00")
    if reviewed_at < recorded_at:
        raise ManifestValidationError("protocol review cannot predate the recorded responses")
    verdicts = review["topic_verdicts"]
    if not isinstance(verdicts, dict) or set(verdicts) != set(PROTOCOL_TEACH_BACK_TOPICS):
        raise ManifestValidationError("protocol review topic verdict set mismatch")
    if any(verdicts[topic] != "pass" for topic in PROTOCOL_TEACH_BACK_TOPICS):
        raise ManifestValidationError("protocol review topic verdict must be pass for every topic")
    if review["overall_verdict"] != "pass":
        raise ManifestValidationError("protocol review overall_verdict must be pass")
    revision_requests = review["open_revision_requests"]
    if not isinstance(revision_requests, list) or any(
        not isinstance(item, str) or not item.strip() for item in revision_requests
    ):
        raise ManifestValidationError("protocol review revision requests must be a text array")
    if revision_requests:
        raise ManifestValidationError("protocol review has open revision requests")


def assert_scientific_contract_ready(
    charter: dict[str, Any],
    roi_map: dict[str, Any],
    contract: dict[str, Any],
    *,
    final_charter_sha256: str | None,
) -> None:
    """Fail closed unless the frozen scientific manifest contract is internally valid."""
    validate_roi_binding(charter, roi_map)
    pending_gates = charter.get("pending_human_gates")
    if (
        not isinstance(pending_gates, list)
        or any(not isinstance(gate, str) or not gate for gate in pending_gates)
        or len(set(pending_gates)) != len(pending_gates)
    ):
        raise ManifestValidationError("pending_human_gates must be a unique text array")
    features = charter.get("features", {})
    if features.get("normalization") == "PENDING_D26" or any(
        "D26" in gate for gate in charter.get("pending_human_gates", [])
    ):
        raise ManifestValidationError("D26 normalization estimand is unresolved")
    normalization = features.get("normalization")
    if not isinstance(normalization, dict) or (
        normalization.get("decision") != "D26_A_strict_inductive"
        or normalization.get("fit_scope") != "permitted_model_training_sessions_only"
        or normalization.get("label_access") is not False
        or normalization.get("held_session_distribution_access") is not False
        or charter.get("learners", {}).get("M1", {}).get("extra_scaler") is not False
    ):
        raise ManifestValidationError(
            "D26 strict-inductive normalization contract is not satisfied"
        )
    if (
        features.get("analysis_window_s") == "PENDING_D29"
        or features.get("time_bins_s") == "PENDING_D29"
        or any("D29" in gate for gate in charter.get("pending_human_gates", []))
    ):
        raise ManifestValidationError("D29 analysis window and midpoint are unresolved")

    if contract.get("execution_gate_contract") != FROZEN_EXECUTION_GATES:
        raise ManifestValidationError("execution gate contract drifted")
    if charter.get("execution_gates") != FROZEN_EXECUTION_GATES:
        raise ManifestValidationError("charter execution gates are not frozen")

    decisions = charter.get("scientific_decisions")
    decision_gates = contract.get("scientific_decision_gates")
    if not isinstance(decisions, dict) or set(decisions) != {"D26", "D29", "D42", "D44"}:
        raise ManifestValidationError(
            "scientific_decisions must contain exactly D26, D29, D42, and D44"
        )
    if decisions["D26"] != normalization.get("decision"):
        raise ManifestValidationError("D26 decision registry does not match normalization")
    if decisions["D29"] != "D29_A_300_to_1600_ms_exact_950_ms_midpoint":
        raise ManifestValidationError("D29 decision registry does not match the frozen window")
    if not isinstance(decision_gates, dict) or set(decision_gates) != {"D42", "D44"}:
        raise ManifestValidationError("scientific decision gate contract drifted")
    for decision_id in ("D42", "D44"):
        gate_contract = decision_gates[decision_id]
        expected_gate_fields = {
            "pending_gate",
            "allowed_resolved_values",
        }
        if decision_id == "D42":
            expected_gate_fields.add("requires_versioned_replan_before_execution")
        if not isinstance(gate_contract, dict) or set(gate_contract) != expected_gate_fields:
            raise ManifestValidationError(f"{decision_id} gate contract drifted")
        pending_gate = gate_contract["pending_gate"]
        allowed_values = gate_contract["allowed_resolved_values"]
        value = decisions.get(decision_id)
        if (
            not isinstance(pending_gate, str)
            or not isinstance(allowed_values, list)
            or not allowed_values
            or any(not isinstance(item, str) or not item for item in allowed_values)
            or len(set(allowed_values)) != len(allowed_values)
        ):
            raise ManifestValidationError(f"{decision_id} gate contract is invalid")
        if value not in allowed_values:
            raise ManifestValidationError(f"{decision_id} scientific decision is unresolved")
        if pending_gate in pending_gates:
            raise ManifestValidationError(
                f"{decision_id} is resolved but remains in pending_human_gates"
            )
    if decisions["D42"] == "D42_A_omit_balanced_accuracy":
        estimands = charter.get("estimands")
        expected_policy = {
            "decision": "D42_A_omit_balanced_accuracy",
            "included": False,
            "status": "deliberately_omitted_not_computed",
            "threshold": None,
        }
        if not isinstance(estimands, dict) or (
            estimands.get("balanced_accuracy_policy") != expected_policy
            or "balanced_accuracy" in estimands.get("secondary", [])
        ):
            raise ManifestValidationError(
                "D42-A deliberate balanced-accuracy omission policy drifted"
            )
    replan_values = decision_gates["D42"][
        "requires_versioned_replan_before_execution"
    ]
    if (
        not isinstance(replan_values, list)
        or not replan_values
        or any(
            value not in decision_gates["D42"]["allowed_resolved_values"]
            for value in replan_values
        )
    ):
        raise ManifestValidationError("D42 stop/replan gate contract is invalid")
    if decisions["D42"] in replan_values:
        raise ManifestValidationError(
            "D42-C requires stop/replan before the current contract can generate manifests"
        )

    _require_d44_c_charter_contract(charter, contract)

    if charter.get("status") != "frozen_pre_result":
        raise ManifestValidationError("final analysis charter is not frozen")
    remaining_gates = set(pending_gates) - {"Sofia_protocol_teach_back"}
    if remaining_gates:
        raise ManifestValidationError(
            f"unrecognized or unresolved human gates remain: {sorted(remaining_gates)}"
        )

    if not isinstance(final_charter_sha256, str) or not SHA256_RE.fullmatch(
        final_charter_sha256
    ):
        raise ManifestValidationError("final_charter_sha256 must be a SHA-256")
    if canonical_document_sha256(charter) != final_charter_sha256:
        raise ManifestValidationError("final charter canonical digest mismatch")
    if set(contract.get("real_manifest_generation_requires", [])) != {
        "final_charter_sha256",
        "D26_resolved",
        "D29_resolved",
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
        "bounded_preflight_and_capacity_policy_frozen",
        "roi_map_sha256_valid",
        "validated_Sofia_protocol_teach_back_receipt",
    }:
        raise ManifestValidationError("manifest generation requirements drifted from the contract")


def assert_generation_ready(
    charter: dict[str, Any],
    roi_map: dict[str, Any],
    contract: dict[str, Any],
    *,
    protocol_teach_back_receipt: Mapping[str, Any] | None,
    protocol_teach_back_transcript: Mapping[str, Any] | None,
    final_charter_sha256: str | None,
    decision_log_sha256: str | None,
    source_transcript_sha256: str | None,
) -> None:
    """Fail closed until scientific and private human execution gates are resolved."""
    assert_scientific_contract_ready(
        charter,
        roi_map,
        contract,
        final_charter_sha256=final_charter_sha256,
    )
    pending_gates = charter["pending_human_gates"]
    if any(
        "teach_back" in gate for gate in pending_gates
    ) and protocol_teach_back_receipt is None:
        raise ManifestValidationError("Sofia protocol teach-back is unresolved")
    if not isinstance(protocol_teach_back_receipt, dict):
        raise ManifestValidationError("protocol teach-back receipt must be an object")
    validate_protocol_teach_back_receipt(
        protocol_teach_back_receipt,
        final_charter_sha256=final_charter_sha256,
        expected_decision_log_sha256=decision_log_sha256,
        expected_source_transcript_sha256=source_transcript_sha256,
        transcript=protocol_teach_back_transcript,
        charter=charter,
    )
