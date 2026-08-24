"""Outcome-blind preflight receipts and fail-closed full-run capacity authorization.

The module validates synthetic or future technical resource evidence. It never reads EEG data,
fits a model, computes a performance metric, opens a held-out outcome, or chooses D42/D44.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import math
from typing import Any

from peers.m1_manifests import (
    FROZEN_EXECUTION_GATES,
    GIT_COMMIT_RE,
    SHA256_RE,
    canonical_document_sha256,
)


PREFLIGHT_CONTRACT_SHA256 = (
    "b5aaa57f04b031565d4f11afa50f993a8baeba1524cab491d57febc9c89cc7e7"
)
CONDITIONS = ("ocular-removed", "ocular-kept")
STAGE_IDS = (
    "load_validate_inputs",
    "build_validate_feature_tables",
    "fit_predict_without_scoring",
    "publish_readback_artifacts",
)
ROLE_ORDER = ("median", "maximum")
D42_VALUES = {
    "D42_A_omit_balanced_accuracy",
    "D42_B_fixed_0_5_threshold",
    "D42_C_training_only_fitted_threshold",
}
D44_VALUES = {
    "D44_A_fixed_M1_C",
    "D44_B_strict_nested_session_selection",
    "D44_C_outer_training_selected_C_qualified_meta_predictions",
}
CURRENT_CONTRACT_STOP_REPLAN_VALUES = (
    "D42_C_training_only_fitted_threshold",
)
GIB = 1024**3
WALL_ROUND_DIGITS = 6
CANDIDATE_FIELDS = (
    "subject_id",
    "eligible_trials_by_condition",
    "input_bytes_by_condition",
    "trial_manifest_sha256_by_condition",
    "total_eligible_trials",
    "total_input_bytes",
)
MEASUREMENT_FIELDS = (
    "subject_id",
    "roles",
    "condition",
    "stage_wall_seconds",
    "total_wall_seconds",
    "peak_resident_memory_bytes",
    "input_bytes",
    "cache_bytes_written",
    "cache_artifact_set_sha256",
    "cache_entries_atomic_hash_valid",
    "disk_available_bytes_before",
    "disk_available_bytes_after",
    "parallel_worker_count",
    "longest_stage_wall_seconds",
    "longest_no_progress_seconds",
    "transient_retry_count",
)
FORECAST_FIELDS = (
    "wall_time_formula_id",
    "remaining_cache_formula_id",
    "eligible_subject_count",
    "measured_subject_count",
    "measured_subject_condition_run_count",
    "preflight_total_wall_seconds",
    "maximum_subject_preflight_wall_seconds",
    "maximum_peak_resident_memory_bytes",
    "longest_stage_wall_seconds",
    "longest_no_progress_seconds",
    "scaled_full_run_forecast_wall_seconds",
    "projected_remaining_cache_bytes",
    "host_safe_memory_floor_bytes",
    "preflight_parallel_worker_count",
)
PREFLIGHT_RECEIPT_FIELDS = (
    "schema_id",
    "schema_version",
    "receipt_id",
    "started_at_utc",
    "completed_at_utc",
    "charter_id",
    "charter_version",
    "charter_sha256",
    "execution_gate_contract_sha256",
    "protocol_teach_back_receipt_sha256",
    "code_commit",
    "dirty_diff_sha256",
    "environment_lock_sha256",
    "source_inventory_sha256",
    "candidate_inventory",
    "candidate_inventory_sha256",
    "selected_roles",
    "measurements",
    "measurements_sha256",
    "forecast",
    "outcome_access_attestation",
)
CAPACITY_RECEIPT_FIELDS = (
    "schema_id",
    "schema_version",
    "receipt_id",
    "human_gate_id",
    "preflight_receipt_sha256",
    "decided_by",
    "authorship_attestation",
    "source_thread_id",
    "source_message_sha256",
    "recorded_at_utc",
    "decision",
    "reason_in_Sofia_words",
    "selected_current_run_cutoff_utc",
    "acknowledged_budget",
)
BUDGET_FIELDS = (
    "per_subject_timeout_seconds",
    "full_run_timeout_seconds",
    "no_progress_abort_seconds",
    "minimum_disk_available_bytes",
    "host_safe_memory_floor_bytes",
    "parallel_worker_count",
    "transient_retry_limit",
    "resume_policy",
    "partial_outcome_inspection_allowed",
    "scientific_decisions_allowed",
)


class PreflightValidationError(ValueError):
    """Preflight, capacity, or full-run authorization evidence is invalid."""


def _require_exact_fields(value: Any, expected: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise PreflightValidationError(f"{label} must be an object")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise PreflightValidationError(
            f"{label} exact fields mismatch; missing={sorted(wanted - actual)}; "
            f"extra={sorted(actual - wanted)}"
        )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any, label: str) -> int:
    if not _is_int(value) or value <= 0:
        raise PreflightValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not _is_int(value) or value < 0:
        raise PreflightValidationError(f"{label} must be a nonnegative integer")
    return value


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightValidationError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise PreflightValidationError(f"{label} must be a finite positive number")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightValidationError(f"{label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise PreflightValidationError(f"{label} must be a finite nonnegative number")
    return result


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PreflightValidationError(f"{label} must be a SHA-256")
    return value


def _require_git_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or not GIT_COMMIT_RE.fullmatch(value):
        raise PreflightValidationError(f"{label} must be a full Git commit")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PreflightValidationError(f"{label} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise PreflightValidationError(f"{label} must be a valid UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise PreflightValidationError(f"{label} must be UTC")
    return parsed


def _utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightValidationError(f"{label} must be nonempty text")
    return value


def _list_digest(label: str, rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_document_sha256({"kind": label, "rows": list(rows)})


def candidate_inventory_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _list_digest("preflight_candidate_inventory", rows)


def measurements_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return _list_digest("preflight_measurements", rows)


def preflight_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return canonical_document_sha256(receipt)


def capacity_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return canonical_document_sha256(receipt)


def validate_preflight_contract(contract: Mapping[str, Any]) -> None:
    if canonical_document_sha256(contract) != PREFLIGHT_CONTRACT_SHA256:
        raise PreflightValidationError("preflight contract digest mismatch")
    if contract.get("execution_gate_contract") != FROZEN_EXECUTION_GATES:
        raise PreflightValidationError("preflight execution gate contract drifted")
    if tuple(contract.get("stage_ids", [])) != STAGE_IDS:
        raise PreflightValidationError("preflight stage contract drifted")
    if tuple(contract.get("current_contract_stop_replan_values", [])) != (
        CURRENT_CONTRACT_STOP_REPLAN_VALUES
    ):
        raise PreflightValidationError("preflight stop/replan contract drifted")
    if contract.get("full_run_authorization", {}).get("scientific_decisions_allowed") is not False:
        raise PreflightValidationError("full-run authorization may not make scientific decisions")


def _validate_candidate_inventory(rows: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise PreflightValidationError(f"{label} must be a nonempty array")
    if rows != sorted(rows, key=lambda row: row.get("subject_id", "")):
        raise PreflightValidationError(f"{label} must use canonical subject_id order")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_label = f"{label}[{index}]"
        _require_exact_fields(row, CANDIDATE_FIELDS, row_label)
        subject_id = _require_nonempty_text(row["subject_id"], f"{row_label} subject_id")
        if subject_id in seen:
            raise PreflightValidationError(f"{label} has duplicate subject_id {subject_id}")
        seen.add(subject_id)
        condition_counts = row["eligible_trials_by_condition"]
        condition_bytes = row["input_bytes_by_condition"]
        condition_hashes = row["trial_manifest_sha256_by_condition"]
        for mapping, field in (
            (condition_counts, "eligible_trials_by_condition"),
            (condition_bytes, "input_bytes_by_condition"),
            (condition_hashes, "trial_manifest_sha256_by_condition"),
        ):
            _require_exact_fields(mapping, CONDITIONS, f"{row_label} {field}")
        counts = [
            _positive_int(condition_counts[condition], f"{row_label} {condition} trials")
            for condition in CONDITIONS
        ]
        inputs = [
            _positive_int(condition_bytes[condition], f"{row_label} {condition} input_bytes")
            for condition in CONDITIONS
        ]
        for condition in CONDITIONS:
            _require_sha256(
                condition_hashes[condition],
                f"{row_label} {condition} trial manifest",
            )
        if row["total_eligible_trials"] != sum(counts):
            raise PreflightValidationError(f"{label} total_eligible_trials mismatch")
        if row["total_input_bytes"] != sum(inputs):
            raise PreflightValidationError(f"{label} total_input_bytes mismatch")
        validated.append(dict(row))
    return validated


def _expected_selected_roles(inventory: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    workload_order = sorted(
        inventory,
        key=lambda row: (
            row["total_eligible_trials"],
            row["total_input_bytes"],
            row["subject_id"],
        ),
    )
    return {
        "median_subject_id": workload_order[(len(workload_order) - 1) // 2]["subject_id"],
        "maximum_subject_id": workload_order[-1]["subject_id"],
    }


def validate_candidate_inventory(rows: Any) -> list[dict[str, Any]]:
    """Public strict validator for an outcome-blind real preflight inventory."""
    return _validate_candidate_inventory(rows, "candidate inventory")


def select_preflight_roles(rows: Any) -> dict[str, str]:
    """Validate an inventory and recompute the frozen lower-median/maximum subjects."""
    return _expected_selected_roles(validate_candidate_inventory(rows))


def _validate_measurements(
    rows: Any,
    inventory: Sequence[Mapping[str, Any]],
    selected_roles: Mapping[str, str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise PreflightValidationError("measurements must be a nonempty array")
    condition_rank = {condition: index for index, condition in enumerate(CONDITIONS)}
    if rows != sorted(
        rows,
        key=lambda row: (
            row.get("subject_id", ""),
            condition_rank.get(row.get("condition"), len(CONDITIONS)),
        ),
    ):
        raise PreflightValidationError("measurements must use canonical subject/condition order")
    inventory_by_subject = {row["subject_id"]: row for row in inventory}
    role_by_subject: dict[str, list[str]] = {}
    for role in ROLE_ORDER:
        subject_id = selected_roles[f"{role}_subject_id"]
        role_by_subject.setdefault(subject_id, []).append(role)
    expected_pairs = {
        (subject_id, condition)
        for subject_id in role_by_subject
        for condition in CONDITIONS
    }
    seen_pairs: set[tuple[str, str]] = set()
    worker_counts: set[int] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        label = f"measurements[{index}]"
        _require_exact_fields(row, MEASUREMENT_FIELDS, label)
        subject_id = _require_nonempty_text(row["subject_id"], f"{label} subject_id")
        condition = row["condition"]
        pair = (subject_id, condition)
        if pair not in expected_pairs:
            raise PreflightValidationError(f"measurement row is outside selected subjects: {pair}")
        if pair in seen_pairs:
            raise PreflightValidationError(f"duplicate measurement row: {pair}")
        seen_pairs.add(pair)
        if row["roles"] != role_by_subject[subject_id]:
            raise PreflightValidationError(f"measurement roles mismatch for {subject_id}")
        stage_times = row["stage_wall_seconds"]
        _require_exact_fields(stage_times, STAGE_IDS, f"{label} stage_wall_seconds")
        stage_values = [
            _positive_float(stage_times[stage], f"{label} stage {stage}")
            for stage in STAGE_IDS
        ]
        total_wall = _positive_float(row["total_wall_seconds"], f"{label} total_wall_seconds")
        if total_wall < max(stage_values):
            raise PreflightValidationError(f"{label} total wall is shorter than a stage")
        _positive_int(row["peak_resident_memory_bytes"], f"{label} peak memory")
        input_bytes = _positive_int(row["input_bytes"], f"{label} input_bytes")
        expected_input = inventory_by_subject[subject_id]["input_bytes_by_condition"][condition]
        if input_bytes != expected_input:
            raise PreflightValidationError(f"{label} input_bytes mismatch")
        _nonnegative_int(row["cache_bytes_written"], f"{label} cache_bytes_written")
        _require_sha256(
            row["cache_artifact_set_sha256"], f"{label} cache artifact set"
        )
        if row["cache_entries_atomic_hash_valid"] is not True:
            raise PreflightValidationError(
                f"{label} cache entries must be atomically completed and hash-valid"
            )
        _positive_int(row["disk_available_bytes_before"], f"{label} disk before")
        _positive_int(row["disk_available_bytes_after"], f"{label} disk after")
        workers = _positive_int(row["parallel_worker_count"], f"{label} workers")
        worker_counts.add(workers)
        recorded_longest = _positive_float(
            row["longest_stage_wall_seconds"], f"{label} longest stage"
        )
        if not math.isclose(recorded_longest, max(stage_values), abs_tol=1e-9):
            raise PreflightValidationError(f"{label} longest stage mismatch")
        no_progress = _nonnegative_float(
            row["longest_no_progress_seconds"], f"{label} no progress"
        )
        if no_progress > total_wall:
            raise PreflightValidationError(f"{label} no-progress interval exceeds total wall")
        retries = _nonnegative_int(row["transient_retry_count"], f"{label} retry count")
        if retries > FROZEN_EXECUTION_GATES["full_locked_run"]["transient_retry_limit"]:
            raise PreflightValidationError(f"{label} retry count exceeds the frozen retry limit")
        validated.append(dict(row))
    if seen_pairs != expected_pairs:
        missing_pairs = sorted(expected_pairs - seen_pairs)
        raise PreflightValidationError(
            f"measurement subject-condition coverage mismatch; missing={missing_pairs}"
        )
    if len(worker_counts) != 1:
        raise PreflightValidationError(
            "parallel worker count must be identical across measurements"
        )
    return validated


def build_preflight_forecast(
    inventory: Sequence[Mapping[str, Any]],
    measurements: Sequence[Mapping[str, Any]],
    *,
    host_safe_memory_floor_bytes: int,
) -> dict[str, Any]:
    """Derive the exact conservative D47 forecast from outcome-blind workload measurements."""
    _positive_int(host_safe_memory_floor_bytes, "host_safe_memory_floor_bytes")
    inventory_by_subject = {row["subject_id"]: row for row in inventory}
    selected_roles = _expected_selected_roles(inventory)
    selected_subjects = set(selected_roles.values())
    maximum_subject = selected_roles["maximum_subject_id"]
    workers = {row["parallel_worker_count"] for row in measurements}
    if len(workers) != 1:
        raise PreflightValidationError("forecast requires one preflight parallel worker count")
    maximum_peak_memory = max(
        row["peak_resident_memory_bytes"] for row in measurements
    )
    if host_safe_memory_floor_bytes < maximum_peak_memory:
        raise PreflightValidationError(
            "host safe memory floor is below the observed peak resident memory"
        )

    wall_forecast = 0.0
    cache_forecast = 0
    for condition in CONDITIONS:
        condition_measurements = [row for row in measurements if row["condition"] == condition]
        if not condition_measurements:
            raise PreflightValidationError(f"forecast lacks measurements for {condition}")
        max_seconds_per_trial = max(
            row["total_wall_seconds"]
            / inventory_by_subject[row["subject_id"]]["eligible_trials_by_condition"][condition]
            for row in condition_measurements
        )
        max_seconds_per_byte = max(
            row["total_wall_seconds"] / row["input_bytes"]
            for row in condition_measurements
        )
        all_trials = sum(
            row["eligible_trials_by_condition"][condition] for row in inventory
        )
        all_bytes = sum(row["input_bytes_by_condition"][condition] for row in inventory)
        wall_forecast += max(
            max_seconds_per_trial * all_trials,
            max_seconds_per_byte * all_bytes,
        )

        remaining = [row for row in inventory if row["subject_id"] not in selected_subjects]
        if remaining:
            max_cache_per_trial = max(
                row["cache_bytes_written"]
                / inventory_by_subject[row["subject_id"]]["eligible_trials_by_condition"][
                    condition
                ]
                for row in condition_measurements
            )
            max_cache_per_byte = max(
                row["cache_bytes_written"] / row["input_bytes"]
                for row in condition_measurements
            )
            remaining_trials = sum(
                row["eligible_trials_by_condition"][condition] for row in remaining
            )
            remaining_bytes = sum(
                row["input_bytes_by_condition"][condition] for row in remaining
            )
            cache_forecast += math.ceil(
                max(
                    max_cache_per_trial * remaining_trials,
                    max_cache_per_byte * remaining_bytes,
                )
            )

    maximum_subject_wall = sum(
        row["total_wall_seconds"]
        for row in measurements
        if row["subject_id"] == maximum_subject
    )
    return {
        "wall_time_formula_id": (
            "max_observed_seconds_per_trial_or_input_byte_by_condition_"
            "times_full_condition_workload_then_sum"
        ),
        "remaining_cache_formula_id": (
            "ceil_max_observed_cache_bytes_per_trial_or_input_byte_by_condition_"
            "times_unmeasured_condition_workload_then_sum"
        ),
        "eligible_subject_count": len(inventory),
        "measured_subject_count": len(selected_subjects),
        "measured_subject_condition_run_count": len(measurements),
        "preflight_total_wall_seconds": round(
            sum(row["total_wall_seconds"] for row in measurements), WALL_ROUND_DIGITS
        ),
        "maximum_subject_preflight_wall_seconds": round(
            maximum_subject_wall, WALL_ROUND_DIGITS
        ),
        "maximum_peak_resident_memory_bytes": maximum_peak_memory,
        "longest_stage_wall_seconds": round(
            max(row["longest_stage_wall_seconds"] for row in measurements),
            WALL_ROUND_DIGITS,
        ),
        "longest_no_progress_seconds": round(
            max(row["longest_no_progress_seconds"] for row in measurements),
            WALL_ROUND_DIGITS,
        ),
        "scaled_full_run_forecast_wall_seconds": round(
            wall_forecast, WALL_ROUND_DIGITS
        ),
        "projected_remaining_cache_bytes": cache_forecast,
        "host_safe_memory_floor_bytes": host_safe_memory_floor_bytes,
        "preflight_parallel_worker_count": next(iter(workers)),
    }


def build_acknowledged_budget(preflight_receipt: Mapping[str, Any]) -> dict[str, Any]:
    forecast = preflight_receipt["forecast"]
    return {
        "per_subject_timeout_seconds": round(
            2.0 * forecast["maximum_subject_preflight_wall_seconds"], WALL_ROUND_DIGITS
        ),
        "full_run_timeout_seconds": round(
            1.3 * forecast["scaled_full_run_forecast_wall_seconds"], WALL_ROUND_DIGITS
        ),
        "no_progress_abort_seconds": round(
            2.0 * forecast["longest_stage_wall_seconds"], WALL_ROUND_DIGITS
        ),
        "minimum_disk_available_bytes": math.ceil(
            1.2 * forecast["projected_remaining_cache_bytes"] + 5 * GIB
        ),
        "host_safe_memory_floor_bytes": forecast["host_safe_memory_floor_bytes"],
        "parallel_worker_count": forecast["preflight_parallel_worker_count"],
        "transient_retry_limit": 1,
        "resume_policy": "atomically_completed_hash_valid_cache_entries_only",
        "partial_outcome_inspection_allowed": False,
        "scientific_decisions_allowed": False,
    }


def validate_preflight_budget_receipt(
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    charter: Mapping[str, Any],
    expected_final_charter_sha256: str,
    expected_protocol_teach_back_receipt_sha256: str,
    expected_code_commit: str,
    expected_dirty_diff_sha256: str | None,
    expected_environment_lock_sha256: str,
    expected_source_inventory_sha256: str,
    expected_candidate_inventory: Sequence[Mapping[str, Any]],
) -> None:
    validate_preflight_contract(contract)
    _require_exact_fields(receipt, PREFLIGHT_RECEIPT_FIELDS, "preflight receipt")
    if receipt["schema_id"] != "neuroai-m1-preflight-budget-receipt-v1":
        raise PreflightValidationError("preflight receipt schema_id mismatch")
    if receipt["schema_version"] != "1.0":
        raise PreflightValidationError("preflight receipt schema_version mismatch")
    _require_nonempty_text(receipt["receipt_id"], "preflight receipt_id")
    started = _parse_utc(receipt["started_at_utc"], "preflight started_at_utc")
    completed = _parse_utc(receipt["completed_at_utc"], "preflight completed_at_utc")
    if completed <= started:
        raise PreflightValidationError("preflight completion must follow start")

    if charter.get("status") != "frozen_pre_result":
        raise PreflightValidationError("preflight requires a frozen pre-result charter")
    decisions = charter.get("scientific_decisions")
    if not isinstance(decisions, Mapping):
        raise PreflightValidationError("charter scientific decision registry is missing")
    if decisions.get("D42") not in D42_VALUES or decisions.get("D44") not in D44_VALUES:
        raise PreflightValidationError("D42 and D44 must be resolved before preflight")
    if decisions["D42"] in CURRENT_CONTRACT_STOP_REPLAN_VALUES:
        raise PreflightValidationError(
            "D42-C requires stop/replan before the current contract can run preflight"
        )
    pending_human_gates = charter.get("pending_human_gates")
    allowed_pending_human_gates = ([], ["Sofia_protocol_teach_back"])
    if (
        not isinstance(pending_human_gates, list)
        or pending_human_gates not in allowed_pending_human_gates
    ):
        raise PreflightValidationError(
            "preflight charter has an unresolved human gate other than the externally bound "
            "protocol teach-back"
        )
    _require_sha256(expected_final_charter_sha256, "expected final charter")
    if canonical_document_sha256(charter) != expected_final_charter_sha256:
        raise PreflightValidationError("final charter canonical digest mismatch")
    if receipt["charter_id"] != charter.get("charter_id"):
        raise PreflightValidationError("preflight charter_id binding mismatch")
    if receipt["charter_version"] != charter.get("version"):
        raise PreflightValidationError("preflight charter_version binding mismatch")

    expected_bindings = {
        "charter_sha256": expected_final_charter_sha256,
        "protocol_teach_back_receipt_sha256": expected_protocol_teach_back_receipt_sha256,
        "code_commit": expected_code_commit,
        "dirty_diff_sha256": expected_dirty_diff_sha256,
        "environment_lock_sha256": expected_environment_lock_sha256,
        "source_inventory_sha256": expected_source_inventory_sha256,
    }
    for field in (
        "charter_sha256",
        "protocol_teach_back_receipt_sha256",
        "environment_lock_sha256",
        "source_inventory_sha256",
    ):
        _require_sha256(receipt[field], f"preflight {field}")
        _require_sha256(expected_bindings[field], f"expected {field}")
    _require_git_commit(receipt["code_commit"], "preflight code_commit")
    _require_git_commit(expected_code_commit, "expected code_commit")
    if receipt["dirty_diff_sha256"] is not None:
        _require_sha256(receipt["dirty_diff_sha256"], "preflight dirty_diff_sha256")
    if expected_dirty_diff_sha256 is not None:
        _require_sha256(expected_dirty_diff_sha256, "expected dirty_diff_sha256")
    for field, expected in expected_bindings.items():
        if receipt[field] != expected:
            raise PreflightValidationError(f"preflight {field} binding mismatch")

    execution_gate_sha = canonical_document_sha256(FROZEN_EXECUTION_GATES)
    if receipt["execution_gate_contract_sha256"] != execution_gate_sha:
        raise PreflightValidationError("preflight execution gate contract binding mismatch")

    inventory = _validate_candidate_inventory(
        receipt["candidate_inventory"], "candidate inventory"
    )
    expected_inventory = _validate_candidate_inventory(
        list(expected_candidate_inventory), "expected candidate inventory"
    )
    if inventory != expected_inventory:
        raise PreflightValidationError("candidate inventory external binding mismatch")
    if receipt["candidate_inventory_sha256"] != candidate_inventory_sha256(inventory):
        raise PreflightValidationError("candidate inventory digest mismatch")

    _require_exact_fields(
        receipt["selected_roles"],
        ("median_subject_id", "maximum_subject_id"),
        "selected roles",
    )
    expected_roles = _expected_selected_roles(inventory)
    if receipt["selected_roles"] != expected_roles:
        raise PreflightValidationError("selected roles do not match the frozen workload rule")
    measured = _validate_measurements(receipt["measurements"], inventory, expected_roles)
    if receipt["measurements_sha256"] != measurements_sha256(measured):
        raise PreflightValidationError("measurements digest mismatch")

    _require_exact_fields(receipt["forecast"], FORECAST_FIELDS, "preflight forecast")
    expected_forecast = build_preflight_forecast(
        inventory,
        measured,
        host_safe_memory_floor_bytes=_positive_int(
            receipt["forecast"]["host_safe_memory_floor_bytes"],
            "forecast host_safe_memory_floor_bytes",
        ),
    )
    if receipt["forecast"] != expected_forecast:
        raise PreflightValidationError("preflight forecast does not recompute")

    _require_exact_fields(
        receipt["outcome_access_attestation"],
        (
            "outcome_metrics_computed",
            "held_out_outcomes_inspected",
            "interpretation_performed",
            "attested_by_role",
        ),
        "outcome access attestation",
    )
    expected_attestation = {
        "outcome_metrics_computed": False,
        "held_out_outcomes_inspected": False,
        "interpretation_performed": False,
        "attested_by_role": "technical_operator",
    }
    if receipt["outcome_access_attestation"] != expected_attestation:
        raise PreflightValidationError("outcome access attestation is not fail-closed")


def validate_capacity_receipt(
    receipt: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    expected_source_message_sha256: str,
) -> None:
    validate_preflight_contract(contract)
    _require_exact_fields(receipt, CAPACITY_RECEIPT_FIELDS, "capacity receipt")
    fixed = {
        "schema_id": "neuroai-m1-capacity-decision-receipt-v1",
        "schema_version": "1.0",
        "human_gate_id": "Sofia_capacity_go",
        "decided_by": "Sofia Lindelow",
        "authorship_attestation": "verbatim_Sofia_capacity_decision",
    }
    for field, expected in fixed.items():
        if receipt[field] != expected:
            raise PreflightValidationError(f"capacity {field} must equal {expected!r}")
    _require_nonempty_text(receipt["receipt_id"], "capacity receipt_id")
    _require_nonempty_text(receipt["source_thread_id"], "capacity source_thread_id")
    _require_sha256(receipt["source_message_sha256"], "capacity source_message_sha256")
    _require_sha256(expected_source_message_sha256, "expected source_message_sha256")
    if receipt["source_message_sha256"] != expected_source_message_sha256:
        raise PreflightValidationError("capacity source_message_sha256 binding mismatch")
    expected_preflight_sha = preflight_receipt_sha256(preflight_receipt)
    if receipt["preflight_receipt_sha256"] != expected_preflight_sha:
        raise PreflightValidationError("capacity preflight receipt binding mismatch")
    recorded_at = _parse_utc(receipt["recorded_at_utc"], "capacity recorded_at_utc")
    completed_at = _parse_utc(
        preflight_receipt.get("completed_at_utc"), "preflight completed_at_utc"
    )
    if recorded_at < completed_at:
        raise PreflightValidationError("capacity decision cannot predate preflight completion")
    reason = _require_nonempty_text(
        receipt["reason_in_Sofia_words"], "capacity reason_in_Sofia_words"
    )
    if reason.strip().casefold() in {"tbd", "todo", "pending", "placeholder"} or len(
        reason.strip()
    ) < 20:
        raise PreflightValidationError("capacity reason must be reviewable Sofia-owned text")
    if receipt["decision"] not in {"go", "no_go"}:
        raise PreflightValidationError("capacity decision must be go or no_go")
    cutoff = receipt["selected_current_run_cutoff_utc"]
    if receipt["decision"] == "go":
        parsed_cutoff = _parse_utc(cutoff, "capacity cutoff")
        if parsed_cutoff <= recorded_at:
            raise PreflightValidationError("capacity cutoff must follow the recorded decision")
    elif cutoff is not None:
        raise PreflightValidationError("no_go capacity receipt must have a null cutoff")
    _require_exact_fields(receipt["acknowledged_budget"], BUDGET_FIELDS, "acknowledged budget")
    if receipt["acknowledged_budget"] != build_acknowledged_budget(preflight_receipt):
        raise PreflightValidationError("acknowledged budget does not match preflight evidence")


def authorize_full_locked_run(
    preflight_receipt: Mapping[str, Any],
    capacity_receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    charter: Mapping[str, Any],
    expected_final_charter_sha256: str,
    expected_protocol_teach_back_receipt_sha256: str,
    expected_code_commit: str,
    expected_dirty_diff_sha256: str | None,
    expected_environment_lock_sha256: str,
    expected_source_inventory_sha256: str,
    expected_candidate_inventory: Sequence[Mapping[str, Any]],
    expected_capacity_source_message_sha256: str,
    run_started_at_utc: str,
    disk_available_bytes: int,
    host_available_memory_bytes: int,
    parallel_worker_count: int,
) -> dict[str, Any]:
    """Authorize only the frozen run budget; never choose science or inspect partial outcomes."""
    validate_preflight_budget_receipt(
        preflight_receipt,
        contract,
        charter=charter,
        expected_final_charter_sha256=expected_final_charter_sha256,
        expected_protocol_teach_back_receipt_sha256=expected_protocol_teach_back_receipt_sha256,
        expected_code_commit=expected_code_commit,
        expected_dirty_diff_sha256=expected_dirty_diff_sha256,
        expected_environment_lock_sha256=expected_environment_lock_sha256,
        expected_source_inventory_sha256=expected_source_inventory_sha256,
        expected_candidate_inventory=expected_candidate_inventory,
    )
    validate_capacity_receipt(
        capacity_receipt,
        preflight_receipt,
        contract,
        expected_source_message_sha256=expected_capacity_source_message_sha256,
    )
    if capacity_receipt["decision"] != "go":
        raise PreflightValidationError("Sofia capacity decision is not go")
    run_started = _parse_utc(run_started_at_utc, "full-run start")
    decision_time = _parse_utc(capacity_receipt["recorded_at_utc"], "capacity decision time")
    if run_started < decision_time:
        raise PreflightValidationError("full-run start cannot predate the capacity decision")
    cutoff = _parse_utc(
        capacity_receipt["selected_current_run_cutoff_utc"], "Sofia cutoff"
    )
    if run_started > cutoff:
        raise PreflightValidationError("full-run start is after Sofia's cutoff")
    budget = build_acknowledged_budget(preflight_receipt)
    disk_available = _positive_int(disk_available_bytes, "disk_available_bytes")
    if disk_available < budget["minimum_disk_available_bytes"]:
        raise PreflightValidationError("full-run disk headroom is below the acknowledged floor")
    available_memory = _positive_int(
        host_available_memory_bytes, "host_available_memory_bytes"
    )
    if available_memory < budget["host_safe_memory_floor_bytes"]:
        raise PreflightValidationError("full-run available memory is below the memory floor")
    workers = _positive_int(parallel_worker_count, "parallel_worker_count")
    if workers != budget["parallel_worker_count"]:
        raise PreflightValidationError("full-run parallel worker count drifted from preflight")
    timeout_deadline = run_started + timedelta(seconds=budget["full_run_timeout_seconds"])
    effective_deadline = min(timeout_deadline, cutoff)
    return {
        "schema_id": "neuroai-m1-full-run-authorization-v1",
        "schema_version": "1.0",
        "authorized_at_utc": run_started_at_utc,
        "charter_sha256": expected_final_charter_sha256,
        "code_commit": expected_code_commit,
        "preflight_receipt_sha256": preflight_receipt_sha256(preflight_receipt),
        "capacity_receipt_sha256": capacity_receipt_sha256(capacity_receipt),
        "effective_deadline_utc": _utc_string(effective_deadline),
        "budget": budget,
        "runtime_capacity_checked": {
            "disk_available_bytes": disk_available,
            "host_available_memory_bytes": available_memory,
            "parallel_worker_count": workers,
        },
        "partial_outcome_inspection_allowed": False,
        "scientific_decisions_allowed": False,
        "failure_state": "IN_PROGRESS",
    }
