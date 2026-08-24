"""Synthetic tests for the outcome-blind preflight and full-run capacity gate.

No fixture loads a real feature, label, fit, probability, metric, or held-out outcome.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from peers.m1_manifests import canonical_document_sha256, load_json_unique
from peers.m1_preflight import (
    PreflightValidationError,
    authorize_full_locked_run,
    build_acknowledged_budget,
    build_preflight_forecast,
    candidate_inventory_sha256,
    capacity_receipt_sha256,
    measurements_sha256,
    preflight_receipt_sha256,
    validate_capacity_receipt,
    validate_preflight_budget_receipt,
    validate_preflight_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "m1-preflight-contract.json"
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
COMMIT = "f" * 40
STAGES = (
    "load_validate_inputs",
    "build_validate_feature_tables",
    "fit_predict_without_scoring",
    "publish_readback_artifacts",
)


@pytest.fixture(scope="module")
def contract() -> dict:
    return load_json_unique(CONTRACT_PATH)


def final_charter() -> dict:
    charter = load_json_unique(CHARTER_PATH)
    charter["status"] = "frozen_pre_result"
    charter["scientific_decisions"]["D42"] = "D42_A_omit_balanced_accuracy"
    charter["scientific_decisions"]["D44"] = "D44_A_fixed_M1_C"
    charter["pending_human_gates"] = []
    return charter


def candidate_inventory() -> list[dict]:
    rows = []
    for subject_id, removed_trials, kept_trials, removed_bytes, kept_bytes in (
        ("sub-LTP001", 80, 75, 1_000, 900),
        ("sub-LTP002", 100, 90, 1_500, 1_400),
        ("sub-LTP003", 120, 110, 2_000, 1_900),
    ):
        rows.append(
            {
                "subject_id": subject_id,
                "eligible_trials_by_condition": {
                    "ocular-removed": removed_trials,
                    "ocular-kept": kept_trials,
                },
                "input_bytes_by_condition": {
                    "ocular-removed": removed_bytes,
                    "ocular-kept": kept_bytes,
                },
                "trial_manifest_sha256_by_condition": {
                    "ocular-removed": SHA_A,
                    "ocular-kept": SHA_B,
                },
                "total_eligible_trials": removed_trials + kept_trials,
                "total_input_bytes": removed_bytes + kept_bytes,
            }
        )
    return rows


def measurement(
    subject_id: str,
    condition: str,
    *,
    total_wall_seconds: float,
    input_bytes: int,
    cache_bytes_written: int,
    roles: list[str],
) -> dict:
    scale = total_wall_seconds / 100.0
    stage_wall_seconds = {
        "load_validate_inputs": 10.0 * scale,
        "build_validate_feature_tables": 25.0 * scale,
        "fit_predict_without_scoring": 45.0 * scale,
        "publish_readback_artifacts": 15.0 * scale,
    }
    return {
        "subject_id": subject_id,
        "roles": roles,
        "condition": condition,
        "stage_wall_seconds": stage_wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "peak_resident_memory_bytes": 500_000_000 + int(total_wall_seconds * 1_000),
        "input_bytes": input_bytes,
        "cache_bytes_written": cache_bytes_written,
        "cache_artifact_set_sha256": SHA_C,
        "cache_entries_atomic_hash_valid": True,
        "disk_available_bytes_before": 80_000_000_000,
        "disk_available_bytes_after": 79_000_000_000,
        "parallel_worker_count": 2,
        "longest_stage_wall_seconds": max(stage_wall_seconds.values()),
        "longest_no_progress_seconds": 3.0 * scale,
        "transient_retry_count": 0,
    }


def measurements() -> list[dict]:
    return [
        measurement(
            "sub-LTP002",
            "ocular-removed",
            total_wall_seconds=60.0,
            input_bytes=1_500,
            cache_bytes_written=300,
            roles=["median"],
        ),
        measurement(
            "sub-LTP002",
            "ocular-kept",
            total_wall_seconds=66.0,
            input_bytes=1_400,
            cache_bytes_written=280,
            roles=["median"],
        ),
        measurement(
            "sub-LTP003",
            "ocular-removed",
            total_wall_seconds=90.0,
            input_bytes=2_000,
            cache_bytes_written=400,
            roles=["maximum"],
        ),
        measurement(
            "sub-LTP003",
            "ocular-kept",
            total_wall_seconds=96.0,
            input_bytes=1_900,
            cache_bytes_written=380,
            roles=["maximum"],
        ),
    ]


def preflight_receipt(contract: dict) -> tuple[dict, dict]:
    charter = final_charter()
    inventory = candidate_inventory()
    measured = measurements()
    receipt = {
        "schema_id": "neuroai-m1-preflight-budget-receipt-v1",
        "schema_version": "1.0",
        "receipt_id": "synthetic-preflight-run",
        "started_at_utc": "2026-08-16T20:00:00Z",
        "completed_at_utc": "2026-08-16T20:10:00Z",
        "charter_id": charter["charter_id"],
        "charter_version": charter["version"],
        "charter_sha256": canonical_document_sha256(charter),
        "execution_gate_contract_sha256": canonical_document_sha256(
            contract["execution_gate_contract"]
        ),
        "protocol_teach_back_receipt_sha256": SHA_A,
        "code_commit": COMMIT,
        "dirty_diff_sha256": None,
        "environment_lock_sha256": SHA_B,
        "source_inventory_sha256": SHA_C,
        "candidate_inventory": inventory,
        "candidate_inventory_sha256": candidate_inventory_sha256(inventory),
        "selected_roles": {
            "median_subject_id": "sub-LTP002",
            "maximum_subject_id": "sub-LTP003",
        },
        "measurements": measured,
        "measurements_sha256": measurements_sha256(measured),
        "forecast": build_preflight_forecast(
            inventory,
            measured,
            host_safe_memory_floor_bytes=2_000_000_000,
        ),
        "outcome_access_attestation": {
            "outcome_metrics_computed": False,
            "held_out_outcomes_inspected": False,
            "interpretation_performed": False,
            "attested_by_role": "technical_operator",
        },
    }
    return receipt, charter


def validate_preflight(receipt: dict, charter: dict, contract: dict) -> None:
    validate_preflight_budget_receipt(
        receipt,
        contract,
        charter=charter,
        expected_final_charter_sha256=canonical_document_sha256(charter),
        expected_protocol_teach_back_receipt_sha256=SHA_A,
        expected_code_commit=COMMIT,
        expected_dirty_diff_sha256=None,
        expected_environment_lock_sha256=SHA_B,
        expected_source_inventory_sha256=SHA_C,
        expected_candidate_inventory=candidate_inventory(),
    )


def capacity_receipt(preflight: dict) -> dict:
    return {
        "schema_id": "neuroai-m1-capacity-decision-receipt-v1",
        "schema_version": "1.0",
        "receipt_id": "synthetic-capacity-decision",
        "human_gate_id": "Sofia_capacity_go",
        "preflight_receipt_sha256": preflight_receipt_sha256(preflight),
        "decided_by": "Sofia Lindelow",
        "authorship_attestation": "verbatim_Sofia_capacity_decision",
        "source_thread_id": "synthetic-test-thread",
        "source_message_sha256": SHA_D,
        "recorded_at_utc": "2026-08-16T20:15:00Z",
        "decision": "go",
        "reason_in_Sofia_words": (
            "Synthetic capacity reason for structural testing only; no interpretation is present."
        ),
        "selected_current_run_cutoff_utc": "2026-08-16T23:00:00Z",
        "acknowledged_budget": build_acknowledged_budget(preflight),
    }


def test_contract_binds_d47_stage_and_receipt_rules(contract):
    validate_preflight_contract(contract)

    assert tuple(contract["stage_ids"]) == STAGES
    assert contract["execution_gate_contract"]["preflight"][
        "outcome_metrics_or_interpretation_allowed"
    ] is False
    assert contract["execution_gate_contract"]["full_locked_run"][
        "partial_outcome_inspection_allowed"
    ] is False
    assert contract["full_run_authorization"]["scientific_decisions_allowed"] is False
    assert contract["current_contract_stop_replan_values"] == [
        "D42_C_training_only_fitted_threshold"
    ]


def test_valid_preflight_receipt_recomputes_selection_measurements_and_forecast(contract):
    receipt, charter = preflight_receipt(contract)

    validate_preflight(receipt, charter, contract)

    assert receipt["selected_roles"] == {
        "median_subject_id": "sub-LTP002",
        "maximum_subject_id": "sub-LTP003",
    }
    assert receipt["forecast"]["maximum_subject_preflight_wall_seconds"] == 186.0
    assert receipt["forecast"]["maximum_peak_resident_memory_bytes"] > 500_000_000
    assert receipt["forecast"]["preflight_parallel_worker_count"] == 2
    assert receipt["forecast"]["projected_remaining_cache_bytes"] > 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda receipt: receipt.update(auc=0.5), "exact fields"),
        (
            lambda receipt: receipt["outcome_access_attestation"].update(
                held_out_outcomes_inspected=True
            ),
            "outcome access",
        ),
        (
            lambda receipt: receipt["selected_roles"].update(
                median_subject_id="sub-LTP001"
            ),
            "selected roles",
        ),
        (
            lambda receipt: receipt["candidate_inventory"][0].update(
                total_eligible_trials=999
            ),
            "candidate inventory",
        ),
        (
            lambda receipt: receipt["measurements"][0].update(input_bytes=1_501),
            "input_bytes",
        ),
        (
            lambda receipt: receipt["measurements"][0]["stage_wall_seconds"].pop(
                "publish_readback_artifacts"
            ),
            "stage",
        ),
        (
            lambda receipt: receipt["measurements"][0].update(transient_retry_count=2),
            "retry",
        ),
        (
            lambda receipt: receipt["measurements"][0].update(
                cache_entries_atomic_hash_valid=False
            ),
            "cache entries",
        ),
        (
            lambda receipt: receipt["forecast"].update(
                scaled_full_run_forecast_wall_seconds=1.0
            ),
            "forecast",
        ),
        (lambda receipt: receipt.update(code_commit="0" * 40), "binding mismatch"),
    ],
)
def test_preflight_receipt_rejects_drift(contract, mutation, message):
    receipt, charter = preflight_receipt(contract)
    mutation(receipt)

    with pytest.raises(PreflightValidationError, match=message):
        validate_preflight(receipt, charter, contract)


def test_preflight_accepts_actual_frozen_charter_with_external_teach_back_receipt(contract):
    receipt, _ = preflight_receipt(contract)
    charter = load_json_unique(CHARTER_PATH)
    receipt["charter_version"] = charter["version"]
    receipt["charter_sha256"] = canonical_document_sha256(charter)

    validate_preflight_budget_receipt(
        receipt,
        contract,
        charter=charter,
        expected_final_charter_sha256=canonical_document_sha256(charter),
        expected_protocol_teach_back_receipt_sha256=SHA_A,
        expected_code_commit=COMMIT,
        expected_dirty_diff_sha256=None,
        expected_environment_lock_sha256=SHA_B,
        expected_source_inventory_sha256=SHA_C,
        expected_candidate_inventory=candidate_inventory(),
    )

    unresolved = deepcopy(charter)
    unresolved["pending_human_gates"].append("Sofia_capacity_go")
    with pytest.raises(PreflightValidationError, match="unresolved human gate"):
        validate_preflight_budget_receipt(
            receipt,
            contract,
            charter=unresolved,
            expected_final_charter_sha256=canonical_document_sha256(unresolved),
            expected_protocol_teach_back_receipt_sha256=SHA_A,
            expected_code_commit=COMMIT,
            expected_dirty_diff_sha256=None,
            expected_environment_lock_sha256=SHA_B,
            expected_source_inventory_sha256=SHA_C,
            expected_candidate_inventory=candidate_inventory(),
        )


def test_preflight_rejects_unreplanned_d42_c(contract):
    receipt, charter = preflight_receipt(contract)
    charter["scientific_decisions"]["D42"] = (
        "D42_C_training_only_fitted_threshold"
    )

    with pytest.raises(PreflightValidationError, match="D42-C requires stop/replan"):
        validate_preflight_budget_receipt(
            receipt,
            contract,
            charter=charter,
            expected_final_charter_sha256=canonical_document_sha256(charter),
            expected_protocol_teach_back_receipt_sha256=SHA_A,
            expected_code_commit=COMMIT,
            expected_dirty_diff_sha256=None,
            expected_environment_lock_sha256=SHA_B,
            expected_source_inventory_sha256=SHA_C,
            expected_candidate_inventory=candidate_inventory(),
        )


def test_forecast_rejects_memory_floor_below_observed_peak(contract):
    receipt, _ = preflight_receipt(contract)

    with pytest.raises(PreflightValidationError, match="observed peak"):
        build_preflight_forecast(
            receipt["candidate_inventory"],
            receipt["measurements"],
            host_safe_memory_floor_bytes=1,
        )


def test_capacity_receipt_is_human_owned_and_preflight_bound(contract):
    preflight, charter = preflight_receipt(contract)
    validate_preflight(preflight, charter, contract)
    capacity = capacity_receipt(preflight)

    validate_capacity_receipt(
        capacity,
        preflight,
        contract,
        expected_source_message_sha256=SHA_D,
    )

    assert capacity["acknowledged_budget"]["full_run_timeout_seconds"] == round(
        1.3 * preflight["forecast"]["scaled_full_run_forecast_wall_seconds"], 6
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt: receipt.update(preflight_receipt_sha256=SHA_E),
            "preflight receipt binding",
        ),
        (
            lambda receipt: receipt.update(source_message_sha256=SHA_E),
            "source_message_sha256 binding",
        ),
        (
            lambda receipt: receipt.update(authorship_attestation="advisor_drafted"),
            "authorship_attestation",
        ),
        (
            lambda receipt: receipt.update(reason_in_Sofia_words="TBD"),
            "reason",
        ),
        (
            lambda receipt: receipt["acknowledged_budget"].update(
                transient_retry_limit=2
            ),
            "acknowledged budget",
        ),
        (
            lambda receipt: receipt.update(
                selected_current_run_cutoff_utc="2026-08-16T20:14:00Z"
            ),
            "cutoff",
        ),
    ],
)
def test_capacity_receipt_rejects_authorship_provenance_or_budget_drift(
    contract, mutation, message
):
    preflight, _ = preflight_receipt(contract)
    receipt = capacity_receipt(preflight)
    mutation(receipt)

    with pytest.raises(PreflightValidationError, match=message):
        validate_capacity_receipt(
            receipt,
            preflight,
            contract,
            expected_source_message_sha256=SHA_D,
        )


def test_no_go_capacity_receipt_is_valid_evidence_but_cannot_authorize(contract):
    preflight, charter = preflight_receipt(contract)
    capacity = capacity_receipt(preflight)
    capacity["decision"] = "no_go"
    capacity["selected_current_run_cutoff_utc"] = None
    validate_capacity_receipt(
        capacity,
        preflight,
        contract,
        expected_source_message_sha256=SHA_D,
    )

    with pytest.raises(PreflightValidationError, match="capacity decision is not go"):
        authorize_full_locked_run(
            preflight,
            capacity,
            contract,
            charter=charter,
            expected_final_charter_sha256=canonical_document_sha256(charter),
            expected_protocol_teach_back_receipt_sha256=SHA_A,
            expected_code_commit=COMMIT,
            expected_dirty_diff_sha256=None,
            expected_environment_lock_sha256=SHA_B,
            expected_source_inventory_sha256=SHA_C,
            expected_candidate_inventory=candidate_inventory(),
            expected_capacity_source_message_sha256=SHA_D,
            run_started_at_utc="2026-08-16T20:20:00Z",
            disk_available_bytes=100_000_000_000,
            host_available_memory_bytes=10_000_000_000,
            parallel_worker_count=2,
        )


def authorize(contract: dict, **overrides) -> dict:
    preflight, charter = preflight_receipt(contract)
    capacity = capacity_receipt(preflight)
    kwargs = {
        "run_started_at_utc": "2026-08-16T20:20:00Z",
        "disk_available_bytes": 100_000_000_000,
        "host_available_memory_bytes": 10_000_000_000,
        "parallel_worker_count": 2,
    }
    kwargs.update(overrides)
    return authorize_full_locked_run(
        preflight,
        capacity,
        contract,
        charter=charter,
        expected_final_charter_sha256=canonical_document_sha256(charter),
        expected_protocol_teach_back_receipt_sha256=SHA_A,
        expected_code_commit=COMMIT,
        expected_dirty_diff_sha256=None,
        expected_environment_lock_sha256=SHA_B,
        expected_source_inventory_sha256=SHA_C,
        expected_candidate_inventory=candidate_inventory(),
        expected_capacity_source_message_sha256=SHA_D,
        **kwargs,
    )


def test_authorization_derives_exact_budget_and_effective_deadline(contract):
    authorization = authorize(contract)
    preflight, _ = preflight_receipt(contract)
    capacity = capacity_receipt(preflight)
    budget = capacity["acknowledged_budget"]
    run_start = datetime(2026, 8, 16, 20, 20, tzinfo=timezone.utc)
    timeout_deadline = run_start + timedelta(seconds=budget["full_run_timeout_seconds"])
    cutoff = datetime(2026, 8, 16, 23, 0, tzinfo=timezone.utc)

    assert authorization["budget"] == budget
    assert authorization["effective_deadline_utc"] == min(
        timeout_deadline, cutoff
    ).isoformat().replace("+00:00", "Z")
    assert authorization["preflight_receipt_sha256"] == preflight_receipt_sha256(
        preflight
    )
    assert authorization["capacity_receipt_sha256"] == capacity_receipt_sha256(
        capacity
    )
    assert authorization["partial_outcome_inspection_allowed"] is False
    assert authorization["scientific_decisions_allowed"] is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"disk_available_bytes": 1}, "disk headroom"),
        ({"host_available_memory_bytes": 1}, "memory floor"),
        ({"parallel_worker_count": 3}, "parallel worker"),
        ({"run_started_at_utc": "2026-08-16T23:01:00Z"}, "cutoff"),
    ],
)
def test_authorization_fails_closed_on_runtime_capacity_drift(
    contract, overrides, message
):
    with pytest.raises(PreflightValidationError, match=message):
        authorize(contract, **overrides)


def test_contract_drift_is_rejected_before_receipt_validation(contract):
    changed = deepcopy(contract)
    changed["execution_gate_contract"]["full_locked_run"]["transient_retry_limit"] = 2

    with pytest.raises(PreflightValidationError, match="contract digest"):
        validate_preflight_contract(changed)


def test_receipt_digests_change_under_tampering(contract):
    preflight, _ = preflight_receipt(contract)
    capacity = capacity_receipt(preflight)
    preflight_digest = preflight_receipt_sha256(preflight)
    capacity_digest = capacity_receipt_sha256(capacity)

    preflight["measurements"][0]["total_wall_seconds"] += 1.0
    capacity["reason_in_Sofia_words"] += " Changed."

    assert preflight_receipt_sha256(preflight) != preflight_digest
    assert capacity_receipt_sha256(capacity) != capacity_digest
