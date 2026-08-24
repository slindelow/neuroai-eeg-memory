"""Synthetic tests for the pre-result M1 manifest safety boundary.

These tests exercise structure, provenance, and split invariants only. They never load a
real trial label, build an EEG feature, fit a model, or inspect held-out performance.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

import peers.m1_manifests as manifest_module
from peers.m1_manifests import (
    ManifestValidationError,
    _rename_directory_noreplace,
    assert_generation_ready,
    canonical_document_sha256,
    canonical_manifest_bytes,
    canonical_row_sha256,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    event_provenance_sha256,
    load_manifest_bundle,
    load_json_unique,
    manifest_sha256,
    validate_cross_condition_intersection,
    validate_cross_fit_key_provenance,
    validate_exact_key_alignment,
    validate_manifest,
    validate_prediction_key_sets,
    validate_protocol_teach_back_receipt,
    validate_protocol_teach_back_transcript,
    validate_roi_binding,
    write_manifest_bundle_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "m1-manifest-contracts.json"
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
ROI_PATH = ROOT / "docs" / "roi-map-m1-v1.json"
SHA = "a" * 64
FINAL_CHARTER_SHA = "c" * 64
DECISION_LOG_SHA = "d" * 64
PROTOCOL_TOPICS = (
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
EXPECTED_EXECUTION_GATES = {
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


@pytest.fixture(scope="module")
def contract():
    return load_json_unique(CONTRACT_PATH)


def trial_row(
    *,
    item_num: int,
    epoch_row: int,
    condition: str = "ocular-removed",
    recalled: bool | None = None,
    condition_survivor: bool = True,
    outer_role: str = "train",
    session: int = 0,
    trial: int = 1,
) -> dict:
    return {
        "subject_id": "sub-LTP001",
        "session": session,
        "trial": trial,
        "item_num": item_num,
        "condition": condition,
        "serial_position": item_num,
        "list_number": trial,
        "onset_s": float(item_num),
        "recalled": item_num % 2 == 0 if recalled is None else recalled,
        "source_events_id": f"sub-LTP001_ses-{session}_events.tsv",
        "source_events_sha256": SHA,
        "derivative_id": f"sub-LTP001_ses-{session}_{condition}_epo.fif",
        "derivative_sha256": SHA,
        "epoch_row": epoch_row,
        "included": True,
        "exclusion_reason_code": "INCLUDED",
        "condition_survivor": condition_survivor,
        "outer_role": outer_role,
    }


def manifest(kind: str, rows: list[dict]) -> dict:
    schema_slug = {
        "inner_fold": "inner-fold",
        "cross_condition_intersection": "cross-condition-intersection",
    }.get(kind, kind)
    return {
        "schema_id": f"neuroai-m1-{schema_slug}-v1",
        "schema_version": "1.2" if kind == "attrition" else "1.0",
        "manifest_kind": kind,
        "created_at_utc": "2026-08-16T20:00:00Z",
        "charter_id": "neuroai-m1-v1",
        "charter_version": "1.0-test",
        "charter_sha256": SHA,
        "roi_map_id": "neuroai-m1-roi-v1",
        "roi_map_sha256": SHA,
        "code_commit": "b" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "synthetic-test-only",
        "source_inventory_sha256": SHA,
        "rows": rows,
    }


def protocol_teach_back_transcript() -> dict:
    messages = []
    for index, topic in enumerate(PROTOCOL_TOPICS, start=1):
        text = f"Synthetic Sofia-owned response for {topic}; no project conclusion is present."
        messages.append(
            {
                "message_id": f"sofia-t{index}",
                "role": "Sofia Lindelow",
                "topic": topic,
                "kind": "answer",
                "text": text,
                "text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                "in_reply_to_message_id": None,
                "disposition": "accepted",
            }
        )
    prompt = "Do you approve this complete verbatim capture?"
    messages.append(
        {
            "message_id": "advisor-approval-prompt",
            "role": "scientific_advisor",
            "topic": "capture_approval",
            "kind": "capture_approval_prompt",
            "text": prompt,
            "text_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
            "in_reply_to_message_id": None,
            "disposition": "context",
        }
    )
    approval = "i approve"
    messages.append(
        {
            "message_id": "sofia-capture-approval",
            "role": "Sofia Lindelow",
            "topic": "capture_approval",
            "kind": "capture_approval",
            "text": approval,
            "text_sha256": sha256(approval.encode("utf-8")).hexdigest(),
            "in_reply_to_message_id": "advisor-approval-prompt",
            "disposition": "approval",
        }
    )
    return {
        "schema_id": "neuroai-m1-protocol-teach-back-transcript-v1",
        "schema_version": "1.0",
        "transcript_id": "synthetic-protocol-teach-back",
        "source_thread_id": "synthetic-test-thread",
        "response_owner": "Sofia Lindelow",
        "authorship_attestation": "verbatim_Sofia_owned_words",
        "capture_approval_message_id": "sofia-capture-approval",
        "messages": messages,
    }


def protocol_teach_back_receipt(*, charter_sha256: str = FINAL_CHARTER_SHA) -> dict:
    transcript = protocol_teach_back_transcript()
    response_message_ids = {
        topic: [f"sofia-t{index}"] for index, topic in enumerate(PROTOCOL_TOPICS, start=1)
    }
    by_id = {message["message_id"]: message for message in transcript["messages"]}
    responses = {
        topic: "\n\n".join(by_id[message_id]["text"] for message_id in message_ids)
        for topic, message_ids in response_message_ids.items()
    }
    responses_sha256 = sha256(
        json.dumps(
            responses,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()
    return {
        "schema_id": "neuroai-m1-protocol-teach-back-v1",
        "schema_version": "1.1",
        "receipt_id": "neuroai-m1-v1-protocol-teach-back",
        "human_gate_id": "Sofia_protocol_teach_back",
        "charter_id": "neuroai-m1-v1",
        "charter_version": "1.0-test-frozen",
        "charter_sha256": charter_sha256,
        "decision_log_sha256": DECISION_LOG_SHA,
        "answered_by": "Sofia Lindelow",
        "authorship_attestation": "verbatim_Sofia_owned_words",
        "source_thread_id": "synthetic-test-thread",
        "source_transcript_sha256": canonical_document_sha256(transcript),
        "capture_approval_message_id": "sofia-capture-approval",
        "response_message_ids": response_message_ids,
        "recorded_at_utc": "2026-08-16T20:00:00Z",
        "responses": responses,
        "responses_sha256": responses_sha256,
        "advisor_review": {
            "reviewed_at_utc": "2026-08-16T20:05:00Z",
            "reviewed_by_role": "scientific_advisor",
            "topic_verdicts": {topic: "pass" for topic in PROTOCOL_TOPICS},
            "overall_verdict": "pass",
            "open_revision_requests": [],
        },
    }


def validate_protocol_receipt(
    receipt: dict,
    *,
    final_charter_sha256: str = FINAL_CHARTER_SHA,
    decision_log_sha256: str = DECISION_LOG_SHA,
    source_transcript_sha256: str | None = None,
) -> None:
    transcript = protocol_teach_back_transcript()
    if source_transcript_sha256 is None:
        source_transcript_sha256 = canonical_document_sha256(transcript)
    validate_protocol_teach_back_receipt(
        receipt,
        final_charter_sha256=final_charter_sha256,
        expected_decision_log_sha256=decision_log_sha256,
        expected_source_transcript_sha256=source_transcript_sha256,
        transcript=transcript,
    )


def test_canonical_bytes_ignore_input_row_order(contract):
    rows = [trial_row(item_num=2, epoch_row=1), trial_row(item_num=1, epoch_row=0)]

    first = canonical_manifest_bytes(manifest("trial", rows), contract)
    second = canonical_manifest_bytes(manifest("trial", list(reversed(rows))), contract)

    assert first == second
    assert first.endswith(b"\n")
    assert manifest_sha256(manifest("trial", rows), contract) == manifest_sha256(
        manifest("trial", list(reversed(rows))), contract
    )


def test_validator_rejects_noncanonical_stored_row_order(contract):
    rows = [trial_row(item_num=2, epoch_row=1), trial_row(item_num=1, epoch_row=0)]

    with pytest.raises(ManifestValidationError, match="canonical row order"):
        validate_manifest(manifest("trial", rows), contract)


def test_duplicate_trial_key_is_rejected_with_key(contract):
    rows = [trial_row(item_num=1, epoch_row=0), trial_row(item_num=1, epoch_row=1)]

    with pytest.raises(ManifestValidationError, match="sub-LTP001.*0.*1.*1.*ocular-removed"):
        canonical_manifest_bytes(manifest("trial", rows), contract)


def test_duplicate_epoch_identity_is_rejected(contract):
    rows = [trial_row(item_num=1, epoch_row=0), trial_row(item_num=2, epoch_row=0)]

    with pytest.raises(ManifestValidationError, match="duplicate epoch identity"):
        canonical_manifest_bytes(manifest("trial", rows), contract)


@pytest.mark.parametrize("field", ["auc", "prob_m1", "selected_hyperparameter"])
def test_forbidden_performance_or_model_field_is_rejected(contract, field):
    row = trial_row(item_num=1, epoch_row=0)
    row[field] = 0.9

    with pytest.raises(ManifestValidationError, match="forbidden field"):
        canonical_manifest_bytes(manifest("trial", [row]), contract)


def test_unknown_and_wrong_type_fields_are_rejected(contract):
    unknown = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    unknown["helpful_note"] = "not part of the frozen envelope"
    with pytest.raises(ManifestValidationError, match="unknown envelope fields"):
        canonical_manifest_bytes(unknown, contract)

    wrong_type = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    wrong_type["rows"][0]["trial"] = True
    with pytest.raises(ManifestValidationError, match="trial.*integer"):
        canonical_manifest_bytes(wrong_type, contract)


def test_nonfinite_value_and_digest_tamper_are_rejected(contract):
    bad_value = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    bad_value["rows"][0]["onset_s"] = float("nan")
    with pytest.raises(ManifestValidationError, match="non-finite"):
        canonical_manifest_bytes(bad_value, contract)

    original = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    expected = manifest_sha256(original, contract)
    tampered = deepcopy(original)
    tampered["rows"][0]["recalled"] = not tampered["rows"][0]["recalled"]
    with pytest.raises(ManifestValidationError, match="digest mismatch"):
        validate_manifest(tampered, contract, expected_sha256=expected)


def test_schema_and_external_binding_tamper_are_rejected(contract):
    wrong_schema = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    wrong_schema["schema_id"] = "neuroai-m1-not-the-trial-schema-v1"
    with pytest.raises(ManifestValidationError, match="schema_id mismatch"):
        validate_manifest(wrong_schema, contract)

    good = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    with pytest.raises(ManifestValidationError, match="binding mismatch for charter_sha256"):
        validate_manifest(good, contract, expected_bindings={"charter_sha256": "c" * 64})


def test_outer_test_must_be_highest_eligible_session(contract):
    row = {
        "subject_id": "sub-LTP001",
        "condition": "ocular-removed",
        "sessions_present": [0, 1, 2],
        "sessions_technically_eligible": [0, 1, 2],
        "outer_train_sessions": [0, 2],
        "outer_test_session": 1,
        "eligible": True,
        "eligibility_reason_code": "ELIGIBLE",
        "per_session_counts": {},
        "source_sha256s": {"events": SHA, "derivatives": SHA},
        "channel_schema_id": "hydrocel-125-v1",
    }

    with pytest.raises(ManifestValidationError, match="highest technically eligible session"):
        canonical_manifest_bytes(manifest("cohort", [row]), contract)


def test_inner_fold_cannot_fit_the_validation_or_outer_test_session(contract):
    rows = []
    for fold_id, validation, fit in [("fold-0", 0, [1]), ("fold-1", 1, [0, 2])]:
        rows.append(
            {
                "subject_id": "sub-LTP001",
                "condition": "ocular-removed",
                "inner_fold_id": fold_id,
                "fit_sessions": fit,
                "validation_session": validation,
                "outer_test_session": 2,
                "fit_class_counts": {"0": 5, "1": 5},
                "validation_class_counts": {"0": 5, "1": 5},
                "fit_keys_sha256": SHA,
                "validation_keys_sha256": SHA,
                "eligible": True,
                "feasibility_reason_code": "ELIGIBLE",
            }
        )

    with pytest.raises(ManifestValidationError, match="outer test session"):
        canonical_manifest_bytes(manifest("inner_fold", rows), contract)


def test_eligible_inner_fold_requires_both_classes(contract):
    rows = []
    for fold_id, validation, fit in [("fold-0", 0, [1]), ("fold-1", 1, [0])]:
        rows.append(
            {
                "subject_id": "sub-LTP001",
                "condition": "ocular-removed",
                "inner_fold_id": fold_id,
                "fit_sessions": fit,
                "validation_session": validation,
                "outer_test_session": 2,
                "fit_class_counts": {"0": 5, "1": 5},
                "validation_class_counts": {"0": 5, "1": 0 if fold_id == "fold-0" else 5},
                "fit_keys_sha256": SHA,
                "validation_keys_sha256": SHA,
                "eligible": True,
                "feasibility_reason_code": "ELIGIBLE",
            }
        )

    with pytest.raises(ManifestValidationError, match="validation_class_counts.*both classes"):
        canonical_manifest_bytes(manifest("inner_fold", rows), contract)


def attrition_row(
    condition: str,
    flow_id: str,
    count_unit: str,
    stage_index: int,
    stage: str,
    input_count: int,
    excluded_by_reason: dict[str, int],
    survivor_count: int,
) -> dict:
    return {
        "condition": condition,
        "flow_id": flow_id,
        "count_unit": count_unit,
        "stage_index": stage_index,
        "stage": stage,
        "input_count": input_count,
        "excluded_by_reason": excluded_by_reason,
        "survivor_count": survivor_count,
    }


def valid_attrition_rows() -> list[dict]:
    rows = []
    for condition in ("ocular-kept", "ocular-removed"):
        rows.extend(
            [
                attrition_row(
                    condition,
                    "trial_analysis",
                    "trial",
                    0,
                    "behavioral_event_filter",
                    12,
                    {"PRACTICE_OR_NONPOSITIVE_TRIAL": 1, "NEGATIVE_SAMPLE": 1},
                    10,
                ),
                attrition_row(
                    condition,
                    "trial_analysis",
                    "trial",
                    1,
                    "condition_technical_survival",
                    10,
                    {"MISSING_DERIVATIVE": 1},
                    9,
                ),
                attrition_row(
                    condition,
                    "trial_analysis",
                    "trial",
                    2,
                    "event_epoch_alignment",
                    9,
                    {"AUTOREJECT_DROPPED": 1},
                    8,
                ),
                attrition_row(
                    condition,
                    "trial_analysis",
                    "trial",
                    3,
                    "eligible_subject_trials",
                    8,
                    {"TOO_FEW_SESSIONS": 2},
                    6,
                ),
                attrition_row(
                    condition,
                    "subject_condition_eligibility",
                    "subject_condition",
                    0,
                    "candidate_subject_conditions",
                    3,
                    {},
                    3,
                ),
                attrition_row(
                    condition,
                    "subject_condition_eligibility",
                    "subject_condition",
                    1,
                    "technical_session_eligibility",
                    3,
                    {"MISSING_DERIVATIVE": 1},
                    2,
                ),
                attrition_row(
                    condition,
                    "subject_condition_eligibility",
                    "subject_condition",
                    2,
                    "minimum_complete_sessions",
                    2,
                    {},
                    2,
                ),
                attrition_row(
                    condition,
                    "subject_condition_eligibility",
                    "subject_condition",
                    3,
                    "outer_partition_class_feasibility",
                    2,
                    {},
                    2,
                ),
                attrition_row(
                    condition,
                    "subject_condition_eligibility",
                    "subject_condition",
                    4,
                    "inner_fold_class_feasibility",
                    2,
                    {"SINGLE_CLASS_INNER_FOLD": 1},
                    1,
                ),
                attrition_row(
                    condition,
                    "cross_condition_sensitivity",
                    "trial",
                    0,
                    "cross_condition_intersection",
                    6,
                    {"ABSENT_FROM_OTHER_CONDITION": 1},
                    5,
                ),
            ]
        )
    return sorted(rows, key=lambda row: (row["condition"], row["flow_id"], row["stage_index"]))


def test_attrition_flows_support_multiple_reasons_and_reconcile(contract):
    attrition = manifest("attrition", valid_attrition_rows())
    validate_manifest(attrition, contract)
    assert manifest_sha256(attrition, contract)

    bad_sum = deepcopy(attrition)
    row = next(row for row in bad_sum["rows"] if row["stage"] == "behavioral_event_filter")
    row["survivor_count"] = 11
    with pytest.raises(ManifestValidationError, match="stage counts do not reconcile"):
        validate_manifest(bad_sum, contract)


def test_attrition_distinguishes_autoreject_drop_from_alignment_failure(contract):
    attrition = manifest("attrition", valid_attrition_rows())
    stage = next(
        row
        for row in attrition["rows"]
        if row["condition"] == "ocular-removed"
        and row["flow_id"] == "trial_analysis"
        and row["stage_index"] == 2
    )
    assert stage["excluded_by_reason"] == {"AUTOREJECT_DROPPED": 1}
    assert contract["manifest_schemas"]["attrition"]["flows"]["trial_analysis"][
        "stages"
    ][2]["reason_code_order"] == [
        "AUTOREJECT_DROPPED",
        "TRIAL_EPOCH_KEY_MISMATCH",
    ]

    alignment_failure = deepcopy(attrition)
    failed_stage = next(
        row
        for row in alignment_failure["rows"]
        if row["condition"] == "ocular-removed"
        and row["flow_id"] == "trial_analysis"
        and row["stage_index"] == 2
    )
    failed_stage["excluded_by_reason"] = {"TRIAL_EPOCH_KEY_MISMATCH": 1}
    validate_manifest(alignment_failure, contract)


def test_attrition_rejects_continuity_unit_reason_or_stage_drift(contract):
    attrition = manifest("attrition", valid_attrition_rows())

    bad_continuity = deepcopy(attrition)
    row = next(
        row
        for row in bad_continuity["rows"]
        if row["condition"] == "ocular-kept" and row["stage_index"] == 1
        and row["flow_id"] == "trial_analysis"
    )
    row["input_count"], row["survivor_count"] = 9, 8
    with pytest.raises(ManifestValidationError, match="stage continuity"):
        validate_manifest(bad_continuity, contract)

    bad_unit = deepcopy(attrition)
    bad_unit["rows"][0]["count_unit"] = "file"
    with pytest.raises(ManifestValidationError, match="count_unit mismatch"):
        validate_manifest(bad_unit, contract)

    bad_reason = deepcopy(attrition)
    bad_reason["rows"][0]["excluded_by_reason"] = {"OTHER": 1}
    bad_reason["rows"][0]["survivor_count"] = bad_reason["rows"][0]["input_count"] - 1
    with pytest.raises(ManifestValidationError, match="unknown attrition reason.*OTHER"):
        validate_manifest(bad_reason, contract)

    missing_stage = deepcopy(attrition)
    missing_stage["rows"] = missing_stage["rows"][1:]
    with pytest.raises(ManifestValidationError, match="attrition stage set mismatch"):
        validate_manifest(missing_stage, contract)


def test_attrition_cross_condition_input_binds_to_condition_survivors(contract):
    attrition = manifest("attrition", valid_attrition_rows())
    row = next(
        row
        for row in attrition["rows"]
        if row["condition"] == "ocular-removed"
        and row["flow_id"] == "cross_condition_sensitivity"
    )
    row["input_count"], row["survivor_count"] = 7, 6
    with pytest.raises(ManifestValidationError, match="cross-flow input binding"):
        validate_manifest(attrition, contract)


def _synthetic_final_charter() -> dict:
    charter = load_json_unique(CHARTER_PATH)
    charter["status"] = "frozen_pre_result"
    charter["version"] = "1.0-test-frozen"
    charter["scientific_decisions"]["D42"] = "D42_A_omit_balanced_accuracy"
    charter["scientific_decisions"]["D44"] = (
        "D44_C_outer_training_selected_C_qualified_meta_predictions"
    )
    charter["pending_human_gates"] = ["Sofia_protocol_teach_back"]
    return charter


def test_charter_and_manifest_contract_bind_exact_execution_gates(contract):
    charter = load_json_unique(CHARTER_PATH)

    assert charter["execution_gates"] == EXPECTED_EXECUTION_GATES
    assert contract["execution_gate_contract"] == EXPECTED_EXECUTION_GATES


@pytest.mark.parametrize(
    "mutation",
    [
        lambda gates: gates["preflight"].update(
            outcome_metrics_or_interpretation_allowed=True
        ),
        lambda gates: gates["full_locked_run"].update(transient_retry_limit=2),
        lambda gates: gates["full_locked_run"].update(
            cutoff_source="expired_calendar_deadline"
        ),
        lambda gates: gates["failure"].update(protocol_weakening_allowed=True),
    ],
)
def test_generation_ready_rejects_execution_gate_drift(contract, mutation):
    charter = _synthetic_final_charter()
    mutation(charter["execution_gates"])
    final_charter_sha256 = canonical_document_sha256(charter)

    with pytest.raises(ManifestValidationError, match="execution gate"):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            contract,
            protocol_teach_back_receipt=protocol_teach_back_receipt(
                charter_sha256=final_charter_sha256
            ),
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )


def test_generation_ready_rejects_manifest_execution_gate_contract_drift(contract):
    charter = _synthetic_final_charter()
    changed_contract = deepcopy(contract)
    changed_contract["execution_gate_contract"]["full_locked_run"][
        "transient_retry_limit"
    ] = 2
    final_charter_sha256 = canonical_document_sha256(charter)

    with pytest.raises(ManifestValidationError, match="execution gate contract drifted"):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            changed_contract,
            protocol_teach_back_receipt=protocol_teach_back_receipt(
                charter_sha256=final_charter_sha256
            ),
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )


def test_generation_ready_rejects_d42_c_until_a_replanned_contract_exists(contract):
    assert contract["scientific_decision_gates"]["D42"][
        "requires_versioned_replan_before_execution"
    ] == ["D42_C_training_only_fitted_threshold"]
    charter = _synthetic_final_charter()
    charter["scientific_decisions"]["D42"] = (
        "D42_C_training_only_fitted_threshold"
    )
    final_charter_sha256 = canonical_document_sha256(charter)

    with pytest.raises(ManifestValidationError, match="D42-C requires stop/replan"):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            contract,
            protocol_teach_back_receipt=protocol_teach_back_receipt(
                charter_sha256=final_charter_sha256
            ),
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda policy, secondary: secondary.append("balanced_accuracy"),
        lambda policy, secondary: policy.update(included=True),
        lambda policy, secondary: policy.update(threshold=0.5),
        lambda policy, secondary: policy.update(status="pending"),
    ],
)
def test_generation_ready_rejects_d42_a_omission_policy_drift(contract, mutation):
    charter = _synthetic_final_charter()
    mutation(
        charter["estimands"]["balanced_accuracy_policy"],
        charter["estimands"]["secondary"],
    )
    final_charter_sha256 = canonical_document_sha256(charter)

    with pytest.raises(ManifestValidationError, match="D42-A"):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            contract,
            protocol_teach_back_receipt=protocol_teach_back_receipt(
                charter_sha256=final_charter_sha256
            ),
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )


def test_actual_charter_has_all_scientific_decisions_and_fails_before_teach_back(
    contract,
):
    charter = load_json_unique(CHARTER_PATH)
    roi_map = load_json_unique(ROI_PATH)

    d26_pending = deepcopy(charter)
    d26_pending["features"]["normalization"] = "PENDING_D26"
    d26_pending["pending_human_gates"].insert(0, "D26_normalization_estimand")
    with pytest.raises(ManifestValidationError, match="D26"):
        assert_generation_ready(
            d26_pending,
            roi_map,
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )

    d29_pending = deepcopy(charter)
    d29_pending["features"]["analysis_window_s"] = "PENDING_D29"
    d29_pending["features"]["time_bins_s"] = "PENDING_D29"
    d29_pending["pending_human_gates"].insert(0, "D29_window_and_midpoint")
    with pytest.raises(ManifestValidationError, match="D29"):
        assert_generation_ready(
            d29_pending,
            roi_map,
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )

    assert charter["features"]["normalization"]["decision"] == (
        "D26_A_strict_inductive"
    )
    assert charter["scientific_decisions"]["D42"] == (
        "D42_A_omit_balanced_accuracy"
    )
    assert charter["scientific_decisions"]["D44"] == (
        "D44_C_outer_training_selected_C_qualified_meta_predictions"
    )
    assert "D42_balanced_accuracy_handling" not in charter["pending_human_gates"]
    assert "balanced_accuracy" not in charter["estimands"]["secondary"]
    assert charter["estimands"]["balanced_accuracy_policy"] == {
        "decision": "D42_A_omit_balanced_accuracy",
        "included": False,
        "status": "deliberately_omitted_not_computed",
        "threshold": None,
    }
    assert "D44_M1_hyperparameter_selection_scope" not in charter[
        "pending_human_gates"
    ]
    assert charter["learners"]["M3"]["meta_training_predictions"] == (
        "session_out_coefficients_and_transforms_conditional_on_"
        "outer_training_selected_C"
    )
    assert charter["cross_fitting"] == {
        "unit": "session",
        "in_sample_first_level_predictions_allowed": False,
        "record_row_fit_key_hashes": True,
        "outer_test_first_level_fit_scope": "all_outer_training_rows_only",
        "meta_fit_scope": (
            "session_out_coefficients_and_transforms_conditional_on_"
            "outer_training_selected_C"
        ),
        "M1_C_selection_scope": (
            "outer_training_wide_inner_session_C_selection_including_"
            "meta_prediction_session"
        ),
        "predicted_session_labels_may_influence_selected_C": True,
        "outer_test_session_used_for_selection": False,
        "fully_session_cross_fitted_claim_allowed": False,
        "provenance_contract": "neuroai-m1-d44-c-cross-fit-provenance-v1",
    }

    d42_pending = deepcopy(charter)
    d42_pending["scientific_decisions"]["D42"] = None
    d42_pending["pending_human_gates"].insert(
        0, "D42_balanced_accuracy_handling"
    )
    with pytest.raises(ManifestValidationError, match="D42"):
        assert_generation_ready(
            d42_pending,
            roi_map,
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )

    d44_pending = deepcopy(charter)
    d44_pending["scientific_decisions"]["D44"] = None
    d44_pending["pending_human_gates"].insert(
        0, "D44_M1_hyperparameter_selection_scope"
    )
    with pytest.raises(ManifestValidationError, match="D44"):
        assert_generation_ready(
            d44_pending,
            roi_map,
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )

    final_charter_sha256 = canonical_document_sha256(charter)
    with pytest.raises(ManifestValidationError, match="teach-back is unresolved"):
        assert_generation_ready(
            charter,
            roi_map,
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )

    assert charter["features"]["analysis_window_s"] == [0.3, 1.6]
    assert charter["features"]["time_bins_s"] == [[0.3, 0.95], [0.95, 1.6]]

    transductive = deepcopy(charter)
    transductive["features"]["normalization"][
        "held_session_distribution_access"
    ] = True
    with pytest.raises(ManifestValidationError, match="strict-inductive"):
        assert_generation_ready(
            transductive,
            roi_map,
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )


def test_protocol_teach_back_receipt_accepts_complete_bound_synthetic_fixture(contract):
    receipt = protocol_teach_back_receipt()

    validate_protocol_receipt(receipt)
    receipt_contract = contract["protocol_teach_back_receipt"]
    charter = load_json_unique(CHARTER_PATH)
    assert receipt_contract["response_topics"] == list(PROTOCOL_TOPICS)
    assert set(receipt_contract["exact_fields"]) == set(receipt)
    assert charter["protocol_teach_back"]["required_topic_count"] == len(PROTOCOL_TOPICS)
    assert charter["protocol_teach_back"]["response_owner"] == "Sofia Lindelow"


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda receipt: receipt.update(answered_by="scientific advisor"), "answered_by"),
        (
            lambda receipt: receipt.update(authorship_attestation="advisor_drafted"),
            "authorship_attestation",
        ),
        (lambda receipt: receipt.update(extra_field=True), "unknown protocol receipt"),
        (
            lambda receipt: receipt.update(source_transcript_sha256="not-a-hash"),
            "source_transcript_sha256",
        ),
        (
            lambda receipt: receipt["responses"].pop("failure_mode"),
            "response topic set",
        ),
        (
            lambda receipt: receipt["responses"].update(failure_mode="TBD"),
            "placeholder",
        ),
        (
            lambda receipt: receipt["advisor_review"]["topic_verdicts"].update(
                failure_mode="revise"
            ),
            "topic verdict",
        ),
        (
            lambda receipt: receipt["advisor_review"]["open_revision_requests"].append(
                "Explain one concrete failure mode."
            ),
            "revision requests",
        ),
        (
            lambda receipt: receipt["advisor_review"].update(
                reviewed_at_utc="2026-08-16T19:00:00Z"
            ),
            "predate",
        ),
    ],
)
def test_protocol_teach_back_receipt_rejects_incomplete_or_unaccepted_evidence(
    mutation, error
):
    receipt = protocol_teach_back_receipt()
    mutation(receipt)

    with pytest.raises(ManifestValidationError, match=error):
        validate_protocol_receipt(receipt)


def test_protocol_teach_back_receipt_rejects_stale_or_tampered_bindings():
    stale = protocol_teach_back_receipt(charter_sha256="b" * 64)
    with pytest.raises(ManifestValidationError, match="charter_sha256 binding"):
        validate_protocol_receipt(stale)

    tampered = protocol_teach_back_receipt()
    tampered["responses"]["failure_mode"] += " Tampered after review."
    with pytest.raises(ManifestValidationError, match="derived from selected transcript"):
        validate_protocol_receipt(tampered)

    receipt = protocol_teach_back_receipt()
    with pytest.raises(ManifestValidationError, match="decision_log_sha256 binding"):
        validate_protocol_receipt(receipt, decision_log_sha256="f" * 64)
    with pytest.raises(ManifestValidationError, match="source_transcript_sha256 binding"):
        validate_protocol_receipt(receipt, source_transcript_sha256="f" * 64)


def test_generation_ready_rejects_boolean_and_requires_bound_teach_back_receipt(contract):
    charter = _synthetic_final_charter()
    roi_map = load_json_unique(ROI_PATH)
    final_charter_sha256 = canonical_document_sha256(charter)

    with pytest.raises(ManifestValidationError, match="receipt must be an object"):
        assert_generation_ready(
            charter,
            roi_map,
            contract,
            protocol_teach_back_receipt=True,
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )

    assert_generation_ready(
        charter,
        roi_map,
        contract,
        protocol_teach_back_receipt=protocol_teach_back_receipt(
            charter_sha256=final_charter_sha256
        ),
        final_charter_sha256=final_charter_sha256,
        decision_log_sha256=DECISION_LOG_SHA,
        protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
    )


def test_generation_ready_recomputes_final_charter_digest(contract):
    charter = _synthetic_final_charter()
    roi_map = load_json_unique(ROI_PATH)
    original_sha256 = canonical_document_sha256(charter)
    receipt = protocol_teach_back_receipt(charter_sha256=original_sha256)

    changed = deepcopy(charter)
    changed["claim_ceiling"] += " Synthetic drift."
    with pytest.raises(ManifestValidationError, match="canonical digest mismatch"):
        assert_generation_ready(
            changed,
            roi_map,
            contract,
            protocol_teach_back_receipt=receipt,
            final_charter_sha256=original_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )


def test_generation_ready_rejects_unknown_scientific_decision_value(contract):
    charter = _synthetic_final_charter()
    charter["scientific_decisions"]["D42"] = "D42_UNDECLARED_SHORTCUT"
    with pytest.raises(ManifestValidationError, match="D42"):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda charter: charter["scientific_decisions"].pop("D29"), "exactly D26"),
        (
            lambda charter: charter["scientific_decisions"].update(D26="D26_B"),
            "D26 decision registry",
        ),
        (
            lambda charter: charter["scientific_decisions"].update(D29="D29_B"),
            "D29 decision registry",
        ),
    ],
)
def test_generation_ready_rejects_decision_registry_drift(contract, mutation, error):
    charter = _synthetic_final_charter()
    mutation(charter)
    with pytest.raises(ManifestValidationError, match=error):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            contract,
            protocol_teach_back_receipt=None,
            final_charter_sha256=None,
            decision_log_sha256=None,
            protocol_teach_back_transcript=None,
            source_transcript_sha256=None,
        )


def test_generation_ready_does_not_depend_on_decision_gate_object_order(contract):
    charter = _synthetic_final_charter()
    final_charter_sha256 = canonical_document_sha256(charter)
    reordered = deepcopy(contract)
    reordered["scientific_decision_gates"] = {
        key: reordered["scientific_decision_gates"][key] for key in ("D44", "D42")
    }

    assert_generation_ready(
        charter,
        load_json_unique(ROI_PATH),
        reordered,
        protocol_teach_back_receipt=protocol_teach_back_receipt(
            charter_sha256=final_charter_sha256
        ),
        final_charter_sha256=final_charter_sha256,
        decision_log_sha256=DECISION_LOG_SHA,
        protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
    )


def test_roi_binding_rejects_overlap_and_accepts_the_frozen_documents():
    charter = load_json_unique(CHARTER_PATH)
    roi_map = load_json_unique(ROI_PATH)
    validate_roi_binding(charter, roi_map)

    overlapping_charter = deepcopy(charter)
    overlapping = deepcopy(roi_map)
    repeated_channel = overlapping["rois"]["LAI"]["channels"][0]
    overlapping["rois"]["LAS"]["channels"][0] = repeated_channel
    overlapping_charter["features"]["roi_channels"]["LAS"][0] = repeated_channel
    with pytest.raises(ManifestValidationError, match="overlap"):
        validate_roi_binding(overlapping_charter, overlapping)


def test_json_loader_rejects_duplicate_object_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"charter_id":"first","charter_id":"second"}\n')

    with pytest.raises(ManifestValidationError, match="duplicate JSON key.*charter_id"):
        load_json_unique(path)


def test_canonical_output_round_trips_as_json(contract):
    data = canonical_manifest_bytes(
        manifest("trial", [trial_row(item_num=1, epoch_row=0)]), contract
    )
    decoded = json.loads(data)
    assert decoded["rows"][0]["item_num"] == 1


def test_atomic_manifest_bundle_round_trip_is_canonical_and_immutable(tmp_path, contract):
    unsorted = manifest(
        "trial",
        [trial_row(item_num=2, epoch_row=1), trial_row(item_num=1, epoch_row=0)],
    )
    bundle = tmp_path / "trial-manifest"
    digest = write_manifest_bundle_atomic(bundle, unsorted, contract)

    assert {path.name for path in bundle.iterdir()} == {"manifest.json", "manifest.sha256"}
    assert (bundle / "manifest.sha256").read_text() == f"{digest}\n"
    loaded = load_manifest_bundle(bundle, contract)
    assert [row["item_num"] for row in loaded["rows"]] == [1, 2]

    with pytest.raises(ManifestValidationError, match="already exists"):
        write_manifest_bundle_atomic(bundle, unsorted, contract)


def test_noreplace_rename_cannot_replace_an_existing_empty_directory(tmp_path):
    source = tmp_path / "complete-temporary-bundle"
    source.mkdir()
    (source / "payload").write_text("new evidence\n")
    destination = tmp_path / "existing-bundle"
    destination.mkdir()

    with pytest.raises(ManifestValidationError, match="already exists"):
        _rename_directory_noreplace(source, destination)

    assert (source / "payload").read_text() == "new evidence\n"
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_manifest_bundle_contract_requires_exclusive_rename(contract):
    bundle_contract = contract["manifest_bundle"]

    assert contract["version"] == "1.8-frozen"
    assert bundle_contract["ordinary_rename_allowed"] is False
    assert bundle_contract["exclusive_rename_implementations"] == {
        "darwin": "renamex_np_RENAME_EXCL",
        "linux": "renameat2_RENAME_NOREPLACE",
    }
    assert (
        bundle_contract["unsupported_platform_or_filesystem_policy"]
        == "fail_closed_without_publication"
    )


def test_noreplace_rename_fails_closed_on_unsupported_platform(tmp_path, monkeypatch):
    source = tmp_path / "complete-temporary-bundle"
    source.mkdir()
    destination = tmp_path / "unpublished-bundle"
    monkeypatch.setattr(manifest_module.sys, "platform", "unsupported-test-platform")

    with pytest.raises(ManifestValidationError, match="unsupported on platform"):
        _rename_directory_noreplace(source, destination)

    assert source.is_dir()
    assert not destination.exists()


def test_manifest_bundle_publication_race_preserves_competing_destination(
    tmp_path, contract, monkeypatch
):
    valid = manifest("trial", [trial_row(item_num=1, epoch_row=0)])
    bundle = tmp_path / "raced-bundle"
    original_rename = manifest_module._rename_directory_noreplace

    def create_destination_before_publish(source, destination):
        destination.mkdir()
        return original_rename(source, destination)

    monkeypatch.setattr(
        manifest_module,
        "_rename_directory_noreplace",
        create_destination_before_publish,
    )
    with pytest.raises(ManifestValidationError, match="already exists"):
        write_manifest_bundle_atomic(bundle, valid, contract)

    assert bundle.is_dir()
    assert list(bundle.iterdir()) == []
    assert list(tmp_path.glob(".raced-bundle.tmp-*")) == []


def test_manifest_bundle_rejects_partial_extra_noncanonical_or_digest_tamper(tmp_path, contract):
    valid = manifest("trial", [trial_row(item_num=1, epoch_row=0)])

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "manifest.json").write_bytes(canonical_manifest_bytes(valid, contract))
    with pytest.raises(ManifestValidationError, match="exactly manifest.json and manifest.sha256"):
        load_manifest_bundle(partial, contract)

    extra = tmp_path / "extra"
    write_manifest_bundle_atomic(extra, valid, contract)
    (extra / "unexpected.txt").write_text("not permitted\n")
    with pytest.raises(ManifestValidationError, match="exactly manifest.json and manifest.sha256"):
        load_manifest_bundle(extra, contract)

    noncanonical = tmp_path / "noncanonical"
    write_manifest_bundle_atomic(noncanonical, valid, contract)
    decoded = json.loads((noncanonical / "manifest.json").read_text())
    (noncanonical / "manifest.json").write_text(json.dumps(decoded, indent=2) + "\n")
    with pytest.raises(ManifestValidationError, match="not canonical bytes"):
        load_manifest_bundle(noncanonical, contract)

    bad_digest = tmp_path / "bad-digest"
    write_manifest_bundle_atomic(bad_digest, valid, contract)
    (bad_digest / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(ManifestValidationError, match="digest mismatch"):
        load_manifest_bundle(bad_digest, contract)


def test_invalid_manifest_never_publishes_bundle(tmp_path, contract):
    invalid = manifest(
        "trial",
        [trial_row(item_num=1, epoch_row=0), trial_row(item_num=1, epoch_row=1)],
    )
    bundle = tmp_path / "invalid"
    with pytest.raises(ManifestValidationError, match="duplicate trial row key"):
        write_manifest_bundle_atomic(bundle, invalid, contract)
    assert not bundle.exists()


def test_exact_key_alignment_ignores_order_and_rejects_set_drift():
    key_1 = ("sub-LTP001", 0, 1, 1)
    key_2 = ("sub-LTP001", 0, 1, 2)
    validate_exact_key_alignment(
        [key_1, key_2], [key_2, key_1], expected_name="events", observed_name="epochs"
    )

    with pytest.raises(ManifestValidationError, match="missing.*item_num=2"):
        validate_exact_key_alignment(
            [key_1, key_2], [key_1], expected_name="events", observed_name="epochs"
        )
    with pytest.raises(ManifestValidationError, match="extra.*item_num=2"):
        validate_exact_key_alignment(
            [key_1], [key_1, key_2], expected_name="events", observed_name="epochs"
        )
    with pytest.raises(ManifestValidationError, match="duplicate epochs key"):
        validate_exact_key_alignment(
            [key_1], [key_1, key_1], expected_name="events", observed_name="epochs"
        )


def condition_manifests() -> tuple[dict, dict, dict]:
    removed_rows = [
        trial_row(item_num=1, epoch_row=0, condition="ocular-removed"),
        trial_row(item_num=2, epoch_row=1, condition="ocular-removed"),
    ]
    kept_rows = [
        trial_row(item_num=1, epoch_row=10, condition="ocular-kept"),
        trial_row(item_num=2, epoch_row=11, condition="ocular-kept"),
        trial_row(
            item_num=3,
            epoch_row=12,
            condition="ocular-kept",
            condition_survivor=False,
            outer_role="excluded",
        ),
    ]
    removed = manifest("trial", removed_rows)
    kept = manifest("trial", kept_rows)
    intersection_rows = []
    for removed_row, kept_row in zip(removed_rows, kept_rows[:2], strict=True):
        intersection_rows.append(
            {
                "subject_id": removed_row["subject_id"],
                "session": removed_row["session"],
                "trial": removed_row["trial"],
                "item_num": removed_row["item_num"],
                "ocular_removed_row_sha256": canonical_row_sha256(removed_row),
                "ocular_kept_row_sha256": canonical_row_sha256(kept_row),
                "recalled": removed_row["recalled"],
                "event_provenance_sha256": event_provenance_sha256(removed_row),
                "ocular_removed_epoch_row": removed_row["epoch_row"],
                "ocular_kept_epoch_row": kept_row["epoch_row"],
            }
        )
    intersection = manifest("cross_condition_intersection", intersection_rows)
    return removed, kept, intersection


def test_cross_condition_intersection_is_exact_and_provenance_bound(contract):
    removed, kept, intersection = condition_manifests()
    validate_cross_condition_intersection(intersection, removed, kept, contract)

    missing = deepcopy(intersection)
    missing["rows"] = missing["rows"][:1]
    with pytest.raises(ManifestValidationError, match="missing.*item_num=2"):
        validate_cross_condition_intersection(missing, removed, kept, contract)

    changed_label = deepcopy(kept)
    changed_label["rows"][0]["recalled"] = not changed_label["rows"][0]["recalled"]
    with pytest.raises(ManifestValidationError, match="recall label disagreement"):
        validate_cross_condition_intersection(intersection, removed, changed_label, contract)


def test_cross_condition_intersection_rejects_row_or_event_tamper(contract):
    removed, kept, intersection = condition_manifests()

    bad_row_hash = deepcopy(intersection)
    bad_row_hash["rows"][0]["ocular_kept_row_sha256"] = "0" * 64
    with pytest.raises(ManifestValidationError, match="ocular-kept row hash"):
        validate_cross_condition_intersection(bad_row_hash, removed, kept, contract)

    bad_event_hash = deepcopy(intersection)
    bad_event_hash["rows"][0]["event_provenance_sha256"] = "0" * 64
    with pytest.raises(ManifestValidationError, match="event provenance hash"):
        validate_cross_condition_intersection(bad_event_hash, removed, kept, contract)


def test_prediction_keys_must_equal_outer_test_survivors_for_all_three_models(contract):
    rows = [
        trial_row(item_num=1, epoch_row=0, outer_role="test"),
        trial_row(item_num=2, epoch_row=1, outer_role="test"),
        trial_row(item_num=3, epoch_row=2, outer_role="train"),
    ]
    trial_manifest = manifest("trial", rows)
    expected = [canonical_trial_key(row) for row in rows[:2]]
    validate_prediction_key_sets(
        trial_manifest,
        {"B3": list(reversed(expected)), "M1": expected, "M3": expected},
        contract,
    )

    missing = {"B3": expected, "M1": expected[:1], "M3": expected}
    with pytest.raises(ManifestValidationError, match="M1 predictions.*missing.*item_num=2"):
        validate_prediction_key_sets(trial_manifest, missing, contract)

    duplicate = {"B3": expected + [expected[0]], "M1": expected, "M3": expected}
    with pytest.raises(ManifestValidationError, match="duplicate B3 predictions key"):
        validate_prediction_key_sets(trial_manifest, duplicate, contract)


def test_prediction_key_gate_requires_exact_model_set(contract):
    row = trial_row(item_num=1, epoch_row=0, outer_role="test")
    key = canonical_trial_key(row)

    with pytest.raises(ManifestValidationError, match="prediction model set"):
        validate_prediction_key_sets(
            manifest("trial", [row]), {"B3": [key], "M1": [key]}, contract
        )


def cross_fit_manifests_and_keys() -> tuple[dict, dict, dict]:
    rows = [
        trial_row(item_num=item, epoch_row=item - 1, session=session, outer_role=role)
        for session, role in [(0, "train"), (1, "train"), (2, "test")]
        for item in (1, 2)
    ]
    keys_by_session = {
        session: [canonical_trial_key(row) for row in rows if row["session"] == session]
        for session in (0, 1, 2)
    }
    fold_rows = []
    for fold_id, validation_session, fit_session in [
        ("fold-0", 0, 1),
        ("fold-1", 1, 0),
    ]:
        fold_rows.append(
            {
                "subject_id": "sub-LTP001",
                "condition": "ocular-removed",
                "inner_fold_id": fold_id,
                "fit_sessions": [fit_session],
                "validation_session": validation_session,
                "outer_test_session": 2,
                "fit_class_counts": {"0": 1, "1": 1},
                "validation_class_counts": {"0": 1, "1": 1},
                "fit_keys_sha256": canonical_trial_key_set_sha256(keys_by_session[fit_session]),
                "validation_keys_sha256": canonical_trial_key_set_sha256(
                    keys_by_session[validation_session]
                ),
                "eligible": True,
                "feasibility_reason_code": "ELIGIBLE",
            }
        )
    by_fold = {
        "fold-0": {
            "fit_keys": keys_by_session[1],
            "predicted_keys": keys_by_session[0],
        },
        "fold-1": {
            "fit_keys": keys_by_session[0],
            "predicted_keys": keys_by_session[1],
        },
    }
    provenance = {"B3": deepcopy(by_fold), "M1": deepcopy(by_fold)}
    return manifest("trial", rows), manifest("inner_fold", fold_rows), provenance


def d44_c_cross_fit_provenance(contract) -> tuple[dict, dict, dict, dict]:
    trials, folds, fold_keys_by_model = cross_fit_manifests_and_keys()
    charter = _synthetic_final_charter()
    outer_train_rows = [
        row
        for row in trials["rows"]
        if row["condition_survivor"] and row["outer_role"] == "train"
    ]
    outer_train_keys = [canonical_trial_key(row) for row in outer_train_rows]
    outer_train_sessions = sorted({row["session"] for row in outer_train_rows})
    record = {
        "schema_id": "neuroai-m1-d44-c-cross-fit-provenance-v1",
        "schema_version": "1.0",
        "subject_id": "sub-LTP001",
        "condition": "ocular-removed",
        "outer_test_session": 2,
        "decision": "D44_C_outer_training_selected_C_qualified_meta_predictions",
        "meta_prediction_claim": (
            "session_out_coefficients_and_transforms_conditional_on_"
            "outer_training_selected_C"
        ),
        "fully_session_cross_fitted": False,
        "fold_keys_by_model": fold_keys_by_model,
        "m1_hyperparameter_selection": {
            "scope": (
                "outer_training_wide_inner_session_C_selection_including_"
                "meta_prediction_session"
            ),
            "candidate_C_grid": deepcopy(charter["learners"]["M1"]["C_grid"]),
            "selection_metric": charter["splits"]["selection_metric"],
            "tie_break": charter["splits"]["tie_break"],
            "selected_C": 0.1,
            "fit_sessions": outer_train_sessions,
            "validation_sessions": outer_train_sessions,
            "fit_keys_sha256": canonical_trial_key_set_sha256(outer_train_keys),
            "validation_keys_sha256": canonical_trial_key_set_sha256(
                outer_train_keys
            ),
            "inner_fold_manifest_sha256": manifest_sha256(folds, contract),
            "predicted_sessions_in_selection_validation": outer_train_sessions,
            "predicted_session_labels_may_influence_selected_C": True,
            "outer_test_session_used_for_selection": False,
            "one_selected_C_shared_by_all_meta_folds": True,
        },
    }
    return trials, folds, charter, record


def test_d44_c_cross_fit_provenance_is_complete_qualified_and_hash_bound(contract):
    trials, folds, charter, record = d44_c_cross_fit_provenance(contract)

    manifest_module.validate_d44_c_cross_fit_provenance(
        trials, folds, record, charter, contract
    )
    payload = manifest_module.canonical_d44_c_cross_fit_provenance_bytes(
        trials, folds, record, charter, contract
    )
    assert manifest_module.d44_c_cross_fit_provenance_sha256(
        trials, folds, record, charter, contract
    ) == sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(fully_session_cross_fitted=True),
        lambda record: record.update(meta_prediction_claim="fully_session_cross_fitted"),
        lambda record: record["m1_hyperparameter_selection"].update(
            scope="predicted_session_excluded"
        ),
        lambda record: record["m1_hyperparameter_selection"].update(
            selected_C=1000.0
        ),
        lambda record: record["m1_hyperparameter_selection"].update(
            validation_sessions=[0]
        ),
        lambda record: record["m1_hyperparameter_selection"].update(
            predicted_sessions_in_selection_validation=[0]
        ),
        lambda record: record["m1_hyperparameter_selection"].update(
            predicted_session_labels_may_influence_selected_C=False
        ),
        lambda record: record["m1_hyperparameter_selection"].update(
            outer_test_session_used_for_selection=True
        ),
        lambda record: record["m1_hyperparameter_selection"].update(
            inner_fold_manifest_sha256="0" * 64
        ),
    ],
)
def test_d44_c_cross_fit_provenance_rejects_selection_or_claim_drift(
    contract, mutation
):
    trials, folds, charter, record = d44_c_cross_fit_provenance(contract)
    mutation(record)

    with pytest.raises(ManifestValidationError, match="D44-C"):
        manifest_module.validate_d44_c_cross_fit_provenance(
            trials, folds, record, charter, contract
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda charter: charter["cross_fitting"].update(
            fully_session_cross_fitted_claim_allowed=True
        ),
        lambda charter: charter["cross_fitting"].update(
            predicted_session_labels_may_influence_selected_C=False
        ),
        lambda charter: charter["learners"]["M3"].update(
            meta_training_predictions="fully_session_cross_fitted"
        ),
    ],
)
def test_generation_ready_rejects_d44_c_semantic_drift(contract, mutation):
    charter = _synthetic_final_charter()
    mutation(charter)
    final_charter_sha256 = canonical_document_sha256(charter)

    with pytest.raises(ManifestValidationError, match="D44-C"):
        assert_generation_ready(
            charter,
            load_json_unique(ROI_PATH),
            contract,
            protocol_teach_back_receipt=protocol_teach_back_receipt(
                charter_sha256=final_charter_sha256
            ),
            final_charter_sha256=final_charter_sha256,
            decision_log_sha256=DECISION_LOG_SHA,
            protocol_teach_back_transcript=protocol_teach_back_transcript(),
            source_transcript_sha256=canonical_document_sha256(protocol_teach_back_transcript()),
        )


def test_cross_fit_key_provenance_is_session_disjoint_complete_and_hash_bound(contract):
    trials, folds, provenance = cross_fit_manifests_and_keys()
    validate_cross_fit_key_provenance(trials, folds, provenance, contract)

    leaked = deepcopy(provenance)
    leaked["M1"]["fold-0"]["fit_keys"].append(
        leaked["M1"]["fold-0"]["predicted_keys"][0]
    )
    with pytest.raises(ManifestValidationError, match="M1 fold-0 fit keys.*extra"):
        validate_cross_fit_key_provenance(trials, folds, leaked, contract)

    missing = deepcopy(provenance)
    missing["B3"]["fold-1"]["predicted_keys"] = missing["B3"]["fold-1"][
        "predicted_keys"
    ][:1]
    with pytest.raises(ManifestValidationError, match="B3 fold-1 predicted keys.*missing"):
        validate_cross_fit_key_provenance(trials, folds, missing, contract)


def test_cross_fit_key_provenance_rejects_hash_or_model_set_drift(contract):
    trials, folds, provenance = cross_fit_manifests_and_keys()

    bad_hash = deepcopy(folds)
    bad_hash["rows"][0]["fit_keys_sha256"] = "0" * 64
    with pytest.raises(ManifestValidationError, match="fit_keys_sha256 mismatch.*fold-0"):
        validate_cross_fit_key_provenance(trials, bad_hash, provenance, contract)

    incomplete_models = {"B3": provenance["B3"]}
    with pytest.raises(ManifestValidationError, match="cross-fit model set"):
        validate_cross_fit_key_provenance(trials, folds, incomplete_models, contract)
