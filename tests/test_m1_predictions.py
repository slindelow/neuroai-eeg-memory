"""Synthetic tests for the pre-result paired outer-test prediction-table boundary.

The fixtures are invented. They enforce but do not change resolved D26-A/D29-A/D42-A/D44-C, and
they do not fit a model, use a real label or prediction, calculate a metric, or inspect any
held-out outcome.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

import peers.m1_evaluation as m1_evaluation
from peers.m1_features import feature_schema_sha256
from peers.m1_manifests import (
    canonical_row_sha256,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    event_provenance_sha256,
    load_json_unique,
    manifest_sha256,
)
from peers.m1_predictions import (
    PredictionTableValidationError,
    build_condition_comparison_table,
    canonical_condition_comparison_table_bytes,
    canonical_prediction_table_bytes,
    condition_comparison_bundle_contract_sha256,
    condition_comparison_table_sha256,
    learner_contract_sha256,
    load_condition_comparison_bundle,
    load_prediction_table_bundle,
    model_provenance_sha256,
    prediction_table_bundle_contract_sha256,
    prediction_table_sha256,
    validate_condition_comparison_table,
    validate_prediction_table,
    write_condition_comparison_bundle_atomic,
    write_prediction_table_bundle_atomic,
)
from peers.m1_result_storage import (
    EvaluationResultBundleValidationError,
    load_common_intersection_subject_evaluation_bundle,
    load_primary_aggregate_bundle,
    load_subject_evaluation_bundle,
    result_record_bundle_contract_sha256,
    write_common_intersection_subject_evaluation_bundle_atomic,
    write_primary_aggregate_bundle_atomic,
    write_subject_evaluation_bundle_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
FEATURE_CONTRACT_PATH = ROOT / "docs" / "m1-feature-table-contract.json"
MANIFEST_CONTRACT_PATH = ROOT / "docs" / "m1-manifest-contracts.json"
PREDICTION_CONTRACT_PATH = ROOT / "docs" / "m1-prediction-table-contract.json"
EVALUATION_CONTRACT_PATH = ROOT / "docs" / "m1-evaluation-contract.json"
SHA = "a" * 64
EVALUATION_PROVENANCE = {
    "evaluator_code_commit": "5" * 40,
    "evaluator_dirty_diff_sha256": None,
    "evaluator_package_versions_sha256": "6" * 64,
    "evaluation_command": "synthetic-evaluation-test-only",
}
AGGREGATE_PROVENANCE = {
    "aggregator_code_commit": "7" * 40,
    "aggregator_dirty_diff_sha256": None,
    "aggregator_package_versions_sha256": "8" * 64,
    "aggregation_command": "synthetic-primary-aggregation-test-only",
}
COMPARISON_PROVENANCE = {
    "created_at_utc": "2026-08-16T23:25:00Z",
    "comparator_code_commit": "9" * 40,
    "comparator_dirty_diff_sha256": None,
    "comparator_uv_lock_sha256": "a" * 64,
    "comparator_package_versions_sha256": "b" * 64,
    "comparison_command": "synthetic-condition-comparison-test-only",
    "comparator_python_version": "3.12.synthetic",
}


@pytest.fixture(scope="module")
def resolved_charter():
    return load_json_unique(CHARTER_PATH)


@pytest.fixture()
def pending_charter(resolved_charter):
    charter = deepcopy(resolved_charter)
    charter["features"]["analysis_window_s"] = "PENDING_D29"
    charter["features"]["time_bins_s"] = "PENDING_D29"
    charter["pending_human_gates"].insert(0, "D29_window_and_midpoint")
    return charter


def with_d26_pending(charter):
    pending = deepcopy(charter)
    pending["features"]["normalization"] = "PENDING_D26"
    pending["pending_human_gates"].insert(0, "D26_normalization_estimand")
    return pending


@pytest.fixture(scope="module")
def feature_contract():
    return load_json_unique(FEATURE_CONTRACT_PATH)


@pytest.fixture(scope="module")
def manifest_contract():
    return load_json_unique(MANIFEST_CONTRACT_PATH)


@pytest.fixture(scope="module")
def prediction_contract():
    return load_json_unique(PREDICTION_CONTRACT_PATH)


@pytest.fixture(scope="module")
def evaluation_contract():
    return load_json_unique(EVALUATION_CONTRACT_PATH)


def trial_row(
    *,
    session: int,
    item_num: int,
    role: str,
    subject_id: str = "sub-SYN001",
    condition: str = "ocular-removed",
) -> dict:
    return {
        "subject_id": subject_id,
        "session": session,
        "trial": item_num,
        "item_num": item_num,
        "condition": condition,
        "serial_position": item_num,
        "list_number": item_num,
        "onset_s": float(item_num),
        "recalled": item_num % 2 == 0,
        "source_events_id": f"{subject_id}_ses-{session}_events.tsv",
        "source_events_sha256": SHA,
        "derivative_id": f"{subject_id}_ses-{session}_{condition}_epo.fif",
        "derivative_sha256": SHA,
        "epoch_row": item_num - 1,
        "included": True,
        "exclusion_reason_code": "INCLUDED",
        "condition_survivor": True,
        "outer_role": role,
    }


def synthetic_trial_manifest(
    *,
    subject_id: str = "sub-SYN001",
    condition: str = "ocular-removed",
    item_nums: tuple[int, ...] = (1, 2, 3, 4),
    outer_test_session: int = 2,
):
    rows = [
        trial_row(
            session=session,
            item_num=item_num,
            role=role,
            subject_id=subject_id,
            condition=condition,
        )
        for session, role in (
            (0, "train"),
            (1, "train"),
            (outer_test_session, "test"),
        )
        for item_num in item_nums
    ]
    return {
        "schema_id": "neuroai-m1-trial-v1",
        "schema_version": "1.0",
        "manifest_kind": "trial",
        "created_at_utc": "2026-08-16T22:30:00Z",
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


def synthetic_model_provenance(charter, trial_manifest):
    train_rows = [row for row in trial_manifest["rows"] if row["outer_role"] == "train"]
    test_rows = [row for row in trial_manifest["rows"] if row["outer_role"] == "test"]
    train_keys = [canonical_trial_key(row) for row in train_rows]
    test_keys = [canonical_trial_key(row) for row in test_rows]
    fit_keys_sha256 = canonical_trial_key_set_sha256(train_keys)
    prediction_keys_sha256 = canonical_trial_key_set_sha256(test_keys)
    provenance = {}
    for index, model_id in enumerate(("B3", "M1", "M3"), start=1):
        provenance[model_id] = {
            "model_id": model_id,
            "fit_role": (
                "qualified_session_out_meta_training"
                if model_id == "M3"
                else "outer_training_refit"
            ),
            "fit_sessions": [0, 1],
            "fit_keys_sha256": fit_keys_sha256,
            "prediction_keys_sha256": prediction_keys_sha256,
            "learner_contract_sha256": learner_contract_sha256(charter, model_id),
            "selected_hyperparameters_sha256": str(index) * 64,
            "transform_chain_sha256": str(index + 3) * 64,
            "transform_fit_keys_sha256": str(index + 6) * 64,
            "model_artifact_sha256": str(index + 2) * 64,
            "upstream_cross_fit_sha256": "f" * 64 if model_id == "M3" else None,
            "seed": 0,
        }
    return provenance


def prediction_table(
    charter,
    feature_contract,
    manifest_contract,
    prediction_contract,
    *,
    subject_id="sub-SYN001",
    condition="ocular-removed",
    item_nums=(1, 2, 3, 4),
    prob_b3=None,
    prob_m1=None,
    prob_m3=None,
    outer_test_session=2,
):
    trial_manifest = synthetic_trial_manifest(
        subject_id=subject_id,
        condition=condition,
        item_nums=tuple(item_nums),
        outer_test_session=outer_test_session,
    )
    model_provenance = synthetic_model_provenance(charter, trial_manifest)
    model_provenance_digest = model_provenance_sha256(model_provenance, prediction_contract)
    test_rows = [row for row in trial_manifest["rows"] if row["outer_role"] == "test"]
    model_digests = {
        model_id: model_provenance_sha256(
            {model_id: provenance}, prediction_contract, require_all_models=False
        )
        for model_id, provenance in model_provenance.items()
    }
    outer_fold_id = (
        f"outer__{subject_id}__{condition}__test-{outer_test_session}"
    )
    rows = [
        {
            "subject_id": row["subject_id"],
            "session": row["session"],
            "trial": row["trial"],
            "item_num": row["item_num"],
            "condition": row["condition"],
            "outer_fold_id": outer_fold_id,
            "recalled": row["recalled"],
            "prob_b3": (
                prob_b3[index]
                if prob_b3 is not None
                else 0.25 + 0.5 * float(row["recalled"])
            ),
            "prob_m1": (
                prob_m1[index]
                if prob_m1 is not None
                else 0.35 + 0.3 * float(row["recalled"])
            ),
            "prob_m3": (
                prob_m3[index]
                if prob_m3 is not None
                else 0.20 + 0.6 * float(row["recalled"])
            ),
            "b3_provenance_sha256": model_digests["B3"],
            "m1_provenance_sha256": model_digests["M1"],
            "m3_provenance_sha256": model_digests["M3"],
        }
        for index, row in enumerate(test_rows)
    ]
    expected_bindings = {
        "charter_sha256": "c" * 64,
        "roi_map_sha256": "d" * 64,
        "trial_manifest_sha256": manifest_sha256(trial_manifest, manifest_contract),
        "fold_manifest_sha256": "e" * 64,
        "feature_schema_sha256": feature_schema_sha256(charter, feature_contract),
        "feature_table_sha256": "1" * 64,
        "cross_fit_provenance_sha256": "f" * 64,
        "model_provenance_sha256": model_provenance_digest,
        "uv_lock_sha256": "2" * 64,
        "package_versions_sha256": "3" * 64,
    }
    table = {
        "schema_id": prediction_contract["logical_table"]["schema_id"],
        "schema_version": prediction_contract["logical_table"]["schema_version"],
        "created_at_utc": "2026-08-16T22:45:00Z",
        "condition": condition,
        "subject_id": subject_id,
        "outer_fold_id": outer_fold_id,
        "outer_test_session": outer_test_session,
        **expected_bindings,
        "code_commit": "4" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "synthetic-test-only",
        "python_version": "3.12.synthetic",
        "model_provenance": model_provenance,
        "rows": rows,
    }
    return table, trial_manifest, expected_bindings


def validate_synthetic(table, trial_manifest, bindings, charter, contracts):
    feature_contract, manifest_contract, prediction_contract = contracts
    validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )


def rewrite_prediction_bundle_manifest(destination, **updates):
    manifest_path = destination / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(updates)
    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    digest = sha256(payload).hexdigest()
    manifest_path.write_bytes(payload)
    (destination / "bundle.sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def rewrite_raw_result_record_and_manifest(destination, record_payload):
    (destination / "record.json").write_bytes(record_payload)
    return rewrite_prediction_bundle_manifest(
        destination,
        record_json_sha256=sha256(record_payload).hexdigest(),
    )


def rewrite_result_record_and_manifest(destination, record):
    record_payload = (
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    return rewrite_raw_result_record_and_manifest(destination, record_payload)


@pytest.fixture()
def contracts(feature_contract, manifest_contract, prediction_contract):
    return feature_contract, manifest_contract, prediction_contract


def test_contract_remains_blocked_by_human_scientific_gates(prediction_contract):
    assert prediction_contract["real_prediction_generation_requires"] == [
        "final_charter_sha256",
        "D26_resolved",
        "D29_resolved",
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
        "bounded_preflight_and_capacity_policy_frozen",
        "frozen_real_manifests_and_feature_tables",
        "validated_Sofia_protocol_teach_back_receipt",
        "completed_model_fits_with_verified_provenance",
    ]
    assert prediction_contract["implementation"]["real_predictions_generated"] is False
    assert prediction_contract["implementation"]["metrics_calculated"] is False


def test_prediction_contract_matches_charter_model_stack(
    resolved_charter, prediction_contract
):
    assert prediction_contract["model_provenance"]["required_models"] == [
        "B3",
        "M1",
        "M3",
    ]
    assert prediction_contract["logical_table"]["probability_fields"] == [
        "prob_b3",
        "prob_m1",
        "prob_m3",
    ]
    assert resolved_charter["learners"]["M3"]["inputs"] == ["prob_b3", "prob_m1"]
    assert resolved_charter["learners"]["M3"]["raw_features_allowed"] is False


def test_prediction_contract_binds_d44_c_qualified_meta_provenance(
    resolved_charter, prediction_contract
):
    assert prediction_contract["resolved_scientific_inputs"] == [
        "D26_A_strict_inductive_normalization",
        "D29_A_300_to_1600_ms_exact_950_ms_midpoint",
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
    ]
    assert prediction_contract["scientific_choices_not_selected_by_this_contract"] == []
    assert prediction_contract["model_provenance"]["fit_roles"]["M3"] == (
        "qualified_session_out_meta_training"
    )
    assert prediction_contract["D44_C_cross_fit_provenance"] == {
        "schema_id": "neuroai-m1-d44-c-cross-fit-provenance-v1",
        "decision": "D44_C_outer_training_selected_C_qualified_meta_predictions",
        "meta_prediction_claim": (
            "session_out_coefficients_and_transforms_conditional_on_"
            "outer_training_selected_C"
        ),
        "fully_session_cross_fitted_claim_allowed": False,
        "predicted_session_labels_may_influence_selected_C": True,
        "outer_test_session_used_for_selection": False,
        "table_binding_field": "cross_fit_provenance_sha256",
        "M3_upstream_binding_required": True,
    }
    assert resolved_charter["scientific_decisions"]["D44"] == (
        "D44_C_outer_training_selected_C_qualified_meta_predictions"
    )


@pytest.mark.parametrize("field", ["learner", "splits", "cross_fitting", "seed"])
def test_learner_digest_changes_with_contract_drift(resolved_charter, field):
    changed = deepcopy(resolved_charter)
    if field == "learner":
        changed["learners"]["M1"]["max_iter"] = 1999
    elif field == "splits":
        changed["splits"]["selection_metric"] = "synthetic-different-metric"
    elif field == "cross_fitting":
        changed["cross_fitting"]["meta_fit_scope"] = "synthetic-different-scope"
    else:
        changed["determinism"]["global_seed"] = 1

    assert learner_contract_sha256(changed, "M1") != learner_contract_sha256(
        resolved_charter, "M1"
    )


def test_valid_prediction_table_pairs_labels_probabilities_and_provenance(
    resolved_charter, contracts
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    validate_synthetic(table, trial_manifest, bindings, resolved_charter, contracts)
    assert prediction_table_sha256(
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        expected_bindings=bindings,
    )


def test_prediction_table_canonical_bytes_sort_rows_but_stored_table_requires_order(
    resolved_charter, contracts
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    reversed_table = deepcopy(table)
    reversed_table["rows"].reverse()

    assert canonical_prediction_table_bytes(
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        expected_bindings=bindings,
    ) == canonical_prediction_table_bytes(
        reversed_table,
        trial_manifest,
        resolved_charter,
        *contracts,
        expected_bindings=bindings,
    )
    with pytest.raises(PredictionTableValidationError, match="canonical row order"):
        validate_synthetic(
            reversed_table,
            trial_manifest,
            bindings,
            resolved_charter,
            contracts,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_label"])
def test_prediction_table_rejects_key_and_label_drift(resolved_charter, contracts, mutation):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    if mutation == "missing":
        table["rows"].pop()
    elif mutation == "extra":
        extra = deepcopy(table["rows"][-1])
        extra["trial"] = 5
        extra["item_num"] = 5
        table["rows"].append(extra)
    elif mutation == "duplicate":
        table["rows"].append(deepcopy(table["rows"][-1]))
    else:
        table["rows"][0]["recalled"] = not table["rows"][0]["recalled"]

    with pytest.raises(PredictionTableValidationError):
        validate_synthetic(table, trial_manifest, bindings, resolved_charter, contracts)


@pytest.mark.parametrize(
    ("value", "pattern"),
    [
        (0, "float64"),
        (float("nan"), "finite"),
        (-0.01, "closed interval"),
        (1.01, "closed interval"),
    ],
)
def test_prediction_table_rejects_invalid_probabilities(
    resolved_charter, contracts, value, pattern
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    table["rows"][0]["prob_m1"] = value

    with pytest.raises(PredictionTableValidationError, match=pattern):
        validate_synthetic(table, trial_manifest, bindings, resolved_charter, contracts)


@pytest.mark.parametrize(
    "mutation",
    ["model_set", "fit_keys", "fit_sessions", "fit_role", "prediction_keys"],
)
def test_prediction_table_rejects_model_fit_and_prediction_provenance_drift(
    resolved_charter, contracts, mutation
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    if mutation == "model_set":
        table["model_provenance"].pop("M1")
    elif mutation == "fit_keys":
        table["model_provenance"]["B3"]["fit_keys_sha256"] = "9" * 64
    elif mutation == "fit_sessions":
        table["model_provenance"]["M1"]["fit_sessions"] = [0]
    elif mutation == "fit_role":
        table["model_provenance"]["M3"]["fit_role"] = "outer_training_refit"
    else:
        table["model_provenance"]["M3"]["prediction_keys_sha256"] = "9" * 64

    with pytest.raises(PredictionTableValidationError):
        validate_synthetic(table, trial_manifest, bindings, resolved_charter, contracts)


@pytest.mark.parametrize("mutation", ["row_link", "top_digest", "artifact"])
def test_prediction_table_rejects_provenance_link_and_external_binding_drift(
    resolved_charter, contracts, mutation
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    if mutation == "row_link":
        table["rows"][0]["m1_provenance_sha256"] = "9" * 64
    elif mutation == "top_digest":
        table["model_provenance_sha256"] = "9" * 64
    else:
        table["model_provenance"]["M1"]["model_artifact_sha256"] = "9" * 64

    with pytest.raises(PredictionTableValidationError):
        validate_synthetic(table, trial_manifest, bindings, resolved_charter, contracts)


@pytest.mark.parametrize("field", ["auc", "delta_auc", "result_category", "coefficient"])
def test_prediction_table_rejects_metric_result_and_model_parameter_fields(
    resolved_charter, contracts, field
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    table[field] = 0.5

    with pytest.raises(PredictionTableValidationError, match="extra"):
        validate_synthetic(table, trial_manifest, bindings, resolved_charter, contracts)


def test_prediction_table_rejects_pending_d26_and_d29(
    pending_charter, contracts
):
    d26_pending = with_d26_pending(pending_charter)
    with pytest.raises(PredictionTableValidationError, match="PENDING_D26"):
        learner_contract_sha256(d26_pending, "M1")

    with pytest.raises(PredictionTableValidationError, match="PENDING_D29"):
        learner_contract_sha256(pending_charter, "M1")


def test_prediction_bundle_contract_is_typed_atomic_and_decision_independent(
    prediction_contract,
):
    bundle = prediction_contract["storage_bundle"]

    assert prediction_contract["version"] == "1.3-draft"
    assert bundle["atomic_publication"] == (
        "temporary_directory_fsync_then_atomic_noreplace_rename"
    )
    assert bundle["exact_files"] == [
        "metadata.json",
        "row-records.json",
        "probabilities.npy",
        "bundle.json",
        "bundle.sha256",
    ]
    assert bundle["probabilities"]["field_order"] == [
        "prob_b3",
        "prob_m1",
        "prob_m3",
    ]
    assert bundle["probabilities"]["dtype"] == "little_endian_float64"
    assert bundle["probabilities"]["allow_pickle"] is False
    assert bundle["probabilities"]["shape"] == "n_rows_by_3"
    assert bundle["metric_or_result_fields_allowed"] is False
    assert bundle["real_publication_allowed_before_final_readiness"] is False


def test_prediction_bundle_round_trip_is_exact_typed_and_immutable(
    tmp_path, resolved_charter, contracts
):
    feature_contract, manifest_contract, prediction_contract = contracts
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    destination = tmp_path / "prediction-bundle"

    digest = write_prediction_table_bundle_atomic(
        destination,
        table,
        trial_manifest,
        resolved_charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )

    assert {path.name for path in destination.iterdir()} == set(
        prediction_contract["storage_bundle"]["exact_files"]
    )
    with (destination / "probabilities.npy").open("rb") as handle:
        probabilities = np.load(handle, allow_pickle=False)
        assert handle.read(1) == b""
    assert probabilities.dtype.str == "<f8"
    assert probabilities.flags.c_contiguous
    assert probabilities.shape == (4, 3)

    loaded = load_prediction_table_bundle(
        destination,
        trial_manifest,
        resolved_charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
        expected_bundle_sha256=digest,
    )
    assert loaded == table

    with pytest.raises(PredictionTableValidationError, match="already exists"):
        write_prediction_table_bundle_atomic(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
        )


def test_prediction_bundle_readback_rejects_tamper_extra_files_and_wrong_external_digest(
    tmp_path, resolved_charter, contracts
):
    feature_contract, manifest_contract, prediction_contract = contracts
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)

    def publish(name):
        destination = tmp_path / name
        digest = write_prediction_table_bundle_atomic(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
        )
        return destination, digest

    destination, digest = publish("wrong-external")
    with pytest.raises(PredictionTableValidationError, match="external bundle digest"):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256="0" * 64,
        )

    destination, digest = publish("tampered-probabilities")
    values_path = destination / "probabilities.npy"
    payload = bytearray(values_path.read_bytes())
    payload[-1] ^= 1
    values_path.write_bytes(payload)
    with pytest.raises(PredictionTableValidationError, match="probabilities.npy digest"):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )

    destination, digest = publish("extra-file")
    (destination / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(PredictionTableValidationError, match="exact files"):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def test_prediction_bundle_rejects_contract_drift(
    tmp_path, resolved_charter, contracts
):
    feature_contract, manifest_contract, prediction_contract = contracts
    changed = deepcopy(prediction_contract)
    changed["storage_bundle"]["probabilities"]["allow_pickle"] = True
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)

    with pytest.raises(PredictionTableValidationError, match="bundle contract drifted"):
        write_prediction_table_bundle_atomic(
            tmp_path / "prediction-bundle",
            table,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            changed,
            expected_bindings=bindings,
        )


@pytest.mark.parametrize("unsafe_kind", ["object_array", "trailing_bytes"])
def test_prediction_bundle_rejects_unsafe_npy_even_when_all_hashes_are_rewritten(
    tmp_path, resolved_charter, contracts, unsafe_kind
):
    feature_contract, manifest_contract, prediction_contract = contracts
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    destination = tmp_path / unsafe_kind
    write_prediction_table_bundle_atomic(
        destination,
        table,
        trial_manifest,
        resolved_charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )
    values_path = destination / "probabilities.npy"
    if unsafe_kind == "object_array":
        with values_path.open("wb") as handle:
            np.save(handle, np.array([[object()]], dtype=object), allow_pickle=True)
    else:
        with values_path.open("ab") as handle:
            handle.write(b"unexpected trailing byte")
    digest = rewrite_prediction_bundle_manifest(
        destination,
        probabilities_npy_sha256=sha256(values_path.read_bytes()).hexdigest(),
    )

    message = "safe NumPy array" if unsafe_kind == "object_array" else "trailing bytes"
    with pytest.raises(PredictionTableValidationError, match=message):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def test_prediction_bundle_revalidates_labels_after_consistent_member_rehash(
    tmp_path, resolved_charter, contracts
):
    feature_contract, manifest_contract, prediction_contract = contracts
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    destination = tmp_path / "label-tamper"
    write_prediction_table_bundle_atomic(
        destination,
        table,
        trial_manifest,
        resolved_charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )
    records_path = destination / "row-records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    records["rows"][0]["recalled"] = not records["rows"][0]["recalled"]
    records_payload = (
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    records_path.write_bytes(records_payload)
    digest = rewrite_prediction_bundle_manifest(
        destination,
        row_records_sha256=sha256(records_payload).hexdigest(),
    )

    with pytest.raises(PredictionTableValidationError, match="recall label mismatch"):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def test_prediction_bundle_rejects_rehashed_probability_drift_against_logical_digest(
    tmp_path, resolved_charter, contracts
):
    feature_contract, manifest_contract, prediction_contract = contracts
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    destination = tmp_path / "probability-rehash"
    write_prediction_table_bundle_atomic(
        destination,
        table,
        trial_manifest,
        resolved_charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )
    values_path = destination / "probabilities.npy"
    with values_path.open("rb") as handle:
        probabilities = np.load(handle, allow_pickle=False)
    probabilities[0, 1] = 0.55
    with values_path.open("wb") as handle:
        np.save(handle, probabilities, allow_pickle=False)
    digest = rewrite_prediction_bundle_manifest(
        destination,
        probabilities_npy_sha256=sha256(values_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(
        PredictionTableValidationError, match="reconstructed logical-table digest"
    ):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def test_prediction_bundle_rejects_symlink_member(
    tmp_path, resolved_charter, contracts
):
    feature_contract, manifest_contract, prediction_contract = contracts
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    destination = tmp_path / "symlink-member"
    digest = write_prediction_table_bundle_atomic(
        destination,
        table,
        trial_manifest,
        resolved_charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )
    metadata_path = destination / "metadata.json"
    target = tmp_path / "metadata-target.json"
    metadata_path.rename(target)
    metadata_path.symlink_to(target)

    with pytest.raises(PredictionTableValidationError, match="regular file"):
        load_prediction_table_bundle(
            destination,
            trial_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def evaluate_synthetic(
    table,
    trial_manifest,
    bindings,
    charter,
    contracts,
    evaluation_contract,
):
    return m1_evaluation.evaluate_subject_prediction_table(
        table,
        trial_manifest,
        charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        evaluation_provenance=EVALUATION_PROVENANCE,
    )


def validate_evaluation_synthetic(
    record,
    table,
    trial_manifest,
    bindings,
    charter,
    contracts,
    evaluation_contract,
):
    m1_evaluation.validate_subject_evaluation_record(
        record,
        table,
        trial_manifest,
        charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )


def test_evaluation_contract_binds_d42_a_omission_and_keeps_real_work_deferred(
    resolved_charter, evaluation_contract
):
    assert evaluation_contract["implementation"]["data_scope"] == "synthetic_only"
    assert evaluation_contract["implementation"]["real_metrics_calculated"] is False
    assert evaluation_contract["metric_scope"]["balanced_accuracy"] == (
        "OMITTED_BY_D42_A_NOT_COMPUTED"
    )
    assert evaluation_contract["resolved_scientific_inputs"] == [
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
    ]
    assert evaluation_contract["scientific_choices_not_selected_by_this_contract"] == []
    assert resolved_charter["scientific_decisions"]["D42"] == (
        "D42_A_omit_balanced_accuracy"
    )
    assert "D42_balanced_accuracy_handling" not in resolved_charter["pending_human_gates"]
    assert "balanced_accuracy" not in resolved_charter["estimands"]["secondary"]
    assert resolved_charter["estimands"]["balanced_accuracy_policy"] == {
        "decision": "D42_A_omit_balanced_accuracy",
        "included": False,
        "status": "deliberately_omitted_not_computed",
        "threshold": None,
    }
    assert evaluation_contract["real_evaluation_requires"] == [
        "final_charter_sha256",
        "D26_resolved",
        "D29_resolved",
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
        "bounded_preflight_and_capacity_policy_frozen",
        "validated_real_prediction_table_bundles",
        "validated_Sofia_protocol_teach_back_receipt",
        "independent_evaluator_code_and_environment_bindings",
    ]
    assert evaluation_contract["forbidden_actions"] == [
        "change_resolved_D26_or_D29",
        "add_or_compute_balanced_accuracy_without_versioned_charter_replan",
        "accept_arbitrary_arrays_without_a_validated_prediction_table",
        "filter_rows_by_metric_or_observed_performance",
        "inspect_real_M1_or_M3_outcomes",
    ]


def test_result_record_bundle_contract_is_atomic_d42_a_bound_and_fail_closed(
    evaluation_contract,
):
    bundle = evaluation_contract["result_record_storage_bundle"]

    assert bundle["atomic_publication"] == (
        "temporary_directory_fsync_then_atomic_noreplace_rename"
    )
    assert bundle["exact_files"] == ["record.json", "bundle.json", "bundle.sha256"]
    assert bundle["record_fields_source"] == (
        "resolved_evaluation_contract_exact_fields"
    )
    assert bundle["balanced_accuracy_policy"] == (
        "D42_A_deliberately_omitted_no_threshold_or_fields"
    )
    assert bundle["external_bundle_sha256_required_on_readback"] is True
    assert bundle["real_publication_allowed_before_final_readiness"] is False
    assert set(bundle["record_kinds"]) == {
        "subject_evaluation",
        "common_intersection_subject_evaluation",
        "primary_aggregate",
        "common_intersection_aggregate",
    }
    assert result_record_bundle_contract_sha256(evaluation_contract)


def test_result_storage_layout_is_stable_but_full_identity_rejects_d42_field_drift(
    evaluation_contract,
):
    changed = deepcopy(evaluation_contract)
    changed["metric_scope"]["balanced_accuracy"] = "D42_B_FIXED_0_5_TEST_ONLY"
    changed["subject_record"]["exact_fields"].append("balanced_accuracy_b3")
    changed["subject_record"]["metric_fields"].append("balanced_accuracy_b3")

    assert result_record_bundle_contract_sha256(changed) == (
        result_record_bundle_contract_sha256(evaluation_contract)
    )
    assert m1_evaluation.evaluation_contract_sha256(changed) != (
        m1_evaluation.evaluation_contract_sha256(evaluation_contract)
    )
    assert "balanced_accuracy_b3" not in json.dumps(
        evaluation_contract["result_record_storage_bundle"], sort_keys=True
    )


@pytest.mark.parametrize(
    "balanced_accuracy_state",
    [
        "PENDING_EXPLICIT_THRESHOLD_CONTRACT_NOT_COMPUTED",
        "D42_B_FIXED_0_5_TEST_ONLY",
    ],
)
def test_subject_evaluation_rejects_non_d42_a_metric_policy(
    balanced_accuracy_state,
    resolved_charter,
    contracts,
    evaluation_contract,
):
    table, trial_manifest, bindings = prediction_table(
        resolved_charter, *contracts
    )
    changed = deepcopy(evaluation_contract)
    changed["metric_scope"]["balanced_accuracy"] = balanced_accuracy_state

    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError,
        match="D42-A",
    ):
        evaluate_synthetic(
            table,
            trial_manifest,
            bindings,
            resolved_charter,
            contracts,
            changed,
        )


def test_subject_evaluation_bundle_round_trip_is_exact_atomic_and_immutable(
    tmp_path, resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(
        resolved_charter, *contracts
    )
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    destination = tmp_path / "subject-result"

    digest = write_subject_evaluation_bundle_atomic(
        destination,
        record,
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )

    assert {path.name for path in destination.iterdir()} == {
        "record.json",
        "bundle.json",
        "bundle.sha256",
    }
    loaded = load_subject_evaluation_bundle(
        destination,
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
        expected_bundle_sha256=digest,
    )
    assert loaded == record

    with pytest.raises(
        EvaluationResultBundleValidationError, match="already exists"
    ):
        write_subject_evaluation_bundle_atomic(
            destination,
            record,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
        )


def test_subject_evaluation_bundle_rejects_authentication_and_membership_drift(
    tmp_path, resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(
        resolved_charter, *contracts
    )
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )

    def publish(name):
        destination = tmp_path / name
        digest = write_subject_evaluation_bundle_atomic(
            destination,
            record,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
        )
        return destination, digest

    destination, _ = publish("external-digest")
    with pytest.raises(
        EvaluationResultBundleValidationError, match="external bundle digest"
    ):
        load_subject_evaluation_bundle(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
            expected_bundle_sha256="0" * 64,
        )

    destination, digest = publish("extra-member")
    (destination / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluationResultBundleValidationError, match="exact files"):
        load_subject_evaluation_bundle(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
            expected_bundle_sha256=digest,
        )

    destination, digest = publish("symlink-record")
    record_path = destination / "record.json"
    target = tmp_path / "result-record-target.json"
    record_path.rename(target)
    record_path.symlink_to(target)
    with pytest.raises(EvaluationResultBundleValidationError, match="regular file"):
        load_subject_evaluation_bundle(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
            expected_bundle_sha256=digest,
        )


def test_subject_evaluation_bundle_rejects_consistently_rehashed_metric_drift(
    tmp_path, resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(
        resolved_charter, *contracts
    )
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    destination = tmp_path / "metric-drift"
    write_subject_evaluation_bundle_atomic(
        destination,
        record,
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
    changed = deepcopy(record)
    changed["brier_m3"] = float(changed["brier_m3"] + 0.01)
    rewritten_digest = rewrite_result_record_and_manifest(destination, changed)

    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError,
        match="complete recomputation",
    ):
        load_subject_evaluation_bundle(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
            expected_bundle_sha256=rewritten_digest,
        )


def test_subject_evaluation_bundle_rejects_rehashed_duplicate_key_record(
    tmp_path, resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(
        resolved_charter, *contracts
    )
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    destination = tmp_path / "duplicate-key"
    write_subject_evaluation_bundle_atomic(
        destination,
        record,
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
    original = (destination / "record.json").read_bytes()
    duplicate = b'{"schema_id":"duplicate",' + original[1:]
    rewritten_digest = rewrite_raw_result_record_and_manifest(
        destination, duplicate
    )

    with pytest.raises(EvaluationResultBundleValidationError, match="duplicate"):
        load_subject_evaluation_bundle(
            destination,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            evaluation_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
            expected_bundle_sha256=rewritten_digest,
        )


def test_result_record_bundle_rejects_storage_contract_drift(
    tmp_path, resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(
        resolved_charter, *contracts
    )
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    changed_contract = deepcopy(evaluation_contract)
    changed_contract["result_record_storage_bundle"]["balanced_accuracy_policy"] = (
        "silently_choose_a_threshold"
    )

    with pytest.raises(
        EvaluationResultBundleValidationError, match="contract drifted"
    ):
        write_subject_evaluation_bundle_atomic(
            tmp_path / "contract-drift",
            record,
            table,
            trial_manifest,
            resolved_charter,
            *contracts,
            changed_contract,
            expected_bindings=bindings,
            expected_evaluation_provenance=EVALUATION_PROVENANCE,
        )


def test_subject_evaluation_consumes_one_validated_table(
    resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )

    assert record["prediction_table_sha256"] == prediction_table_sha256(
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        expected_bindings=bindings,
    )
    assert record["n_trials"] == len(table["rows"]) == 4
    assert record["n_recalled"] == 2
    assert record["n_not_recalled"] == 2
    assert record["auc_b3"] == pytest.approx(1.0)
    assert record["auc_m1"] == pytest.approx(1.0)
    assert record["auc_m3"] == pytest.approx(1.0)
    assert record["delta_auc_m3_minus_b3"] == pytest.approx(0.0)
    assert record["idi_m3_vs_b3"] == pytest.approx(0.1)
    assert record["continuous_nri_m3_vs_b3"] == pytest.approx(2.0)
    assert record["brier_b3"] == pytest.approx(0.0625)
    assert record["brier_m1"] == pytest.approx(0.1225)
    assert record["brier_m3"] == pytest.approx(0.04)
    assert all("balanced_accuracy" not in field for field in record)


def test_subject_evaluation_passes_every_canonical_row_to_incremental_metrics(
    resolved_charter, contracts, evaluation_contract, monkeypatch
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    observed = {}
    original = m1_evaluation.incremental_value

    def capture(y_true, prob_old, prob_new):
        observed["labels"] = y_true.tolist()
        observed["old"] = prob_old.tolist()
        observed["new"] = prob_new.tolist()
        return original(y_true, prob_old, prob_new)

    monkeypatch.setattr(m1_evaluation, "incremental_value", capture)
    evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )

    assert observed == {
        "labels": [row["recalled"] for row in table["rows"]],
        "old": [row["prob_b3"] for row in table["rows"]],
        "new": [row["prob_m3"] for row in table["rows"]],
    }


@pytest.mark.parametrize("mutation", ["row_order", "label", "probability", "binding"])
def test_subject_evaluation_rejects_invalid_prediction_table_first(
    resolved_charter, contracts, evaluation_contract, mutation
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    if mutation == "row_order":
        table["rows"].reverse()
    elif mutation == "label":
        table["rows"][0]["recalled"] = not table["rows"][0]["recalled"]
    elif mutation == "probability":
        table["rows"][0]["prob_m3"] = float("nan")
    else:
        table["feature_table_sha256"] = "9" * 64

    with pytest.raises(PredictionTableValidationError):
        evaluate_synthetic(
            table,
            trial_manifest,
            bindings,
            resolved_charter,
            contracts,
            evaluation_contract,
        )


@pytest.mark.parametrize("mutation", ["metric", "count", "digest", "extra"])
def test_subject_evaluation_record_recomputes_and_rejects_drift(
    resolved_charter, contracts, evaluation_contract, mutation
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    if mutation == "metric":
        record["delta_auc_m3_minus_b3"] = 0.5
    elif mutation == "count":
        record["n_trials"] = 1
    elif mutation == "digest":
        record["prediction_table_sha256"] = "9" * 64
    else:
        record["balanced_accuracy_m3"] = 0.5

    with pytest.raises(m1_evaluation.EvaluationConsumerValidationError):
        validate_evaluation_synthetic(
            record,
            table,
            trial_manifest,
            bindings,
            resolved_charter,
            contracts,
            evaluation_contract,
        )


def test_subject_evaluation_digest_is_deterministic_and_metric_sensitive(
    resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    digest = m1_evaluation.subject_evaluation_sha256(
        record,
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
    assert digest == m1_evaluation.subject_evaluation_sha256(
        deepcopy(record),
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )

    changed_table = deepcopy(table)
    changed_table["rows"][0]["prob_m3"] = 0.10
    changed_record = evaluate_synthetic(
        changed_table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    changed_digest = m1_evaluation.subject_evaluation_sha256(
        changed_record,
        changed_table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
    assert changed_digest != digest


def test_subject_evaluation_fails_closed_at_d26_then_d29(
    pending_charter, resolved_charter, contracts, evaluation_contract
):
    table, trial_manifest, bindings = prediction_table(resolved_charter, *contracts)
    with pytest.raises(PredictionTableValidationError, match="PENDING_D26"):
        evaluate_synthetic(
            table,
            trial_manifest,
            bindings,
            with_d26_pending(pending_charter),
            contracts,
            evaluation_contract,
        )

    with pytest.raises(PredictionTableValidationError, match="PENDING_D29"):
        evaluate_synthetic(
            table,
            trial_manifest,
            bindings,
            pending_charter,
            contracts,
            evaluation_contract,
        )


def patterned_probabilities(item_nums, direction):
    baseline_pattern = [0.4, 0.6, 0.6, 0.4]
    positive_pattern = [0.2, 0.8, 0.3, 0.7]
    adverse_pattern = [0.8, 0.2, 0.7, 0.3]
    repeats = (len(item_nums) + 3) // 4
    baseline = (baseline_pattern * repeats)[: len(item_nums)]
    if direction == "positive":
        m3 = (positive_pattern * repeats)[: len(item_nums)]
    elif direction == "adverse":
        m3 = (adverse_pattern * repeats)[: len(item_nums)]
    else:
        m3 = baseline.copy()
    return [float(value) for value in baseline], [0.5] * len(item_nums), [
        float(value) for value in m3
    ]


def evaluated_subject_record(
    subject_id,
    direction,
    resolved_charter,
    contracts,
    evaluation_contract,
    *,
    item_nums=(1, 2, 3, 4),
    condition="ocular-removed",
):
    prob_b3, prob_m1, prob_m3 = patterned_probabilities(item_nums, direction)
    table, trial_manifest, bindings = prediction_table(
        resolved_charter,
        *contracts,
        subject_id=subject_id,
        condition=condition,
        item_nums=item_nums,
        prob_b3=prob_b3,
        prob_m1=prob_m1,
        prob_m3=prob_m3,
    )
    record = evaluate_synthetic(
        table,
        trial_manifest,
        bindings,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    digest = m1_evaluation.subject_evaluation_sha256(
        record,
        table,
        trial_manifest,
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_bindings=bindings,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
    return record, digest


def aggregate_synthetic(records, record_digests, charter, evaluation_contract):
    return m1_evaluation.aggregate_primary_subject_records(
        records,
        charter,
        evaluation_contract,
        expected_charter_sha256="c" * 64,
        expected_record_sha256_by_subject=record_digests,
        aggregate_provenance=AGGREGATE_PROVENANCE,
    )


def test_primary_aggregate_bundle_round_trip_and_rehashed_drift_rejection(
    tmp_path, resolved_charter, contracts, evaluation_contract
):
    records = []
    digests = {}
    for index, direction in enumerate(("positive", "adverse"), start=1):
        subject_id = f"sub-SYN{index:03d}"
        record, digest = evaluated_subject_record(
            subject_id,
            direction,
            resolved_charter,
            contracts,
            evaluation_contract,
        )
        records.append(record)
        digests[subject_id] = digest
    aggregate = aggregate_synthetic(
        records,
        digests,
        resolved_charter,
        evaluation_contract,
    )
    destination = tmp_path / "primary-aggregate-result"
    bundle_digest = write_primary_aggregate_bundle_atomic(
        destination,
        aggregate,
        records,
        resolved_charter,
        evaluation_contract,
        expected_charter_sha256="c" * 64,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )

    loaded = load_primary_aggregate_bundle(
        destination,
        list(reversed(records)),
        resolved_charter,
        evaluation_contract,
        expected_charter_sha256="c" * 64,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
        expected_bundle_sha256=bundle_digest,
    )
    assert loaded == aggregate

    changed = deepcopy(aggregate)
    changed["primary_category"] = "positive_increment"
    rewritten_digest = rewrite_result_record_and_manifest(destination, changed)
    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError,
        match="complete recomputation",
    ):
        load_primary_aggregate_bundle(
            destination,
            records,
            resolved_charter,
            evaluation_contract,
            expected_charter_sha256="c" * 64,
            expected_record_sha256_by_subject=digests,
            expected_aggregate_provenance=AGGREGATE_PROVENANCE,
            expected_bundle_sha256=rewritten_digest,
        )


def test_primary_aggregate_contract_fixes_estimand_and_keeps_sensitivity_separate(
    evaluation_contract,
):
    aggregate = evaluation_contract["primary_aggregate"]
    assert aggregate["condition"] == "ocular-removed"
    assert aggregate["estimand"] == "unweighted_mean_subject_auc_M3_minus_refit_B3"
    assert aggregate["bootstrap"] == {
        "unit": "subject",
        "resamples": 2000,
        "seed": 0,
        "interval_percentiles": [2.5, 97.5],
        "numpy_percentile_method": "linear",
    }
    assert evaluation_contract["metric_scope"]["ocular_condition_discordance"] == (
        "IMPLEMENTED_SYNTHETIC_ONLY"
    )


def test_primary_aggregate_is_unweighted_over_subjects(
    resolved_charter, contracts, evaluation_contract
):
    short, short_digest = evaluated_subject_record(
        "sub-SYN001",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    long, long_digest = evaluated_subject_record(
        "sub-SYN002",
        "adverse",
        resolved_charter,
        contracts,
        evaluation_contract,
        item_nums=tuple(range(1, 9)),
    )
    result = aggregate_synthetic(
        [long, short],
        {"sub-SYN001": short_digest, "sub-SYN002": long_digest},
        resolved_charter,
        evaluation_contract,
    )

    assert short["delta_auc_m3_minus_b3"] == pytest.approx(0.5)
    assert long["delta_auc_m3_minus_b3"] == pytest.approx(-0.5)
    assert result["mean_delta_auc_m3_minus_b3"] == pytest.approx(0.0)
    trial_weighted = (
        short["n_trials"] * short["delta_auc_m3_minus_b3"]
        + long["n_trials"] * long["delta_auc_m3_minus_b3"]
    ) / (short["n_trials"] + long["n_trials"])
    assert trial_weighted == pytest.approx(-1 / 6)
    assert result["mean_delta_auc_m3_minus_b3"] != pytest.approx(trial_weighted)
    assert result["primary_category"] == "no_detectable_increment__imprecise"


@pytest.mark.parametrize(
    ("directions", "expected_category"),
    [
        (("positive", "positive"), "positive_increment"),
        (("adverse", "adverse"), "adverse_increment"),
        (("positive", "adverse"), "no_detectable_increment__imprecise"),
    ],
)
def test_primary_aggregate_uses_only_interval_categories(
    resolved_charter,
    contracts,
    evaluation_contract,
    directions,
    expected_category,
):
    records = []
    digests = {}
    for index, direction in enumerate(directions, start=1):
        subject_id = f"sub-SYN{index:03d}"
        record, digest = evaluated_subject_record(
            subject_id,
            direction,
            resolved_charter,
            contracts,
            evaluation_contract,
        )
        records.append(record)
        digests[subject_id] = digest
    result = aggregate_synthetic(
        records,
        digests,
        resolved_charter,
        evaluation_contract,
    )
    assert result["primary_category"] == expected_category
    assert result["equivalence_claim_allowed"] is False
    assert result["secondary_can_create_positive_path"] is False


def test_primary_bootstrap_matches_independent_seeded_subject_resampling(
    resolved_charter, contracts, evaluation_contract
):
    directions = ("positive", "null", "adverse")
    records = []
    digests = {}
    for index, direction in enumerate(directions, start=1):
        subject_id = f"sub-SYN{index:03d}"
        record, digest = evaluated_subject_record(
            subject_id,
            direction,
            resolved_charter,
            contracts,
            evaluation_contract,
        )
        records.append(record)
        digests[subject_id] = digest
    result = aggregate_synthetic(
        records,
        digests,
        resolved_charter,
        evaluation_contract,
    )

    values = np.asarray(
        [record["delta_auc_m3_minus_b3"] for record in records], dtype=float
    )
    rng = np.random.default_rng(0)
    draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
    expected_low, expected_high = np.percentile(
        draws, [2.5, 97.5], method="linear"
    )
    assert result["ci_low"] == pytest.approx(expected_low)
    assert result["ci_high"] == pytest.approx(expected_high)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "digest"])
def test_primary_aggregate_rejects_cohort_and_record_binding_drift(
    resolved_charter, contracts, evaluation_contract, mutation
):
    record, digest = evaluated_subject_record(
        "sub-SYN001",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    records = [record]
    digests = {"sub-SYN001": digest}
    if mutation == "missing":
        digests = {}
    elif mutation == "extra":
        digests["sub-SYN999"] = "9" * 64
    elif mutation == "duplicate":
        records.append(deepcopy(record))
    else:
        digests["sub-SYN001"] = "9" * 64

    with pytest.raises(m1_evaluation.EvaluationConsumerValidationError):
        aggregate_synthetic(records, digests, resolved_charter, evaluation_contract)


def test_primary_aggregate_rejects_sensitivity_condition(
    resolved_charter, contracts, evaluation_contract
):
    record, digest = evaluated_subject_record(
        "sub-SYN001",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
        condition="ocular-kept",
    )
    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError, match="ocular-removed"
    ):
        aggregate_synthetic(
            [record],
            {"sub-SYN001": digest},
            resolved_charter,
            evaluation_contract,
        )


def test_primary_aggregate_record_recomputes_and_hashes_deterministically(
    resolved_charter, contracts, evaluation_contract
):
    records = []
    digests = {}
    for index, direction in enumerate(("positive", "adverse"), start=1):
        subject_id = f"sub-SYN{index:03d}"
        record, digest = evaluated_subject_record(
            subject_id,
            direction,
            resolved_charter,
            contracts,
            evaluation_contract,
        )
        records.append(record)
        digests[subject_id] = digest
    aggregate = aggregate_synthetic(
        list(reversed(records)),
        digests,
        resolved_charter,
        evaluation_contract,
    )
    digest = m1_evaluation.primary_aggregate_sha256(
        aggregate,
        records,
        resolved_charter,
        evaluation_contract,
        expected_charter_sha256="c" * 64,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )
    assert digest == m1_evaluation.primary_aggregate_sha256(
        deepcopy(aggregate),
        list(reversed(records)),
        resolved_charter,
        evaluation_contract,
        expected_charter_sha256="c" * 64,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )

    changed = deepcopy(aggregate)
    changed["mean_delta_auc_m3_minus_b3"] = 0.25
    with pytest.raises(m1_evaluation.EvaluationConsumerValidationError):
        m1_evaluation.validate_primary_aggregate_record(
            changed,
            records,
            resolved_charter,
            evaluation_contract,
            expected_charter_sha256="c" * 64,
            expected_record_sha256_by_subject=digests,
            expected_aggregate_provenance=AGGREGATE_PROVENANCE,
        )


def test_primary_aggregate_fails_closed_at_d26_then_d29(
    pending_charter, resolved_charter, contracts, evaluation_contract
):
    record, digest = evaluated_subject_record(
        "sub-SYN001",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    with pytest.raises(PredictionTableValidationError, match="PENDING_D26"):
        aggregate_synthetic(
            [record],
            {"sub-SYN001": digest},
            with_d26_pending(pending_charter),
            evaluation_contract,
        )

    with pytest.raises(PredictionTableValidationError, match="PENDING_D29"):
        aggregate_synthetic(
            [record], {"sub-SYN001": digest}, pending_charter, evaluation_contract
        )


def test_primary_aggregate_rejects_bootstrap_contract_drift(
    resolved_charter, contracts, evaluation_contract
):
    record, digest = evaluated_subject_record(
        "sub-SYN001",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    changed_contract = deepcopy(evaluation_contract)
    changed_contract["primary_aggregate"]["bootstrap"]["seed"] = 1
    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError, match="bootstrap"
    ):
        aggregate_synthetic(
            [record], {"sub-SYN001": digest}, resolved_charter, changed_contract
        )


@pytest.mark.parametrize("changed_field", ["evaluator_code_commit", "evaluation_command"])
def test_primary_aggregate_rejects_mixed_evaluator_provenance(
    resolved_charter, contracts, evaluation_contract, changed_field
):
    first, first_digest = evaluated_subject_record(
        "sub-SYN001",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    second, _ = evaluated_subject_record(
        "sub-SYN002",
        "positive",
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    second[changed_field] = (
        "9" * 40 if changed_field == "evaluator_code_commit" else "different-command"
    )
    second_digest = m1_evaluation.subject_record_content_sha256(
        second, evaluation_contract
    )
    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError,
        match="inconsistent evaluator provenance",
    ):
        aggregate_synthetic(
            [first, second],
            {"sub-SYN001": first_digest, "sub-SYN002": second_digest},
            resolved_charter,
            evaluation_contract,
        )


def cross_condition_intersection_manifest(removed_manifest, kept_manifest):
    removed_by_key = {
        canonical_trial_key(row): row
        for row in removed_manifest["rows"]
        if row["condition_survivor"]
    }
    kept_by_key = {
        canonical_trial_key(row): row
        for row in kept_manifest["rows"]
        if row["condition_survivor"]
    }
    rows = []
    for key in sorted(set(removed_by_key) & set(kept_by_key)):
        removed = removed_by_key[key]
        kept = kept_by_key[key]
        assert event_provenance_sha256(removed) == event_provenance_sha256(kept)
        rows.append(
            {
                "subject_id": removed["subject_id"],
                "session": removed["session"],
                "trial": removed["trial"],
                "item_num": removed["item_num"],
                "ocular_removed_row_sha256": canonical_row_sha256(removed),
                "ocular_kept_row_sha256": canonical_row_sha256(kept),
                "recalled": removed["recalled"],
                "event_provenance_sha256": event_provenance_sha256(removed),
                "ocular_removed_epoch_row": removed["epoch_row"],
                "ocular_kept_epoch_row": kept["epoch_row"],
            }
        )
    return {
        "schema_id": "neuroai-m1-cross-condition-intersection-v1",
        "schema_version": "1.0",
        "manifest_kind": "cross_condition_intersection",
        "created_at_utc": "2026-08-16T23:25:00Z",
        "charter_id": removed_manifest["charter_id"],
        "charter_version": removed_manifest["charter_version"],
        "charter_sha256": removed_manifest["charter_sha256"],
        "roi_map_id": removed_manifest["roi_map_id"],
        "roi_map_sha256": removed_manifest["roi_map_sha256"],
        "code_commit": "b" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "synthetic-intersection-test-only",
        "source_inventory_sha256": SHA,
        "rows": rows,
    }


def condition_comparison_inputs(
    resolved_charter,
    contracts,
    *,
    kept_item_nums=(1, 2, 3, 4),
    kept_subject_id="sub-SYN001",
    kept_outer_test_session=2,
):
    feature_contract, manifest_contract, prediction_contract = contracts
    removed_table, removed_manifest, removed_bindings = prediction_table(
        resolved_charter,
        *contracts,
        condition="ocular-removed",
        prob_b3=[0.4, 0.6, 0.6, 0.4],
        prob_m1=[0.3, 0.7, 0.7, 0.3],
        prob_m3=[0.2, 0.8, 0.8, 0.2],
    )
    kept_table, kept_manifest, kept_bindings = prediction_table(
        resolved_charter,
        *contracts,
        condition="ocular-kept",
        subject_id=kept_subject_id,
        item_nums=kept_item_nums,
        outer_test_session=kept_outer_test_session,
        prob_b3=[0.45, 0.55, 0.55, 0.45, 0.5][: len(kept_item_nums)],
        prob_m1=[0.35, 0.65, 0.65, 0.35, 0.5][: len(kept_item_nums)],
        prob_m3=[0.25, 0.75, 0.75, 0.25, 0.5][: len(kept_item_nums)],
    )
    intersection = cross_condition_intersection_manifest(
        removed_manifest, kept_manifest
    )
    expected_source_digests = {
        "ocular-removed": prediction_table_sha256(
            removed_table,
            removed_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=removed_bindings,
        ),
        "ocular-kept": prediction_table_sha256(
            kept_table,
            kept_manifest,
            resolved_charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_bindings=kept_bindings,
        ),
    }
    return {
        "removed_table": removed_table,
        "kept_table": kept_table,
        "removed_manifest": removed_manifest,
        "kept_manifest": kept_manifest,
        "intersection": intersection,
        "removed_bindings": removed_bindings,
        "kept_bindings": kept_bindings,
        "expected_intersection_sha256": manifest_sha256(
            intersection, manifest_contract
        ),
        "expected_source_digests": expected_source_digests,
    }


def build_comparison_synthetic(inputs, charter, contracts):
    return build_condition_comparison_table(
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        comparison_provenance=COMPARISON_PROVENANCE,
    )


def validate_comparison_synthetic(record, inputs, charter, contracts):
    validate_condition_comparison_table(
        record,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
    )


def write_comparison_bundle_synthetic(destination, record, inputs, charter, contracts):
    return write_condition_comparison_bundle_atomic(
        destination,
        record,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
    )


def load_comparison_bundle_synthetic(
    destination, digest, inputs, charter, contracts
):
    return load_condition_comparison_bundle(
        destination,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        expected_bundle_sha256=digest,
    )


def evaluate_common_intersection_synthetic(
    table,
    inputs,
    charter,
    contracts,
    evaluation_contract,
):
    return m1_evaluation.evaluate_common_intersection_subject(
        table,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        evaluation_contract,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        evaluation_provenance=EVALUATION_PROVENANCE,
    )


def validate_common_intersection_evaluation_synthetic(
    record,
    table,
    inputs,
    charter,
    contracts,
    evaluation_contract,
):
    m1_evaluation.validate_common_intersection_subject_record(
        record,
        table,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        evaluation_contract,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )


def write_common_subject_result_bundle_synthetic(
    destination,
    record,
    table,
    inputs,
    charter,
    contracts,
    evaluation_contract,
):
    return write_common_intersection_subject_evaluation_bundle_atomic(
        destination,
        record,
        table,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        evaluation_contract,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )


def load_common_subject_result_bundle_synthetic(
    destination,
    digest,
    table,
    inputs,
    charter,
    contracts,
    evaluation_contract,
):
    return load_common_intersection_subject_evaluation_bundle(
        destination,
        table,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        charter,
        *contracts,
        evaluation_contract,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
        expected_bundle_sha256=digest,
    )


def test_condition_comparison_contract_is_metric_free_and_intersection_bound(
    prediction_contract,
):
    contract = prediction_contract["condition_comparison_table"]
    assert contract["schema_id"] == "neuroai-m1-condition-comparison-table-v1"
    assert contract["source_scope"] == "exact_common_outer_test_intersection"
    assert contract["metric_fields_allowed"] is False
    assert set(contract["probability_fields"]) == {
        "ocular_removed_prob_b3",
        "ocular_removed_prob_m1",
        "ocular_removed_prob_m3",
        "ocular_kept_prob_b3",
        "ocular_kept_prob_m1",
        "ocular_kept_prob_m3",
    }


def test_condition_comparison_bundle_contract_is_typed_atomic_and_decision_independent(
    prediction_contract,
):
    bundle = prediction_contract["condition_comparison_storage_bundle"]

    assert prediction_contract["version"] == "1.3-draft"
    assert bundle["atomic_publication"] == (
        "temporary_directory_fsync_then_atomic_noreplace_rename"
    )
    assert bundle["exact_files"] == [
        "metadata.json",
        "row-records.json",
        "probabilities.npy",
        "bundle.json",
        "bundle.sha256",
    ]
    assert bundle["probabilities"]["field_order"] == [
        "ocular_removed_prob_b3",
        "ocular_removed_prob_m1",
        "ocular_removed_prob_m3",
        "ocular_kept_prob_b3",
        "ocular_kept_prob_m1",
        "ocular_kept_prob_m3",
    ]
    assert bundle["probabilities"]["dtype"] == "little_endian_float64"
    assert bundle["probabilities"]["shape"] == "n_rows_by_6"
    assert bundle["probabilities"]["allow_pickle"] is False
    assert bundle["metric_or_result_fields_allowed"] is False
    assert bundle["real_publication_allowed_before_final_readiness"] is False
    assert condition_comparison_bundle_contract_sha256(prediction_contract)


def test_condition_comparison_bundle_round_trip_is_exact_typed_and_immutable(
    tmp_path, resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    destination = tmp_path / "comparison-bundle"

    digest = write_comparison_bundle_synthetic(
        destination, record, inputs, resolved_charter, contracts
    )

    assert {path.name for path in destination.iterdir()} == set(
        contracts[2]["condition_comparison_storage_bundle"]["exact_files"]
    )
    with (destination / "probabilities.npy").open("rb") as handle:
        probabilities = np.load(handle, allow_pickle=False)
        assert handle.read(1) == b""
    assert probabilities.dtype.str == "<f8"
    assert probabilities.flags.c_contiguous
    assert probabilities.shape == (4, 6)

    loaded = load_comparison_bundle_synthetic(
        destination, digest, inputs, resolved_charter, contracts
    )
    assert loaded == record

    with pytest.raises(PredictionTableValidationError, match="already exists"):
        write_comparison_bundle_synthetic(
            destination, record, inputs, resolved_charter, contracts
        )


def test_condition_comparison_bundle_rejects_tamper_extra_and_wrong_external_digest(
    tmp_path, resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)

    def publish(name):
        destination = tmp_path / name
        digest = write_comparison_bundle_synthetic(
            destination, record, inputs, resolved_charter, contracts
        )
        return destination, digest

    destination, _ = publish("wrong-external")
    with pytest.raises(PredictionTableValidationError, match="external bundle digest"):
        load_comparison_bundle_synthetic(
            destination, "0" * 64, inputs, resolved_charter, contracts
        )

    destination, digest = publish("tampered-probabilities")
    values_path = destination / "probabilities.npy"
    payload = bytearray(values_path.read_bytes())
    payload[-1] ^= 1
    values_path.write_bytes(payload)
    with pytest.raises(PredictionTableValidationError, match="probabilities.npy digest"):
        load_comparison_bundle_synthetic(
            destination, digest, inputs, resolved_charter, contracts
        )

    destination, digest = publish("extra-file")
    (destination / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(PredictionTableValidationError, match="exact files"):
        load_comparison_bundle_synthetic(
            destination, digest, inputs, resolved_charter, contracts
        )


def test_condition_comparison_bundle_rejects_contract_drift(
    tmp_path, resolved_charter, contracts
):
    changed_contracts = list(contracts)
    changed_prediction_contract = deepcopy(contracts[2])
    changed_prediction_contract["condition_comparison_storage_bundle"][
        "probabilities"
    ]["allow_pickle"] = True
    changed_contracts[2] = changed_prediction_contract
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)

    with pytest.raises(PredictionTableValidationError, match="bundle contract drifted"):
        write_comparison_bundle_synthetic(
            tmp_path / "comparison-bundle",
            record,
            inputs,
            resolved_charter,
            tuple(changed_contracts),
        )


@pytest.mark.parametrize("unsafe_kind", ["object_array", "trailing_bytes"])
def test_condition_comparison_bundle_rejects_unsafe_npy_with_rewritten_hashes(
    tmp_path, resolved_charter, contracts, unsafe_kind
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    destination = tmp_path / unsafe_kind
    write_comparison_bundle_synthetic(
        destination, record, inputs, resolved_charter, contracts
    )
    values_path = destination / "probabilities.npy"
    if unsafe_kind == "object_array":
        with values_path.open("wb") as handle:
            np.save(handle, np.array([[object()]], dtype=object), allow_pickle=True)
    else:
        with values_path.open("ab") as handle:
            handle.write(b"unexpected trailing byte")
    digest = rewrite_prediction_bundle_manifest(
        destination,
        probabilities_npy_sha256=sha256(values_path.read_bytes()).hexdigest(),
    )

    message = "safe NumPy array" if unsafe_kind == "object_array" else "trailing bytes"
    with pytest.raises(PredictionTableValidationError, match=message):
        load_comparison_bundle_synthetic(
            destination, digest, inputs, resolved_charter, contracts
        )


def test_condition_comparison_bundle_revalidates_labels_after_member_rehash(
    tmp_path, resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    destination = tmp_path / "label-tamper"
    write_comparison_bundle_synthetic(
        destination, record, inputs, resolved_charter, contracts
    )
    records_path = destination / "row-records.json"
    row_records = json.loads(records_path.read_text(encoding="utf-8"))
    row_records["rows"][0]["recalled"] = not row_records["rows"][0]["recalled"]
    records_payload = (
        json.dumps(
            row_records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    records_path.write_bytes(records_payload)
    digest = rewrite_prediction_bundle_manifest(
        destination,
        row_records_sha256=sha256(records_payload).hexdigest(),
    )

    with pytest.raises(PredictionTableValidationError):
        load_comparison_bundle_synthetic(
            destination, digest, inputs, resolved_charter, contracts
        )


def test_condition_comparison_bundle_rejects_rehashed_probability_drift(
    tmp_path, resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    destination = tmp_path / "probability-rehash"
    write_comparison_bundle_synthetic(
        destination, record, inputs, resolved_charter, contracts
    )
    values_path = destination / "probabilities.npy"
    with values_path.open("rb") as handle:
        probabilities = np.load(handle, allow_pickle=False)
    probabilities[0, 5] = 0.57
    with values_path.open("wb") as handle:
        np.save(handle, probabilities, allow_pickle=False)
    digest = rewrite_prediction_bundle_manifest(
        destination,
        probabilities_npy_sha256=sha256(values_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(PredictionTableValidationError):
        load_comparison_bundle_synthetic(
            destination, digest, inputs, resolved_charter, contracts
        )


def test_condition_comparison_bundle_rejects_symlink_member(
    tmp_path, resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    destination = tmp_path / "symlink-member"
    digest = write_comparison_bundle_synthetic(
        destination, record, inputs, resolved_charter, contracts
    )
    metadata_path = destination / "metadata.json"
    target = tmp_path / "comparison-metadata-target.json"
    metadata_path.rename(target)
    metadata_path.symlink_to(target)

    with pytest.raises(PredictionTableValidationError, match="regular file"):
        load_comparison_bundle_synthetic(
            destination, digest, inputs, resolved_charter, contracts
        )


def test_condition_comparison_pairs_only_exact_common_outer_test_rows(
    resolved_charter, contracts
):
    inputs = condition_comparison_inputs(
        resolved_charter, contracts, kept_item_nums=(1, 2, 3, 4, 5)
    )
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)

    assert record["n_rows"] == 4
    assert [row["item_num"] for row in record["rows"]] == [1, 2, 3, 4]
    assert [row["ocular_removed_prob_m3"] for row in record["rows"]] == [
        0.2,
        0.8,
        0.8,
        0.2,
    ]
    assert [row["ocular_kept_prob_m3"] for row in record["rows"]] == [
        0.25,
        0.75,
        0.75,
        0.25,
    ]
    assert not any("auc" in field or "category" in field for field in record)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "label", "probability", "source_link", "metric"],
)
def test_condition_comparison_record_rejects_row_or_result_drift(
    resolved_charter, contracts, mutation
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    changed = deepcopy(record)
    if mutation == "missing":
        changed["rows"] = changed["rows"][:-1]
        changed["n_rows"] -= 1
    elif mutation == "extra":
        changed["rows"].append(deepcopy(changed["rows"][0]))
        changed["rows"][-1]["item_num"] = 999
        changed["n_rows"] += 1
    elif mutation == "label":
        changed["rows"][0]["recalled"] = not changed["rows"][0]["recalled"]
    elif mutation == "probability":
        changed["rows"][0]["ocular_kept_prob_m3"] = 0.99
    elif mutation == "source_link":
        changed["rows"][0]["ocular_kept_prediction_row_sha256"] = "0" * 64
    else:
        changed["delta_auc"] = 0.5

    with pytest.raises(PredictionTableValidationError):
        validate_comparison_synthetic(changed, inputs, resolved_charter, contracts)


def test_condition_comparison_rejects_stale_source_or_intersection_digest(
    resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    stale_source = deepcopy(inputs)
    stale_source["expected_source_digests"]["ocular-kept"] = "0" * 64
    with pytest.raises(PredictionTableValidationError, match="source prediction-table digest"):
        build_comparison_synthetic(stale_source, resolved_charter, contracts)

    stale_intersection = deepcopy(inputs)
    stale_intersection["expected_intersection_sha256"] = "0" * 64
    with pytest.raises(PredictionTableValidationError, match="intersection manifest digest"):
        build_comparison_synthetic(stale_intersection, resolved_charter, contracts)


def test_condition_comparison_rejects_tampered_intersection_before_pairing(
    resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    inputs["intersection"]["rows"][0]["event_provenance_sha256"] = "0" * 64
    inputs["expected_intersection_sha256"] = manifest_sha256(
        inputs["intersection"], contracts[1]
    )
    with pytest.raises(PredictionTableValidationError, match="event provenance"):
        build_comparison_synthetic(inputs, resolved_charter, contracts)


def test_condition_comparison_requires_ordered_condition_roles(
    resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    inputs["removed_table"], inputs["kept_table"] = (
        inputs["kept_table"],
        inputs["removed_table"],
    )
    with pytest.raises(PredictionTableValidationError, match="ocular-removed"):
        build_comparison_synthetic(inputs, resolved_charter, contracts)


@pytest.mark.parametrize(
    ("input_kwargs", "message"),
    [
        ({"kept_subject_id": "sub-SYN999"}, "subjects do not match"),
        ({"kept_outer_test_session": 3}, "common outer-test session"),
    ],
)
def test_condition_comparison_rejects_subject_or_session_mismatch(
    resolved_charter, contracts, input_kwargs, message
):
    inputs = condition_comparison_inputs(
        resolved_charter, contracts, **input_kwargs
    )
    with pytest.raises(PredictionTableValidationError, match=message):
        build_comparison_synthetic(inputs, resolved_charter, contracts)


def test_condition_comparison_fails_closed_at_d26_then_d29(
    pending_charter, resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    with pytest.raises(PredictionTableValidationError, match="PENDING_D26"):
        build_comparison_synthetic(inputs, with_d26_pending(pending_charter), contracts)

    with pytest.raises(PredictionTableValidationError, match="PENDING_D29"):
        build_comparison_synthetic(inputs, pending_charter, contracts)


def test_condition_comparison_recomputes_and_hashes_deterministically(
    resolved_charter, contracts
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    record = build_comparison_synthetic(inputs, resolved_charter, contracts)
    validate_comparison_synthetic(record, inputs, resolved_charter, contracts)
    canonical = canonical_condition_comparison_table_bytes(
        record,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        resolved_charter,
        *contracts,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
    )
    digest = condition_comparison_table_sha256(
        record,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        resolved_charter,
        *contracts,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
    )
    assert canonical.endswith(b"\n")
    assert len(digest) == 64


def test_common_intersection_subject_bundle_round_trip_and_rehashed_drift_rejection(
    tmp_path,
    resolved_charter,
    contracts,
    evaluation_contract,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    record = evaluate_common_intersection_synthetic(
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    destination = tmp_path / "common-subject-result"
    digest = write_common_subject_result_bundle_synthetic(
        destination,
        record,
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )

    loaded = load_common_subject_result_bundle_synthetic(
        destination,
        digest,
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    assert loaded == record

    changed = deepcopy(record)
    changed["ocular_kept_brier_m3"] = float(
        changed["ocular_kept_brier_m3"] + 0.01
    )
    rewritten_digest = rewrite_result_record_and_manifest(destination, changed)
    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError,
        match="complete recomputation",
    ):
        load_common_subject_result_bundle_synthetic(
            destination,
            rewritten_digest,
            table,
            inputs,
            resolved_charter,
            contracts,
            evaluation_contract,
        )


def test_common_intersection_subject_contract_is_paired_and_category_free(
    evaluation_contract,
):
    contract = evaluation_contract["common_intersection_subject_record"]
    assert contract["schema_id"] == (
        "neuroai-m1-common-intersection-subject-evaluation-v1"
    )
    assert contract["condition_order"] == ["ocular-removed", "ocular-kept"]
    assert len(contract["metric_fields_by_condition"]["ocular-removed"]) == 11
    assert len(contract["metric_fields_by_condition"]["ocular-kept"]) == 11
    assert not any(
        "balanced_accuracy" in field
        or "category" in field
        or "discord" in field
        for field in contract["exact_fields"]
    )
    assert evaluation_contract["metric_scope"][
        "ocular_condition_discordance"
    ] == "IMPLEMENTED_SYNTHETIC_ONLY"


@pytest.mark.parametrize("mutation", ["metric_group", "discordance_scope"])
def test_common_intersection_subject_rejects_evaluation_contract_drift(
    resolved_charter,
    contracts,
    evaluation_contract,
    mutation,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    changed_contract = deepcopy(evaluation_contract)
    if mutation == "metric_group":
        del changed_contract["common_intersection_subject_record"][
            "metric_fields_by_condition"
        ]["ocular-kept"]
    else:
        changed_contract["metric_scope"]["ocular_condition_discordance"] = (
            "IMPLEMENTED_TOO_EARLY"
        )

    with pytest.raises(m1_evaluation.EvaluationConsumerValidationError):
        evaluate_common_intersection_synthetic(
            table,
            inputs,
            resolved_charter,
            contracts,
            changed_contract,
        )


def test_common_intersection_subject_consumes_exact_paired_table(
    resolved_charter,
    contracts,
    evaluation_contract,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    record = evaluate_common_intersection_synthetic(
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )

    assert record["condition_comparison_table_sha256"] == (
        condition_comparison_table_sha256(
            table,
            inputs["removed_table"],
            inputs["kept_table"],
            inputs["removed_manifest"],
            inputs["kept_manifest"],
            inputs["intersection"],
            resolved_charter,
            *contracts,
            expected_removed_bindings=inputs["removed_bindings"],
            expected_kept_bindings=inputs["kept_bindings"],
            expected_intersection_manifest_sha256=inputs[
                "expected_intersection_sha256"
            ],
            expected_prediction_table_sha256_by_condition=inputs[
                "expected_source_digests"
            ],
            expected_comparison_provenance=COMPARISON_PROVENANCE,
        )
    )
    assert record["n_trials"] == len(table["rows"]) == 4
    assert record["n_recalled"] == 2
    assert record["n_not_recalled"] == 2
    assert record["ocular_removed_auc_b3"] == pytest.approx(0.5)
    assert record["ocular_removed_auc_m3"] == pytest.approx(0.5)
    assert record["ocular_removed_delta_auc_m3_minus_b3"] == pytest.approx(0.0)
    assert record["ocular_removed_brier_b3"] == pytest.approx(0.26)
    assert record["ocular_removed_brier_m3"] == pytest.approx(0.34)
    assert record["ocular_kept_auc_b3"] == pytest.approx(0.5)
    assert record["ocular_kept_auc_m3"] == pytest.approx(0.5)
    assert record["ocular_kept_delta_auc_m3_minus_b3"] == pytest.approx(0.0)
    assert record["ocular_kept_brier_b3"] == pytest.approx(0.2525)
    assert record["ocular_kept_brier_m3"] == pytest.approx(0.3125)


def test_common_intersection_subject_passes_every_row_to_both_conditions(
    resolved_charter,
    contracts,
    evaluation_contract,
    monkeypatch,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    observed = []
    original = m1_evaluation.incremental_value

    def capture(y_true, prob_old, prob_new):
        observed.append(
            (y_true.tolist(), prob_old.tolist(), prob_new.tolist())
        )
        return original(y_true, prob_old, prob_new)

    monkeypatch.setattr(m1_evaluation, "incremental_value", capture)
    evaluate_common_intersection_synthetic(
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )

    labels = [row["recalled"] for row in table["rows"]]
    assert observed == [
        (
            labels,
            [row["ocular_removed_prob_b3"] for row in table["rows"]],
            [row["ocular_removed_prob_m3"] for row in table["rows"]],
        ),
        (
            labels,
            [row["ocular_kept_prob_b3"] for row in table["rows"]],
            [row["ocular_kept_prob_m3"] for row in table["rows"]],
        ),
    ]


@pytest.mark.parametrize("mutation", ["row", "source", "result"])
def test_common_intersection_subject_rejects_invalid_table_before_metrics(
    resolved_charter,
    contracts,
    evaluation_contract,
    mutation,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    if mutation == "row":
        table["rows"][0]["ocular_kept_prob_m3"] = float("nan")
    elif mutation == "source":
        table["ocular_kept_prediction_table_sha256"] = "0" * 64
    else:
        table["condition_discordant"] = False

    with pytest.raises(PredictionTableValidationError):
        evaluate_common_intersection_synthetic(
            table,
            inputs,
            resolved_charter,
            contracts,
            evaluation_contract,
        )


@pytest.mark.parametrize(
    "mutation",
    ["metric", "count", "digest", "identity", "provenance", "extra"],
)
def test_common_intersection_subject_record_recomputes_and_rejects_drift(
    resolved_charter,
    contracts,
    evaluation_contract,
    mutation,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    record = evaluate_common_intersection_synthetic(
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    if mutation == "metric":
        record["ocular_kept_delta_auc_m3_minus_b3"] = 0.5
    elif mutation == "count":
        record["n_trials"] = 1
    elif mutation == "digest":
        record["condition_comparison_table_sha256"] = "0" * 64
    elif mutation == "identity":
        record["subject_id"] = "sub-SYN999"
    elif mutation == "provenance":
        record["evaluator_code_commit"] = "0" * 40
    else:
        record["condition_discordant"] = False

    with pytest.raises(m1_evaluation.EvaluationConsumerValidationError):
        validate_common_intersection_evaluation_synthetic(
            record,
            table,
            inputs,
            resolved_charter,
            contracts,
            evaluation_contract,
        )


def test_common_intersection_subject_rejects_one_class_intersection(
    resolved_charter,
    contracts,
    evaluation_contract,
):
    inputs = condition_comparison_inputs(
        resolved_charter,
        contracts,
        kept_item_nums=(1, 3, 5, 6),
    )
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    assert [row["recalled"] for row in table["rows"]] == [False, False]

    with pytest.raises(
        m1_evaluation.EvaluationConsumerValidationError,
        match="requires both recall classes",
    ):
        evaluate_common_intersection_synthetic(
            table,
            inputs,
            resolved_charter,
            contracts,
            evaluation_contract,
        )


def test_common_intersection_subject_fails_closed_at_d26_then_d29(
    pending_charter,
    resolved_charter,
    contracts,
    evaluation_contract,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    with pytest.raises(PredictionTableValidationError, match="PENDING_D26"):
        evaluate_common_intersection_synthetic(
            table,
            inputs,
            with_d26_pending(pending_charter),
            contracts,
            evaluation_contract,
        )

    with pytest.raises(PredictionTableValidationError, match="PENDING_D29"):
        evaluate_common_intersection_synthetic(
            table,
            inputs,
            pending_charter,
            contracts,
            evaluation_contract,
        )


def test_common_intersection_subject_recomputes_and_hashes_deterministically(
    resolved_charter,
    contracts,
    evaluation_contract,
):
    inputs = condition_comparison_inputs(resolved_charter, contracts)
    table = build_comparison_synthetic(inputs, resolved_charter, contracts)
    record = evaluate_common_intersection_synthetic(
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    validate_common_intersection_evaluation_synthetic(
        record,
        table,
        inputs,
        resolved_charter,
        contracts,
        evaluation_contract,
    )
    digest = m1_evaluation.common_intersection_subject_sha256(
        record,
        table,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
    assert len(digest) == 64
    assert digest == m1_evaluation.common_intersection_subject_content_sha256(
        record,
        evaluation_contract,
    )
    assert digest == m1_evaluation.common_intersection_subject_sha256(
        deepcopy(record),
        table,
        inputs["removed_table"],
        inputs["kept_table"],
        inputs["removed_manifest"],
        inputs["kept_manifest"],
        inputs["intersection"],
        resolved_charter,
        *contracts,
        evaluation_contract,
        expected_removed_bindings=inputs["removed_bindings"],
        expected_kept_bindings=inputs["kept_bindings"],
        expected_intersection_manifest_sha256=inputs[
            "expected_intersection_sha256"
        ],
        expected_prediction_table_sha256_by_condition=inputs[
            "expected_source_digests"
        ],
        expected_comparison_provenance=COMPARISON_PROVENANCE,
        expected_evaluation_provenance=EVALUATION_PROVENANCE,
    )
