"""Synthetic contract tests for the paired ocular-condition aggregate.

All records are invented. The suite does not choose D42, compute balanced accuracy, or read a
real label, feature, prediction, metric, or held-out outcome.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from peers.m1_evaluation import (
    EvaluationConsumerValidationError,
    aggregate_common_intersection_subject_records,
    canonical_common_intersection_aggregate_bytes,
    common_intersection_aggregate_sha256,
    common_intersection_subject_content_sha256,
    evaluation_contract_sha256,
    validate_common_intersection_aggregate_record,
)
from peers.m1_result_storage import (
    load_common_intersection_aggregate_bundle,
    write_common_intersection_aggregate_bundle_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
EVALUATION_CONTRACT_PATH = ROOT / "docs" / "m1-evaluation-contract.json"

CHARTER_SHA256 = "c" * 64
AGGREGATE_PROVENANCE = {
    "aggregator_code_commit": "a" * 40,
    "aggregator_dirty_diff_sha256": None,
    "aggregator_package_versions_sha256": "b" * 64,
    "aggregation_command": "synthetic-common-intersection-aggregate",
}


@pytest.fixture(scope="module")
def charter():
    return json.loads(CHARTER_PATH.read_text())


@pytest.fixture(scope="module")
def evaluation_contract():
    return json.loads(EVALUATION_CONTRACT_PATH.read_text())


def paired_subject_record(
    subject_id,
    removed_delta,
    kept_delta,
    evaluation_contract,
    *,
    n_trials=4,
    evaluation_command="synthetic-subject-evaluation",
):
    contract = evaluation_contract["common_intersection_subject_record"]
    record = {
        "schema_id": contract["schema_id"],
        "schema_version": contract["schema_version"],
        "subject_id": subject_id,
        "outer_test_session": 2,
        "ocular_removed_outer_fold_id": f"{subject_id}::ocular-removed::outer-session-2",
        "ocular_kept_outer_fold_id": f"{subject_id}::ocular-kept::outer-session-2",
        "charter_sha256": CHARTER_SHA256,
        "condition_comparison_table_sha256": subject_id.encode().hex().ljust(64, "0")[:64],
        "evaluation_contract_sha256": evaluation_contract_sha256(evaluation_contract),
        "evaluator_code_commit": "d" * 40,
        "evaluator_dirty_diff_sha256": None,
        "evaluator_package_versions_sha256": "e" * 64,
        "evaluation_command": evaluation_command,
        "n_trials": n_trials,
        "n_recalled": n_trials // 2,
        "n_not_recalled": n_trials - n_trials // 2,
    }
    for field in contract["metric_fields"]:
        record[field] = 0.5
    record["ocular_removed_delta_auc_m3_minus_b3"] = float(removed_delta)
    record["ocular_kept_delta_auc_m3_minus_b3"] = float(kept_delta)
    return record


def records_and_digests(specs, evaluation_contract):
    records = []
    digests = {}
    for index, (removed_delta, kept_delta, n_trials) in enumerate(specs, start=1):
        subject_id = f"sub-SYN{index:03d}"
        record = paired_subject_record(
            subject_id,
            removed_delta,
            kept_delta,
            evaluation_contract,
            n_trials=n_trials,
        )
        records.append(record)
        digests[subject_id] = common_intersection_subject_content_sha256(
            record, evaluation_contract
        )
    return records, digests


def aggregate(records, digests, charter, evaluation_contract):
    return aggregate_common_intersection_subject_records(
        records,
        charter,
        evaluation_contract,
        expected_charter_sha256=CHARTER_SHA256,
        expected_record_sha256_by_subject=digests,
        aggregate_provenance=AGGREGATE_PROVENANCE,
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
    (destination / "record.json").write_bytes(record_payload)
    manifest_path = destination / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_json_sha256"] = sha256(record_payload).hexdigest()
    manifest_payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    bundle_digest = sha256(manifest_payload).hexdigest()
    manifest_path.write_bytes(manifest_payload)
    (destination / "bundle.sha256").write_text(
        f"{bundle_digest}\n", encoding="ascii"
    )
    return bundle_digest


def test_common_intersection_aggregate_bundle_round_trip_and_drift_rejection(
    tmp_path, charter, evaluation_contract
):
    records, digests = records_and_digests(
        [(0.5, 0.25, 4), (-0.5, -0.25, 4)], evaluation_contract
    )
    result = aggregate(records, digests, charter, evaluation_contract)
    destination = tmp_path / "common-aggregate-result"
    digest = write_common_intersection_aggregate_bundle_atomic(
        destination,
        result,
        records,
        charter,
        evaluation_contract,
        expected_charter_sha256=CHARTER_SHA256,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )

    loaded = load_common_intersection_aggregate_bundle(
        destination,
        list(reversed(records)),
        charter,
        evaluation_contract,
        expected_charter_sha256=CHARTER_SHA256,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
        expected_bundle_sha256=digest,
    )
    assert loaded == result

    changed = deepcopy(result)
    changed["condition_discordant"] = not changed["condition_discordant"]
    rewritten_digest = rewrite_result_record_and_manifest(destination, changed)
    with pytest.raises(
        EvaluationConsumerValidationError, match="complete recomputation"
    ):
        load_common_intersection_aggregate_bundle(
            destination,
            records,
            charter,
            evaluation_contract,
            expected_charter_sha256=CHARTER_SHA256,
            expected_record_sha256_by_subject=digests,
            expected_aggregate_provenance=AGGREGATE_PROVENANCE,
            expected_bundle_sha256=rewritten_digest,
        )


def test_contract_fixes_paired_subject_bootstrap_and_discordance(
    charter, evaluation_contract
):
    contract = evaluation_contract["common_intersection_aggregate"]
    assert contract["source_scope"] == (
        "complete_exact_common_intersection_subject_record_set"
    )
    assert contract["condition_order"] == ["ocular-removed", "ocular-kept"]
    assert contract["bootstrap"] == {
        "unit": "subject",
        "paired_condition_resampling": True,
        "resamples": 2000,
        "seed": 0,
        "interval_percentiles": [2.5, 97.5],
        "numpy_percentile_method": "linear",
    }
    assert contract["condition_discordance_rule"] == charter["result_categories"][
        "condition_discordant"
    ]
    assert contract["balanced_accuracy_included"] is False
    assert contract["primary_category_unchanged_by_sensitivity"] is True
    assert evaluation_contract["metric_scope"]["ocular_condition_discordance"] == (
        "IMPLEMENTED_SYNTHETIC_ONLY"
    )


def test_aggregate_is_unweighted_and_uses_one_paired_subject_set(
    charter, evaluation_contract
):
    records, digests = records_and_digests(
        [(0.5, 0.25, 4), (-0.5, -0.25, 12)], evaluation_contract
    )
    result = aggregate(list(reversed(records)), digests, charter, evaluation_contract)

    assert result["ocular_removed_mean_delta_auc_m3_minus_b3"] == pytest.approx(0.0)
    assert result["ocular_kept_mean_delta_auc_m3_minus_b3"] == pytest.approx(0.0)
    assert [binding["subject_id"] for binding in result["subject_record_bindings"]] == [
        "sub-SYN001",
        "sub-SYN002",
    ]
    trial_weighted_removed = (4 * 0.5 + 12 * -0.5) / 16
    assert trial_weighted_removed == pytest.approx(-0.25)
    assert result["ocular_removed_mean_delta_auc_m3_minus_b3"] != pytest.approx(
        trial_weighted_removed
    )


def test_both_conditions_use_the_same_seeded_subject_resamples(
    charter, evaluation_contract
):
    specs = [(0.5, 0.25, 4), (0.0, -0.1, 4), (-0.5, 0.3, 4)]
    records, digests = records_and_digests(specs, evaluation_contract)
    result = aggregate(records, digests, charter, evaluation_contract)

    values = np.asarray([[removed, kept] for removed, kept, _ in specs], dtype=float)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(values), size=(2000, len(values)))
    draws = values[indices].mean(axis=1)
    expected = np.percentile(draws, [2.5, 97.5], axis=0, method="linear")

    assert result["ocular_removed_ci_low"] == pytest.approx(expected[0, 0])
    assert result["ocular_removed_ci_high"] == pytest.approx(expected[1, 0])
    assert result["ocular_kept_ci_low"] == pytest.approx(expected[0, 1])
    assert result["ocular_kept_ci_high"] == pytest.approx(expected[1, 1])


@pytest.mark.parametrize(
    ("specs", "expected_removed", "expected_kept", "expected_discordant"),
    [
        (
            [(0.5, -0.5, 4), (-0.4, 0.4, 4)],
            "no_detectable_increment__imprecise",
            "no_detectable_increment__imprecise",
            True,
        ),
        (
            [(0.5, 0.1, 4), (0.5, -0.05, 4)],
            "positive_increment",
            "no_detectable_increment__imprecise",
            True,
        ),
        (
            [(0.5, 0.25, 4), (0.5, 0.25, 4)],
            "positive_increment",
            "positive_increment",
            False,
        ),
    ],
)
def test_discordance_uses_opposite_point_sign_or_different_interval_category(
    charter,
    evaluation_contract,
    specs,
    expected_removed,
    expected_kept,
    expected_discordant,
):
    records, digests = records_and_digests(specs, evaluation_contract)
    result = aggregate(records, digests, charter, evaluation_contract)

    assert result["ocular_removed_category"] == expected_removed
    assert result["ocular_kept_category"] == expected_kept
    assert result["condition_discordant"] is expected_discordant
    assert result["primary_category_unchanged_by_sensitivity"] is True
    assert result["secondary_can_create_positive_path"] is False
    assert result["equivalence_claim_allowed"] is False
    assert result["balanced_accuracy_included"] is False


def test_exact_zero_is_not_treated_as_an_opposite_sign(
    charter, evaluation_contract
):
    records, digests = records_and_digests(
        [(0.0, 0.1, 4), (0.0, -0.1, 4)], evaluation_contract
    )
    result = aggregate(records, digests, charter, evaluation_contract)

    assert result["ocular_removed_mean_delta_auc_m3_minus_b3"] == 0.0
    assert result["ocular_kept_mean_delta_auc_m3_minus_b3"] == 0.0
    assert result["ocular_removed_category"] == result["ocular_kept_category"]
    assert result["condition_discordant"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "digest"])
def test_aggregate_rejects_subject_set_and_digest_drift(
    charter, evaluation_contract, mutation
):
    records, digests = records_and_digests([(0.5, 0.25, 4)], evaluation_contract)
    if mutation == "missing":
        digests = {}
    elif mutation == "extra":
        digests["sub-SYN999"] = "9" * 64
    elif mutation == "duplicate":
        records.append(deepcopy(records[0]))
    else:
        digests["sub-SYN001"] = "9" * 64

    with pytest.raises(EvaluationConsumerValidationError):
        aggregate(records, digests, charter, evaluation_contract)


def test_aggregate_rejects_mixed_subject_evaluator_provenance(
    charter, evaluation_contract
):
    records, digests = records_and_digests(
        [(0.5, 0.25, 4), (0.5, 0.25, 4)], evaluation_contract
    )
    records[1]["evaluation_command"] = "different-evaluator"
    digests["sub-SYN002"] = common_intersection_subject_content_sha256(
        records[1], evaluation_contract
    )

    with pytest.raises(
        EvaluationConsumerValidationError, match="inconsistent evaluator provenance"
    ):
        aggregate(records, digests, charter, evaluation_contract)


@pytest.mark.parametrize(
    "mutation",
    ["metric", "category", "discordance", "binding", "extra", "balanced_accuracy"],
)
def test_aggregate_record_recomputes_and_rejects_result_drift(
    charter, evaluation_contract, mutation
):
    records, digests = records_and_digests(
        [(0.5, 0.25, 4), (0.5, 0.25, 4)], evaluation_contract
    )
    result = aggregate(records, digests, charter, evaluation_contract)
    changed = deepcopy(result)
    if mutation == "metric":
        changed["ocular_kept_ci_low"] = -0.5
    elif mutation == "category":
        changed["ocular_kept_category"] = "adverse_increment"
    elif mutation == "discordance":
        changed["condition_discordant"] = True
    elif mutation == "binding":
        changed["subject_record_bindings"][0][
            "common_intersection_subject_evaluation_sha256"
        ] = "9" * 64
    elif mutation == "extra":
        changed["primary_category"] = "positive_increment"
    else:
        changed["ocular_removed_balanced_accuracy"] = 0.75

    with pytest.raises(EvaluationConsumerValidationError):
        validate_common_intersection_aggregate_record(
            changed,
            records,
            charter,
            evaluation_contract,
            expected_charter_sha256=CHARTER_SHA256,
            expected_record_sha256_by_subject=digests,
            expected_aggregate_provenance=AGGREGATE_PROVENANCE,
        )


def test_aggregate_recomputes_serializes_and_hashes_deterministically(
    charter, evaluation_contract
):
    records, digests = records_and_digests(
        [(0.5, 0.25, 4), (-0.5, -0.25, 4)], evaluation_contract
    )
    result = aggregate(list(reversed(records)), digests, charter, evaluation_contract)
    canonical = canonical_common_intersection_aggregate_bytes(
        result,
        records,
        charter,
        evaluation_contract,
        expected_charter_sha256=CHARTER_SHA256,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )
    digest = common_intersection_aggregate_sha256(
        result,
        list(reversed(records)),
        charter,
        evaluation_contract,
        expected_charter_sha256=CHARTER_SHA256,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )

    assert canonical.endswith(b"\n")
    assert len(digest) == 64
    assert digest == common_intersection_aggregate_sha256(
        deepcopy(result),
        records,
        charter,
        evaluation_contract,
        expected_charter_sha256=CHARTER_SHA256,
        expected_record_sha256_by_subject=digests,
        expected_aggregate_provenance=AGGREGATE_PROVENANCE,
    )


def test_aggregate_fails_closed_on_charter_or_contract_drift(
    charter, evaluation_contract
):
    records, digests = records_and_digests([(0.5, 0.25, 4)], evaluation_contract)

    pending = deepcopy(charter)
    pending["features"]["analysis_window_s"] = "PENDING_D29"
    pending["features"]["time_bins_s"] = "PENDING_D29"
    with pytest.raises(Exception, match="PENDING_D29"):
        aggregate(records, digests, pending, evaluation_contract)

    changed_contract = deepcopy(evaluation_contract)
    changed_contract["common_intersection_aggregate"]["bootstrap"]["seed"] = 1
    with pytest.raises(EvaluationConsumerValidationError, match="bootstrap"):
        aggregate(records, digests, charter, changed_contract)


@pytest.mark.parametrize("field", ["estimand", "uncertainty", "result_categories"])
def test_aggregate_rejects_scientific_contract_drift(
    charter, evaluation_contract, field
):
    records, digests = records_and_digests([(0.5, 0.25, 4)], evaluation_contract)
    changed_contract = deepcopy(evaluation_contract)
    if field == "result_categories":
        changed_contract["common_intersection_aggregate"][field] = {
            "positive_increment": "mean_gt_0"
        }
    else:
        changed_contract["common_intersection_aggregate"][field] = "changed"

    with pytest.raises(EvaluationConsumerValidationError):
        aggregate(records, digests, charter, changed_contract)
