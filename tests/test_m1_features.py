"""Synthetic tests for the pre-result M1 feature-table safety boundary.

The fixtures are invented. They enforce but do not change D26, and they do not read a real epoch,
choose D29, build an EEG feature, fit a model, or inspect any held-out outcome.
"""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from peers.m1_features import (
    FeatureTableValidationError,
    canonical_feature_table_bytes,
    canonical_normalization_transform_provenance_bytes,
    derive_feature_names,
    feature_schema_sha256,
    feature_table_sha256,
    fit_strict_inductive_normalizer,
    load_feature_table_bundle,
    normalization_contract_sha256,
    normalization_statistics_evidence,
    normalization_transform_provenance_sha256,
    validate_feature_table,
    validate_normalization_transform_provenance,
    write_feature_table_bundle_atomic,
)
from peers.m1_manifests import (
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    load_json_unique,
    manifest_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
CONTRACT_PATH = ROOT / "docs" / "m1-feature-table-contract.json"
MANIFEST_CONTRACT_PATH = ROOT / "docs" / "m1-manifest-contracts.json"
SHA = "a" * 64


@pytest.fixture(scope="module")
def feature_contract():
    return load_json_unique(CONTRACT_PATH)


@pytest.fixture(scope="module")
def frozen_charter():
    return load_json_unique(CHARTER_PATH)


@pytest.fixture()
def resolved_charter(frozen_charter):
    return deepcopy(frozen_charter)


@pytest.fixture(scope="module")
def manifest_contract():
    return load_json_unique(MANIFEST_CONTRACT_PATH)


def synthetic_keys():
    return [
        ("sub-SYN001", 0, 1, 1),
        ("sub-SYN001", 0, 2, 2),
    ]


def expected_bindings(charter):
    return {
        "charter_sha256": "b" * 64,
        "roi_map_sha256": "c" * 64,
        "trial_manifest_sha256": "d" * 64,
        "fold_manifest_sha256": "e" * 64,
        "source_inventory_sha256": "f" * 64,
        "transform_provenance_sha256": "1" * 64,
        "feature_schema_sha256": feature_schema_sha256(charter),
        "uv_lock_sha256": "2" * 64,
        "package_versions_sha256": "3" * 64,
    }


def feature_table(charter, feature_contract):
    keys = synthetic_keys()
    names = derive_feature_names(charter)
    bindings = expected_bindings(charter)
    rows = [
        {
            "subject_id": subject_id,
            "session": session,
            "trial": trial,
            "item_num": item_num,
            "values": [float(index) for index in range(len(names))],
        }
        for subject_id, session, trial, item_num in keys
    ]
    return {
        "schema_id": feature_contract["logical_table"]["schema_id"],
        "schema_version": feature_contract["logical_table"]["schema_version"],
        "created_at_utc": "2026-08-16T22:30:00Z",
        "condition": "ocular-removed",
        "subject_id": "sub-SYN001",
        "partition_id": "synthetic-partition-only",
        **bindings,
        "expected_key_set_sha256": canonical_trial_key_set_sha256(keys),
        "code_commit": "4" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "synthetic-test-only",
        "python_version": "3.12.synthetic",
        "feature_names": names,
        "rows": rows,
    }


def rewrite_bundle_manifest(destination, **updates):
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


def test_resolved_d26_and_d29_are_bound_and_synthetic_pending_values_fail_closed(
    frozen_charter, feature_contract
):
    d26_pending = deepcopy(frozen_charter)
    d26_pending["features"]["normalization"] = "PENDING_D26"
    d26_pending["pending_human_gates"].insert(0, "D26_normalization_estimand")
    with pytest.raises(FeatureTableValidationError, match="PENDING_D26"):
        feature_schema_sha256(d26_pending)

    d29_pending = deepcopy(frozen_charter)
    d29_pending["features"]["analysis_window_s"] = "PENDING_D29"
    d29_pending["features"]["time_bins_s"] = "PENDING_D29"
    with pytest.raises(FeatureTableValidationError, match="PENDING_D29"):
        feature_schema_sha256(d29_pending)

    assert frozen_charter["features"]["analysis_window_s"] == [0.3, 1.6]
    assert frozen_charter["features"]["time_bins_s"] == [
        [0.3, 0.95],
        [0.95, 1.6],
    ]
    assert feature_schema_sha256(frozen_charter, feature_contract)

    assert feature_contract["real_feature_generation_requires"] == [
        "final_charter_sha256",
        "D26_resolved",
        "D29_resolved",
        "D42_A_omit_balanced_accuracy",
        "D44_C_outer_training_selected_C_qualified_meta_predictions",
        "bounded_preflight_and_capacity_policy_frozen",
        "roi_map_sha256_valid",
        "trial_and_fold_manifests_frozen",
        "validated_Sofia_protocol_teach_back_receipt",
    ]
    assert feature_contract["implementation"]["real_features_generated"] is False


def test_feature_axis_names_are_exact_stable_and_decision_independent(resolved_charter):
    names = derive_feature_names(resolved_charter)

    assert len(names) == 128
    assert len(set(names)) == 128
    assert names[0] == "eeg__time-t0__roi-LAI__freq-f00"
    assert names[7] == "eeg__time-t0__roi-LAI__freq-f07"
    assert names[8] == "eeg__time-t0__roi-LAS__freq-f00"
    assert names[-1] == "eeg__time-t1__roi-RPS__freq-f07"
    assert feature_schema_sha256(resolved_charter) == feature_schema_sha256(
        deepcopy(resolved_charter)
    )


def test_feature_axis_contract_and_charter_are_identical(
    resolved_charter, feature_contract
):
    features = resolved_charter["features"]
    axes = feature_contract["feature_axis_contract"]

    assert axes["time_bin_ids"] == features["time_bin_ids"]
    assert axes["frequency_ids"] == features["frequency_ids"]
    assert axes["feature_order"] == features["feature_order"]
    assert axes["feature_name_template"] == features["feature_name_template"]
    assert axes["expected_feature_count"] == features["expected_feature_count"]
    assert "feature_schema_sha256" in resolved_charter["provenance"]["cache_key_fields"]


@pytest.mark.parametrize("field", ["analysis_window_s", "frequencies_hz"])
def test_feature_schema_digest_changes_with_scientific_or_numeric_schema_drift(
    resolved_charter, field
):
    changed = deepcopy(resolved_charter)
    if field == "analysis_window_s":
        changed["features"][field] = [0.2, 1.6]
        changed["features"]["time_bins_s"][0][0] = 0.2
    else:
        changed["features"][field][0] = 2.0000000001

    assert feature_schema_sha256(changed) != feature_schema_sha256(resolved_charter)


def synthetic_manifest(kind, rows):
    schema_slug = "inner-fold" if kind == "inner_fold" else kind
    return {
        "schema_id": f"neuroai-m1-{schema_slug}-v1",
        "schema_version": "1.0",
        "manifest_kind": kind,
        "created_at_utc": "2026-08-16T22:00:00Z",
        "charter_id": "neuroai-m1-v1",
        "charter_version": "1.0-test",
        "charter_sha256": "b" * 64,
        "roi_map_id": "neuroai-m1-roi-v1",
        "roi_map_sha256": "c" * 64,
        "code_commit": "4" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "synthetic-test-only",
        "source_inventory_sha256": "d" * 64,
        "rows": rows,
    }


def synthetic_normalization_manifests():
    trial_rows = []
    for session, role in ((0, "train"), (1, "train"), (2, "test")):
        for item_num in (1, 2):
            trial_rows.append(
                {
                    "subject_id": "sub-SYN001",
                    "session": session,
                    "trial": item_num,
                    "item_num": item_num,
                    "condition": "ocular-removed",
                    "serial_position": item_num,
                    "list_number": item_num,
                    "onset_s": float(item_num),
                    "recalled": item_num == 2,
                    "source_events_id": f"sub-SYN001_ses-{session}_events.tsv",
                    "source_events_sha256": SHA,
                    "derivative_id": f"sub-SYN001_ses-{session}_ocular-removed_epo.fif",
                    "derivative_sha256": SHA,
                    "epoch_row": item_num - 1,
                    "included": True,
                    "exclusion_reason_code": "INCLUDED",
                    "condition_survivor": True,
                    "outer_role": role,
                }
            )
    keys_by_session = {
        session: [
            canonical_trial_key(row) for row in trial_rows if row["session"] == session
        ]
        for session in (0, 1, 2)
    }
    fold_rows = []
    for fold_id, validation_session, fit_session in (
        ("fold-0", 0, 1),
        ("fold-1", 1, 0),
    ):
        fold_rows.append(
            {
                "subject_id": "sub-SYN001",
                "condition": "ocular-removed",
                "inner_fold_id": fold_id,
                "fit_sessions": [fit_session],
                "validation_session": validation_session,
                "outer_test_session": 2,
                "fit_class_counts": {"0": 1, "1": 1},
                "validation_class_counts": {"0": 1, "1": 1},
                "fit_keys_sha256": canonical_trial_key_set_sha256(
                    keys_by_session[fit_session]
                ),
                "validation_keys_sha256": canonical_trial_key_set_sha256(
                    keys_by_session[validation_session]
                ),
                "eligible": True,
                "feasibility_reason_code": "ELIGIBLE",
            }
        )
    return (
        synthetic_manifest("trial", trial_rows),
        synthetic_manifest("inner_fold", fold_rows),
        keys_by_session,
    )


def normalization_expected_bindings(
    charter, feature_contract, manifest_contract, trial_manifest, fold_manifest
):
    return {
        "charter_sha256": "b" * 64,
        "roi_map_sha256": "c" * 64,
        "trial_manifest_sha256": manifest_sha256(trial_manifest, manifest_contract),
        "inner_fold_manifest_sha256": manifest_sha256(fold_manifest, manifest_contract),
        "feature_schema_sha256": feature_schema_sha256(charter, feature_contract),
        "normalization_contract_sha256": normalization_contract_sha256(charter),
        "uv_lock_sha256": "e" * 64,
        "package_versions_sha256": "f" * 64,
    }


def normalization_transform_provenance(charter, feature_contract, manifest_contract):
    trial_manifest, fold_manifest, keys = synthetic_normalization_manifests()
    bindings = normalization_expected_bindings(
        charter, feature_contract, manifest_contract, trial_manifest, fold_manifest
    )
    records = []
    for fold_id, fit_session, apply_session in (
        ("fold-0", 1, 0),
        ("fold-1", 0, 1),
    ):
        invented_log_power = np.arange(2 * 40 * 8 * 5, dtype=np.float64).reshape(
            2, 40, 8, 5
        )
        statistic_evidence = normalization_statistics_evidence(
            fit_strict_inductive_normalizer(invented_log_power + fit_session)
        )
        records.append(
            {
                "transform_id": f"inner__{fold_id}",
                "stage": "inner_model_selection",
                "inner_fold_id": fold_id,
                "fit_sessions": [fit_session],
                "apply_sessions": [apply_session],
                "fit_keys_sha256": canonical_trial_key_set_sha256(keys[fit_session]),
                "apply_keys_sha256": canonical_trial_key_set_sha256(keys[apply_session]),
                "n_fit_trials": 2,
                "n_apply_trials": 2,
                "statistic_axes": ["trial", "time"],
                "preserved_axes": ["channel", "frequency"],
                "ddof": 0,
                "zero_standard_deviation_policy": "replace_with_1_then_center",
                "label_access": False,
                "held_session_distribution_access": False,
                **statistic_evidence,
            }
        )
    outer_invented_log_power = np.arange(4 * 40 * 8 * 5, dtype=np.float64).reshape(
        4, 40, 8, 5
    )
    outer_statistic_evidence = normalization_statistics_evidence(
        fit_strict_inductive_normalizer(outer_invented_log_power)
    )
    records.append(
        {
            "transform_id": "outer__refit",
            "stage": "outer_refit",
            "inner_fold_id": None,
            "fit_sessions": [0, 1],
            "apply_sessions": [2],
            "fit_keys_sha256": canonical_trial_key_set_sha256(keys[0] + keys[1]),
            "apply_keys_sha256": canonical_trial_key_set_sha256(keys[2]),
            "n_fit_trials": 4,
            "n_apply_trials": 2,
            "statistic_axes": ["trial", "time"],
            "preserved_axes": ["channel", "frequency"],
            "ddof": 0,
            "zero_standard_deviation_policy": "replace_with_1_then_center",
            "label_access": False,
            "held_session_distribution_access": False,
            **outer_statistic_evidence,
        }
    )
    channel_order = [
        channel
        for roi_id in charter["features"]["roi_order"]
        for channel in charter["features"]["roi_channels"][roi_id]
    ]
    provenance = {
        "schema_id": feature_contract["normalization_transform_provenance"]["schema_id"],
        "schema_version": feature_contract["normalization_transform_provenance"][
            "schema_version"
        ],
        "created_at_utc": "2026-08-16T22:15:00Z",
        "subject_id": "sub-SYN001",
        "condition": "ocular-removed",
        "outer_test_session": 2,
        **bindings,
        "code_commit": "4" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "synthetic-normalization-test-only",
        "python_version": "3.12.synthetic",
        "statistic_dtype": "float64",
        "statistic_channel_order": channel_order,
        "statistic_frequency_ids": charter["features"]["frequency_ids"],
        "statistic_shape": [40, 8],
        "records": records,
    }
    return provenance, trial_manifest, fold_manifest, bindings


def test_normalization_contract_binds_strict_inductive_d26_and_no_second_scaler(
    frozen_charter,
):
    digest = normalization_contract_sha256(frozen_charter)
    assert len(digest) == 64
    assert frozen_charter["features"]["normalization"]["decision"] == (
        "D26_A_strict_inductive"
    )
    assert frozen_charter["features"]["normalization"][
        "held_session_distribution_access"
    ] is False
    assert frozen_charter["learners"]["M1"]["extra_scaler"] is False

    transductive = deepcopy(frozen_charter)
    transductive["features"]["normalization"]["held_session_distribution_access"] = True
    with pytest.raises(FeatureTableValidationError, match="held-session distribution"):
        normalization_contract_sha256(transductive)

    second_scaler = deepcopy(frozen_charter)
    second_scaler["learners"]["M1"]["extra_scaler"] = True
    with pytest.raises(FeatureTableValidationError, match="extra scaler"):
        normalization_contract_sha256(second_scaler)

    altered_estimator = deepcopy(frozen_charter)
    altered_estimator["features"]["normalization"]["ddof"] = 1
    with pytest.raises(FeatureTableValidationError, match="exact strict-inductive"):
        normalization_contract_sha256(altered_estimator)


def test_normalization_transform_provenance_binds_exact_inner_and_outer_key_scopes(
    resolved_charter, feature_contract, manifest_contract
):
    provenance, trials, folds, bindings = normalization_transform_provenance(
        resolved_charter, feature_contract, manifest_contract
    )
    validate_normalization_transform_provenance(
        provenance,
        resolved_charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        expected_bindings=bindings,
    )
    assert normalization_transform_provenance_sha256(
        provenance,
        resolved_charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        expected_bindings=bindings,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["records"][0].update(fit_sessions=[0]), "fit sessions"),
        (lambda value: value["records"][0].update(apply_sessions=[2]), "apply sessions"),
        (
            lambda value: value["records"][-1].update(fit_sessions=[0, 1, 2]),
            "fit sessions",
        ),
        (
            lambda value: value["records"][-1].update(
                held_session_distribution_access=True
            ),
            "held-session distribution",
        ),
        (lambda value: value["records"][0].update(label_access=True), "label access"),
        (lambda value: value["records"][0].update(fit_keys_sha256="9" * 64), "fit keys"),
        (
            lambda value: value["records"][0].update(n_zero_std_cells=321),
            "zero-standard-deviation count",
        ),
        (
            lambda value: value["records"][0].update(n_fit_timepoints=0),
            "fit-timepoint count",
        ),
        (lambda value: value.update(statistic_shape=[8, 40]), "statistic shape"),
    ],
)
def test_normalization_transform_provenance_rejects_leakage_and_axis_drift(
    resolved_charter, feature_contract, manifest_contract, mutation, message
):
    provenance, trials, folds, bindings = normalization_transform_provenance(
        resolved_charter, feature_contract, manifest_contract
    )
    mutation(provenance)
    with pytest.raises(FeatureTableValidationError, match=message):
        validate_normalization_transform_provenance(
            provenance,
            resolved_charter,
            feature_contract,
            manifest_contract,
            trial_manifest=trials,
            inner_fold_manifest=folds,
            expected_bindings=bindings,
        )


def test_normalization_transform_provenance_is_canonical_and_binding_tamper_fails(
    resolved_charter, feature_contract, manifest_contract
):
    provenance, trials, folds, bindings = normalization_transform_provenance(
        resolved_charter, feature_contract, manifest_contract
    )
    reversed_records = deepcopy(provenance)
    reversed_records["records"].reverse()
    original_bytes = canonical_normalization_transform_provenance_bytes(
        provenance,
        resolved_charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        expected_bindings=bindings,
    )
    assert original_bytes == canonical_normalization_transform_provenance_bytes(
        reversed_records,
        resolved_charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        expected_bindings=bindings,
    )
    with pytest.raises(FeatureTableValidationError, match="canonical record order"):
        validate_normalization_transform_provenance(
            reversed_records,
            resolved_charter,
            feature_contract,
            manifest_contract,
            trial_manifest=trials,
            inner_fold_manifest=folds,
            expected_bindings=bindings,
        )

    tampered = deepcopy(provenance)
    tampered["normalization_contract_sha256"] = "9" * 64
    with pytest.raises(FeatureTableValidationError, match="binding mismatch"):
        validate_normalization_transform_provenance(
            tampered,
            resolved_charter,
            feature_contract,
            manifest_contract,
            trial_manifest=trials,
            inner_fold_manifest=folds,
            expected_bindings=bindings,
        )


def test_valid_feature_table_has_exact_keys_values_and_provenance(
    resolved_charter, feature_contract
):
    table = feature_table(resolved_charter, feature_contract)
    bindings = expected_bindings(resolved_charter)

    validate_feature_table(
        table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    )
    assert feature_table_sha256(
        table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    )


def test_feature_table_canonical_bytes_sort_rows_but_stored_validation_requires_order(
    resolved_charter, feature_contract
):
    table = feature_table(resolved_charter, feature_contract)
    reversed_table = deepcopy(table)
    reversed_table["rows"].reverse()
    bindings = expected_bindings(resolved_charter)

    assert canonical_feature_table_bytes(
        table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    ) == canonical_feature_table_bytes(
        reversed_table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    )
    with pytest.raises(FeatureTableValidationError, match="canonical row order"):
        validate_feature_table(
            reversed_table,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "label"])
def test_feature_table_rejects_key_drift_duplicates_and_outcome_fields(
    resolved_charter, feature_contract, mutation
):
    table = feature_table(resolved_charter, feature_contract)
    if mutation == "missing":
        table["rows"].pop()
    elif mutation == "extra":
        extra = deepcopy(table["rows"][-1])
        extra["trial"] = 3
        extra["item_num"] = 3
        table["rows"].append(extra)
    elif mutation == "duplicate":
        table["rows"].append(deepcopy(table["rows"][-1]))
    else:
        table["rows"][0]["recalled"] = True

    with pytest.raises(FeatureTableValidationError):
        validate_feature_table(
            table,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=expected_bindings(resolved_charter),
        )


@pytest.mark.parametrize("mutation", ["integer", "nonfinite", "short", "axis_order"])
def test_feature_table_rejects_non_float64_nonfinite_and_axis_drift(
    resolved_charter, feature_contract, mutation
):
    table = feature_table(resolved_charter, feature_contract)
    if mutation == "integer":
        table["rows"][0]["values"][0] = 0
    elif mutation == "nonfinite":
        table["rows"][0]["values"][0] = float("nan")
    elif mutation == "short":
        table["rows"][0]["values"].pop()
    else:
        table["feature_names"][0], table["feature_names"][1] = (
            table["feature_names"][1],
            table["feature_names"][0],
        )

    with pytest.raises(FeatureTableValidationError):
        validate_feature_table(
            table,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=expected_bindings(resolved_charter),
        )


def test_feature_table_rejects_binding_subject_condition_and_key_hash_drift(
    resolved_charter, feature_contract
):
    original = feature_table(resolved_charter, feature_contract)
    mutations = []

    binding = deepcopy(original)
    binding["trial_manifest_sha256"] = "9" * 64
    mutations.append(binding)

    subject = deepcopy(original)
    subject["rows"][0]["subject_id"] = "sub-SYN999"
    mutations.append(subject)

    condition = deepcopy(original)
    condition["condition"] = "not-a-condition"
    mutations.append(condition)

    key_hash = deepcopy(original)
    key_hash["expected_key_set_sha256"] = "8" * 64
    mutations.append(key_hash)

    for table in mutations:
        with pytest.raises(FeatureTableValidationError):
            validate_feature_table(
                table,
                resolved_charter,
                feature_contract,
                expected_keys=synthetic_keys(),
                expected_bindings=expected_bindings(resolved_charter),
            )


def test_feature_bundle_contract_is_typed_atomic_and_decision_independent(feature_contract):
    bundle = feature_contract["storage_bundle"]

    assert feature_contract["version"] == "1.5-draft"
    assert bundle["atomic_publication"] == (
        "temporary_directory_fsync_then_atomic_noreplace_rename"
    )
    assert bundle["exact_files"] == [
        "metadata.json",
        "trial-keys.json",
        "values.npy",
        "bundle.json",
        "bundle.sha256",
    ]
    assert bundle["values"]["format"] == "npy"
    assert bundle["values"]["dtype"] == "little_endian_float64"
    assert bundle["values"]["allow_pickle"] is False
    assert bundle["values"]["shape"] == "n_rows_by_128"
    assert bundle["outcome_fields_allowed"] is False
    assert bundle["real_publication_allowed_before_final_readiness"] is False


def test_feature_bundle_round_trip_is_exact_typed_and_immutable(
    tmp_path, resolved_charter, feature_contract
):
    table = feature_table(resolved_charter, feature_contract)
    bindings = expected_bindings(resolved_charter)
    destination = tmp_path / "feature-bundle"

    digest = write_feature_table_bundle_atomic(
        destination,
        table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    )

    assert {path.name for path in destination.iterdir()} == set(
        feature_contract["storage_bundle"]["exact_files"]
    )
    with (destination / "values.npy").open("rb") as handle:
        values = np.load(handle, allow_pickle=False)
        assert handle.read(1) == b""
    assert values.dtype.str == "<f8"
    assert values.flags.c_contiguous
    assert values.shape == (2, 128)

    loaded = load_feature_table_bundle(
        destination,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
        expected_bundle_sha256=digest,
    )
    assert loaded == table

    with pytest.raises(FeatureTableValidationError, match="already exists"):
        write_feature_table_bundle_atomic(
            destination,
            table,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
        )


def test_feature_bundle_readback_rejects_tamper_extra_files_and_wrong_external_digest(
    tmp_path, resolved_charter, feature_contract
):
    table = feature_table(resolved_charter, feature_contract)
    bindings = expected_bindings(resolved_charter)

    def publish(name):
        destination = tmp_path / name
        digest = write_feature_table_bundle_atomic(
            destination,
            table,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
        )
        return destination, digest

    destination, digest = publish("wrong-external")
    with pytest.raises(FeatureTableValidationError, match="external bundle digest"):
        load_feature_table_bundle(
            destination,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
            expected_bundle_sha256="0" * 64,
        )

    destination, digest = publish("tampered-values")
    values_path = destination / "values.npy"
    payload = bytearray(values_path.read_bytes())
    payload[-1] ^= 1
    values_path.write_bytes(payload)
    with pytest.raises(FeatureTableValidationError, match="values.npy digest"):
        load_feature_table_bundle(
            destination,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )

    destination, digest = publish("extra-file")
    (destination / "unexpected.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(FeatureTableValidationError, match="exact files"):
        load_feature_table_bundle(
            destination,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def test_feature_bundle_rejects_contract_drift(tmp_path, resolved_charter, feature_contract):
    changed = deepcopy(feature_contract)
    changed["storage_bundle"]["values"]["allow_pickle"] = True

    with pytest.raises(FeatureTableValidationError, match="bundle contract drifted"):
        write_feature_table_bundle_atomic(
            tmp_path / "feature-bundle",
            feature_table(resolved_charter, feature_contract),
            resolved_charter,
            changed,
            expected_keys=synthetic_keys(),
            expected_bindings=expected_bindings(resolved_charter),
        )


@pytest.mark.parametrize("unsafe_kind", ["object_array", "trailing_bytes"])
def test_feature_bundle_rejects_unsafe_npy_even_when_all_hashes_are_rewritten(
    tmp_path, resolved_charter, feature_contract, unsafe_kind
):
    table = feature_table(resolved_charter, feature_contract)
    bindings = expected_bindings(resolved_charter)
    destination = tmp_path / unsafe_kind
    write_feature_table_bundle_atomic(
        destination,
        table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    )
    values_path = destination / "values.npy"
    if unsafe_kind == "object_array":
        with values_path.open("wb") as handle:
            np.save(handle, np.array([[object()]], dtype=object), allow_pickle=True)
    else:
        with values_path.open("ab") as handle:
            handle.write(b"unexpected trailing byte")
    digest = rewrite_bundle_manifest(
        destination,
        values_npy_sha256=sha256(values_path.read_bytes()).hexdigest(),
    )

    message = "safe NumPy array" if unsafe_kind == "object_array" else "trailing bytes"
    with pytest.raises(FeatureTableValidationError, match=message):
        load_feature_table_bundle(
            destination,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )


def test_feature_bundle_revalidates_trial_keys_after_consistent_member_rehash(
    tmp_path, resolved_charter, feature_contract
):
    table = feature_table(resolved_charter, feature_contract)
    bindings = expected_bindings(resolved_charter)
    destination = tmp_path / "key-tamper"
    write_feature_table_bundle_atomic(
        destination,
        table,
        resolved_charter,
        feature_contract,
        expected_keys=synthetic_keys(),
        expected_bindings=bindings,
    )
    keys_path = destination / "trial-keys.json"
    keys = json.loads(keys_path.read_text(encoding="utf-8"))
    keys["rows"][0]["item_num"] = 99
    keys_payload = (
        json.dumps(
            keys,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    keys_path.write_bytes(keys_payload)
    digest = rewrite_bundle_manifest(
        destination,
        trial_keys_sha256=sha256(keys_payload).hexdigest(),
    )

    with pytest.raises(FeatureTableValidationError, match="feature table rows"):
        load_feature_table_bundle(
            destination,
            resolved_charter,
            feature_contract,
            expected_keys=synthetic_keys(),
            expected_bindings=bindings,
            expected_bundle_sha256=digest,
        )
