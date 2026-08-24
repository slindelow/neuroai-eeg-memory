"""Build validated M1 feature and prediction artifacts from frozen model outputs.

This module only assembles logical artifacts.  It never selects a cohort, changes a
scientific contract, or evaluates held-session predictions.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

import numpy as np

from peers.m1_feature_generation import (
    SessionSpectralSummary,
    combine_training_session_statistics,
    compact_features_for_sessions,
)
from peers.m1_features import (
    derive_feature_names,
    feature_schema_sha256,
    feature_table_sha256,
    normalization_contract_sha256,
    normalization_statistics_evidence,
    normalization_transform_provenance_sha256,
    validate_feature_table,
    validate_normalization_transform_provenance,
)
from peers.m1_manifests import (
    D44_C_DECISION,
    D44_C_META_PREDICTION_CLAIM,
    D44_C_SELECTION_SCOPE,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    d44_c_cross_fit_provenance_sha256,
    manifest_sha256,
    validate_d44_c_cross_fit_provenance,
    validate_manifest,
)
from peers.m1_model_execution import ModelExecutionResult
from peers.m1_predictions import (
    learner_contract_sha256,
    model_provenance_sha256,
    validate_prediction_table,
)


class ArtifactExecutionError(ValueError):
    """A real artifact could not be assembled under the frozen contracts."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    return sha256(payload).hexdigest()


def slice_subject_manifest(
    manifest: Mapping[str, Any],
    subject_id: str,
    manifest_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one authenticated subject manifest without changing its source envelope."""
    validate_manifest(dict(manifest), dict(manifest_contract))
    sliced = deepcopy(dict(manifest))
    sliced["rows"] = [
        deepcopy(row) for row in manifest["rows"] if row["subject_id"] == subject_id
    ]
    if not sliced["rows"]:
        raise ArtifactExecutionError(f"manifest has no rows for {subject_id}")
    validate_manifest(sliced, dict(manifest_contract))
    return sliced


def build_subject_manifest_lineage(
    manifests: Mapping[str, Mapping[str, Any]],
    subject_id: str,
    manifest_contract: Mapping[str, Any],
    *,
    validated_parent_sha256s: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Slice every execution manifest and authenticate it against its frozen parent."""
    required = (
        "trial-ocular-removed",
        "inner-fold-ocular-removed",
        "trial-ocular-kept",
        "inner-fold-ocular-kept",
        "cross-condition-intersection",
    )
    missing = sorted(set(required) - set(manifests))
    if missing:
        raise ArtifactExecutionError(f"subject lineage lacks manifests: {missing}")
    if validated_parent_sha256s is None:
        parent_sha256s = validate_lineage_parent_manifests(
            manifests, manifest_contract
        )
    else:
        parent_sha256s = dict(validated_parent_sha256s)
        if set(parent_sha256s) != set(required):
            raise ArtifactExecutionError("validated parent digest set mismatch")
    slices = {}
    for name in required:
        sliced = deepcopy(dict(manifests[name]))
        sliced["rows"] = [
            deepcopy(row)
            for row in manifests[name]["rows"]
            if row["subject_id"] == subject_id
        ]
        if not sliced["rows"]:
            raise ArtifactExecutionError(f"manifest has no rows for {subject_id}")
        validate_manifest(sliced, dict(manifest_contract))
        slices[name] = sliced
    slice_sha256s = {
        name: manifest_sha256(value, dict(manifest_contract))
        for name, value in slices.items()
    }
    lineage = {
        "schema_id": "neuroai-m1-subject-manifest-lineage-v1",
        "schema_version": "1.0",
        "subject_id": subject_id,
        "slice_rule": "retain_exact_parent_rows_where_subject_id_matches",
        "parent_manifest_sha256s": parent_sha256s,
        "subject_manifest_sha256s": slice_sha256s,
    }
    return slices, lineage


def validate_lineage_parent_manifests(
    manifests: Mapping[str, Mapping[str, Any]],
    manifest_contract: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the five frozen execution parents once and return their digests."""
    required = (
        "trial-ocular-removed",
        "inner-fold-ocular-removed",
        "trial-ocular-kept",
        "inner-fold-ocular-kept",
        "cross-condition-intersection",
    )
    missing = sorted(set(required) - set(manifests))
    if missing:
        raise ArtifactExecutionError(f"subject lineage lacks manifests: {missing}")
    digests = {}
    for name in required:
        validate_manifest(dict(manifests[name]), dict(manifest_contract))
        digests[name] = manifest_sha256(
            dict(manifests[name]), dict(manifest_contract)
        )
    return digests


def build_all_subject_manifest_lineages(
    manifests: Mapping[str, Mapping[str, Any]],
    subject_ids: Sequence[str],
    manifest_contract: Mapping[str, Any],
) -> dict[str, tuple[dict[str, dict[str, Any]], dict[str, Any]]]:
    """Validate frozen parents once, then derive every requested subject lineage."""
    required = (
        "trial-ocular-removed",
        "inner-fold-ocular-removed",
        "trial-ocular-kept",
        "inner-fold-ocular-kept",
        "cross-condition-intersection",
    )
    missing = sorted(set(required) - set(manifests))
    if missing:
        raise ArtifactExecutionError(f"subject lineage lacks manifests: {missing}")
    parent_sha256s = validate_lineage_parent_manifests(
        manifests, manifest_contract
    )
    ordered_subjects = list(subject_ids)
    if ordered_subjects != sorted(ordered_subjects) or len(ordered_subjects) != len(
        set(ordered_subjects)
    ):
        raise ArtifactExecutionError("subject lineage ids must be unique and sorted")
    outputs = {}
    for subject_id in ordered_subjects:
        outputs[subject_id] = build_subject_manifest_lineage(
            manifests,
            subject_id,
            manifest_contract,
            validated_parent_sha256s=parent_sha256s,
        )
    return outputs


def _key_rows(
    trial_manifest: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train = [
        row
        for row in trial_manifest["rows"]
        if row["condition_survivor"] and row["outer_role"] == "train"
    ]
    test = [
        row
        for row in trial_manifest["rows"]
        if row["condition_survivor"] and row["outer_role"] == "test"
    ]
    if not train or not test:
        raise ArtifactExecutionError("subject trial manifest lacks outer train or test rows")
    return train, test


def build_normalization_provenance(
    summaries: Sequence[SessionSpectralSummary],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    created_at_utc: str,
    charter_sha256: str,
    roi_map_sha256: str,
    uv_lock_sha256: str,
    package_versions_sha256: str,
    code_commit: str,
    generator_command: str,
    python_version: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build all inner and outer D26-A transform records from raw session summaries."""
    validate_manifest(dict(trial_manifest), dict(manifest_contract))
    validate_manifest(dict(inner_fold_manifest), dict(manifest_contract))
    train_rows, test_rows = _key_rows(trial_manifest)
    groups = {(row["subject_id"], row["condition"]) for row in trial_manifest["rows"]}
    if len(groups) != 1:
        raise ArtifactExecutionError("normalization provenance requires one subject-condition")
    subject_id, condition = next(iter(groups))
    outer_test_session = next(iter({int(row["session"]) for row in test_rows}))
    bindings = {
        "charter_sha256": charter_sha256,
        "roi_map_sha256": roi_map_sha256,
        "trial_manifest_sha256": manifest_sha256(
            dict(trial_manifest), dict(manifest_contract)
        ),
        "inner_fold_manifest_sha256": manifest_sha256(
            dict(inner_fold_manifest), dict(manifest_contract)
        ),
        "feature_schema_sha256": feature_schema_sha256(charter, feature_contract),
        "normalization_contract_sha256": normalization_contract_sha256(
            charter, feature_contract
        ),
        "uv_lock_sha256": uv_lock_sha256,
        "package_versions_sha256": package_versions_sha256,
    }
    records: list[dict[str, Any]] = []
    folds = sorted(
        [row for row in inner_fold_manifest["rows"] if row["eligible"]],
        key=lambda row: row["inner_fold_id"],
    )
    for fold in folds:
        fit_sessions = list(fold["fit_sessions"])
        apply_sessions = [int(fold["validation_session"])]
        statistics = combine_training_session_statistics(
            summaries, fit_sessions=fit_sessions, charter=charter
        )
        fit_keys, _ = compact_features_for_sessions(
            summaries, statistics, apply_sessions=fit_sessions, charter=charter
        )
        apply_keys, _ = compact_features_for_sessions(
            summaries, statistics, apply_sessions=apply_sessions, charter=charter
        )
        records.append(
            {
                "transform_id": f"inner__{fold['inner_fold_id']}",
                "stage": "inner_model_selection",
                "inner_fold_id": fold["inner_fold_id"],
                "fit_sessions": fit_sessions,
                "apply_sessions": apply_sessions,
                "fit_keys_sha256": canonical_trial_key_set_sha256(fit_keys),
                "apply_keys_sha256": canonical_trial_key_set_sha256(apply_keys),
                "n_fit_trials": len(fit_keys),
                "n_apply_trials": len(apply_keys),
                "statistic_axes": ["trial", "time"],
                "preserved_axes": ["channel", "frequency"],
                "ddof": 0,
                "zero_standard_deviation_policy": "replace_with_1_then_center",
                "label_access": False,
                "held_session_distribution_access": False,
                **normalization_statistics_evidence(statistics),
            }
        )
    outer_fit_sessions = sorted({int(row["session"]) for row in train_rows})
    outer_statistics = combine_training_session_statistics(
        summaries, fit_sessions=outer_fit_sessions, charter=charter
    )
    outer_fit_keys, _ = compact_features_for_sessions(
        summaries,
        outer_statistics,
        apply_sessions=outer_fit_sessions,
        charter=charter,
    )
    outer_apply_keys, _ = compact_features_for_sessions(
        summaries,
        outer_statistics,
        apply_sessions=[outer_test_session],
        charter=charter,
    )
    records.append(
        {
            "transform_id": "outer__refit",
            "stage": "outer_refit",
            "inner_fold_id": None,
            "fit_sessions": outer_fit_sessions,
            "apply_sessions": [outer_test_session],
            "fit_keys_sha256": canonical_trial_key_set_sha256(outer_fit_keys),
            "apply_keys_sha256": canonical_trial_key_set_sha256(outer_apply_keys),
            "n_fit_trials": len(outer_fit_keys),
            "n_apply_trials": len(outer_apply_keys),
            "statistic_axes": ["trial", "time"],
            "preserved_axes": ["channel", "frequency"],
            "ddof": 0,
            "zero_standard_deviation_policy": "replace_with_1_then_center",
            "label_access": False,
            "held_session_distribution_access": False,
            **normalization_statistics_evidence(outer_statistics),
        }
    )
    provenance = {
        "schema_id": feature_contract["normalization_transform_provenance"]["schema_id"],
        "schema_version": feature_contract["normalization_transform_provenance"][
            "schema_version"
        ],
        "created_at_utc": created_at_utc,
        "subject_id": subject_id,
        "condition": condition,
        "outer_test_session": outer_test_session,
        **bindings,
        "code_commit": code_commit,
        "dirty_diff_sha256": None,
        "generator_command": generator_command,
        "python_version": python_version,
        "statistic_dtype": "float64",
        "statistic_channel_order": [
            channel
            for roi in charter["features"]["roi_order"]
            for channel in charter["features"]["roi_channels"][roi]
        ],
        "statistic_frequency_ids": list(charter["features"]["frequency_ids"]),
        "statistic_shape": [40, 8],
        "records": records,
    }
    validate_normalization_transform_provenance(
        provenance,
        charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        expected_bindings=bindings,
    )
    return provenance, bindings


def build_feature_tables(
    summaries: Sequence[SessionSpectralSummary],
    normalization_provenance: Mapping[str, Any],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    source_inventory_sha256: str,
    created_at_utc: str,
    generator_command: str,
    python_version: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Build one complete feature table for every inner and outer transform.

    Inner tables contain all outer-training rows (fit plus validation sessions) under
    that fold's fitted transform.  The outer table contains outer-training plus held
    rows under the outer-refit transform.  The prediction table binds the outer table.
    """
    train_rows, test_rows = _key_rows(trial_manifest)
    outer_test_session = next(iter({int(row["session"]) for row in test_rows}))
    provenance_bindings = {
        field: normalization_provenance[field]
        for field in feature_contract["normalization_transform_provenance"][
            "external_binding_fields"
        ]
    }
    provenance_digest = normalization_transform_provenance_sha256(
        normalization_provenance,
        charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        expected_bindings=provenance_bindings,
    )
    common_bindings = {
        "charter_sha256": normalization_provenance["charter_sha256"],
        "roi_map_sha256": normalization_provenance["roi_map_sha256"],
        "trial_manifest_sha256": normalization_provenance["trial_manifest_sha256"],
        "fold_manifest_sha256": normalization_provenance[
            "inner_fold_manifest_sha256"
        ],
        "source_inventory_sha256": source_inventory_sha256,
        "transform_provenance_sha256": provenance_digest,
        "feature_schema_sha256": normalization_provenance["feature_schema_sha256"],
        "uv_lock_sha256": normalization_provenance["uv_lock_sha256"],
        "package_versions_sha256": normalization_provenance[
            "package_versions_sha256"
        ],
    }
    subject_id = normalization_provenance["subject_id"]
    condition = normalization_provenance["condition"]
    tables: dict[str, dict[str, Any]] = {}
    folds = sorted(
        [row for row in inner_fold_manifest["rows"] if row["eligible"]],
        key=lambda row: row["inner_fold_id"],
    )
    specifications = [
        (
            f"inner__{fold['inner_fold_id']}",
            list(fold["fit_sessions"]),
            sorted({int(row["session"]) for row in train_rows}),
        )
        for fold in folds
    ]
    outer_fit_sessions = sorted({int(row["session"]) for row in train_rows})
    specifications.append(
        (
            "outer__refit",
            outer_fit_sessions,
            outer_fit_sessions + [outer_test_session],
        )
    )
    for partition_id, fit_sessions, apply_sessions in specifications:
        statistics = combine_training_session_statistics(
            summaries, fit_sessions=fit_sessions, charter=charter
        )
        keys, values = compact_features_for_sessions(
            summaries,
            statistics,
            apply_sessions=apply_sessions,
            charter=charter,
        )
        expected_rows = (
            train_rows
            if partition_id.startswith("inner__")
            else train_rows + test_rows
        )
        expected_keys = sorted(canonical_trial_key(row) for row in expected_rows)
        if list(keys) != expected_keys:
            raise ArtifactExecutionError(
                f"{partition_id} feature rows do not match its frozen manifest scope"
            )
        table = {
            "schema_id": feature_contract["logical_table"]["schema_id"],
            "schema_version": feature_contract["logical_table"]["schema_version"],
            "created_at_utc": created_at_utc,
            "condition": condition,
            "subject_id": subject_id,
            "partition_id": partition_id,
            **common_bindings,
            "expected_key_set_sha256": canonical_trial_key_set_sha256(keys),
            "code_commit": normalization_provenance["code_commit"],
            "dirty_diff_sha256": None,
            "generator_command": generator_command,
            "python_version": python_version,
            "feature_names": derive_feature_names(charter),
            "rows": [
                {
                    "subject_id": key[0],
                    "session": key[1],
                    "trial": key[2],
                    "item_num": key[3],
                    "values": [float(value) for value in values[index]],
                }
                for index, key in enumerate(keys)
            ],
        }
        validate_feature_table(
            table,
            charter,
            feature_contract,
            expected_keys=expected_keys,
            expected_bindings=common_bindings,
        )
        tables[partition_id] = table
    return tables, common_bindings


def build_d44_c_provenance(
    model_result: ModelExecutionResult,
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build exact qualified session-out provenance for the M3 meta-training rows."""
    train_rows, test_rows = _key_rows(trial_manifest)
    groups = {(row["subject_id"], row["condition"]) for row in trial_manifest["rows"]}
    if len(groups) != 1:
        raise ArtifactExecutionError("D44-C provenance requires one subject-condition")
    subject_id, condition = next(iter(groups))
    outer_train_sessions = sorted({int(row["session"]) for row in train_rows})
    outer_test_session = next(iter({int(row["session"]) for row in test_rows}))
    folds = sorted(
        [row for row in inner_fold_manifest["rows"] if row["eligible"]],
        key=lambda row: row["inner_fold_id"],
    )
    by_model: dict[str, dict[str, dict[str, list[tuple[str, int, int, int]]]]] = {
        "B3": {},
        "M1": {},
    }
    for fold in folds:
        fit_keys = [
            canonical_trial_key(row)
            for row in train_rows
            if row["session"] in fold["fit_sessions"]
        ]
        predicted_keys = [
            canonical_trial_key(row)
            for row in train_rows
            if row["session"] == fold["validation_session"]
        ]
        for model_id in ("B3", "M1"):
            by_model[model_id][fold["inner_fold_id"]] = {
                "fit_keys": fit_keys,
                "predicted_keys": predicted_keys,
            }
    train_keys = [canonical_trial_key(row) for row in train_rows]
    record = {
        "schema_id": "neuroai-m1-d44-c-cross-fit-provenance-v1",
        "schema_version": "1.0",
        "subject_id": subject_id,
        "condition": condition,
        "outer_test_session": outer_test_session,
        "decision": D44_C_DECISION,
        "meta_prediction_claim": D44_C_META_PREDICTION_CLAIM,
        "fully_session_cross_fitted": False,
        "fold_keys_by_model": by_model,
        "m1_hyperparameter_selection": {
            "scope": D44_C_SELECTION_SCOPE,
            "candidate_C_grid": list(charter["learners"]["M1"]["C_grid"]),
            "selection_metric": charter["splits"]["selection_metric"],
            "tie_break": charter["splits"]["tie_break"],
            "selected_C": float(model_result.selected_m1_c),
            "fit_sessions": outer_train_sessions,
            "validation_sessions": outer_train_sessions,
            "fit_keys_sha256": canonical_trial_key_set_sha256(train_keys),
            "validation_keys_sha256": canonical_trial_key_set_sha256(train_keys),
            "inner_fold_manifest_sha256": manifest_sha256(
                dict(inner_fold_manifest), dict(manifest_contract)
            ),
            "predicted_sessions_in_selection_validation": outer_train_sessions,
            "predicted_session_labels_may_influence_selected_C": True,
            "outer_test_session_used_for_selection": False,
            "one_selected_C_shared_by_all_meta_folds": True,
        },
    }
    validate_d44_c_cross_fit_provenance(
        dict(trial_manifest),
        dict(inner_fold_manifest),
        record,
        charter,
        dict(manifest_contract),
    )
    return record, d44_c_cross_fit_provenance_sha256(
        dict(trial_manifest),
        dict(inner_fold_manifest),
        record,
        charter,
        dict(manifest_contract),
    )


def build_prediction_table(
    model_result: ModelExecutionResult,
    feature_table: Mapping[str, Any],
    feature_bindings: Mapping[str, str],
    d44_provenance: Mapping[str, Any],
    *,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    created_at_utc: str,
    generator_command: str,
    python_version: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Assemble the canonical held-session B3/M1/M3 table without evaluating it."""
    train_rows, test_rows = _key_rows(trial_manifest)
    test_by_key = {canonical_trial_key(row): row for row in test_rows}
    if tuple(model_result.test_keys) != tuple(sorted(test_by_key)):
        raise ArtifactExecutionError("model result keys do not match outer-test manifest")
    train_keys = [canonical_trial_key(row) for row in train_rows]
    fit_keys_sha256 = canonical_trial_key_set_sha256(train_keys)
    prediction_keys_sha256 = canonical_trial_key_set_sha256(model_result.test_keys)
    fit_sessions = sorted({int(row["session"]) for row in train_rows})
    cross_fit_digest = d44_c_cross_fit_provenance_sha256(
        dict(trial_manifest),
        dict(inner_fold_manifest),
        d44_provenance,
        charter,
        dict(manifest_contract),
    )
    normalization_digest = feature_table["transform_provenance_sha256"]
    model_provenance: dict[str, dict[str, Any]] = {}
    for model_id in ("B3", "M1", "M3"):
        transform_chain = model_result.transform_chain_sha256[model_id]
        if model_id == "M1":
            transform_chain = _canonical_sha256(
                {
                    "base_transform_chain_sha256": transform_chain,
                    "normalization_transform_provenance_sha256": normalization_digest,
                }
            )
        model_provenance[model_id] = {
            "model_id": model_id,
            "fit_role": (
                "qualified_session_out_meta_training"
                if model_id == "M3"
                else "outer_training_refit"
            ),
            "fit_sessions": fit_sessions,
            "fit_keys_sha256": fit_keys_sha256,
            "prediction_keys_sha256": prediction_keys_sha256,
            "learner_contract_sha256": learner_contract_sha256(charter, model_id),
            "selected_hyperparameters_sha256": model_result.selected_hyperparameters_sha256[
                model_id
            ],
            "transform_chain_sha256": transform_chain,
            "transform_fit_keys_sha256": fit_keys_sha256,
            "model_artifact_sha256": model_result.model_artifact_sha256[model_id],
            "upstream_cross_fit_sha256": (
                cross_fit_digest if model_id == "M3" else None
            ),
            "seed": int(charter["determinism"]["global_seed"]),
        }
    provenance_digest = model_provenance_sha256(
        model_provenance, prediction_contract
    )
    feature_keys = [
        (row["subject_id"], row["session"], row["trial"], row["item_num"])
        for row in feature_table["rows"]
    ]
    expected_feature_keys = sorted(
        canonical_trial_key(row) for row in train_rows + test_rows
    )
    if feature_table.get("partition_id") != "outer__refit":
        raise ArtifactExecutionError(
            "prediction table must bind the outer-refit feature partition"
        )
    if feature_keys != expected_feature_keys:
        raise ArtifactExecutionError(
            "outer-refit feature table does not cover the complete subject manifest"
        )
    feature_digest = feature_table_sha256(
        feature_table,
        charter,
        feature_contract,
        expected_keys=expected_feature_keys,
        expected_bindings=feature_bindings,
    )
    bindings = {
        "charter_sha256": feature_table["charter_sha256"],
        "roi_map_sha256": feature_table["roi_map_sha256"],
        "trial_manifest_sha256": feature_table["trial_manifest_sha256"],
        "fold_manifest_sha256": feature_table["fold_manifest_sha256"],
        "feature_schema_sha256": feature_table["feature_schema_sha256"],
        "feature_table_sha256": feature_digest,
        "cross_fit_provenance_sha256": cross_fit_digest,
        "model_provenance_sha256": provenance_digest,
        "uv_lock_sha256": feature_table["uv_lock_sha256"],
        "package_versions_sha256": feature_table["package_versions_sha256"],
    }
    subject_id = feature_table["subject_id"]
    condition = feature_table["condition"]
    outer_test_session = next(iter({int(row["session"]) for row in test_rows}))
    outer_fold_id = f"outer__{subject_id}__{condition}__test-{outer_test_session}"
    model_links = {
        model_id: model_provenance_sha256(
            {model_id: model_provenance[model_id]},
            prediction_contract,
            require_all_models=False,
        )
        for model_id in ("B3", "M1", "M3")
    }
    rows = []
    for index, key in enumerate(model_result.test_keys):
        manifest_row = test_by_key[key]
        rows.append(
            {
                "subject_id": key[0],
                "session": key[1],
                "trial": key[2],
                "item_num": key[3],
                "condition": condition,
                "outer_fold_id": outer_fold_id,
                "recalled": bool(manifest_row["recalled"]),
                "prob_b3": float(model_result.prob_b3[index]),
                "prob_m1": float(model_result.prob_m1[index]),
                "prob_m3": float(model_result.prob_m3[index]),
                "b3_provenance_sha256": model_links["B3"],
                "m1_provenance_sha256": model_links["M1"],
                "m3_provenance_sha256": model_links["M3"],
            }
        )
    table = {
        "schema_id": prediction_contract["logical_table"]["schema_id"],
        "schema_version": prediction_contract["logical_table"]["schema_version"],
        "created_at_utc": created_at_utc,
        "condition": condition,
        "subject_id": subject_id,
        "outer_fold_id": outer_fold_id,
        "outer_test_session": outer_test_session,
        **bindings,
        "code_commit": feature_table["code_commit"],
        "dirty_diff_sha256": None,
        "generator_command": generator_command,
        "python_version": python_version,
        "model_provenance": model_provenance,
        "rows": rows,
    }
    validate_prediction_table(
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=bindings,
    )
    return table, bindings
