"""Invented-data tests for full-run feature and prediction artifact assembly."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peers.m1_artifact_execution import (
    ArtifactExecutionError,
    build_d44_c_provenance,
    build_all_subject_manifest_lineages,
    build_feature_tables,
    build_normalization_provenance,
    build_prediction_table,
    build_subject_manifest_lineage,
)
from peers.m1_feature_generation import (
    combine_training_session_statistics,
    compact_features_for_sessions,
    summarize_log_power_session,
)
from peers.m1_manifests import (
    canonical_row_sha256,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    event_provenance_sha256,
    load_json_unique,
    manifest_sha256,
)
from peers.m1_model_execution import execute_frozen_models_without_outer_scoring
from peers.m1_full_execution import (
    FullExecutionError,
    canonical_bytes,
    canonical_sha256,
    publish_full_run_index,
    publish_aggregate_artifacts,
    publish_subject_comparison_artifacts,
    publish_subject_condition_artifacts,
    verify_full_run_from_index,
)


ROOT = Path(__file__).resolve().parents[1]
CHARTER = load_json_unique(ROOT / "docs" / "analysis-charter-m1-v1.json")
FEATURE_CONTRACT = load_json_unique(ROOT / "docs" / "m1-feature-table-contract.json")
MANIFEST_CONTRACT = load_json_unique(ROOT / "docs" / "m1-manifest-contracts.json")
PREDICTION_CONTRACT = load_json_unique(ROOT / "docs" / "m1-prediction-table-contract.json")
EVALUATION_CONTRACT = load_json_unique(ROOT / "docs" / "m1-evaluation-contract.json")
SHA = "a" * 64
ITEMS = tuple(range(1, 9))


def _manifest(kind, rows):
    slug = {
        "inner_fold": "inner-fold",
        "cross_condition_intersection": "cross-condition-intersection",
    }.get(kind, kind)
    return {
        "schema_id": f"neuroai-m1-{slug}-v1",
        "schema_version": "1.0",
        "manifest_kind": kind,
        "created_at_utc": "2026-08-17T20:00:00Z",
        "charter_id": "neuroai-m1-v1",
        "charter_version": "1.3-frozen",
        "charter_sha256": "b" * 64,
        "roi_map_id": "neuroai-m1-roi-v1",
        "roi_map_sha256": "c" * 64,
        "code_commit": "4" * 40,
        "dirty_diff_sha256": None,
        "generator_command": "invented-artifact-test",
        "source_inventory_sha256": "d" * 64,
        "rows": rows,
    }


def _inputs():
    trial_rows = []
    for session, role in ((0, "train"), (1, "train"), (2, "test")):
        for item in ITEMS:
            trial_rows.append(
                {
                    "subject_id": "sub-SYN001",
                    "session": session,
                    "trial": item,
                    "item_num": item,
                    "condition": "ocular-removed",
                    "serial_position": item,
                    "list_number": item,
                    "onset_s": float(item),
                    "recalled": item % 2 == 0,
                    "source_events_id": f"sub-SYN001_ses-{session}_events.tsv",
                    "source_events_sha256": SHA,
                    "derivative_id": f"sub-SYN001_ses-{session}_epo.fif",
                    "derivative_sha256": SHA,
                    "epoch_row": item - 1,
                    "included": True,
                    "exclusion_reason_code": "INCLUDED",
                    "condition_survivor": True,
                    "outer_role": role,
                }
            )
    keys = {
        session: [
            canonical_trial_key(row) for row in trial_rows if row["session"] == session
        ]
        for session in (0, 1, 2)
    }
    folds = []
    for fold_id, validation, fit in (("fold-0", 0, 1), ("fold-1", 1, 0)):
        folds.append(
            {
                "subject_id": "sub-SYN001",
                "condition": "ocular-removed",
                "inner_fold_id": fold_id,
                "fit_sessions": [fit],
                "validation_session": validation,
                "outer_test_session": 2,
                "fit_class_counts": {"0": 4, "1": 4},
                "validation_class_counts": {"0": 4, "1": 4},
                "fit_keys_sha256": canonical_trial_key_set_sha256(keys[fit]),
                "validation_keys_sha256": canonical_trial_key_set_sha256(keys[validation]),
                "eligible": True,
                "feasibility_reason_code": "ELIGIBLE",
            }
        )
    rng = np.random.default_rng(20260817)
    time_axis = np.linspace(-1.0, 2.0, 31, dtype=np.float64)
    summaries = []
    for session in (0, 1, 2):
        values = rng.normal(loc=session * 0.2, size=(len(ITEMS), 40, 8, 31)).astype(np.float64)
        summaries.append(
            summarize_log_power_session(
                values,
                time_axis,
                keys[session],
                subject_id="sub-SYN001",
                session=session,
                condition="ocular-removed",
                derivative_id=f"sub-SYN001_ses-{session}_epo.fif",
                derivative_sha256=SHA,
                charter=CHARTER,
            )
        )
    reference_rows = []
    for subject_index, subject in enumerate(("sub-SYN001", "sub-SYN002", "sub-SYN003")):
        for session in (0, 1, 2):
            for item in ITEMS:
                reference_rows.append(
                    {
                        "subject_id": subject,
                        "session": session,
                        "trial": item,
                        "item_num": item,
                        "item_name": (
                            "apple",
                            "bridge",
                            "candle",
                            "dragon",
                            "engine",
                            "forest",
                            "garden",
                            "harbor",
                        )[item - 1],
                        "serial_position": item,
                        "recalled": bool((item + subject_index) % 2),
                    }
                )
    reference = pd.DataFrame(reference_rows)
    target = reference.loc[reference.subject_id == "sub-SYN001"].copy()
    target["recalled"] = [item % 2 == 0 for item in target["item_num"]]

    def matrix(fit_sessions, apply_sessions):
        statistics = combine_training_session_statistics(
            summaries, fit_sessions=fit_sessions, charter=CHARTER
        )
        return compact_features_for_sessions(
            summaries,
            statistics,
            apply_sessions=apply_sessions,
            charter=CHARTER,
        )

    result = execute_frozen_models_without_outer_scoring(
        target,
        reference,
        charter=CHARTER,
        m1_matrix=matrix,
    )
    return _manifest("trial", trial_rows), _manifest("inner_fold", folds), summaries, result


def test_builds_all_transform_feature_tables_and_valid_prediction_table():
    trials, folds, summaries, result = _inputs()
    provenance, _ = build_normalization_provenance(
        summaries,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        created_at_utc="2026-08-17T21:00:00Z",
        charter_sha256="b" * 64,
        roi_map_sha256="c" * 64,
        uv_lock_sha256="e" * 64,
        package_versions_sha256="f" * 64,
        code_commit="4" * 40,
        generator_command="invented-artifact-test",
        python_version="3.12.synthetic",
    )
    tables, feature_bindings = build_feature_tables(
        summaries,
        provenance,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        source_inventory_sha256="d" * 64,
        created_at_utc="2026-08-17T21:00:01Z",
        generator_command="invented-artifact-test",
        python_version="3.12.synthetic",
    )
    assert list(tables) == ["inner__fold-0", "inner__fold-1", "outer__refit"]
    assert len(tables["inner__fold-0"]["rows"]) == 16
    assert len(tables["inner__fold-1"]["rows"]) == 16
    assert len(tables["outer__refit"]["rows"]) == 24

    d44, _ = build_d44_c_provenance(
        result,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        manifest_contract=MANIFEST_CONTRACT,
    )
    table, bindings = build_prediction_table(
        result,
        tables["outer__refit"],
        feature_bindings,
        d44,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        prediction_contract=PREDICTION_CONTRACT,
        created_at_utc="2026-08-17T21:00:02Z",
        generator_command="invented-artifact-test",
        python_version="3.12.synthetic",
    )
    assert len(table["rows"]) == 8
    assert table["feature_table_sha256"] == bindings["feature_table_sha256"]
    assert table["model_provenance"]["M3"]["upstream_cross_fit_sha256"] == (
        table["cross_fit_provenance_sha256"]
    )
    assert "auc" not in table


def test_subject_manifest_lineage_binds_parent_and_exact_subject_slices():
    trials, folds, _, _ = _inputs()
    intersection_rows = []
    for row in trials["rows"]:
        intersection_rows.append(
            {
                "subject_id": row["subject_id"],
                "session": row["session"],
                "trial": row["trial"],
                "item_num": row["item_num"],
                "ocular_removed_row_sha256": SHA,
                "ocular_kept_row_sha256": SHA,
                "ocular_removed_epoch_row": row["epoch_row"],
                "ocular_kept_epoch_row": row["epoch_row"],
                "recalled": row["recalled"],
                "event_provenance_sha256": SHA,
            }
        )
    manifests = {
        "trial-ocular-removed": trials,
        "inner-fold-ocular-removed": folds,
        "trial-ocular-kept": {
            **trials,
            "rows": [{**row, "condition": "ocular-kept"} for row in trials["rows"]],
        },
        "inner-fold-ocular-kept": {
            **folds,
            "rows": [{**row, "condition": "ocular-kept"} for row in folds["rows"]],
        },
        "cross-condition-intersection": _manifest(
            "cross_condition_intersection", intersection_rows
        ),
    }
    slices, lineage = build_subject_manifest_lineage(
        manifests, "sub-SYN001", MANIFEST_CONTRACT
    )
    assert set(slices) == set(manifests)
    assert lineage["slice_rule"] == "retain_exact_parent_rows_where_subject_id_matches"
    assert set(lineage["parent_manifest_sha256s"]) == set(manifests)
    assert set(lineage["subject_manifest_sha256s"]) == set(manifests)
    all_lineages = build_all_subject_manifest_lineages(
        manifests, ["sub-SYN001"], MANIFEST_CONTRACT
    )
    assert all_lineages["sub-SYN001"][1] == lineage


def test_prediction_rejects_non_outer_or_incomplete_feature_partition():
    trials, folds, summaries, result = _inputs()
    provenance, _ = build_normalization_provenance(
        summaries,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        created_at_utc="2026-08-17T21:00:00Z",
        charter_sha256="b" * 64,
        roi_map_sha256="c" * 64,
        uv_lock_sha256="e" * 64,
        package_versions_sha256="f" * 64,
        code_commit="4" * 40,
        generator_command="invented-artifact-test",
        python_version="3.12.synthetic",
    )
    tables, feature_bindings = build_feature_tables(
        summaries,
        provenance,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        source_inventory_sha256="d" * 64,
        created_at_utc="2026-08-17T21:00:01Z",
        generator_command="invented-artifact-test",
        python_version="3.12.synthetic",
    )
    d44, _ = build_d44_c_provenance(
        result,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        charter=CHARTER,
        manifest_contract=MANIFEST_CONTRACT,
    )
    with np.testing.assert_raises_regex(
        ArtifactExecutionError, "outer-refit feature partition"
    ):
        build_prediction_table(
            result,
            tables["inner__fold-0"],
            feature_bindings,
            d44,
            trial_manifest=trials,
            inner_fold_manifest=folds,
            charter=CHARTER,
            feature_contract=FEATURE_CONTRACT,
            manifest_contract=MANIFEST_CONTRACT,
            prediction_contract=PREDICTION_CONTRACT,
            created_at_utc="2026-08-17T21:00:02Z",
            generator_command="invented-artifact-test",
            python_version="3.12.synthetic",
        )


def test_full_publisher_graph_writes_subject_comparison_and_aggregates(tmp_path):
    trials, folds, summaries, result = _inputs()
    kept_trials = {
        **trials,
        "rows": [{**row, "condition": "ocular-kept"} for row in trials["rows"]],
    }
    kept_folds = {
        **folds,
        "rows": [{**row, "condition": "ocular-kept"} for row in folds["rows"]],
    }
    kept_summaries = [replace(value, condition="ocular-kept") for value in summaries]
    common = dict(
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        prediction_contract=PREDICTION_CONTRACT,
        evaluation_contract=EVALUATION_CONTRACT,
        source_inventory_sha256="d" * 64,
        charter_sha256="b" * 64,
        roi_map_sha256="c" * 64,
        uv_lock_sha256="e" * 64,
        package_versions_sha256="f" * 64,
        code_commit="4" * 40,
        created_at_utc="2026-08-17T22:00:00Z",
        generator_command="invented-full-publisher-test",
        python_version="3.12.synthetic",
    )
    removed = publish_subject_condition_artifacts(
        output_root=tmp_path / "sub-SYN001" / "ocular-removed",
        summaries=summaries,
        model_result=result,
        trial_manifest=trials,
        inner_fold_manifest=folds,
        parent_trial_manifest_sha256=manifest_sha256(trials, MANIFEST_CONTRACT),
        parent_inner_fold_manifest_sha256=manifest_sha256(folds, MANIFEST_CONTRACT),
        **common,
    )
    kept = publish_subject_condition_artifacts(
        output_root=tmp_path / "sub-SYN001" / "ocular-kept",
        summaries=kept_summaries,
        model_result=result,
        trial_manifest=kept_trials,
        inner_fold_manifest=kept_folds,
        parent_trial_manifest_sha256=manifest_sha256(kept_trials, MANIFEST_CONTRACT),
        parent_inner_fold_manifest_sha256=manifest_sha256(kept_folds, MANIFEST_CONTRACT),
        **common,
    )
    intersection_rows = []
    for removed_row, kept_row in zip(trials["rows"], kept_trials["rows"], strict=True):
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
    intersection = _manifest("cross_condition_intersection", intersection_rows)
    comparison = publish_subject_comparison_artifacts(
        output_root=tmp_path / "sub-SYN001" / "comparison",
        ocular_removed=removed,
        ocular_kept=kept,
        intersection_manifest=intersection,
        parent_intersection_manifest_sha256=manifest_sha256(
            intersection, MANIFEST_CONTRACT
        ),
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        prediction_contract=PREDICTION_CONTRACT,
        evaluation_contract=EVALUATION_CONTRACT,
        uv_lock_sha256="e" * 64,
        package_versions_sha256="f" * 64,
        code_commit="4" * 40,
        created_at_utc="2026-08-17T22:00:01Z",
        generator_command="invented-full-publisher-test",
        python_version="3.12.synthetic",
    )
    aggregates = publish_aggregate_artifacts(
        output_root=tmp_path / "aggregates",
        subject_condition_outputs=[removed, kept],
        subject_comparison_outputs=[comparison],
        charter=CHARTER,
        evaluation_contract=EVALUATION_CONTRACT,
        charter_sha256="b" * 64,
        package_versions_sha256="f" * 64,
        code_commit="4" * 40,
        generator_command="invented-full-publisher-test",
    )
    assert aggregates["index"]["subject_ids"] == ["sub-SYN001"]
    assert "balanced_accuracy" not in str(aggregates["primary"])
    assert (tmp_path / "aggregates" / "primary" / "bundle.sha256").is_file()

    parent_manifests = {
        "trial-ocular-removed": trials,
        "inner-fold-ocular-removed": folds,
        "trial-ocular-kept": kept_trials,
        "inner-fold-ocular-kept": kept_folds,
        "cross-condition-intersection": intersection,
    }
    _, lineage = build_subject_manifest_lineage(
        parent_manifests, "sub-SYN001", MANIFEST_CONTRACT
    )
    global_bindings = {
        "code_commit": "4" * 40,
        "charter_sha256": "b" * 64,
        "roi_map_sha256": "c" * 64,
        "manifest_contract_sha256": canonical_sha256(MANIFEST_CONTRACT),
        "feature_contract_sha256": canonical_sha256(FEATURE_CONTRACT),
        "prediction_contract_sha256": canonical_sha256(PREDICTION_CONTRACT),
        "evaluation_contract_sha256": canonical_sha256(EVALUATION_CONTRACT),
        "preflight_receipt_sha256": "6" * 64,
        "capacity_receipt_sha256": "7" * 64,
        "full_run_authorization_sha256": "8" * 64,
        "execution_source_inventory_sha256": "d" * 64,
        "parent_manifest_run_index_sha256": "9" * 64,
    }
    _, run_index_sha256 = publish_full_run_index(
        output_root=tmp_path,
        subject_condition_outputs=[removed, kept],
        subject_comparison_outputs=[comparison],
        aggregate_output=aggregates,
        created_at_utc="2026-08-17T22:00:02Z",
        subject_manifest_lineage_by_subject={"sub-SYN001": lineage},
        generator_command="invented-full-publisher-test",
        **global_bindings,
    )
    receipt, _ = verify_full_run_from_index(
        output_root=tmp_path,
        expected_run_index_sha256=run_index_sha256,
        parent_manifests=parent_manifests,
        charter=CHARTER,
        feature_contract=FEATURE_CONTRACT,
        manifest_contract=MANIFEST_CONTRACT,
        prediction_contract=PREDICTION_CONTRACT,
        evaluation_contract=EVALUATION_CONTRACT,
        expected_global_bindings=global_bindings,
        verification_output=tmp_path / "verification-receipt.json",
        verified_at_utc="2026-08-17T22:00:03Z",
        verifier_code_commit="4" * 40,
        verification_command="invented-full-verifier-test",
    )
    assert receipt["verified_subject_condition_count"] == 2
    assert receipt["result_values_in_receipt"] is False
    assert receipt["all_stored_contracts_revalidated"] is True
    assert receipt["source_features_and_predictions_regenerated"] is False

    normalization_path = (
        tmp_path
        / "sub-SYN001"
        / "ocular-removed"
        / "normalization-provenance.json"
    )
    tampered = load_json_unique(normalization_path)
    tampered["created_at_utc"] = "2026-08-17T22:00:04Z"
    normalization_path.write_bytes(canonical_bytes(tampered))
    with pytest.raises(FullExecutionError, match="JSON digest mismatch"):
        verify_full_run_from_index(
            output_root=tmp_path,
            expected_run_index_sha256=run_index_sha256,
            parent_manifests=parent_manifests,
            charter=CHARTER,
            feature_contract=FEATURE_CONTRACT,
            manifest_contract=MANIFEST_CONTRACT,
            prediction_contract=PREDICTION_CONTRACT,
            evaluation_contract=EVALUATION_CONTRACT,
            expected_global_bindings=global_bindings,
            verification_output=tmp_path / "tampered-verification.json",
            verified_at_utc="2026-08-17T22:00:05Z",
            verifier_code_commit="4" * 40,
            verification_command="invented-full-verifier-tamper-test",
        )
