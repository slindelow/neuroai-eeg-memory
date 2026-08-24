"""Authenticated publication primitives for the authorized NeuroAI M1 full run.

The functions in this module are deliberately explicit: each subject-condition is
published from validated manifests, raw spectral summaries, and an already fitted
``ModelExecutionResult``; every logical artifact is written through its existing
atomic bundle API and read back by the same closed contract.  Nothing here chooses
a scientific option or prints a performance value.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from peers.m1_artifact_execution import (
    build_d44_c_provenance,
    build_feature_tables,
    build_normalization_provenance,
    build_prediction_table,
    build_subject_manifest_lineage,
    validate_lineage_parent_manifests,
)
from peers.m1_evaluation import (
    aggregate_common_intersection_subject_records,
    aggregate_primary_subject_records,
    common_intersection_subject_content_sha256,
    evaluate_common_intersection_subject,
    evaluate_subject_prediction_table,
    subject_record_content_sha256,
)
from peers.m1_feature_generation import SessionSpectralSummary
from peers.m1_features import (
    feature_table_sha256,
    load_feature_table_bundle,
    normalization_transform_provenance_sha256,
    write_feature_table_bundle_atomic,
)
from peers.m1_manifests import (
    canonical_trial_key,
    d44_c_cross_fit_provenance_sha256,
    manifest_sha256,
    validate_d44_c_cross_fit_provenance,
    validate_manifest,
    load_manifest_bundle,
    write_manifest_bundle_atomic,
)
from peers.m1_model_execution import ModelExecutionResult
from peers.m1_predictions import (
    build_condition_comparison_table,
    condition_comparison_table_sha256,
    load_condition_comparison_bundle,
    load_prediction_table_bundle,
    prediction_table_sha256,
    write_condition_comparison_bundle_atomic,
    write_prediction_table_bundle_atomic,
)
from peers.m1_result_storage import (
    load_common_intersection_aggregate_bundle,
    load_common_intersection_subject_evaluation_bundle,
    load_primary_aggregate_bundle,
    load_subject_evaluation_bundle,
    write_common_intersection_aggregate_bundle_atomic,
    write_common_intersection_subject_evaluation_bundle_atomic,
    write_primary_aggregate_bundle_atomic,
    write_subject_evaluation_bundle_atomic,
)


class FullExecutionError(ValueError):
    """The locked result publication graph violated a frozen boundary."""


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FullExecutionError("execution document is not finite canonical JSON") from exc
    return text.encode("utf-8") + b"\n"


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def publish_canonical_json_noreplace(path: Path, value: Any) -> str:
    """Publish canonical JSON with fsync and hard-link no-replace semantics."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    digest = sha256(payload).hexdigest()
    temporary = destination.parent / (
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        os.unlink(temporary)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            os.unlink(temporary)
        raise
    if destination.read_bytes() != payload:
        raise FullExecutionError(f"canonical JSON readback mismatch: {destination}")
    return digest


def _manifest_keys(manifest: Mapping[str, Any]) -> list[tuple[str, int, int, int]]:
    return sorted(canonical_trial_key(row) for row in manifest["rows"])


def _evaluation_provenance(
    *, code_commit: str, package_versions_sha256: str, command: str
) -> dict[str, Any]:
    return {
        "evaluator_code_commit": code_commit,
        "evaluator_dirty_diff_sha256": None,
        "evaluator_package_versions_sha256": package_versions_sha256,
        "evaluation_command": command,
    }


def _aggregate_provenance(
    *, code_commit: str, package_versions_sha256: str, command: str
) -> dict[str, Any]:
    return {
        "aggregator_code_commit": code_commit,
        "aggregator_dirty_diff_sha256": None,
        "aggregator_package_versions_sha256": package_versions_sha256,
        "aggregation_command": command,
    }


def publish_subject_condition_artifacts(
    *,
    output_root: Path,
    summaries: Sequence[SessionSpectralSummary],
    model_result: ModelExecutionResult,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    parent_trial_manifest_sha256: str,
    parent_inner_fold_manifest_sha256: str,
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    source_inventory_sha256: str,
    charter_sha256: str,
    roi_map_sha256: str,
    uv_lock_sha256: str,
    package_versions_sha256: str,
    code_commit: str,
    created_at_utc: str,
    generator_command: str,
    python_version: str,
) -> dict[str, Any]:
    """Publish and read back one complete subject-condition evidence chain."""
    root = Path(output_root)
    if os.path.lexists(root):
        raise FullExecutionError(f"subject-condition destination exists: {root}")
    root.mkdir(parents=True)
    validate_manifest(dict(trial_manifest), dict(manifest_contract))
    validate_manifest(dict(inner_fold_manifest), dict(manifest_contract))
    subject_groups = {
        (row["subject_id"], row["condition"]) for row in trial_manifest["rows"]
    }
    if len(subject_groups) != 1:
        raise FullExecutionError("subject-condition publisher requires one group")
    subject_id, condition = next(iter(subject_groups))

    trial_bundle_sha256 = write_manifest_bundle_atomic(
        root / "trial-manifest", dict(trial_manifest), dict(manifest_contract)
    )
    fold_bundle_sha256 = write_manifest_bundle_atomic(
        root / "inner-fold-manifest",
        dict(inner_fold_manifest),
        dict(manifest_contract),
    )
    trial_sha256 = manifest_sha256(dict(trial_manifest), dict(manifest_contract))
    fold_sha256 = manifest_sha256(dict(inner_fold_manifest), dict(manifest_contract))

    normalization, normalization_bindings = build_normalization_provenance(
        summaries,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        charter=charter,
        feature_contract=feature_contract,
        manifest_contract=manifest_contract,
        created_at_utc=created_at_utc,
        charter_sha256=charter_sha256,
        roi_map_sha256=roi_map_sha256,
        uv_lock_sha256=uv_lock_sha256,
        package_versions_sha256=package_versions_sha256,
        code_commit=code_commit,
        generator_command=generator_command,
        python_version=python_version,
    )
    normalization_sha256 = normalization_transform_provenance_sha256(
        normalization,
        charter,
        feature_contract,
        manifest_contract,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        expected_bindings=normalization_bindings,
    )
    normalization_file_sha256 = publish_canonical_json_noreplace(
        root / "normalization-provenance.json", normalization
    )
    if normalization_file_sha256 != normalization_sha256:
        raise FullExecutionError("normalization logical and file digests diverged")

    feature_tables, feature_bindings = build_feature_tables(
        summaries,
        normalization,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        charter=charter,
        feature_contract=feature_contract,
        manifest_contract=manifest_contract,
        source_inventory_sha256=source_inventory_sha256,
        created_at_utc=created_at_utc,
        generator_command=generator_command,
        python_version=python_version,
    )
    feature_artifacts: dict[str, dict[str, str]] = {}
    (root / "features").mkdir()
    for partition_id, table in feature_tables.items():
        keys = [canonical_trial_key(row) for row in table["rows"]]
        logical_sha256 = feature_table_sha256(
            table,
            charter,
            feature_contract,
            expected_keys=keys,
            expected_bindings=feature_bindings,
        )
        bundle_sha256 = write_feature_table_bundle_atomic(
            root / "features" / partition_id,
            table,
            charter,
            feature_contract,
            expected_keys=keys,
            expected_bindings=feature_bindings,
        )
        feature_artifacts[partition_id] = {
            "logical_sha256": logical_sha256,
            "bundle_sha256": bundle_sha256,
        }

    d44, d44_sha256 = build_d44_c_provenance(
        model_result,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        charter=charter,
        manifest_contract=manifest_contract,
    )
    validate_d44_c_cross_fit_provenance(
        dict(trial_manifest),
        dict(inner_fold_manifest),
        d44,
        charter,
        dict(manifest_contract),
    )
    d44_file_sha256 = publish_canonical_json_noreplace(
        root / "d44-c-cross-fit-provenance.json", d44
    )
    if d44_file_sha256 != d44_sha256:
        raise FullExecutionError("D44-C logical and file digests diverged")

    selection_evidence = {
        "schema_id": "neuroai-m1-inner-selection-evidence-v1",
        "schema_version": "1.0",
        "subject_id": subject_id,
        "condition": condition,
        "selected_m1_c": float(model_result.selected_m1_c),
        "selection_metric": charter["splits"]["selection_metric"],
        "tie_break": charter["splits"]["tie_break"],
        "candidate_scores_by_c": {
            key: [float(value) for value in values]
            for key, values in model_result.inner_selection_auc_by_c.items()
        },
        "inner_fold_manifest_sha256": fold_sha256,
        "outer_test_outcome_access": False,
    }
    selection_evidence_sha256 = publish_canonical_json_noreplace(
        root / "inner-selection-evidence.json", selection_evidence
    )

    prediction_table, prediction_bindings = build_prediction_table(
        model_result,
        feature_tables["outer__refit"],
        feature_bindings,
        d44,
        trial_manifest=trial_manifest,
        inner_fold_manifest=inner_fold_manifest,
        charter=charter,
        feature_contract=feature_contract,
        manifest_contract=manifest_contract,
        prediction_contract=prediction_contract,
        created_at_utc=created_at_utc,
        generator_command=generator_command,
        python_version=python_version,
    )
    prediction_logical_sha256 = prediction_table_sha256(
        prediction_table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=prediction_bindings,
    )
    prediction_bundle_sha256 = write_prediction_table_bundle_atomic(
        root / "predictions",
        prediction_table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_bindings=prediction_bindings,
    )

    evaluation_provenance = _evaluation_provenance(
        code_commit=code_commit,
        package_versions_sha256=package_versions_sha256,
        command=generator_command,
    )
    subject_record = evaluate_subject_prediction_table(
        prediction_table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_bindings=prediction_bindings,
        evaluation_provenance=evaluation_provenance,
    )
    subject_record_sha256 = subject_record_content_sha256(
        subject_record, evaluation_contract
    )
    subject_result_bundle_sha256 = write_subject_evaluation_bundle_atomic(
        root / "subject-evaluation",
        subject_record,
        prediction_table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_bindings=prediction_bindings,
        expected_evaluation_provenance=evaluation_provenance,
    )
    index_record = {
        "subject_id": subject_id,
        "condition": condition,
        "parent_trial_manifest_sha256": parent_trial_manifest_sha256,
        "parent_inner_fold_manifest_sha256": parent_inner_fold_manifest_sha256,
        "trial_manifest_sha256": trial_sha256,
        "trial_manifest_bundle_sha256": trial_bundle_sha256,
        "inner_fold_manifest_sha256": fold_sha256,
        "inner_fold_manifest_bundle_sha256": fold_bundle_sha256,
        "normalization_provenance_sha256": normalization_sha256,
        "feature_artifacts": feature_artifacts,
        "feature_bindings": feature_bindings,
        "d44_c_cross_fit_provenance_sha256": d44_sha256,
        "inner_selection_evidence_sha256": selection_evidence_sha256,
        "prediction_bindings": prediction_bindings,
        "prediction_table_sha256": prediction_logical_sha256,
        "prediction_bundle_sha256": prediction_bundle_sha256,
        "evaluation_provenance": evaluation_provenance,
        "subject_evaluation_sha256": subject_record_sha256,
        "subject_evaluation_bundle_sha256": subject_result_bundle_sha256,
    }
    index_sha256 = publish_canonical_json_noreplace(
        root / "index.json", index_record
    )
    return {
        "root": root,
        "trial_manifest": dict(trial_manifest),
        "inner_fold_manifest": dict(inner_fold_manifest),
        "normalization": normalization,
        "feature_tables": feature_tables,
        "prediction_table": prediction_table,
        "prediction_bindings": prediction_bindings,
        "subject_record": subject_record,
        "index": index_record,
        "index_sha256": index_sha256,
    }


def publish_subject_comparison_artifacts(
    *,
    output_root: Path,
    ocular_removed: Mapping[str, Any],
    ocular_kept: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    parent_intersection_manifest_sha256: str,
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    uv_lock_sha256: str,
    package_versions_sha256: str,
    code_commit: str,
    created_at_utc: str,
    generator_command: str,
    python_version: str,
) -> dict[str, Any]:
    """Publish the exact same-trial ocular comparison and subject evaluation."""
    root = Path(output_root)
    if os.path.lexists(root):
        raise FullExecutionError(f"comparison destination exists: {root}")
    root.mkdir(parents=True)
    validate_manifest(dict(intersection_manifest), dict(manifest_contract))
    intersection_sha256 = manifest_sha256(
        dict(intersection_manifest), dict(manifest_contract)
    )
    intersection_bundle_sha256 = write_manifest_bundle_atomic(
        root / "intersection-manifest",
        dict(intersection_manifest),
        dict(manifest_contract),
    )
    prediction_digests = {
        "ocular-removed": ocular_removed["index"]["prediction_table_sha256"],
        "ocular-kept": ocular_kept["index"]["prediction_table_sha256"],
    }
    comparison_provenance = {
        "created_at_utc": created_at_utc,
        "comparator_code_commit": code_commit,
        "comparator_dirty_diff_sha256": None,
        "comparator_uv_lock_sha256": uv_lock_sha256,
        "comparator_package_versions_sha256": package_versions_sha256,
        "comparison_command": generator_command,
        "comparator_python_version": python_version,
    }
    comparison_table = build_condition_comparison_table(
        ocular_removed["prediction_table"],
        ocular_kept["prediction_table"],
        ocular_removed["trial_manifest"],
        ocular_kept["trial_manifest"],
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=ocular_removed["prediction_bindings"],
        expected_kept_bindings=ocular_kept["prediction_bindings"],
        expected_intersection_manifest_sha256=intersection_sha256,
        expected_prediction_table_sha256_by_condition=prediction_digests,
        comparison_provenance=comparison_provenance,
    )
    comparison_logical_sha256 = condition_comparison_table_sha256(
        comparison_table,
        ocular_removed["prediction_table"],
        ocular_kept["prediction_table"],
        ocular_removed["trial_manifest"],
        ocular_kept["trial_manifest"],
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=ocular_removed["prediction_bindings"],
        expected_kept_bindings=ocular_kept["prediction_bindings"],
        expected_intersection_manifest_sha256=intersection_sha256,
        expected_prediction_table_sha256_by_condition=prediction_digests,
        expected_comparison_provenance=comparison_provenance,
    )
    comparison_bundle_sha256 = write_condition_comparison_bundle_atomic(
        root / "comparison",
        comparison_table,
        ocular_removed["prediction_table"],
        ocular_kept["prediction_table"],
        ocular_removed["trial_manifest"],
        ocular_kept["trial_manifest"],
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        expected_removed_bindings=ocular_removed["prediction_bindings"],
        expected_kept_bindings=ocular_kept["prediction_bindings"],
        expected_intersection_manifest_sha256=intersection_sha256,
        expected_prediction_table_sha256_by_condition=prediction_digests,
        expected_comparison_provenance=comparison_provenance,
    )
    evaluation_provenance = _evaluation_provenance(
        code_commit=code_commit,
        package_versions_sha256=package_versions_sha256,
        command=generator_command,
    )
    subject_record = evaluate_common_intersection_subject(
        comparison_table,
        ocular_removed["prediction_table"],
        ocular_kept["prediction_table"],
        ocular_removed["trial_manifest"],
        ocular_kept["trial_manifest"],
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_removed_bindings=ocular_removed["prediction_bindings"],
        expected_kept_bindings=ocular_kept["prediction_bindings"],
        expected_intersection_manifest_sha256=intersection_sha256,
        expected_prediction_table_sha256_by_condition=prediction_digests,
        expected_comparison_provenance=comparison_provenance,
        evaluation_provenance=evaluation_provenance,
    )
    subject_record_sha256 = common_intersection_subject_content_sha256(
        subject_record, evaluation_contract
    )
    subject_result_bundle_sha256 = (
        write_common_intersection_subject_evaluation_bundle_atomic(
            root / "subject-evaluation",
            subject_record,
            comparison_table,
            ocular_removed["prediction_table"],
            ocular_kept["prediction_table"],
            ocular_removed["trial_manifest"],
            ocular_kept["trial_manifest"],
            intersection_manifest,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            evaluation_contract,
            expected_removed_bindings=ocular_removed["prediction_bindings"],
            expected_kept_bindings=ocular_kept["prediction_bindings"],
            expected_intersection_manifest_sha256=intersection_sha256,
            expected_prediction_table_sha256_by_condition=prediction_digests,
            expected_comparison_provenance=comparison_provenance,
            expected_evaluation_provenance=evaluation_provenance,
        )
    )
    index_record = {
        "subject_id": comparison_table["subject_id"],
        "parent_intersection_manifest_sha256": parent_intersection_manifest_sha256,
        "intersection_manifest_sha256": intersection_sha256,
        "intersection_manifest_bundle_sha256": intersection_bundle_sha256,
        "prediction_table_sha256_by_condition": prediction_digests,
        "comparison_provenance": comparison_provenance,
        "comparison_table_sha256": comparison_logical_sha256,
        "comparison_bundle_sha256": comparison_bundle_sha256,
        "evaluation_provenance": evaluation_provenance,
        "subject_evaluation_sha256": subject_record_sha256,
        "subject_evaluation_bundle_sha256": subject_result_bundle_sha256,
    }
    index_sha256 = publish_canonical_json_noreplace(root / "index.json", index_record)
    return {
        "root": root,
        "intersection_manifest": dict(intersection_manifest),
        "comparison_table": comparison_table,
        "comparison_provenance": comparison_provenance,
        "subject_record": subject_record,
        "index": index_record,
        "index_sha256": index_sha256,
    }


def publish_aggregate_artifacts(
    *,
    output_root: Path,
    subject_condition_outputs: Sequence[Mapping[str, Any]],
    subject_comparison_outputs: Sequence[Mapping[str, Any]],
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    charter_sha256: str,
    package_versions_sha256: str,
    code_commit: str,
    generator_command: str,
) -> dict[str, Any]:
    """Publish the primary and paired-condition aggregates over exact subject sets."""
    root = Path(output_root)
    if os.path.lexists(root):
        raise FullExecutionError(f"aggregate destination exists: {root}")
    root.mkdir(parents=True)
    primary_records = [
        output["subject_record"]
        for output in subject_condition_outputs
        if output["index"]["condition"] == "ocular-removed"
    ]
    common_records = [output["subject_record"] for output in subject_comparison_outputs]
    if not primary_records or len(primary_records) != len(common_records):
        raise FullExecutionError("aggregate subject sets are empty or unequal")
    primary_digests = {
        record["subject_id"]: subject_record_content_sha256(
            record, evaluation_contract
        )
        for record in primary_records
    }
    common_digests = {
        record["subject_id"]: common_intersection_subject_content_sha256(
            record, evaluation_contract
        )
        for record in common_records
    }
    if set(primary_digests) != set(common_digests):
        raise FullExecutionError("primary and comparison aggregates use different subjects")
    aggregate_provenance = _aggregate_provenance(
        code_commit=code_commit,
        package_versions_sha256=package_versions_sha256,
        command=generator_command,
    )
    primary = aggregate_primary_subject_records(
        primary_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=charter_sha256,
        expected_record_sha256_by_subject=primary_digests,
        aggregate_provenance=aggregate_provenance,
    )
    primary_bundle_sha256 = write_primary_aggregate_bundle_atomic(
        root / "primary",
        primary,
        primary_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=charter_sha256,
        expected_record_sha256_by_subject=primary_digests,
        expected_aggregate_provenance=aggregate_provenance,
    )
    common = aggregate_common_intersection_subject_records(
        common_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=charter_sha256,
        expected_record_sha256_by_subject=common_digests,
        aggregate_provenance=aggregate_provenance,
    )
    common_bundle_sha256 = write_common_intersection_aggregate_bundle_atomic(
        root / "common-intersection",
        common,
        common_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=charter_sha256,
        expected_record_sha256_by_subject=common_digests,
        expected_aggregate_provenance=aggregate_provenance,
    )
    index_record = {
        "schema_id": "neuroai-m1-aggregate-artifact-index-v1",
        "schema_version": "1.0",
        "subject_ids": sorted(primary_digests),
        "primary_subject_record_sha256s": dict(sorted(primary_digests.items())),
        "common_intersection_subject_record_sha256s": dict(
            sorted(common_digests.items())
        ),
        "aggregate_provenance": aggregate_provenance,
        "primary_bundle_sha256": primary_bundle_sha256,
        "common_intersection_bundle_sha256": common_bundle_sha256,
    }
    index_sha256 = publish_canonical_json_noreplace(root / "index.json", index_record)
    return {
        "root": root,
        "primary": primary,
        "common_intersection": common,
        "index": index_record,
        "index_sha256": index_sha256,
    }


def publish_full_run_index(
    *,
    output_root: Path,
    subject_condition_outputs: Sequence[Mapping[str, Any]],
    subject_comparison_outputs: Sequence[Mapping[str, Any]],
    aggregate_output: Mapping[str, Any],
    created_at_utc: str,
    code_commit: str,
    charter_sha256: str,
    roi_map_sha256: str,
    manifest_contract_sha256: str,
    feature_contract_sha256: str,
    prediction_contract_sha256: str,
    evaluation_contract_sha256: str,
    preflight_receipt_sha256: str,
    capacity_receipt_sha256: str,
    full_run_authorization_sha256: str,
    execution_source_inventory_sha256: str,
    parent_manifest_run_index_sha256: str,
    subject_manifest_lineage_by_subject: Mapping[str, Mapping[str, Any]],
    generator_command: str,
) -> tuple[dict[str, Any], str]:
    """Bind every child artifact into one immutable, metric-free run index."""
    root = Path(output_root)
    if not root.is_dir() or root.is_symlink():
        raise FullExecutionError("full-run output root must be an existing directory")
    condition_entries: dict[str, dict[str, Any]] = {}
    for output in subject_condition_outputs:
        record = output["index"]
        key = f"{record['subject_id']}::{record['condition']}"
        if key in condition_entries:
            raise FullExecutionError(f"duplicate subject-condition output: {key}")
        condition_entries[key] = {
            "relative_path": Path(output["root"]).relative_to(root).as_posix(),
            "index_sha256": output["index_sha256"],
        }
    comparison_entries: dict[str, dict[str, Any]] = {}
    for output in subject_comparison_outputs:
        subject_id = output["index"]["subject_id"]
        if subject_id in comparison_entries:
            raise FullExecutionError(f"duplicate subject comparison: {subject_id}")
        comparison_entries[subject_id] = {
            "relative_path": Path(output["root"]).relative_to(root).as_posix(),
            "index_sha256": output["index_sha256"],
        }
    subjects = sorted(comparison_entries)
    expected_condition_keys = {
        f"{subject_id}::{condition}"
        for subject_id in subjects
        for condition in ("ocular-removed", "ocular-kept")
    }
    if set(condition_entries) != expected_condition_keys:
        raise FullExecutionError("run index lacks an exact two-condition subject grid")
    if set(subject_manifest_lineage_by_subject) != set(subjects):
        raise FullExecutionError("run index lineage subject set mismatch")
    index = {
        "schema_id": "neuroai-m1-full-run-index-v1",
        "schema_version": "1.0",
        "created_at_utc": created_at_utc,
        "code_commit": code_commit,
        "dirty_diff_sha256": None,
        "generator_command": generator_command,
        "charter_sha256": charter_sha256,
        "roi_map_sha256": roi_map_sha256,
        "manifest_contract_sha256": manifest_contract_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "prediction_contract_sha256": prediction_contract_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256,
        "preflight_receipt_sha256": preflight_receipt_sha256,
        "capacity_receipt_sha256": capacity_receipt_sha256,
        "full_run_authorization_sha256": full_run_authorization_sha256,
        "execution_source_inventory_sha256": execution_source_inventory_sha256,
        "parent_manifest_run_index_sha256": parent_manifest_run_index_sha256,
        "subject_ids": subjects,
        "subject_manifest_lineage_by_subject": {
            key: dict(value)
            for key, value in sorted(subject_manifest_lineage_by_subject.items())
        },
        "subject_condition_artifacts": dict(sorted(condition_entries.items())),
        "subject_comparison_artifacts": dict(sorted(comparison_entries.items())),
        "aggregate_artifacts": {
            "relative_path": Path(aggregate_output["root"])
            .relative_to(root)
            .as_posix(),
            "index_sha256": aggregate_output["index_sha256"],
        },
        "partial_outcome_inspection_allowed": False,
        "scientific_decisions_allowed": False,
        "result_values_in_index": False,
    }
    digest = publish_canonical_json_noreplace(root / "run-index.json", index)
    publish_canonical_json_noreplace(
        root / "run-receipt.json",
        {
            "schema_id": "neuroai-m1-full-run-receipt-v1",
            "schema_version": "1.0",
            "full_run_index_sha256": digest,
            "code_commit": code_commit,
            "preflight_receipt_sha256": preflight_receipt_sha256,
            "capacity_receipt_sha256": capacity_receipt_sha256,
            "full_run_authorization_sha256": full_run_authorization_sha256,
            "publication_complete": True,
            "partial_outcome_inspection_performed": False,
            "interpretation_performed": False,
        },
    )
    return index, digest


def load_canonical_json_expected(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load one duplicate-key-free canonical JSON document at an expected digest."""
    source = Path(path)
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != expected_sha256:
        raise FullExecutionError(f"JSON digest mismatch: {source}")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FullExecutionError(f"invalid JSON: {source}") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise FullExecutionError(f"JSON is not a canonical object: {source}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullExecutionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def verify_full_run_from_index(
    *,
    output_root: Path,
    expected_run_index_sha256: str,
    parent_manifests: Mapping[str, Mapping[str, Any]],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    expected_global_bindings: Mapping[str, str],
    verification_output: Path,
    verified_at_utc: str,
    verifier_code_commit: str,
    verification_command: str,
) -> tuple[dict[str, Any], str]:
    """Reopen every artifact and recompute all downstream comparisons and results."""
    root = Path(output_root)
    index = load_canonical_json_expected(
        root / "run-index.json", expected_run_index_sha256
    )
    required_index_fields = {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "code_commit",
        "dirty_diff_sha256",
        "generator_command",
        "charter_sha256",
        "roi_map_sha256",
        "manifest_contract_sha256",
        "feature_contract_sha256",
        "prediction_contract_sha256",
        "evaluation_contract_sha256",
        "preflight_receipt_sha256",
        "capacity_receipt_sha256",
        "full_run_authorization_sha256",
        "execution_source_inventory_sha256",
        "parent_manifest_run_index_sha256",
        "subject_ids",
        "subject_manifest_lineage_by_subject",
        "subject_condition_artifacts",
        "subject_comparison_artifacts",
        "aggregate_artifacts",
        "partial_outcome_inspection_allowed",
        "scientific_decisions_allowed",
        "result_values_in_index",
    }
    if set(index) != required_index_fields:
        raise FullExecutionError("full-run index fields drifted")
    if index["schema_id"] != "neuroai-m1-full-run-index-v1":
        raise FullExecutionError("unexpected full-run index schema")
    if any(
        index[field] is not False
        for field in (
            "partial_outcome_inspection_allowed",
            "scientific_decisions_allowed",
            "result_values_in_index",
        )
    ):
        raise FullExecutionError("full-run index outcome boundary drifted")
    for field, expected in expected_global_bindings.items():
        if index.get(field) != expected:
            raise FullExecutionError(f"full-run global binding mismatch: {field}")
    subjects = index["subject_ids"]
    if subjects != sorted(subjects) or len(subjects) != len(set(subjects)):
        raise FullExecutionError("full-run subject ids are not canonical")

    expected_parent_sha256s = {
        name: manifest_sha256(dict(value), dict(manifest_contract))
        for name, value in parent_manifests.items()
        if name
        in {
            "trial-ocular-removed",
            "inner-fold-ocular-removed",
            "trial-ocular-kept",
            "inner-fold-ocular-kept",
            "cross-condition-intersection",
        }
    }
    condition_outputs: dict[str, dict[str, Any]] = {}
    verified_child_indexes: dict[str, str] = {}
    parent_sha256s = validate_lineage_parent_manifests(
        parent_manifests, manifest_contract
    )
    for subject_id in subjects:
        _, expected_lineage = build_subject_manifest_lineage(
            parent_manifests,
            subject_id,
            manifest_contract,
            validated_parent_sha256s=parent_sha256s,
        )
        if index["subject_manifest_lineage_by_subject"].get(subject_id) != expected_lineage:
            raise FullExecutionError(f"subject lineage mismatch: {subject_id}")
        if expected_lineage["parent_manifest_sha256s"] != expected_parent_sha256s:
            raise FullExecutionError(f"parent lineage mismatch: {subject_id}")
        for condition in ("ocular-removed", "ocular-kept"):
            key = f"{subject_id}::{condition}"
            locator = index["subject_condition_artifacts"].get(key)
            if not isinstance(locator, Mapping):
                raise FullExecutionError(f"missing subject-condition locator: {key}")
            child_root = root / locator["relative_path"]
            child = load_canonical_json_expected(
                child_root / "index.json", locator["index_sha256"]
            )
            verified_child_indexes[key] = locator["index_sha256"]
            trial_manifest = load_manifest_bundle(
                child_root / "trial-manifest", dict(manifest_contract)
            )
            fold_manifest = load_manifest_bundle(
                child_root / "inner-fold-manifest", dict(manifest_contract)
            )
            if (
                manifest_sha256(trial_manifest, dict(manifest_contract))
                != child["trial_manifest_sha256"]
                or manifest_sha256(fold_manifest, dict(manifest_contract))
                != child["inner_fold_manifest_sha256"]
            ):
                raise FullExecutionError(f"subject manifest digest mismatch: {key}")
            if (
                child["parent_trial_manifest_sha256"]
                != expected_parent_sha256s[f"trial-{condition}"]
                or child["parent_inner_fold_manifest_sha256"]
                != expected_parent_sha256s[f"inner-fold-{condition}"]
            ):
                raise FullExecutionError(f"subject parent manifest mismatch: {key}")
            normalization = load_canonical_json_expected(
                child_root / "normalization-provenance.json",
                child["normalization_provenance_sha256"],
            )
            normalization_bindings = {
                field: normalization[field]
                for field in feature_contract["normalization_transform_provenance"][
                    "external_binding_fields"
                ]
            }
            observed_normalization_sha256 = normalization_transform_provenance_sha256(
                normalization,
                charter,
                feature_contract,
                manifest_contract,
                trial_manifest=trial_manifest,
                inner_fold_manifest=fold_manifest,
                expected_bindings=normalization_bindings,
            )
            if observed_normalization_sha256 != child["normalization_provenance_sha256"]:
                raise FullExecutionError(f"normalization digest mismatch: {key}")
            survivor_rows = [
                row for row in trial_manifest["rows"] if row["condition_survivor"]
            ]
            train_keys = sorted(
                canonical_trial_key(row)
                for row in survivor_rows
                if row["outer_role"] == "train"
            )
            all_keys = sorted(canonical_trial_key(row) for row in survivor_rows)
            feature_tables: dict[str, dict[str, Any]] = {}
            for partition_id, feature_index in child["feature_artifacts"].items():
                expected_keys = (
                    all_keys if partition_id == "outer__refit" else train_keys
                )
                table = load_feature_table_bundle(
                    child_root / "features" / partition_id,
                    charter,
                    feature_contract,
                    expected_keys=expected_keys,
                    expected_bindings=child["feature_bindings"],
                    expected_bundle_sha256=feature_index["bundle_sha256"],
                )
                observed = feature_table_sha256(
                    table,
                    charter,
                    feature_contract,
                    expected_keys=expected_keys,
                    expected_bindings=child["feature_bindings"],
                )
                if observed != feature_index["logical_sha256"]:
                    raise FullExecutionError(f"feature digest mismatch: {key}/{partition_id}")
                feature_tables[partition_id] = table
            d44 = load_canonical_json_expected(
                child_root / "d44-c-cross-fit-provenance.json",
                child["d44_c_cross_fit_provenance_sha256"],
            )
            observed_d44 = d44_c_cross_fit_provenance_sha256(
                trial_manifest,
                fold_manifest,
                d44,
                charter,
                dict(manifest_contract),
            )
            if observed_d44 != child["d44_c_cross_fit_provenance_sha256"]:
                raise FullExecutionError(f"D44-C digest mismatch: {key}")
            load_canonical_json_expected(
                child_root / "inner-selection-evidence.json",
                child["inner_selection_evidence_sha256"],
            )
            prediction_table = load_prediction_table_bundle(
                child_root / "predictions",
                trial_manifest,
                charter,
                feature_contract,
                manifest_contract,
                prediction_contract,
                expected_bindings=child["prediction_bindings"],
                expected_bundle_sha256=child["prediction_bundle_sha256"],
            )
            observed_prediction_sha256 = prediction_table_sha256(
                prediction_table,
                trial_manifest,
                charter,
                feature_contract,
                manifest_contract,
                prediction_contract,
                expected_bindings=child["prediction_bindings"],
            )
            if observed_prediction_sha256 != child["prediction_table_sha256"]:
                raise FullExecutionError(f"prediction digest mismatch: {key}")
            subject_record = load_subject_evaluation_bundle(
                child_root / "subject-evaluation",
                prediction_table,
                trial_manifest,
                charter,
                feature_contract,
                manifest_contract,
                prediction_contract,
                evaluation_contract,
                expected_bindings=child["prediction_bindings"],
                expected_evaluation_provenance=child["evaluation_provenance"],
                expected_bundle_sha256=child["subject_evaluation_bundle_sha256"],
            )
            if (
                subject_record_content_sha256(subject_record, evaluation_contract)
                != child["subject_evaluation_sha256"]
            ):
                raise FullExecutionError(f"subject evaluation digest mismatch: {key}")
            condition_outputs[key] = {
                "trial_manifest": trial_manifest,
                "prediction_table": prediction_table,
                "prediction_bindings": child["prediction_bindings"],
                "subject_record": subject_record,
                "index": child,
            }

    comparison_outputs: dict[str, dict[str, Any]] = {}
    for subject_id in subjects:
        locator = index["subject_comparison_artifacts"].get(subject_id)
        if not isinstance(locator, Mapping):
            raise FullExecutionError(f"missing comparison locator: {subject_id}")
        child_root = root / locator["relative_path"]
        child = load_canonical_json_expected(
            child_root / "index.json", locator["index_sha256"]
        )
        verified_child_indexes[f"{subject_id}::comparison"] = locator["index_sha256"]
        intersection = load_manifest_bundle(
            child_root / "intersection-manifest", dict(manifest_contract)
        )
        if (
            manifest_sha256(intersection, dict(manifest_contract))
            != child["intersection_manifest_sha256"]
            or child["parent_intersection_manifest_sha256"]
            != expected_parent_sha256s["cross-condition-intersection"]
        ):
            raise FullExecutionError(f"intersection lineage mismatch: {subject_id}")
        removed = condition_outputs[f"{subject_id}::ocular-removed"]
        kept = condition_outputs[f"{subject_id}::ocular-kept"]
        prediction_digests = child["prediction_table_sha256_by_condition"]
        comparison_table = load_condition_comparison_bundle(
            child_root / "comparison",
            removed["prediction_table"],
            kept["prediction_table"],
            removed["trial_manifest"],
            kept["trial_manifest"],
            intersection,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_removed_bindings=removed["prediction_bindings"],
            expected_kept_bindings=kept["prediction_bindings"],
            expected_intersection_manifest_sha256=child["intersection_manifest_sha256"],
            expected_prediction_table_sha256_by_condition=prediction_digests,
            expected_comparison_provenance=child["comparison_provenance"],
            expected_bundle_sha256=child["comparison_bundle_sha256"],
        )
        observed_comparison_sha256 = condition_comparison_table_sha256(
            comparison_table,
            removed["prediction_table"],
            kept["prediction_table"],
            removed["trial_manifest"],
            kept["trial_manifest"],
            intersection,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            expected_removed_bindings=removed["prediction_bindings"],
            expected_kept_bindings=kept["prediction_bindings"],
            expected_intersection_manifest_sha256=child["intersection_manifest_sha256"],
            expected_prediction_table_sha256_by_condition=prediction_digests,
            expected_comparison_provenance=child["comparison_provenance"],
        )
        if observed_comparison_sha256 != child["comparison_table_sha256"]:
            raise FullExecutionError(f"comparison digest mismatch: {subject_id}")
        subject_record = load_common_intersection_subject_evaluation_bundle(
            child_root / "subject-evaluation",
            comparison_table,
            removed["prediction_table"],
            kept["prediction_table"],
            removed["trial_manifest"],
            kept["trial_manifest"],
            intersection,
            charter,
            feature_contract,
            manifest_contract,
            prediction_contract,
            evaluation_contract,
            expected_removed_bindings=removed["prediction_bindings"],
            expected_kept_bindings=kept["prediction_bindings"],
            expected_intersection_manifest_sha256=child["intersection_manifest_sha256"],
            expected_prediction_table_sha256_by_condition=prediction_digests,
            expected_comparison_provenance=child["comparison_provenance"],
            expected_evaluation_provenance=child["evaluation_provenance"],
            expected_bundle_sha256=child["subject_evaluation_bundle_sha256"],
        )
        if (
            common_intersection_subject_content_sha256(
                subject_record, evaluation_contract
            )
            != child["subject_evaluation_sha256"]
        ):
            raise FullExecutionError(f"comparison evaluation digest mismatch: {subject_id}")
        comparison_outputs[subject_id] = {
            "subject_record": subject_record,
            "index": child,
        }

    aggregate_locator = index["aggregate_artifacts"]
    aggregate_root = root / aggregate_locator["relative_path"]
    aggregate_index = load_canonical_json_expected(
        aggregate_root / "index.json", aggregate_locator["index_sha256"]
    )
    verified_child_indexes["aggregates"] = aggregate_locator["index_sha256"]
    primary_records = [
        condition_outputs[f"{subject_id}::ocular-removed"]["subject_record"]
        for subject_id in subjects
    ]
    common_records = [comparison_outputs[subject_id]["subject_record"] for subject_id in subjects]
    load_primary_aggregate_bundle(
        aggregate_root / "primary",
        primary_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=index["charter_sha256"],
        expected_record_sha256_by_subject=aggregate_index[
            "primary_subject_record_sha256s"
        ],
        expected_aggregate_provenance=aggregate_index["aggregate_provenance"],
        expected_bundle_sha256=aggregate_index["primary_bundle_sha256"],
    )
    load_common_intersection_aggregate_bundle(
        aggregate_root / "common-intersection",
        common_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=index["charter_sha256"],
        expected_record_sha256_by_subject=aggregate_index[
            "common_intersection_subject_record_sha256s"
        ],
        expected_aggregate_provenance=aggregate_index["aggregate_provenance"],
        expected_bundle_sha256=aggregate_index["common_intersection_bundle_sha256"],
    )
    receipt = {
        "schema_id": "neuroai-m1-independent-verification-receipt-v1",
        "schema_version": "1.0",
        "verified_at_utc": verified_at_utc,
        "verifier_code_commit": verifier_code_commit,
        "verification_command": verification_command,
        "full_run_index_sha256": expected_run_index_sha256,
        "verified_subject_count": len(subjects),
        "verified_subject_condition_count": len(condition_outputs),
        "verified_child_index_sha256s": dict(sorted(verified_child_indexes.items())),
        "primary_aggregate_bundle_sha256": aggregate_index["primary_bundle_sha256"],
        "common_intersection_aggregate_bundle_sha256": aggregate_index[
            "common_intersection_bundle_sha256"
        ],
        "all_stored_contracts_revalidated": True,
        "downstream_comparisons_and_results_recomputed": True,
        "source_features_and_predictions_regenerated": False,
        "publication_complete": True,
        "result_values_in_receipt": False,
        "interpretation_performed": False,
    }
    receipt_sha256 = publish_canonical_json_noreplace(verification_output, receipt)
    return receipt, receipt_sha256
