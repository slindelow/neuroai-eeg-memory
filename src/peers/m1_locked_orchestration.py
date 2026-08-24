"""Single-process, fail-closed orchestration for an authorized NeuroAI M1 run.

This module is intentionally silent about result values.  It enforces the human and
capacity authorization supplied by the CLI, runs one subject at a time, publishes
only authenticated artifacts, and returns the run-index digest for an independent
verification process.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any, Callable, Mapping

from peers.m1_artifact_execution import (
    build_subject_manifest_lineage,
    validate_lineage_parent_manifests,
)
from peers.m1_feature_generation import (
    combine_training_session_statistics,
    compact_features_for_sessions,
    extract_locked_epoch_session,
)
from peers.m1_full_execution import (
    FullExecutionError,
    canonical_sha256,
    publish_aggregate_artifacts,
    publish_canonical_json_noreplace,
    publish_full_run_index,
    publish_subject_comparison_artifacts,
    publish_subject_condition_artifacts,
)
from peers.m1_manifests import canonical_document_sha256, manifest_sha256
from peers.m1_model_execution import execute_frozen_models_without_outer_scoring
from peers.m1_preflight import capacity_receipt_sha256, preflight_receipt_sha256
from peers.m1_preflight_execution import (
    CONDITIONS,
    _package_identity,
    _target_trials,
    build_real_candidate_inventory,
    execution_source_inventory,
)


class LockedRunError(FullExecutionError):
    """The authorized run exceeded its immutable operational contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_utc(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise LockedRunError("authorization contains an invalid deadline") from exc


def _check_runtime_budget(
    *,
    output_root: Path,
    authorization: Mapping[str, Any],
    available_memory_bytes: Callable[[], int],
) -> None:
    budget = authorization["budget"]
    if datetime.now(timezone.utc) > _parse_utc(authorization["effective_deadline_utc"]):
        raise LockedRunError("full-run effective deadline passed")
    if shutil.disk_usage(output_root).free < budget["minimum_disk_available_bytes"]:
        raise LockedRunError("full-run disk headroom fell below the authorized floor")
    if available_memory_bytes() < budget["host_safe_memory_floor_bytes"]:
        raise LockedRunError("full-run available memory fell below the authorized floor")


def _require_stage_time(
    started: float,
    authorization: Mapping[str, Any],
    stage: str,
) -> None:
    elapsed = time.perf_counter() - started
    if elapsed > authorization["budget"]["no_progress_abort_seconds"]:
        raise LockedRunError(
            f"{stage} exceeded the authorized no-progress interval"
        )


def _extract_summaries(
    *,
    subject_id: str,
    condition: str,
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    data_root: Path,
    workers: int,
    authorization: Mapping[str, Any],
) -> list[Any]:
    survivor_rows = [
        row
        for row in trial_manifest["rows"]
        if row["subject_id"] == subject_id
        and row["condition"] == condition
        and row["condition_survivor"] is True
    ]
    summaries = []
    for session in sorted({int(row["session"]) for row in survivor_rows}):
        session_rows = [row for row in survivor_rows if row["session"] == session]
        derivative_ids = {row["derivative_id"] for row in session_rows}
        if len(derivative_ids) != 1:
            raise LockedRunError("one session resolved to multiple derivative files")
        started = time.perf_counter()
        summaries.append(
            extract_locked_epoch_session(
                Path(data_root) / next(iter(derivative_ids)),
                session_rows,
                data_root=Path(data_root),
                charter=charter,
                n_jobs=workers,
            )
        )
        _require_stage_time(started, authorization, "single-session spectral extraction")
    return summaries


def run_authorized_full_cohort(
    *,
    output_root: Path,
    data_root: Path,
    manifests: Mapping[str, Mapping[str, Any]],
    manifest_run_index: Mapping[str, Any],
    manifest_run_index_sha256: str,
    manifest_source_inventory: Mapping[str, Any],
    behavioral_reference: Any,
    behavioral_reference_inventory: Mapping[str, Any],
    behavioral_reference_inventory_sha256: str,
    authorization: Mapping[str, Any],
    preflight_receipt: Mapping[str, Any],
    capacity_receipt: Mapping[str, Any],
    charter: Mapping[str, Any],
    roi_map: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    uv_lock_sha256: str,
    workers: int,
    generator_command: str,
    available_memory_bytes: Callable[[], int],
) -> tuple[dict[str, Any], str]:
    """Execute the exact eligible cohort and return only its metric-free index receipt."""
    root = Path(output_root)
    if os.path.lexists(root):
        raise LockedRunError(f"full-run output already exists: {root}")
    root.mkdir(parents=True)
    code_commit = authorization["code_commit"]
    created_at_utc = utc_now()
    package_versions, package_versions_sha256 = _package_identity()
    execution_inventory, execution_inventory_sha256 = execution_source_inventory(
        manifest_source_inventory, behavioral_reference_inventory
    )
    if behavioral_reference_inventory_sha256 != canonical_sha256(
        behavioral_reference_inventory
    ):
        raise LockedRunError("behavioral reference inventory digest mismatch")
    inventory = build_real_candidate_inventory(
        manifests, manifest_run_index, data_root=Path(data_root)
    )
    expected_subjects = [row["subject_id"] for row in inventory]
    if expected_subjects != sorted(expected_subjects):
        raise LockedRunError("candidate subject order is not canonical")
    parent_manifest_sha256s = validate_lineage_parent_manifests(
        manifests, manifest_contract
    )
    authorization_sha256 = canonical_sha256(authorization)
    publish_canonical_json_noreplace(root / "authorization.json", authorization)
    publish_canonical_json_noreplace(
        root / "behavioral-reference-inventory.json", behavioral_reference_inventory
    )
    publish_canonical_json_noreplace(
        root / "execution-source-inventory.json", execution_inventory
    )

    condition_outputs = []
    comparison_outputs = []
    lineages: dict[str, Mapping[str, Any]] = {}
    try:
        for subject_id in expected_subjects:
            _check_runtime_budget(
                output_root=root,
                authorization=authorization,
                available_memory_bytes=available_memory_bytes,
            )
            subject_started = time.perf_counter()
            slices, lineage = build_subject_manifest_lineage(
                manifests,
                subject_id,
                manifest_contract,
                validated_parent_sha256s=parent_manifest_sha256s,
            )
            lineages[subject_id] = lineage
            by_condition: dict[str, Mapping[str, Any]] = {}
            for condition in CONDITIONS:
                parent_trial = manifests[f"trial-{condition}"]
                parent_fold = manifests[f"inner-fold-{condition}"]
                subject_trial = slices[f"trial-{condition}"]
                subject_fold = slices[f"inner-fold-{condition}"]
                trials = _target_trials(
                    parent_trial,
                    behavioral_reference,
                    subject_id=subject_id,
                    condition=condition,
                )
                summaries = _extract_summaries(
                    subject_id=subject_id,
                    condition=condition,
                    trial_manifest=parent_trial,
                    charter=charter,
                    data_root=Path(data_root),
                    workers=workers,
                    authorization=authorization,
                )

                def matrix(fit_sessions: Any, apply_sessions: Any):
                    statistics = combine_training_session_statistics(
                        summaries, fit_sessions=fit_sessions, charter=charter
                    )
                    return compact_features_for_sessions(
                        summaries,
                        statistics,
                        apply_sessions=apply_sessions,
                        charter=charter,
                    )

                model_started = time.perf_counter()
                model_result = execute_frozen_models_without_outer_scoring(
                    trials,
                    behavioral_reference,
                    charter=charter,
                    m1_matrix=matrix,
                )
                _require_stage_time(
                    model_started, authorization, "subject-condition model fit"
                )
                publication_started = time.perf_counter()
                output = publish_subject_condition_artifacts(
                    output_root=root / "subjects" / subject_id / condition,
                    summaries=summaries,
                    model_result=model_result,
                    trial_manifest=subject_trial,
                    inner_fold_manifest=subject_fold,
                    parent_trial_manifest_sha256=manifest_sha256(
                        dict(parent_trial), dict(manifest_contract)
                    ),
                    parent_inner_fold_manifest_sha256=manifest_sha256(
                        dict(parent_fold), dict(manifest_contract)
                    ),
                    charter=charter,
                    feature_contract=feature_contract,
                    manifest_contract=manifest_contract,
                    prediction_contract=prediction_contract,
                    evaluation_contract=evaluation_contract,
                    source_inventory_sha256=execution_inventory_sha256,
                    charter_sha256=canonical_document_sha256(charter),
                    roi_map_sha256=canonical_document_sha256(roi_map),
                    uv_lock_sha256=uv_lock_sha256,
                    package_versions_sha256=package_versions_sha256,
                    code_commit=code_commit,
                    created_at_utc=created_at_utc,
                    generator_command=generator_command,
                    python_version=platform.python_version(),
                )
                _require_stage_time(
                    publication_started,
                    authorization,
                    "subject-condition artifact publication",
                )
                condition_outputs.append(output)
                by_condition[condition] = output
            comparison = publish_subject_comparison_artifacts(
                output_root=root / "subjects" / subject_id / "comparison",
                ocular_removed=by_condition["ocular-removed"],
                ocular_kept=by_condition["ocular-kept"],
                intersection_manifest=slices["cross-condition-intersection"],
                parent_intersection_manifest_sha256=manifest_sha256(
                    dict(manifests["cross-condition-intersection"]),
                    dict(manifest_contract),
                ),
                charter=charter,
                feature_contract=feature_contract,
                manifest_contract=manifest_contract,
                prediction_contract=prediction_contract,
                evaluation_contract=evaluation_contract,
                uv_lock_sha256=uv_lock_sha256,
                package_versions_sha256=package_versions_sha256,
                code_commit=code_commit,
                created_at_utc=created_at_utc,
                generator_command=generator_command,
                python_version=platform.python_version(),
            )
            comparison_outputs.append(comparison)
            if (
                time.perf_counter() - subject_started
                > authorization["budget"]["per_subject_timeout_seconds"]
            ):
                raise LockedRunError("subject exceeded the authorized timeout")

        aggregates = publish_aggregate_artifacts(
            output_root=root / "aggregates",
            subject_condition_outputs=condition_outputs,
            subject_comparison_outputs=comparison_outputs,
            charter=charter,
            evaluation_contract=evaluation_contract,
            charter_sha256=canonical_document_sha256(charter),
            package_versions_sha256=package_versions_sha256,
            code_commit=code_commit,
            generator_command=generator_command,
        )
        return publish_full_run_index(
            output_root=root,
            subject_condition_outputs=condition_outputs,
            subject_comparison_outputs=comparison_outputs,
            aggregate_output=aggregates,
            created_at_utc=created_at_utc,
            code_commit=code_commit,
            charter_sha256=canonical_document_sha256(charter),
            roi_map_sha256=canonical_document_sha256(roi_map),
            manifest_contract_sha256=canonical_document_sha256(manifest_contract),
            feature_contract_sha256=canonical_document_sha256(feature_contract),
            prediction_contract_sha256=canonical_document_sha256(prediction_contract),
            evaluation_contract_sha256=canonical_document_sha256(evaluation_contract),
            preflight_receipt_sha256=preflight_receipt_sha256(preflight_receipt),
            capacity_receipt_sha256=capacity_receipt_sha256(capacity_receipt),
            full_run_authorization_sha256=authorization_sha256,
            execution_source_inventory_sha256=execution_inventory_sha256,
            parent_manifest_run_index_sha256=manifest_run_index_sha256,
            subject_manifest_lineage_by_subject=lineages,
            generator_command=generator_command,
        )
    except Exception as exc:
        failure = {
            "schema_id": "neuroai-m1-full-run-failure-v1",
            "schema_version": "1.0",
            "recorded_at_utc": utc_now(),
            "code_commit": code_commit,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "completed_subject_condition_count": len(condition_outputs),
            "completed_comparison_count": len(comparison_outputs),
            "partial_outcome_inspection_performed": False,
            "interpretation_performed": False,
            "failure_state": "IN_PROGRESS",
        }
        if not os.path.lexists(root / "failure.json"):
            publish_canonical_json_noreplace(root / "failure.json", failure)
        raise
