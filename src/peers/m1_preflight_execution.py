"""Real, outcome-blind workload preparation for the bounded M1 preflight."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from peers.m1_feature_generation import (
    build_raw_cache_metadata,
    combine_training_session_statistics,
    compact_features_for_sessions,
    extract_locked_epoch_session,
    load_raw_spectral_session_bundle,
    write_raw_spectral_session_bundle_atomic,
)
from peers.m1_features import feature_schema_sha256
from peers.m1_manifests import canonical_document_sha256, manifest_sha256
from peers.m1_model_execution import (
    build_full_behavioral_reference,
    execute_frozen_models_without_outer_scoring,
)
from peers.m1_preflight import (
    build_preflight_forecast,
    candidate_inventory_sha256,
    measurements_sha256,
    preflight_receipt_sha256,
    select_preflight_roles,
    validate_candidate_inventory,
    validate_preflight_budget_receipt,
)


CONDITIONS = ("ocular-removed", "ocular-kept")


class PreflightExecutionError(ValueError):
    """Real preflight inputs violate the frozen outcome-blind workload boundary."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _package_identity() -> tuple[dict[str, str], str]:
    names = ("mne", "numpy", "pandas", "scikit-learn", "scipy", "wordfreq")
    versions = {name: importlib.metadata.version(name) for name in names}
    return versions, _canonical_sha256(versions)


def _target_trials(
    manifest: Mapping[str, Any],
    behavioral_reference: pd.DataFrame,
    *,
    subject_id: str,
    condition: str,
) -> pd.DataFrame:
    rows = [
        row
        for row in manifest["rows"]
        if row["subject_id"] == subject_id
        and row["condition"] == condition
        and row["condition_survivor"] is True
    ]
    if not rows:
        raise PreflightExecutionError("selected subject-condition has no survivor rows")
    frame = pd.DataFrame(rows).sort_values(
        ["subject_id", "session", "trial", "item_num"]
    ).reset_index(drop=True)
    lookup = behavioral_reference.loc[
        :, ["subject_id", "session", "trial", "item_num", "item_name"]
    ]
    frame = frame.merge(
        lookup,
        on=["subject_id", "session", "trial", "item_num"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if frame["item_name"].isna().any():
        raise PreflightExecutionError("survivor row lacks full-reference behavioral word")
    return frame


def build_real_candidate_inventory(
    manifests: Mapping[str, Mapping[str, Any]],
    run_index: Mapping[str, Any],
    *,
    data_root: Path,
) -> list[dict[str, Any]]:
    """Derive common eligible workloads from locked survivor rows and source file sizes."""
    required = {
        "cohort-ocular-removed",
        "cohort-ocular-kept",
        "trial-ocular-removed",
        "trial-ocular-kept",
    }
    if not required.issubset(manifests):
        raise PreflightExecutionError(
            f"candidate inventory lacks manifests: {sorted(required - set(manifests))}"
        )
    bundle_sha256s = run_index.get("bundle_sha256s")
    if not isinstance(bundle_sha256s, Mapping):
        raise PreflightExecutionError("candidate inventory lacks run-index bundle digests")

    eligible_by_condition: dict[str, set[str]] = {}
    survivor_rows_by_condition: dict[str, list[Mapping[str, Any]]] = {}
    for condition in CONDITIONS:
        cohort_name = f"cohort-{condition}"
        trial_name = f"trial-{condition}"
        cohort_rows = manifests[cohort_name].get("rows")
        trial_rows = manifests[trial_name].get("rows")
        if not isinstance(cohort_rows, list) or not isinstance(trial_rows, list):
            raise PreflightExecutionError("candidate inventory manifest rows are invalid")
        eligible = {
            row["subject_id"]
            for row in cohort_rows
            if row.get("eligible") is True
            and row.get("eligibility_reason_code") == "ELIGIBLE"
        }
        survivors = [
            row
            for row in trial_rows
            if row.get("condition_survivor") is True
            and row.get("included") is True
            and row.get("exclusion_reason_code") == "INCLUDED"
        ]
        survivor_subjects = {row["subject_id"] for row in survivors}
        if not eligible.issubset(survivor_subjects):
            raise PreflightExecutionError(
                f"eligible {condition} subject lacks survivor rows"
            )
        eligible_by_condition[condition] = eligible
        survivor_rows_by_condition[condition] = survivors
        digest = bundle_sha256s.get(trial_name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise PreflightExecutionError(
                f"candidate inventory lacks {trial_name} digest"
            )

    common_subjects = sorted(set.intersection(*eligible_by_condition.values()))
    if not common_subjects:
        raise PreflightExecutionError("candidate inventory has no common eligible subjects")
    root = Path(data_root)
    rows: list[dict[str, Any]] = []
    for subject_id in common_subjects:
        counts: dict[str, int] = {}
        input_bytes: dict[str, int] = {}
        manifest_hashes: dict[str, str] = {}
        for condition in CONDITIONS:
            subject_rows = [
                row
                for row in survivor_rows_by_condition[condition]
                if row["subject_id"] == subject_id
            ]
            if not subject_rows:
                raise PreflightExecutionError(
                    f"candidate {subject_id} lacks {condition} survivor rows"
                )
            derivative_ids = sorted({row["derivative_id"] for row in subject_rows})
            size = 0
            for derivative_id in derivative_ids:
                path = root / derivative_id
                if path.is_symlink() or not path.is_file():
                    raise PreflightExecutionError(
                        f"candidate input is not a regular file: {derivative_id}"
                    )
                size += path.stat().st_size
            counts[condition] = len(subject_rows)
            input_bytes[condition] = size
            manifest_hashes[condition] = bundle_sha256s[f"trial-{condition}"]
        rows.append(
            {
                "subject_id": subject_id,
                "eligible_trials_by_condition": counts,
                "input_bytes_by_condition": input_bytes,
                "trial_manifest_sha256_by_condition": manifest_hashes,
                "total_eligible_trials": sum(counts.values()),
                "total_input_bytes": sum(input_bytes.values()),
            }
        )
    try:
        return validate_candidate_inventory(rows)
    except ValueError as exc:
        raise PreflightExecutionError(str(exc)) from exc


def summarize_real_candidate_inventory(
    manifests: Mapping[str, Mapping[str, Any]],
    run_index: Mapping[str, Any],
    *,
    data_root: Path,
) -> dict[str, Any]:
    """Return the authenticated workload digest and frozen selected roles, never outcomes."""
    inventory = build_real_candidate_inventory(manifests, run_index, data_root=data_root)
    return {
        "candidate_inventory": inventory,
        "candidate_inventory_sha256": candidate_inventory_sha256(inventory),
        "selected_roles": select_preflight_roles(inventory),
        "outcome_access": False,
    }


def execution_source_inventory(
    manifest_source_inventory: Mapping[str, Any],
    behavioral_reference_inventory: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Bind the EEG manifest sources and the broader B3 reference as one input identity."""
    document = {
        "schema_id": "neuroai-m1-execution-source-inventory-v1",
        "schema_version": "1.0",
        "m1_manifest_source_inventory_sha256": canonical_document_sha256(
            manifest_source_inventory
        ),
        "behavioral_reference_inventory_sha256": _canonical_sha256(
            behavioral_reference_inventory
        ),
        "behavioral_reference_trial_count": behavioral_reference_inventory["trial_count"],
        "behavioral_reference_subject_count": behavioral_reference_inventory["subject_count"],
    }
    return document, _canonical_sha256(document)


def _publish_canonical_file_noreplace(path: Path, document: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(document)
    digest = sha256(payload).hexdigest()
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            os.unlink(temporary)
        raise
    return digest


def execute_subject_condition_preflight(
    *,
    subject_id: str,
    condition: str,
    roles: list[str],
    candidate_row: Mapping[str, Any],
    behavioral_reference: pd.DataFrame,
    behavioral_reference_inventory_sha256: str,
    manifest_source_inventory_sha256: str,
    execution_source_inventory_sha256: str,
    trial_manifest: Mapping[str, Any],
    inner_fold_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    roi_map: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    code_commit: str,
    uv_lock_sha256: str,
    parallel_worker_count: int,
    generator_command: str,
) -> tuple[dict[str, Any], Path]:
    """Run one measured subject-condition path and publish only unscored evidence."""
    if condition not in CONDITIONS or parallel_worker_count <= 0:
        raise PreflightExecutionError("invalid condition or parallel worker count")
    run_root = Path(output_root) / subject_id / condition
    if os.path.lexists(run_root):
        raise PreflightExecutionError(f"preflight destination already exists: {run_root}")
    run_root.mkdir(parents=True)
    disk_before = shutil.disk_usage(output_root).free
    started = time.perf_counter()
    stage_seconds: dict[str, float] = {}

    stage = time.perf_counter()
    trials = _target_trials(
        trial_manifest,
        behavioral_reference,
        subject_id=subject_id,
        condition=condition,
    )
    trial_digest = manifest_sha256(dict(trial_manifest), dict(manifest_contract))
    fold_digest = manifest_sha256(dict(inner_fold_manifest), dict(manifest_contract))
    charter_digest = canonical_document_sha256(charter)
    roi_digest = canonical_document_sha256(roi_map)
    schema_digest = feature_schema_sha256(charter, feature_contract)
    package_versions, package_digest = _package_identity()
    stage_seconds["load_validate_inputs"] = max(time.perf_counter() - stage, 1e-9)

    stage = time.perf_counter()
    summaries = []
    cache_records = []
    for session in sorted(int(value) for value in trials["session"].unique()):
        session_rows = [
            row
            for row in trial_manifest["rows"]
            if row["subject_id"] == subject_id
            and row["condition"] == condition
            and row["session"] == session
            and row["condition_survivor"] is True
        ]
        derivative_id = session_rows[0]["derivative_id"]
        summary = extract_locked_epoch_session(
            Path(data_root) / derivative_id,
            session_rows,
            data_root=Path(data_root),
            charter=charter,
            n_jobs=parallel_worker_count,
        )
        bindings = {
            "trial_manifest_sha256": trial_digest,
            "source_inventory_sha256": execution_source_inventory_sha256,
            "charter_sha256": charter_digest,
            "roi_map_sha256": roi_digest,
            "feature_schema_sha256": schema_digest,
            "uv_lock_sha256": uv_lock_sha256,
            "package_versions_sha256": package_digest,
            "code_commit": code_commit,
            "dirty_diff_sha256": None,
        }
        metadata = build_raw_cache_metadata(
            summary,
            created_at_utc=_utc_now(),
            bindings=bindings,
            generator_command=generator_command,
            python_version=platform.python_version(),
            mne_version=package_versions["mne"],
            numpy_version=package_versions["numpy"],
            frequencies_hz=charter["features"]["frequencies_hz"],
            n_cycles=[value / 2.0 for value in charter["features"]["frequencies_hz"]],
            decim=charter["features"]["decim"],
            parallel_worker_count=parallel_worker_count,
            channel_order=[
                channel
                for roi in charter["features"]["roi_order"]
                for channel in charter["features"]["roi_channels"][roi]
            ],
            time_bin_ids=charter["features"]["time_bin_ids"],
        )
        bundle_path = run_root / f"raw-session-{session}"
        bundle_digest = write_raw_spectral_session_bundle_atomic(
            bundle_path, summary, metadata, charter, feature_contract
        )
        summaries.append(summary)
        cache_records.append(
            {
                "session": session,
                "relative_path": bundle_path.relative_to(output_root).as_posix(),
                "bundle_sha256": bundle_digest,
                "metadata_bindings": {
                    "derivative_id": summary.derivative_id,
                    "derivative_sha256": summary.derivative_sha256,
                    **bindings,
                },
            }
        )
    stage_seconds["build_validate_feature_tables"] = max(
        time.perf_counter() - stage, 1e-9
    )

    stage = time.perf_counter()
    def matrix(fit_sessions: Any, apply_sessions: Any):
        statistics = combine_training_session_statistics(
            summaries, fit_sessions=fit_sessions, charter=charter
        )
        return compact_features_for_sessions(
            summaries, statistics, apply_sessions=apply_sessions, charter=charter
        )

    model_result = execute_frozen_models_without_outer_scoring(
        trials,
        behavioral_reference,
        charter=charter,
        m1_matrix=matrix,
    )
    stage_seconds["fit_predict_without_scoring"] = max(
        time.perf_counter() - stage, 1e-9
    )

    stage = time.perf_counter()
    for record in cache_records:
        load_raw_spectral_session_bundle(
            output_root / record["relative_path"],
            charter,
            feature_contract,
            expected_bundle_sha256=record["bundle_sha256"],
            expected_metadata_bindings=record["metadata_bindings"],
        )
    execution_evidence = {
        "schema_id": "neuroai-m1-unscored-preflight-artifact-set-v1",
        "schema_version": "1.0",
        "subject_id": subject_id,
        "condition": condition,
        "code_commit": code_commit,
        "charter_sha256": charter_digest,
        "trial_manifest_sha256": trial_digest,
        "inner_fold_manifest_sha256": fold_digest,
        "manifest_source_inventory_sha256": manifest_source_inventory_sha256,
        "behavioral_reference_inventory_sha256": behavioral_reference_inventory_sha256,
        "execution_source_inventory_sha256": execution_source_inventory_sha256,
        "feature_schema_sha256": schema_digest,
        "uv_lock_sha256": uv_lock_sha256,
        "package_versions_sha256": package_digest,
        "raw_session_caches": cache_records,
        "selected_m1_c": model_result.selected_m1_c,
        "inner_selection_evidence_sha256": _canonical_sha256(
            dict(model_result.inner_selection_auc_by_c)
        ),
        "model_artifact_sha256": dict(model_result.model_artifact_sha256),
        "transform_chain_sha256": dict(model_result.transform_chain_sha256),
        "selected_hyperparameters_sha256": dict(
            model_result.selected_hyperparameters_sha256
        ),
        "held_prediction_arrays_sha256": {
            "B3": sha256(model_result.prob_b3.tobytes(order="C")).hexdigest(),
            "M1": sha256(model_result.prob_m1.tobytes(order="C")).hexdigest(),
            "M3": sha256(model_result.prob_m3.tobytes(order="C")).hexdigest(),
        },
        "held_prediction_values_stored": False,
        "outcome_metrics_computed": False,
        "held_out_outcomes_inspected": False,
        "interpretation_performed": False,
    }
    artifact_path = run_root / "artifact-set.json"
    artifact_digest = _publish_canonical_file_noreplace(
        artifact_path, execution_evidence
    )
    if artifact_path.read_bytes() != _canonical_bytes(execution_evidence):
        raise PreflightExecutionError("unscored artifact-set readback mismatch")
    stage_seconds["publish_readback_artifacts"] = max(
        time.perf_counter() - stage, 1e-9
    )
    total = max(time.perf_counter() - started, max(stage_seconds.values()))
    cache_bytes = _directory_size(run_root)
    disk_after = shutil.disk_usage(output_root).free
    return (
        {
            "subject_id": subject_id,
            "roles": roles,
            "condition": condition,
            "stage_wall_seconds": {
                key: round(value, 6) for key, value in stage_seconds.items()
            },
            "total_wall_seconds": round(total, 6),
            "peak_resident_memory_bytes": _peak_rss_bytes(),
            "input_bytes": candidate_row["input_bytes_by_condition"][condition],
            "cache_bytes_written": cache_bytes,
            "cache_artifact_set_sha256": artifact_digest,
            "cache_entries_atomic_hash_valid": True,
            "disk_available_bytes_before": disk_before,
            "disk_available_bytes_after": disk_after,
            "parallel_worker_count": parallel_worker_count,
            "longest_stage_wall_seconds": round(max(stage_seconds.values()), 6),
            "longest_no_progress_seconds": round(max(stage_seconds.values()), 6),
            "transient_retry_count": 0,
        },
        artifact_path,
    )


def run_real_preflight_and_write_receipt(
    *,
    manifests: Mapping[str, Mapping[str, Any]],
    run_index: Mapping[str, Any],
    manifest_source_inventory: Mapping[str, Any],
    charter: Mapping[str, Any],
    roi_map: Mapping[str, Any],
    protocol_teach_back_receipt: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    preflight_contract: Mapping[str, Any],
    data_root: Path,
    output_root: Path,
    code_commit: str,
    uv_lock_sha256: str,
    parallel_worker_count: int,
    host_safe_memory_floor_bytes: int,
    generator_command: str,
) -> tuple[dict[str, Any], str]:
    """Run the frozen two-subject preflight and publish a validated capacity receipt."""
    root = Path(output_root)
    if os.path.lexists(root):
        raise PreflightExecutionError(f"preflight output already exists: {root}")
    root.mkdir(parents=True)
    started_at = _utc_now()
    shared_started = time.perf_counter()
    inventory = build_real_candidate_inventory(manifests, run_index, data_root=data_root)
    roles = select_preflight_roles(inventory)
    behavioral, behavioral_inventory, behavioral_digest = build_full_behavioral_reference(
        data_root
    )
    execution_inventory, execution_inventory_digest = execution_source_inventory(
        manifest_source_inventory, behavioral_inventory
    )
    _publish_canonical_file_noreplace(
        root / "behavioral-reference-inventory.json", behavioral_inventory
    )
    _publish_canonical_file_noreplace(
        root / "execution-source-inventory.json", execution_inventory
    )
    shared_seconds = max(time.perf_counter() - shared_started, 1e-9)

    selected_subjects = sorted(set(roles.values()))
    by_subject = {row["subject_id"]: row for row in inventory}
    role_names: dict[str, list[str]] = {}
    for role in ("median", "maximum"):
        role_names.setdefault(roles[f"{role}_subject_id"], []).append(role)
    measurements = []
    for subject_id in selected_subjects:
        for condition in CONDITIONS:
            measurement, _ = execute_subject_condition_preflight(
                subject_id=subject_id,
                condition=condition,
                roles=role_names[subject_id],
                candidate_row=by_subject[subject_id],
                behavioral_reference=behavioral,
                behavioral_reference_inventory_sha256=behavioral_digest,
                manifest_source_inventory_sha256=canonical_document_sha256(
                    manifest_source_inventory
                ),
                execution_source_inventory_sha256=execution_inventory_digest,
                trial_manifest=manifests[f"trial-{condition}"],
                inner_fold_manifest=manifests[f"inner-fold-{condition}"],
                charter=charter,
                roi_map=roi_map,
                manifest_contract=manifest_contract,
                feature_contract=feature_contract,
                data_root=data_root,
                output_root=root,
                code_commit=code_commit,
                uv_lock_sha256=uv_lock_sha256,
                parallel_worker_count=parallel_worker_count,
                generator_command=generator_command,
            )
            measurements.append(measurement)
    measurements = sorted(
        measurements,
        key=lambda row: (row["subject_id"], CONDITIONS.index(row["condition"])),
    )
    first = measurements[0]
    first["stage_wall_seconds"]["load_validate_inputs"] = round(
        first["stage_wall_seconds"]["load_validate_inputs"] + shared_seconds, 6
    )
    first["total_wall_seconds"] = round(
        first["total_wall_seconds"] + shared_seconds, 6
    )
    first["longest_stage_wall_seconds"] = round(
        max(first["stage_wall_seconds"].values()), 6
    )
    first["longest_no_progress_seconds"] = first["longest_stage_wall_seconds"]
    forecast = build_preflight_forecast(
        inventory,
        measurements,
        host_safe_memory_floor_bytes=host_safe_memory_floor_bytes,
    )
    receipt = {
        "schema_id": "neuroai-m1-preflight-budget-receipt-v1",
        "schema_version": "1.0",
        "receipt_id": f"preflight-{started_at}",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "charter_id": charter["charter_id"],
        "charter_version": charter["version"],
        "charter_sha256": canonical_document_sha256(charter),
        "execution_gate_contract_sha256": canonical_document_sha256(
            charter["execution_gates"]
        ),
        "protocol_teach_back_receipt_sha256": canonical_document_sha256(
            protocol_teach_back_receipt
        ),
        "code_commit": code_commit,
        "dirty_diff_sha256": None,
        "environment_lock_sha256": uv_lock_sha256,
        "source_inventory_sha256": execution_inventory_digest,
        "candidate_inventory": inventory,
        "candidate_inventory_sha256": candidate_inventory_sha256(inventory),
        "selected_roles": roles,
        "measurements": measurements,
        "measurements_sha256": measurements_sha256(measurements),
        "forecast": forecast,
        "outcome_access_attestation": {
            "outcome_metrics_computed": False,
            "held_out_outcomes_inspected": False,
            "interpretation_performed": False,
            "attested_by_role": "technical_operator",
        },
    }
    validate_preflight_budget_receipt(
        receipt,
        preflight_contract,
        charter=charter,
        expected_final_charter_sha256=canonical_document_sha256(charter),
        expected_protocol_teach_back_receipt_sha256=canonical_document_sha256(
            protocol_teach_back_receipt
        ),
        expected_code_commit=code_commit,
        expected_dirty_diff_sha256=None,
        expected_environment_lock_sha256=uv_lock_sha256,
        expected_source_inventory_sha256=execution_inventory_digest,
        expected_candidate_inventory=inventory,
    )
    receipt_digest = _publish_canonical_file_noreplace(
        root / "preflight-receipt.json", receipt
    )
    if receipt_digest != preflight_receipt_sha256(receipt):
        raise PreflightExecutionError("published preflight receipt digest mismatch")
    with (root / "preflight-receipt.sha256").open("xb") as handle:
        handle.write(f"{receipt_digest}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
    return receipt, receipt_digest
