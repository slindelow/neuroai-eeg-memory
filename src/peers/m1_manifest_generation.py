"""Outcome-blind extraction and immutable storage for the real M1 manifests.

This module reads behavioral event identities and cached epoch metadata only to lock the
analysis cohort, trial rows, session folds, processing-condition intersection, and attrition.
It never extracts M1 features, fits a learner, or computes prediction performance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd

from peers.events import build_trial_table, load_events
from peers.m1_manifests import (
    ManifestValidationError,
    _fsync_directory,
    _rename_directory_noreplace,
    _write_fsynced,
    assert_scientific_contract_ready,
    canonical_document_sha256,
    canonical_row_sha256,
    canonical_trial_key,
    canonical_trial_key_set_sha256,
    event_provenance_sha256,
    load_json_unique,
    load_manifest_bundle,
    manifest_sha256,
    validate_cross_condition_intersection,
    validate_manifest,
    write_manifest_bundle_atomic,
)


CONDITIONS = ("ocular-removed", "ocular-kept")
DERIVATIVE_RE = re.compile(
    r"^(?P<subject>sub-LTP\d+)_ses-(?P<session>\d+)_"
    r"(?P<condition>ocular-(?:removed|kept))_epo\.fif$"
)
EXPECTED_CHANNEL_NAMES = tuple(
    [f"E{index}" for index in range(1, 129) if index not in {8, 25, 126, 127}]
    + ["Cz"]
)
RUN_FIXED_FILES = frozenset(
    {"source-inventory.json", "source-inventory.sha256", "index.json", "index.sha256"}
)
RUN_BUNDLE_NAMES = (
    "cohort-ocular-kept",
    "cohort-ocular-removed",
    "inner-fold-ocular-kept",
    "inner-fold-ocular-removed",
    "trial-ocular-kept",
    "trial-ocular-removed",
    "cross-condition-intersection",
    "attrition",
)
RUN_INDEX_FIELDS = frozenset(
    {
        "schema_id",
        "schema_version",
        "created_at_utc",
        "generator_command",
        "code_commit",
        "dirty_diff_sha256",
        "charter_id",
        "charter_version",
        "charter_sha256",
        "roi_map_id",
        "roi_map_sha256",
        "manifest_contract_id",
        "manifest_contract_version",
        "manifest_contract_sha256",
        "source_inventory_sha256",
        "bundle_sha256s",
    }
)


class RealManifestGenerationError(ManifestValidationError):
    """Fail-closed real-source or cross-document error."""


def _canonical_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(path: Path, data_root: Path) -> str:
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise RealManifestGenerationError(
            f"source path is outside the declared data root: {path}"
        ) from exc


def _event_session(path: Path) -> int:
    match = re.fullmatch(r"ses-(\d+)", path.parent.name)
    if match is None:
        raise RealManifestGenerationError(f"invalid event session directory: {path}")
    return int(match.group(1))


def _derivative_identity(path: Path) -> tuple[str, int, str]:
    match = DERIVATIVE_RE.fullmatch(path.name)
    if match is None:
        raise RealManifestGenerationError(f"invalid derivative filename: {path}")
    return (
        match.group("subject"),
        int(match.group("session")),
        match.group("condition"),
    )


def _class_counts(labels: Iterable[Any]) -> dict[str, int]:
    values = np.asarray(list(labels), dtype=bool)
    return {"0": int((~values).sum()), "1": int(values.sum())}


def _has_both_classes(counts: Mapping[str, int]) -> bool:
    return counts.get("0", 0) > 0 and counts.get("1", 0) > 0


def _channel_schema_id(channel_names: Sequence[str]) -> str:
    digest = sha256(_canonical_bytes({"channel_names": list(channel_names)})).hexdigest()
    return f"hydrocel-125-ordered-{digest[:16]}"


def _git_identity(repo_root: Path) -> tuple[str, None]:
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RealManifestGenerationError(
            "real manifest generation requires a clean committed repository"
        )
    return commit, None


def _load_and_verify_readiness(repo_root: Path) -> tuple[dict[str, Any], ...]:
    """Load the public scientific contracts required for manifest generation.

    The authoritative private run also bound a human teach-back receipt and decision-log digest.
    Those personal execution receipts are deliberately excluded from the public companion repo.
    """
    docs = repo_root / "docs"
    charter = load_json_unique(docs / "analysis-charter-m1-v1.json")
    roi_map = load_json_unique(docs / "roi-map-m1-v1.json")
    contract = load_json_unique(docs / "m1-manifest-contracts.json")
    try:
        assert_scientific_contract_ready(
            charter,
            roi_map,
            contract,
            final_charter_sha256=canonical_document_sha256(charter),
        )
    except ManifestValidationError as error:
        raise RealManifestGenerationError(
            f"public scientific manifest contract is not ready: {error}"
        ) from error
    if not roi_map["status"].startswith("frozen"):
        raise RealManifestGenerationError("ROI map must be frozen")
    return charter, roi_map, contract


def _manifest_envelope(
    kind: str,
    rows: list[dict[str, Any]],
    *,
    created_at_utc: str,
    charter: Mapping[str, Any],
    charter_sha256: str,
    roi_map: Mapping[str, Any],
    roi_map_sha256: str,
    contract: Mapping[str, Any],
    code_commit: str,
    generator_command: str,
    source_inventory_sha256: str,
) -> dict[str, Any]:
    schema = contract["manifest_schemas"][kind]
    key_fields = schema["row_key"]
    return {
        "schema_id": schema["schema_id"],
        "schema_version": schema["schema_version"],
        "manifest_kind": kind,
        "created_at_utc": created_at_utc,
        "charter_id": charter["charter_id"],
        "charter_version": charter["version"],
        "charter_sha256": charter_sha256,
        "roi_map_id": roi_map["map_id"],
        "roi_map_sha256": roi_map_sha256,
        "code_commit": code_commit,
        "dirty_diff_sha256": None,
        "generator_command": generator_command,
        "source_inventory_sha256": source_inventory_sha256,
        "rows": sorted(rows, key=lambda row: tuple(row[field] for field in key_fields)),
    }


def _load_candidate_sources(
    data_root: Path,
) -> tuple[
    list[str],
    dict[tuple[str, int], Path],
    dict[tuple[str, int, str], Path],
]:
    derivatives_root = data_root / "derivatives"
    derivative_paths = sorted(derivatives_root.glob("sub-LTP*/*_epo.fif"))
    if not derivative_paths:
        raise RealManifestGenerationError(f"no cached epoch derivatives under {derivatives_root}")
    derivatives: dict[tuple[str, int, str], Path] = {}
    candidate_subjects: set[str] = set()
    for path in derivative_paths:
        identity = _derivative_identity(path)
        if identity in derivatives:
            raise RealManifestGenerationError(f"duplicate derivative identity: {identity}")
        derivatives[identity] = path
        candidate_subjects.add(identity[0])

    events: dict[tuple[str, int], Path] = {}
    for subject in sorted(candidate_subjects):
        paths = sorted((data_root / "peers" / subject).glob("ses-*/*_events.tsv"))
        if not paths:
            raise RealManifestGenerationError(f"candidate subject has no event files: {subject}")
        for path in paths:
            identity = (subject, _event_session(path))
            if identity in events:
                raise RealManifestGenerationError(f"duplicate event identity: {identity}")
            events[identity] = path
    return sorted(candidate_subjects), events, derivatives


def _build_source_inventory(
    data_root: Path,
    event_paths: Mapping[tuple[str, int], Path],
    derivative_paths: Mapping[tuple[str, int, str], Path],
) -> tuple[dict[str, Any], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for (subject, session), path in sorted(event_paths.items()):
        source_id = _source_id(path, data_root)
        digest = _sha256_file(path)
        hashes[source_id] = digest
        rows.append(
            {
                "source_id": source_id,
                "source_kind": "behavioral_events_tsv",
                "subject_id": subject,
                "session": session,
                "condition": None,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    for (subject, session, condition), path in sorted(derivative_paths.items()):
        source_id = _source_id(path, data_root)
        digest = _sha256_file(path)
        hashes[source_id] = digest
        rows.append(
            {
                "source_id": source_id,
                "source_kind": "cached_mne_epochs_fif",
                "subject_id": subject,
                "session": session,
                "condition": condition,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    inventory = {
        "schema_id": "neuroai-m1-source-inventory-v1",
        "schema_version": "1.0",
        "hash_algorithm": "sha256",
        "files": sorted(rows, key=lambda row: row["source_id"]),
    }
    validate_source_inventory(inventory)
    return inventory, hashes


def validate_source_inventory(
    inventory: Mapping[str, Any], data_root: Path | None = None
) -> None:
    expected_fields = {"schema_id", "schema_version", "hash_algorithm", "files"}
    if not isinstance(inventory, Mapping) or set(inventory) != expected_fields:
        raise RealManifestGenerationError("source inventory field set mismatch")
    if inventory["schema_id"] != "neuroai-m1-source-inventory-v1":
        raise RealManifestGenerationError("source inventory schema_id mismatch")
    if inventory["schema_version"] != "1.0" or inventory["hash_algorithm"] != "sha256":
        raise RealManifestGenerationError("source inventory version or hash algorithm mismatch")
    files = inventory["files"]
    if not isinstance(files, list) or not files:
        raise RealManifestGenerationError("source inventory files must be a nonempty array")
    expected_row_fields = {
        "source_id",
        "source_kind",
        "subject_id",
        "session",
        "condition",
        "size_bytes",
        "sha256",
    }
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise RealManifestGenerationError("source inventory row field set mismatch")
        source_id = row["source_id"]
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id.startswith("/")
            or ".." in Path(source_id).parts
            or source_id in seen
        ):
            raise RealManifestGenerationError(f"invalid or duplicate source_id: {source_id}")
        seen.add(source_id)
        if row["source_kind"] not in {"behavioral_events_tsv", "cached_mne_epochs_fif"}:
            raise RealManifestGenerationError("unknown source inventory kind")
        if not isinstance(row["subject_id"], str) or not row["subject_id"].startswith("sub-LTP"):
            raise RealManifestGenerationError("invalid source inventory subject")
        if isinstance(row["session"], bool) or not isinstance(row["session"], int):
            raise RealManifestGenerationError("invalid source inventory session")
        if row["condition"] not in {None, *CONDITIONS}:
            raise RealManifestGenerationError("invalid source inventory condition")
        if isinstance(row["size_bytes"], bool) or not isinstance(row["size_bytes"], int):
            raise RealManifestGenerationError("invalid source inventory size")
        if row["size_bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise RealManifestGenerationError("invalid source inventory size or SHA-256")
        if data_root is not None:
            path = data_root / source_id
            if not path.is_file() or path.is_symlink():
                raise RealManifestGenerationError(
                    f"source inventory file missing or unsafe: {path}"
                )
            if path.stat().st_size != row["size_bytes"] or _sha256_file(path) != row["sha256"]:
                raise RealManifestGenerationError(f"source inventory file drift: {source_id}")
    if files != sorted(files, key=lambda row: row["source_id"]):
        raise RealManifestGenerationError("source inventory is not canonically ordered")


def _load_behavioral_sessions(
    candidate_subjects: Sequence[str],
    event_paths: Mapping[tuple[str, int], Path],
) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[tuple[str, int], pd.DataFrame]]:
    raw_by_session: dict[tuple[str, int], pd.DataFrame] = {}
    trials_by_session: dict[tuple[str, int], pd.DataFrame] = {}
    for identity, path in sorted(event_paths.items()):
        subject, session = identity
        raw = load_events(path)
        observed_subjects = set(raw["subject_id"].astype(str))
        observed_sessions = set(raw["session"].astype(int))
        if observed_subjects != {subject} or observed_sessions != {session}:
            raise RealManifestGenerationError(f"event identity mismatch: {path}")
        trials = build_trial_table(raw)
        if trials.empty or (trials["item_num"] < 1).any():
            raise RealManifestGenerationError(f"invalid aligned trial rows: {path}")
        key_fields = ["subject_id", "session", "trial", "item_num"]
        if trials.duplicated(key_fields).any():
            raise RealManifestGenerationError(f"duplicate behavioral trial key: {path}")
        raw_by_session[identity] = raw
        trials_by_session[identity] = trials
    missing = sorted(set(candidate_subjects) - {key[0] for key in raw_by_session})
    if missing:
        raise RealManifestGenerationError(f"candidate subjects missing behavior: {missing}")
    return raw_by_session, trials_by_session


def _inspect_derivative(
    path: Path,
    behavioral_trials: pd.DataFrame,
    *,
    expected_channel_names: Sequence[str],
) -> tuple[pd.DataFrame, int]:
    try:
        epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    except Exception as exc:
        raise RealManifestGenerationError(f"UNREADABLE_METADATA: {path}") from exc
    if list(epochs.ch_names) != list(expected_channel_names):
        raise RealManifestGenerationError(f"CHANNEL_SCHEMA_MISMATCH: {path}")
    data = epochs.get_data(copy=False)
    if not np.isfinite(data).all():
        raise RealManifestGenerationError(f"NONFINITE_INPUT: {path}")
    metadata = epochs.metadata
    required = {
        "subject_id",
        "session",
        "trial",
        "serial_position",
        "item_name",
        "item_num",
        "onset_s",
        "recalled",
    }
    if metadata is None or not required.issubset(metadata.columns) or len(metadata) != len(epochs):
        raise RealManifestGenerationError(f"UNREADABLE_METADATA: {path}")
    metadata = metadata.loc[:, sorted(required)].reset_index(drop=True)
    keys = ["subject_id", "session", "trial", "item_num"]
    if metadata.duplicated(keys).any():
        raise RealManifestGenerationError(f"TRIAL_EPOCH_KEY_MISMATCH: {path}")
    behavioral = behavioral_trials.set_index(keys, drop=False)
    metadata_indexed = metadata.set_index(keys, drop=False)
    if not metadata_indexed.index.isin(behavioral.index).all():
        raise RealManifestGenerationError(f"TRIAL_EPOCH_KEY_MISMATCH: {path}")
    aligned = behavioral.loc[metadata_indexed.index].reset_index(drop=True)
    comparisons = (
        metadata["serial_position"].astype(int).to_numpy()
        == aligned["serial_position"].astype(int).to_numpy()
    )
    comparisons &= metadata["item_name"].astype(str).to_numpy() == aligned["item_name"].astype(
        str
    ).to_numpy()
    comparisons &= metadata["recalled"].astype(bool).to_numpy() == aligned["recalled"].astype(
        bool
    ).to_numpy()
    comparisons &= np.isclose(
        metadata["onset_s"].astype(float).to_numpy(),
        aligned["onset_s"].astype(float).to_numpy(),
        rtol=0.0,
        atol=1e-9,
    )
    if not comparisons.all():
        raise RealManifestGenerationError(f"TRIAL_EPOCH_KEY_MISMATCH: {path}")
    if not np.array_equal(
        epochs.events[:, 2].astype(bool), metadata["recalled"].astype(bool).to_numpy()
    ):
        raise RealManifestGenerationError(f"TRIAL_EPOCH_KEY_MISMATCH: {path}")
    dropped = len(behavioral_trials) - len(metadata)
    if dropped < 0:
        raise RealManifestGenerationError(f"TRIAL_EPOCH_KEY_MISMATCH: {path}")
    return metadata, dropped


def _cohort_reason(
    sessions: Sequence[int], session_labels: Mapping[int, Sequence[bool]]
) -> tuple[str, list[int], int | None]:
    if len(sessions) < 3:
        return "TOO_FEW_SESSIONS", [], None
    outer_test = max(sessions)
    outer_train = [session for session in sessions if session != outer_test]
    train_counts = _class_counts(
        label for session in outer_train for label in session_labels[session]
    )
    if not _has_both_classes(train_counts):
        return "SINGLE_CLASS_OUTER_TRAIN", outer_train, outer_test
    if not _has_both_classes(_class_counts(session_labels[outer_test])):
        return "SINGLE_CLASS_OUTER_TEST", outer_train, outer_test
    for validation_session in outer_train:
        fit_sessions = [session for session in outer_train if session != validation_session]
        fit_counts = _class_counts(
            label for session in fit_sessions for label in session_labels[session]
        )
        validation_counts = _class_counts(session_labels[validation_session])
        if not _has_both_classes(fit_counts) or not _has_both_classes(validation_counts):
            return "SINGLE_CLASS_INNER_FOLD", outer_train, outer_test
    return "ELIGIBLE", outer_train, outer_test


def build_real_manifest_documents(
    *,
    data_root: Path,
    charter: Mapping[str, Any],
    roi_map: Mapping[str, Any],
    contract: Mapping[str, Any],
    code_commit: str,
    created_at_utc: str,
    generator_command: str,
    expected_channel_names: Sequence[str] = EXPECTED_CHANNEL_NAMES,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build all real manifest documents in memory without fitting or scoring a model."""
    candidate_subjects, event_paths, derivative_paths = _load_candidate_sources(data_root)
    inventory, source_hashes = _build_source_inventory(
        data_root, event_paths, derivative_paths
    )
    source_inventory_sha256 = canonical_document_sha256(inventory)
    raw_by_session, trials_by_session = _load_behavioral_sessions(
        candidate_subjects, event_paths
    )
    charter_sha256 = canonical_document_sha256(charter)
    roi_map_sha256 = canonical_document_sha256(roi_map)
    channel_schema_id = _channel_schema_id(expected_channel_names)

    metadata_by_session_condition: dict[tuple[str, int, str], pd.DataFrame] = {}
    dropped_by_session_condition: dict[tuple[str, int, str], int] = {}
    for identity, path in sorted(derivative_paths.items()):
        subject, session, condition = identity
        behavioral = trials_by_session.get((subject, session))
        if behavioral is None:
            raise RealManifestGenerationError(f"derivative has no event source: {path}")
        metadata, dropped = _inspect_derivative(
            path, behavioral, expected_channel_names=expected_channel_names
        )
        metadata_by_session_condition[identity] = metadata
        dropped_by_session_condition[identity] = dropped

    cohort_rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    cohort_by_condition_subject: dict[tuple[str, str], dict[str, Any]] = {}
    trial_rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }
    inner_rows_by_condition: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in CONDITIONS
    }

    for condition in CONDITIONS:
        for subject in candidate_subjects:
            sessions_present = sorted(
                session for candidate, session in event_paths if candidate == subject
            )
            technically_eligible = sorted(
                session
                for candidate, session, candidate_condition in metadata_by_session_condition
                if candidate == subject and candidate_condition == condition
            )
            session_labels = {
                session: metadata_by_session_condition[(subject, session, condition)][
                    "recalled"
                ].astype(bool).tolist()
                for session in technically_eligible
            }
            reason, outer_train, outer_test = _cohort_reason(
                technically_eligible, session_labels
            )
            eligible = reason == "ELIGIBLE"
            per_session_counts: dict[str, dict[str, Any]] = {}
            row_sources: dict[str, str] = {}
            for session in sessions_present:
                event_path = event_paths[(subject, session)]
                event_id = _source_id(event_path, data_root)
                row_sources[event_id] = source_hashes[event_id]
                behavioral_count = len(trials_by_session[(subject, session)])
                derivative = derivative_paths.get((subject, session, condition))
                if derivative is None:
                    per_session_counts[str(session)] = {
                        "behavioral_aligned": behavioral_count,
                        "matched_epochs": 0,
                        "autoreject_dropped": 0,
                        "class_0": 0,
                        "class_1": 0,
                        "technical_status": "MISSING_DERIVATIVE",
                    }
                    continue
                derivative_id = _source_id(derivative, data_root)
                row_sources[derivative_id] = source_hashes[derivative_id]
                metadata = metadata_by_session_condition[(subject, session, condition)]
                counts = _class_counts(metadata["recalled"].astype(bool))
                per_session_counts[str(session)] = {
                    "behavioral_aligned": behavioral_count,
                    "matched_epochs": len(metadata),
                    "autoreject_dropped": dropped_by_session_condition[
                        (subject, session, condition)
                    ],
                    "class_0": counts["0"],
                    "class_1": counts["1"],
                    "technical_status": "ELIGIBLE",
                }
            cohort_row = {
                "subject_id": subject,
                "condition": condition,
                "sessions_present": sessions_present,
                "sessions_technically_eligible": technically_eligible,
                "outer_train_sessions": outer_train,
                "outer_test_session": outer_test,
                "eligible": eligible,
                "eligibility_reason_code": reason,
                "per_session_counts": per_session_counts,
                "source_sha256s": dict(sorted(row_sources.items())),
                "channel_schema_id": channel_schema_id,
            }
            cohort_rows_by_condition[condition].append(cohort_row)
            cohort_by_condition_subject[(condition, subject)] = cohort_row

            for session in technically_eligible:
                metadata = metadata_by_session_condition[(subject, session, condition)]
                event_path = event_paths[(subject, session)]
                derivative = derivative_paths[(subject, session, condition)]
                event_id = _source_id(event_path, data_root)
                derivative_id = _source_id(derivative, data_root)
                if eligible:
                    outer_role = "test" if session == outer_test else "train"
                else:
                    outer_role = "excluded"
                for epoch_row, record in metadata.iterrows():
                    trial_rows_by_condition[condition].append(
                        {
                            "subject_id": str(record["subject_id"]),
                            "session": int(record["session"]),
                            "trial": int(record["trial"]),
                            "item_num": int(record["item_num"]),
                            "condition": condition,
                            "serial_position": int(record["serial_position"]),
                            "list_number": int(record["trial"]),
                            "onset_s": float(record["onset_s"]),
                            "recalled": bool(record["recalled"]),
                            "source_events_id": event_id,
                            "source_events_sha256": source_hashes[event_id],
                            "derivative_id": derivative_id,
                            "derivative_sha256": source_hashes[derivative_id],
                            "epoch_row": int(epoch_row),
                            "included": True,
                            "exclusion_reason_code": "INCLUDED",
                            "condition_survivor": eligible,
                            "outer_role": outer_role,
                        }
                    )

            if eligible:
                outer_training_rows = [
                    row
                    for row in trial_rows_by_condition[condition]
                    if row["subject_id"] == subject and row["outer_role"] == "train"
                ]
                for validation_session in outer_train:
                    validation_rows = [
                        row
                        for row in outer_training_rows
                        if row["session"] == validation_session
                    ]
                    fit_sessions = [
                        session for session in outer_train if session != validation_session
                    ]
                    fit_rows = [
                        row for row in outer_training_rows if row["session"] in fit_sessions
                    ]
                    inner_rows_by_condition[condition].append(
                        {
                            "subject_id": subject,
                            "condition": condition,
                            "inner_fold_id": f"validation-ses-{validation_session}",
                            "fit_sessions": fit_sessions,
                            "validation_session": validation_session,
                            "outer_test_session": outer_test,
                            "fit_class_counts": _class_counts(
                                row["recalled"] for row in fit_rows
                            ),
                            "validation_class_counts": _class_counts(
                                row["recalled"] for row in validation_rows
                            ),
                            "fit_keys_sha256": canonical_trial_key_set_sha256(
                                canonical_trial_key(row) for row in fit_rows
                            ),
                            "validation_keys_sha256": canonical_trial_key_set_sha256(
                                canonical_trial_key(row) for row in validation_rows
                            ),
                            "eligible": True,
                            "feasibility_reason_code": "ELIGIBLE",
                        }
                    )

    manifests: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        manifests[f"cohort-{condition}"] = _manifest_envelope(
            "cohort",
            cohort_rows_by_condition[condition],
            created_at_utc=created_at_utc,
            charter=charter,
            charter_sha256=charter_sha256,
            roi_map=roi_map,
            roi_map_sha256=roi_map_sha256,
            contract=contract,
            code_commit=code_commit,
            generator_command=generator_command,
            source_inventory_sha256=source_inventory_sha256,
        )
        manifests[f"trial-{condition}"] = _manifest_envelope(
            "trial",
            trial_rows_by_condition[condition],
            created_at_utc=created_at_utc,
            charter=charter,
            charter_sha256=charter_sha256,
            roi_map=roi_map,
            roi_map_sha256=roi_map_sha256,
            contract=contract,
            code_commit=code_commit,
            generator_command=generator_command,
            source_inventory_sha256=source_inventory_sha256,
        )
        manifests[f"inner-fold-{condition}"] = _manifest_envelope(
            "inner_fold",
            inner_rows_by_condition[condition],
            created_at_utc=created_at_utc,
            charter=charter,
            charter_sha256=charter_sha256,
            roi_map=roi_map,
            roi_map_sha256=roi_map_sha256,
            contract=contract,
            code_commit=code_commit,
            generator_command=generator_command,
            source_inventory_sha256=source_inventory_sha256,
        )

    removed_by_key = {
        canonical_trial_key(row): row
        for row in trial_rows_by_condition["ocular-removed"]
        if row["condition_survivor"]
    }
    kept_by_key = {
        canonical_trial_key(row): row
        for row in trial_rows_by_condition["ocular-kept"]
        if row["condition_survivor"]
    }
    intersection_rows = []
    for key in sorted(set(removed_by_key) & set(kept_by_key)):
        removed = removed_by_key[key]
        kept = kept_by_key[key]
        if removed["recalled"] != kept["recalled"]:
            raise RealManifestGenerationError(f"cross-condition label disagreement: {key}")
        intersection_rows.append(
            {
                "subject_id": key[0],
                "session": key[1],
                "trial": key[2],
                "item_num": key[3],
                "ocular_removed_row_sha256": canonical_row_sha256(removed),
                "ocular_kept_row_sha256": canonical_row_sha256(kept),
                "recalled": removed["recalled"],
                "event_provenance_sha256": event_provenance_sha256(removed),
                "ocular_removed_epoch_row": removed["epoch_row"],
                "ocular_kept_epoch_row": kept["epoch_row"],
            }
        )
    manifests["cross-condition-intersection"] = _manifest_envelope(
        "cross_condition_intersection",
        intersection_rows,
        created_at_utc=created_at_utc,
        charter=charter,
        charter_sha256=charter_sha256,
        roi_map=roi_map,
        roi_map_sha256=roi_map_sha256,
        contract=contract,
        code_commit=code_commit,
        generator_command=generator_command,
        source_inventory_sha256=source_inventory_sha256,
    )

    attrition_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        raw_words = [
            raw[raw["trial_type"].eq("WORD")]
            for raw in raw_by_session.values()
        ]
        raw_word_count = sum(len(frame) for frame in raw_words)
        practice_count = sum(int(frame["trial"].le(0).sum()) for frame in raw_words)
        negative_count = sum(
            int((frame["trial"].gt(0) & frame["sample"].lt(0)).sum())
            for frame in raw_words
        )
        aligned_count = sum(len(frame) for frame in trials_by_session.values())
        stage1_reasons: Counter[str] = Counter()
        for (subject, session), trials in trials_by_session.items():
            if (subject, session, condition) not in derivative_paths:
                stage1_reasons["MISSING_DERIVATIVE"] += len(trials)
        technically_surviving = aligned_count - sum(stage1_reasons.values())
        autoreject_dropped = sum(
            dropped
            for (subject, session, candidate_condition), dropped in (
                dropped_by_session_condition.items()
            )
            if candidate_condition == condition
        )
        matched_count = technically_surviving - autoreject_dropped
        final_survivors = sum(
            1 for row in trial_rows_by_condition[condition] if row["condition_survivor"]
        )
        stage3_reasons: Counter[str] = Counter()
        for cohort_row in cohort_rows_by_condition[condition]:
            if cohort_row["eligible"]:
                continue
            excluded_trials = sum(
                trial_row["subject_id"] == cohort_row["subject_id"]
                for trial_row in trial_rows_by_condition[condition]
            )
            stage3_reasons[cohort_row["eligibility_reason_code"]] += excluded_trials
        attrition_rows.extend(
            [
                {
                    "condition": condition,
                    "flow_id": "trial_analysis",
                    "count_unit": "trial",
                    "stage_index": 0,
                    "stage": "behavioral_event_filter",
                    "input_count": raw_word_count,
                    "excluded_by_reason": {
                        key: value
                        for key, value in (
                            ("PRACTICE_OR_NONPOSITIVE_TRIAL", practice_count),
                            ("NEGATIVE_SAMPLE", negative_count),
                        )
                        if value
                    },
                    "survivor_count": aligned_count,
                },
                {
                    "condition": condition,
                    "flow_id": "trial_analysis",
                    "count_unit": "trial",
                    "stage_index": 1,
                    "stage": "condition_technical_survival",
                    "input_count": aligned_count,
                    "excluded_by_reason": dict(stage1_reasons),
                    "survivor_count": technically_surviving,
                },
                {
                    "condition": condition,
                    "flow_id": "trial_analysis",
                    "count_unit": "trial",
                    "stage_index": 2,
                    "stage": "event_epoch_alignment",
                    "input_count": technically_surviving,
                    "excluded_by_reason": (
                        {"AUTOREJECT_DROPPED": autoreject_dropped}
                        if autoreject_dropped
                        else {}
                    ),
                    "survivor_count": matched_count,
                },
                {
                    "condition": condition,
                    "flow_id": "trial_analysis",
                    "count_unit": "trial",
                    "stage_index": 3,
                    "stage": "eligible_subject_trials",
                    "input_count": matched_count,
                    "excluded_by_reason": dict(stage3_reasons),
                    "survivor_count": final_survivors,
                },
            ]
        )

        cohort_rows = cohort_rows_by_condition[condition]
        too_few = sum(
            row["eligibility_reason_code"] == "TOO_FEW_SESSIONS" for row in cohort_rows
        )
        outer_reasons = Counter(
            row["eligibility_reason_code"]
            for row in cohort_rows
            if row["eligibility_reason_code"]
            in {"SINGLE_CLASS_OUTER_TRAIN", "SINGLE_CLASS_OUTER_TEST"}
        )
        inner_failures = sum(
            row["eligibility_reason_code"] == "SINGLE_CLASS_INNER_FOLD"
            for row in cohort_rows
        )
        after_minimum = len(cohort_rows) - too_few
        after_outer = after_minimum - sum(outer_reasons.values())
        attrition_rows.extend(
            [
                {
                    "condition": condition,
                    "flow_id": "subject_condition_eligibility",
                    "count_unit": "subject_condition",
                    "stage_index": 0,
                    "stage": "candidate_subject_conditions",
                    "input_count": len(cohort_rows),
                    "excluded_by_reason": {},
                    "survivor_count": len(cohort_rows),
                },
                {
                    "condition": condition,
                    "flow_id": "subject_condition_eligibility",
                    "count_unit": "subject_condition",
                    "stage_index": 1,
                    "stage": "technical_session_eligibility",
                    "input_count": len(cohort_rows),
                    "excluded_by_reason": {},
                    "survivor_count": len(cohort_rows),
                },
                {
                    "condition": condition,
                    "flow_id": "subject_condition_eligibility",
                    "count_unit": "subject_condition",
                    "stage_index": 2,
                    "stage": "minimum_complete_sessions",
                    "input_count": len(cohort_rows),
                    "excluded_by_reason": ({"TOO_FEW_SESSIONS": too_few} if too_few else {}),
                    "survivor_count": after_minimum,
                },
                {
                    "condition": condition,
                    "flow_id": "subject_condition_eligibility",
                    "count_unit": "subject_condition",
                    "stage_index": 3,
                    "stage": "outer_partition_class_feasibility",
                    "input_count": after_minimum,
                    "excluded_by_reason": dict(outer_reasons),
                    "survivor_count": after_outer,
                },
                {
                    "condition": condition,
                    "flow_id": "subject_condition_eligibility",
                    "count_unit": "subject_condition",
                    "stage_index": 4,
                    "stage": "inner_fold_class_feasibility",
                    "input_count": after_outer,
                    "excluded_by_reason": (
                        {"SINGLE_CLASS_INNER_FOLD": inner_failures}
                        if inner_failures
                        else {}
                    ),
                    "survivor_count": after_outer - inner_failures,
                },
            ]
        )

        other = CONDITIONS[1] if condition == CONDITIONS[0] else CONDITIONS[0]
        condition_keys = {
            canonical_trial_key(row)
            for row in trial_rows_by_condition[condition]
            if row["condition_survivor"]
        }
        other_keys = {
            canonical_trial_key(row)
            for row in trial_rows_by_condition[other]
            if row["condition_survivor"]
        }
        absent = len(condition_keys - other_keys)
        attrition_rows.append(
            {
                "condition": condition,
                "flow_id": "cross_condition_sensitivity",
                "count_unit": "trial",
                "stage_index": 0,
                "stage": "cross_condition_intersection",
                "input_count": len(condition_keys),
                "excluded_by_reason": ({"ABSENT_FROM_OTHER_CONDITION": absent} if absent else {}),
                "survivor_count": len(condition_keys & other_keys),
            }
        )

    manifests["attrition"] = _manifest_envelope(
        "attrition",
        attrition_rows,
        created_at_utc=created_at_utc,
        charter=charter,
        charter_sha256=charter_sha256,
        roi_map=roi_map,
        roi_map_sha256=roi_map_sha256,
        contract=contract,
        code_commit=code_commit,
        generator_command=generator_command,
        source_inventory_sha256=source_inventory_sha256,
    )
    validate_source_inventory(inventory, data_root=data_root)
    validate_real_manifest_documents(inventory, manifests, contract)
    return inventory, manifests


def validate_real_manifest_documents(
    inventory: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    validate_source_inventory(inventory)
    if set(manifests) != set(RUN_BUNDLE_NAMES):
        raise RealManifestGenerationError("real manifest bundle-name set mismatch")
    inventory_sha256 = canonical_document_sha256(inventory)
    bindings: dict[str, Any] | None = None
    for name in RUN_BUNDLE_NAMES:
        manifest = manifests[name]
        validate_manifest(manifest, contract)
        if manifest["source_inventory_sha256"] != inventory_sha256:
            raise RealManifestGenerationError(f"source inventory binding mismatch: {name}")
        current = {field: manifest[field] for field in contract["binding_fields"]}
        if bindings is None:
            bindings = current
        elif current != bindings:
            raise RealManifestGenerationError(f"cross-manifest binding mismatch: {name}")
    validate_cross_condition_intersection(
        manifests["cross-condition-intersection"],
        manifests["trial-ocular-removed"],
        manifests["trial-ocular-kept"],
        contract,
    )
    _validate_cohort_trial_fold_relationships(manifests)
    _validate_source_references(inventory, manifests)
    _validate_attrition_relationships(manifests)


def _validate_cohort_trial_fold_relationships(
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    for condition in CONDITIONS:
        cohorts = {
            row["subject_id"]: row for row in manifests[f"cohort-{condition}"]["rows"]
        }
        trials = manifests[f"trial-{condition}"]["rows"]
        folds = manifests[f"inner-fold-{condition}"]["rows"]
        if any(row["condition"] != condition for row in (*cohorts.values(), *trials, *folds)):
            raise RealManifestGenerationError(f"condition drift in {condition} manifests")
        by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in trials:
            if row["subject_id"] not in cohorts:
                raise RealManifestGenerationError("trial subject absent from cohort")
            by_subject[row["subject_id"]].append(row)
        folds_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in folds:
            folds_by_subject[row["subject_id"]].append(row)
        for subject, cohort in cohorts.items():
            rows = by_subject.get(subject, [])
            session_counts = cohort["per_session_counts"]
            if set(session_counts) != {str(session) for session in cohort["sessions_present"]}:
                raise RealManifestGenerationError("cohort per-session count key drift")
            derived_eligible_sessions = []
            for session in cohort["sessions_present"]:
                counts = session_counts[str(session)]
                expected_fields = {
                    "behavioral_aligned",
                    "matched_epochs",
                    "autoreject_dropped",
                    "class_0",
                    "class_1",
                    "technical_status",
                }
                if not isinstance(counts, Mapping) or set(counts) != expected_fields:
                    raise RealManifestGenerationError("cohort per-session field drift")
                numeric = [
                    counts[field]
                    for field in (
                        "behavioral_aligned",
                        "matched_epochs",
                        "autoreject_dropped",
                        "class_0",
                        "class_1",
                    )
                ]
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in numeric
                ):
                    raise RealManifestGenerationError("cohort per-session count is invalid")
                session_rows = [row for row in rows if row["session"] == session]
                if counts["matched_epochs"] != len(session_rows):
                    raise RealManifestGenerationError("cohort matched-epoch count drift")
                if counts["class_0"] + counts["class_1"] != counts["matched_epochs"]:
                    raise RealManifestGenerationError("cohort class-count total drift")
                if set(row["epoch_row"] for row in session_rows) != set(
                    range(counts["matched_epochs"])
                ):
                    raise RealManifestGenerationError("trial epoch-row coverage drift")
                if counts["technical_status"] == "ELIGIBLE":
                    derived_eligible_sessions.append(session)
                    if (
                        counts["matched_epochs"] + counts["autoreject_dropped"]
                        != counts["behavioral_aligned"]
                    ):
                        raise RealManifestGenerationError("AutoReject attrition count drift")
                elif counts["technical_status"] == "MISSING_DERIVATIVE":
                    if any(
                        counts[field]
                        for field in (
                            "matched_epochs",
                            "autoreject_dropped",
                            "class_0",
                            "class_1",
                        )
                    ):
                        raise RealManifestGenerationError("missing derivative has epoch counts")
                else:
                    raise RealManifestGenerationError("unknown per-session technical status")
            if derived_eligible_sessions != cohort["sessions_technically_eligible"]:
                raise RealManifestGenerationError("cohort eligible-session derivation drift")
            survivor_rows = [row for row in rows if row["condition_survivor"]]
            if cohort["eligible"]:
                expected_roles = {
                    session: (
                        "test" if session == cohort["outer_test_session"] else "train"
                    )
                    for session in cohort["sessions_technically_eligible"]
                }
                if any(row["outer_role"] != expected_roles[row["session"]] for row in rows):
                    raise RealManifestGenerationError("eligible trial outer-role drift")
                if len(survivor_rows) != len(rows):
                    raise RealManifestGenerationError("eligible cohort has non-survivor trial")
                subject_folds = folds_by_subject.get(subject, [])
                if len(subject_folds) != len(cohort["outer_train_sessions"]):
                    raise RealManifestGenerationError("inner-fold count does not match cohort")
                for fold in subject_folds:
                    fit_rows = [
                        row
                        for row in rows
                        if row["outer_role"] == "train"
                        and row["session"] in fold["fit_sessions"]
                    ]
                    validation_rows = [
                        row
                        for row in rows
                        if row["outer_role"] == "train"
                        and row["session"] == fold["validation_session"]
                    ]
                    if fold["fit_keys_sha256"] != canonical_trial_key_set_sha256(
                        canonical_trial_key(row) for row in fit_rows
                    ) or fold["validation_keys_sha256"] != canonical_trial_key_set_sha256(
                        canonical_trial_key(row) for row in validation_rows
                    ):
                        raise RealManifestGenerationError("inner-fold key digest drift")
                    if fold["fit_class_counts"] != _class_counts(
                        row["recalled"] for row in fit_rows
                    ) or fold["validation_class_counts"] != _class_counts(
                        row["recalled"] for row in validation_rows
                    ):
                        raise RealManifestGenerationError("inner-fold class-count drift")
            else:
                if survivor_rows or folds_by_subject.get(subject):
                    raise RealManifestGenerationError("ineligible cohort has survivors or folds")
                if any(row["outer_role"] != "excluded" for row in rows):
                    raise RealManifestGenerationError("ineligible trial must be excluded")


def _validate_source_references(
    inventory: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    inventory_by_id = {row["source_id"]: row for row in inventory["files"]}
    for condition in CONDITIONS:
        for cohort in manifests[f"cohort-{condition}"]["rows"]:
            for source_id, digest in cohort["source_sha256s"].items():
                source = inventory_by_id.get(source_id)
                if source is None or source["sha256"] != digest:
                    raise RealManifestGenerationError("cohort source inventory reference drift")
                if source["subject_id"] != cohort["subject_id"]:
                    raise RealManifestGenerationError("cohort source subject drift")
        for row in manifests[f"trial-{condition}"]["rows"]:
            event_source = inventory_by_id.get(row["source_events_id"])
            derivative_source = inventory_by_id.get(row["derivative_id"])
            if (
                event_source is None
                or event_source["source_kind"] != "behavioral_events_tsv"
                or event_source["sha256"] != row["source_events_sha256"]
                or event_source["subject_id"] != row["subject_id"]
                or event_source["session"] != row["session"]
            ):
                raise RealManifestGenerationError("trial event source reference drift")
            if (
                derivative_source is None
                or derivative_source["source_kind"] != "cached_mne_epochs_fif"
                or derivative_source["sha256"] != row["derivative_sha256"]
                or derivative_source["subject_id"] != row["subject_id"]
                or derivative_source["session"] != row["session"]
                or derivative_source["condition"] != condition
            ):
                raise RealManifestGenerationError("trial derivative source reference drift")


def _validate_attrition_relationships(
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    attrition = {
        (row["condition"], row["flow_id"], row["stage_index"]): row
        for row in manifests["attrition"]["rows"]
    }
    intersection_count = len(manifests["cross-condition-intersection"]["rows"])
    for condition in CONDITIONS:
        cohorts = manifests[f"cohort-{condition}"]["rows"]
        trials = manifests[f"trial-{condition}"]["rows"]
        aligned_count = sum(
            counts["behavioral_aligned"]
            for cohort in cohorts
            for counts in cohort["per_session_counts"].values()
        )
        technically_surviving = sum(
            counts["behavioral_aligned"]
            for cohort in cohorts
            for counts in cohort["per_session_counts"].values()
            if counts["technical_status"] == "ELIGIBLE"
        )
        autoreject_dropped = sum(
            counts["autoreject_dropped"]
            for cohort in cohorts
            for counts in cohort["per_session_counts"].values()
        )
        trial_flow = {
            stage: attrition[(condition, "trial_analysis", stage)] for stage in range(4)
        }
        if trial_flow[0]["survivor_count"] != aligned_count:
            raise RealManifestGenerationError("behavioral attrition survivor drift")
        if trial_flow[1]["survivor_count"] != technically_surviving:
            raise RealManifestGenerationError("technical attrition survivor drift")
        if trial_flow[2]["excluded_by_reason"].get("AUTOREJECT_DROPPED", 0) != autoreject_dropped:
            raise RealManifestGenerationError("AutoReject attrition ledger drift")
        if trial_flow[2]["survivor_count"] != len(trials):
            raise RealManifestGenerationError("trial-manifest attrition count drift")
        final_survivors = sum(row["condition_survivor"] for row in trials)
        if trial_flow[3]["survivor_count"] != final_survivors:
            raise RealManifestGenerationError("condition-survivor attrition count drift")

        subject_flow = {
            stage: attrition[(condition, "subject_condition_eligibility", stage)]
            for stage in range(5)
        }
        if subject_flow[0]["input_count"] != len(cohorts):
            raise RealManifestGenerationError("candidate cohort attrition count drift")
        if subject_flow[4]["survivor_count"] != sum(row["eligible"] for row in cohorts):
            raise RealManifestGenerationError("eligible cohort attrition count drift")
        cross = attrition[(condition, "cross_condition_sensitivity", 0)]
        if (
            cross["input_count"] != final_survivors
            or cross["survivor_count"] != intersection_count
        ):
            raise RealManifestGenerationError("cross-condition attrition count drift")


def _run_index(
    *,
    created_at_utc: str,
    generator_command: str,
    code_commit: str,
    charter: Mapping[str, Any],
    roi_map: Mapping[str, Any],
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_id": "neuroai-m1-real-manifest-run-v1",
        "schema_version": "1.0",
        "created_at_utc": created_at_utc,
        "generator_command": generator_command,
        "code_commit": code_commit,
        "dirty_diff_sha256": None,
        "charter_id": charter["charter_id"],
        "charter_version": charter["version"],
        "charter_sha256": canonical_document_sha256(charter),
        "roi_map_id": roi_map["map_id"],
        "roi_map_sha256": canonical_document_sha256(roi_map),
        "manifest_contract_id": contract["contract_id"],
        "manifest_contract_version": contract["version"],
        "manifest_contract_sha256": canonical_document_sha256(contract),
        "source_inventory_sha256": canonical_document_sha256(inventory),
        "bundle_sha256s": {
            name: manifest_sha256(manifests[name], contract) for name in RUN_BUNDLE_NAMES
        },
    }


def validate_run_index(
    index: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    inventory: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(index, Mapping) or set(index) != RUN_INDEX_FIELDS:
        raise RealManifestGenerationError("real manifest run index field set mismatch")
    if (
        index["schema_id"] != "neuroai-m1-real-manifest-run-v1"
        or index["schema_version"] != "1.0"
    ):
        raise RealManifestGenerationError("real manifest run index schema mismatch")
    text_fields = (
        "created_at_utc",
        "generator_command",
        "code_commit",
        "charter_id",
        "charter_version",
        "roi_map_id",
        "manifest_contract_id",
        "manifest_contract_version",
    )
    if any(not isinstance(index[field], str) or not index[field] for field in text_fields):
        raise RealManifestGenerationError("real manifest run index text field is invalid")
    try:
        parsed_created_at = datetime.fromisoformat(
            index["created_at_utc"].removesuffix("Z") + "+00:00"
        )
    except ValueError as exc:
        raise RealManifestGenerationError(
            "real manifest run index timestamp is invalid"
        ) from exc
    if (
        not index["created_at_utc"].endswith("Z")
        or parsed_created_at.utcoffset() != timezone.utc.utcoffset(None)
    ):
        raise RealManifestGenerationError("real manifest run index timestamp must be UTC")
    if not re.fullmatch(r"[0-9a-f]{40}", index["code_commit"]):
        raise RealManifestGenerationError("real manifest run code commit is invalid")
    if index["dirty_diff_sha256"] is not None:
        raise RealManifestGenerationError("real manifest run must bind a clean code state")
    digest_fields = (
        "charter_sha256",
        "roi_map_sha256",
        "manifest_contract_sha256",
        "source_inventory_sha256",
    )
    if any(not re.fullmatch(r"[0-9a-f]{64}", index[field]) for field in digest_fields):
        raise RealManifestGenerationError("real manifest run index digest is invalid")
    bundle_sha256s = index["bundle_sha256s"]
    if not isinstance(bundle_sha256s, Mapping) or set(bundle_sha256s) != set(RUN_BUNDLE_NAMES):
        raise RealManifestGenerationError("real manifest run bundle digest set mismatch")
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in bundle_sha256s.values()):
        raise RealManifestGenerationError("real manifest run bundle digest is invalid")
    expected = {
        "manifest_contract_id": contract["contract_id"],
        "manifest_contract_version": contract["version"],
        "manifest_contract_sha256": canonical_document_sha256(contract),
        "source_inventory_sha256": canonical_document_sha256(inventory),
    }
    for field, value in expected.items():
        if index[field] != value:
            raise RealManifestGenerationError(f"real manifest run index {field} mismatch")
    first = manifests[RUN_BUNDLE_NAMES[0]]
    manifest_bindings = {
        "charter_id": first["charter_id"],
        "charter_version": first["charter_version"],
        "charter_sha256": first["charter_sha256"],
        "roi_map_id": first["roi_map_id"],
        "roi_map_sha256": first["roi_map_sha256"],
        "code_commit": first["code_commit"],
        "dirty_diff_sha256": first["dirty_diff_sha256"],
    }
    for field, value in manifest_bindings.items():
        if index[field] != value:
            raise RealManifestGenerationError(f"real manifest run index {field} binding mismatch")
    for name, manifest in manifests.items():
        if manifest["created_at_utc"] != index["created_at_utc"]:
            raise RealManifestGenerationError(
                f"real manifest run index timestamp binding mismatch: {name}"
            )
        if manifest["generator_command"] != index["generator_command"]:
            raise RealManifestGenerationError(
                f"real manifest run index generator binding mismatch: {name}"
            )
    expected_bundle_sha256s = {
        name: manifest_sha256(dict(manifests[name]), dict(contract))
        for name in RUN_BUNDLE_NAMES
    }
    if dict(bundle_sha256s) != expected_bundle_sha256s:
        raise RealManifestGenerationError("real manifest run index bundle digest mismatch")


def _write_canonical_pair(directory: Path, stem: str, document: Mapping[str, Any]) -> str:
    payload = _canonical_bytes(document)
    digest = sha256(payload).hexdigest()
    _write_fsynced(directory / f"{stem}.json", payload)
    _write_fsynced(directory / f"{stem}.sha256", f"{digest}\n".encode("ascii"))
    return digest


def write_real_manifest_run_atomic(
    destination: Path,
    *,
    inventory: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    index: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> str:
    """Publish the complete run through one immutable native exclusive directory rename."""
    validate_real_manifest_documents(inventory, manifests, contract)
    validate_run_index(
        index, contract=contract, inventory=inventory, manifests=manifests
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RealManifestGenerationError(f"real manifest run already exists: {destination}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        inventory_digest = _write_canonical_pair(staging, "source-inventory", inventory)
        for name in RUN_BUNDLE_NAMES:
            write_manifest_bundle_atomic(staging / name, dict(manifests[name]), dict(contract))
        index_digest = _write_canonical_pair(staging, "index", index)
        if index["source_inventory_sha256"] != inventory_digest:
            raise RealManifestGenerationError("run index source inventory digest mismatch")
        if index["bundle_sha256s"] != {
            name: manifest_sha256(dict(manifests[name]), dict(contract))
            for name in RUN_BUNDLE_NAMES
        }:
            raise RealManifestGenerationError("run index bundle digest mismatch")
        _fsync_directory(staging)
        _rename_directory_noreplace(staging, destination)
        _fsync_directory(destination.parent)
        return index_digest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_canonical_pair(directory: Path, stem: str) -> tuple[dict[str, Any], str]:
    document_path = directory / f"{stem}.json"
    digest_path = directory / f"{stem}.sha256"
    for path in (document_path, digest_path):
        if not path.is_file() or path.is_symlink():
            raise RealManifestGenerationError(f"missing or unsafe run member: {path}")
    document = load_json_unique(document_path)
    payload = _canonical_bytes(document)
    if document_path.read_bytes() != payload:
        raise RealManifestGenerationError(f"noncanonical run member: {document_path}")
    digest = digest_path.read_text(encoding="ascii")
    if not re.fullmatch(r"[0-9a-f]{64}\n", digest):
        raise RealManifestGenerationError(f"invalid run sidecar: {digest_path}")
    actual = sha256(payload).hexdigest()
    if digest.strip() != actual:
        raise RealManifestGenerationError(f"run member digest mismatch: {stem}")
    return document, actual


def load_real_manifest_run(
    directory: Path,
    contract: Mapping[str, Any],
    *,
    expected_index_sha256: str | None = None,
    data_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise RealManifestGenerationError("real manifest run must be a regular directory")
    expected_members = RUN_FIXED_FILES | set(RUN_BUNDLE_NAMES)
    actual_members = {path.name for path in directory.iterdir()}
    if actual_members != expected_members:
        raise RealManifestGenerationError("real manifest run member set mismatch")
    inventory, inventory_digest = _load_canonical_pair(directory, "source-inventory")
    index, index_digest = _load_canonical_pair(directory, "index")
    if expected_index_sha256 is not None and index_digest != expected_index_sha256:
        raise RealManifestGenerationError("real manifest run external index digest mismatch")
    validate_source_inventory(inventory, data_root=data_root)
    if index.get("source_inventory_sha256") != inventory_digest:
        raise RealManifestGenerationError("real manifest run inventory binding mismatch")
    if index.get("manifest_contract_sha256") != canonical_document_sha256(contract):
        raise RealManifestGenerationError("real manifest run contract binding mismatch")
    manifests: dict[str, dict[str, Any]] = {}
    common_bindings = {
        "charter_id": index["charter_id"],
        "charter_version": index["charter_version"],
        "charter_sha256": index["charter_sha256"],
        "roi_map_id": index["roi_map_id"],
        "roi_map_sha256": index["roi_map_sha256"],
        "code_commit": index["code_commit"],
        "dirty_diff_sha256": index["dirty_diff_sha256"],
        "source_inventory_sha256": index["source_inventory_sha256"],
    }
    for name in RUN_BUNDLE_NAMES:
        manifest = load_manifest_bundle(
            directory / name,
            dict(contract),
            expected_bindings=common_bindings,
        )
        if manifest_sha256(manifest, dict(contract)) != index["bundle_sha256s"][name]:
            raise RealManifestGenerationError(f"run index bundle digest mismatch: {name}")
        manifests[name] = manifest
    validate_real_manifest_documents(inventory, manifests, contract)
    validate_run_index(
        index, contract=contract, inventory=inventory, manifests=manifests
    )
    return inventory, manifests, index


def generate_real_manifest_run(
    *,
    repo_root: Path,
    data_root: Path,
    destination: Path,
    generator_command: str,
    created_at_utc: str | None = None,
) -> str:
    """Check all pre-result gates, build the outcome-blind documents, and publish once."""
    repo_root = Path(repo_root)
    data_root = Path(data_root)
    code_commit, _ = _git_identity(repo_root)
    charter, roi_map, contract = _load_and_verify_readiness(repo_root)
    if created_at_utc is None:
        created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        )
    inventory, manifests = build_real_manifest_documents(
        data_root=data_root,
        charter=charter,
        roi_map=roi_map,
        contract=contract,
        code_commit=code_commit,
        created_at_utc=created_at_utc,
        generator_command=generator_command,
    )
    index = _run_index(
        created_at_utc=created_at_utc,
        generator_command=generator_command,
        code_commit=code_commit,
        charter=charter,
        roi_map=roi_map,
        contract=contract,
        inventory=inventory,
        manifests=manifests,
    )
    return write_real_manifest_run_atomic(
        destination,
        inventory=inventory,
        manifests=manifests,
        index=index,
        contract=contract,
    )


def verify_real_manifest_run_against_sources(
    *,
    repo_root: Path,
    data_root: Path,
    directory: Path,
    expected_index_sha256: str,
    expected_channel_names: Sequence[str] = EXPECTED_CHANNEL_NAMES,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    """Rebuild every document from local sources and require byte-equivalent identities."""
    repo_root = Path(repo_root)
    charter, roi_map, contract = _load_and_verify_readiness(repo_root)
    inventory, manifests, index = load_real_manifest_run(
        directory,
        contract,
        expected_index_sha256=expected_index_sha256,
        data_root=Path(data_root),
    )
    if index["charter_sha256"] != canonical_document_sha256(charter):
        raise RealManifestGenerationError("real manifest run charter drift")
    if index["roi_map_sha256"] != canonical_document_sha256(roi_map):
        raise RealManifestGenerationError("real manifest run ROI map drift")
    rebuilt_inventory, rebuilt_manifests = build_real_manifest_documents(
        data_root=Path(data_root),
        charter=charter,
        roi_map=roi_map,
        contract=contract,
        code_commit=index["code_commit"],
        created_at_utc=index["created_at_utc"],
        generator_command=index["generator_command"],
        expected_channel_names=expected_channel_names,
    )
    if rebuilt_inventory != inventory:
        raise RealManifestGenerationError("real source inventory does not reproduce")
    for name in RUN_BUNDLE_NAMES:
        if rebuilt_manifests[name] != manifests[name]:
            raise RealManifestGenerationError(f"real manifest does not reproduce: {name}")
    return inventory, manifests, index
