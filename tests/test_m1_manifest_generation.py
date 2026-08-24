"""Synthetic integration tests for outcome-blind real-manifest extraction."""

from copy import deepcopy
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from peers.events import build_trial_table, load_events
from peers.m1_manifest_generation import (
    RealManifestGenerationError,
    _load_and_verify_readiness,
    _run_index,
    build_real_manifest_documents,
    load_real_manifest_run,
    validate_real_manifest_documents,
    verify_real_manifest_run_against_sources,
    write_real_manifest_run_atomic,
)
from peers.m1_manifests import load_json_unique


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs" / "m1-manifest-contracts.json"
CHARTER_PATH = ROOT / "docs" / "analysis-charter-m1-v1.json"
ROI_PATH = ROOT / "docs" / "roi-map-m1-v1.json"
CHANNELS = ("E1", "E2")
CREATED_AT = "2026-08-17T21:00:00Z"
COMMAND = "synthetic-outcome-blind-generation"
COMMIT = "b" * 40


def _write_public_contracts(
    repo_root: Path,
    *,
    charter: dict | None = None,
    roi_map: dict | None = None,
    contract: dict | None = None,
) -> None:
    docs = repo_root / "docs"
    docs.mkdir(parents=True)
    documents = {
        "analysis-charter-m1-v1.json": charter or load_json_unique(CHARTER_PATH),
        "roi-map-m1-v1.json": roi_map or load_json_unique(ROI_PATH),
        "m1-manifest-contracts.json": contract or load_json_unique(CONTRACT_PATH),
    }
    for name, document in documents.items():
        (docs / name).write_text(
            json.dumps(document, indent=2) + "\n",
            encoding="utf-8",
        )


def test_public_readiness_accepts_frozen_scientific_contract_without_private_receipts(
    tmp_path: Path,
) -> None:
    _write_public_contracts(tmp_path)

    charter, roi_map, contract = _load_and_verify_readiness(tmp_path)

    assert charter["scientific_decisions"]["D26"] == "D26_A_strict_inductive"
    assert roi_map["status"].startswith("frozen")
    assert contract["scientific_decision_gates"]["D44"]["allowed_resolved_values"]


def test_public_readiness_rejects_scientific_decision_drift(tmp_path: Path) -> None:
    charter = load_json_unique(CHARTER_PATH)
    charter["features"]["normalization"]["held_session_distribution_access"] = True
    _write_public_contracts(tmp_path, charter=charter)

    with pytest.raises(RealManifestGenerationError, match="D26 strict-inductive"):
        _load_and_verify_readiness(tmp_path)


def test_public_readiness_rejects_roi_binding_drift(tmp_path: Path) -> None:
    roi_map = load_json_unique(ROI_PATH)
    roi_map["rois"][roi_map["roi_order"][0]]["channels"][0] = "E128"
    _write_public_contracts(tmp_path, roi_map=roi_map)

    with pytest.raises(RealManifestGenerationError, match="ROI channels do not match"):
        _load_and_verify_readiness(tmp_path)


def _write_events(path: Path, subject: str, session: int) -> None:
    rows = []
    for position, (item_num, recalled) in enumerate(
        [(101, False), (102, True), (103, False), (104, True)], start=1
    ):
        rows.append(
            {
                "onset": float(position),
                "duration": 1.0,
                "sample": position * 100,
                "trial_type": "WORD",
                "subject": subject.removeprefix("sub-"),
                "session": session,
                "trial": 1,
                "item_name": f"WORD{item_num}",
                "item_num": item_num,
            }
        )
        if recalled:
            rows.append(
                {
                    "onset": 20.0 + position,
                    "duration": 0.0,
                    "sample": 2000 + position,
                    "trial_type": "REC_WORD",
                    "subject": subject.removeprefix("sub-"),
                    "session": session,
                    "trial": 1,
                    "item_name": f"WORD{item_num}",
                    "item_num": item_num,
                }
            )
    rows.append(
        {
            "onset": -0.01,
            "duration": 1.0,
            "sample": -1,
            "trial_type": "WORD",
            "subject": subject.removeprefix("sub-"),
            "session": session,
            "trial": 2,
            "item_name": "UNALIGNED",
            "item_num": 105,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_epochs(
    path: Path,
    event_path: Path,
    *,
    dropped_item: int | None,
    channels: tuple[str, ...] = CHANNELS,
    nonfinite: bool = False,
) -> None:
    metadata = build_trial_table(load_events(event_path))
    if dropped_item is not None:
        metadata = metadata[metadata["item_num"] != dropped_item]
    metadata = metadata.reset_index(drop=True)
    data = np.arange(len(metadata) * len(channels) * 5, dtype=float).reshape(
        len(metadata), len(channels), 5
    )
    if nonfinite:
        data[0, 0, 0] = np.nan
    info = mne.create_info(list(channels), sfreq=100.0, ch_types="eeg")
    event_codes = metadata["recalled"].astype(int).to_numpy()
    events = np.column_stack(
        [np.arange(len(metadata), dtype=int) * 10 + 100, np.zeros(len(metadata), int), event_codes]
    )
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"forgotten": 0, "recalled": 1},
        tmin=0.0,
        metadata=metadata,
        verbose="ERROR",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(path, overwrite=True, verbose="ERROR")


def _make_sources(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    subjects = {"sub-LTP001": (0, 1, 2), "sub-LTP002": (0,)}
    for subject, sessions in subjects.items():
        for session in sessions:
            event_path = (
                data_root
                / "peers"
                / subject
                / f"ses-{session}"
                / f"{subject}_ses-{session}_task-ltpFR_events.tsv"
            )
            _write_events(event_path, subject, session)
            for condition in ("ocular-removed", "ocular-kept"):
                dropped_item = 101 if condition == "ocular-removed" else 104
                derivative = (
                    data_root
                    / "derivatives"
                    / subject
                    / f"{subject}_ses-{session}_{condition}_epo.fif"
                )
                _write_epochs(derivative, event_path, dropped_item=dropped_item)
    return data_root


def _documents(tmp_path: Path):
    data_root = _make_sources(tmp_path)
    charter = load_json_unique(CHARTER_PATH)
    roi_map = load_json_unique(ROI_PATH)
    contract = load_json_unique(CONTRACT_PATH)
    inventory, manifests = build_real_manifest_documents(
        data_root=data_root,
        charter=charter,
        roi_map=roi_map,
        contract=contract,
        code_commit=COMMIT,
        created_at_utc=CREATED_AT,
        generator_command=COMMAND,
        expected_channel_names=CHANNELS,
    )
    return data_root, charter, roi_map, contract, inventory, manifests


def test_real_extractor_reconciles_autoreject_attrition_folds_and_intersection(tmp_path):
    _, _, _, contract, inventory, manifests = _documents(tmp_path)
    validate_real_manifest_documents(inventory, manifests, contract)

    removed_cohort = manifests["cohort-ocular-removed"]["rows"]
    eligible = next(row for row in removed_cohort if row["subject_id"] == "sub-LTP001")
    ineligible = next(row for row in removed_cohort if row["subject_id"] == "sub-LTP002")
    assert eligible["eligible"] is True
    assert eligible["outer_train_sessions"] == [0, 1]
    assert eligible["outer_test_session"] == 2
    assert ineligible["eligibility_reason_code"] == "TOO_FEW_SESSIONS"

    attrition = manifests["attrition"]["rows"]
    alignment = next(
        row
        for row in attrition
        if row["condition"] == "ocular-removed"
        and row["flow_id"] == "trial_analysis"
        and row["stage_index"] == 2
    )
    assert alignment["excluded_by_reason"] == {"AUTOREJECT_DROPPED": 4}
    subject_gate = next(
        row
        for row in attrition
        if row["condition"] == "ocular-removed"
        and row["flow_id"] == "subject_condition_eligibility"
        and row["stage_index"] == 2
    )
    assert subject_gate["excluded_by_reason"] == {"TOO_FEW_SESSIONS": 1}

    removed_keys = {
        (row["subject_id"], row["session"], row["trial"], row["item_num"])
        for row in manifests["trial-ocular-removed"]["rows"]
        if row["condition_survivor"]
    }
    kept_keys = {
        (row["subject_id"], row["session"], row["trial"], row["item_num"])
        for row in manifests["trial-ocular-kept"]["rows"]
        if row["condition_survivor"]
    }
    intersection_keys = {
        (row["subject_id"], row["session"], row["trial"], row["item_num"])
        for row in manifests["cross-condition-intersection"]["rows"]
    }
    assert intersection_keys == removed_keys & kept_keys
    assert len(manifests["inner-fold-ocular-removed"]["rows"]) == 2


def test_real_run_round_trip_authenticates_sources_and_is_immutable(tmp_path):
    data_root, charter, roi_map, contract, inventory, manifests = _documents(tmp_path)
    index = _run_index(
        created_at_utc=CREATED_AT,
        generator_command=COMMAND,
        code_commit=COMMIT,
        charter=charter,
        roi_map=roi_map,
        contract=contract,
        inventory=inventory,
        manifests=manifests,
    )
    destination = tmp_path / "manifest-run"
    digest = write_real_manifest_run_atomic(
        destination,
        inventory=inventory,
        manifests=manifests,
        index=index,
        contract=contract,
    )
    loaded_inventory, loaded_manifests, loaded_index = load_real_manifest_run(
        destination,
        contract,
        expected_index_sha256=digest,
        data_root=data_root,
    )
    assert loaded_inventory == inventory
    assert loaded_manifests == manifests
    assert loaded_index == index
    reproduced = verify_real_manifest_run_against_sources(
        repo_root=ROOT,
        data_root=data_root,
        directory=destination,
        expected_index_sha256=digest,
        expected_channel_names=CHANNELS,
    )
    assert reproduced == (inventory, manifests, index)
    with pytest.raises(RealManifestGenerationError, match="already exists"):
        write_real_manifest_run_atomic(
            destination,
            inventory=inventory,
            manifests=manifests,
            index=index,
            contract=contract,
        )

    event_path = next((data_root / "peers").glob("sub-LTP*/ses-*/*_events.tsv"))
    event_path.write_text(event_path.read_text() + "\n")
    with pytest.raises(RealManifestGenerationError, match="source inventory file drift"):
        load_real_manifest_run(destination, contract, data_root=data_root)


def test_real_extractor_fails_closed_on_channel_nonfinite_or_metadata_drift(tmp_path):
    data_root = _make_sources(tmp_path)
    event_path = next((data_root / "peers" / "sub-LTP001").glob("ses-*/*_events.tsv"))
    derivative = (
        data_root
        / "derivatives"
        / "sub-LTP001"
        / f"sub-LTP001_{event_path.parent.name}_ocular-removed_epo.fif"
    )
    _write_epochs(derivative, event_path, dropped_item=101, channels=("E1", "E3"))
    with pytest.raises(RealManifestGenerationError, match="CHANNEL_SCHEMA_MISMATCH"):
        build_real_manifest_documents(
            data_root=data_root,
            charter=load_json_unique(CHARTER_PATH),
            roi_map=load_json_unique(ROI_PATH),
            contract=load_json_unique(CONTRACT_PATH),
            code_commit=COMMIT,
            created_at_utc=CREATED_AT,
            generator_command=COMMAND,
            expected_channel_names=CHANNELS,
        )

    _write_epochs(derivative, event_path, dropped_item=101, nonfinite=True)
    with pytest.raises(RealManifestGenerationError, match="NONFINITE_INPUT"):
        build_real_manifest_documents(
            data_root=data_root,
            charter=load_json_unique(CHARTER_PATH),
            roi_map=load_json_unique(ROI_PATH),
            contract=load_json_unique(CONTRACT_PATH),
            code_commit=COMMIT,
            created_at_utc=CREATED_AT,
            generator_command=COMMAND,
            expected_channel_names=CHANNELS,
        )


def test_cross_document_validator_rejects_fold_digest_tamper(tmp_path):
    _, _, _, contract, inventory, manifests = _documents(tmp_path)
    tampered = deepcopy(manifests)
    tampered["inner-fold-ocular-removed"]["rows"][0]["fit_keys_sha256"] = "f" * 64
    with pytest.raises(RealManifestGenerationError, match="inner-fold key digest drift"):
        validate_real_manifest_documents(inventory, tampered, contract)


def test_cross_document_validator_rejects_attrition_and_source_reference_tamper(tmp_path):
    _, _, _, contract, inventory, manifests = _documents(tmp_path)
    attrition_tamper = deepcopy(manifests)
    row = next(
        row
        for row in attrition_tamper["attrition"]["rows"]
        if row["condition"] == "ocular-removed"
        and row["flow_id"] == "trial_analysis"
        and row["stage_index"] == 2
    )
    row["excluded_by_reason"]["AUTOREJECT_DROPPED"] -= 1
    row["survivor_count"] += 1
    next_stage = next(
        candidate
        for candidate in attrition_tamper["attrition"]["rows"]
        if candidate["condition"] == "ocular-removed"
        and candidate["flow_id"] == "trial_analysis"
        and candidate["stage_index"] == 3
    )
    next_stage["input_count"] += 1
    next_stage["survivor_count"] += 1
    cross_stage = next(
        candidate
        for candidate in attrition_tamper["attrition"]["rows"]
        if candidate["condition"] == "ocular-removed"
        and candidate["flow_id"] == "cross_condition_sensitivity"
    )
    cross_stage["input_count"] += 1
    cross_stage["excluded_by_reason"]["ABSENT_FROM_OTHER_CONDITION"] += 1
    with pytest.raises(RealManifestGenerationError, match="AutoReject attrition ledger drift"):
        validate_real_manifest_documents(inventory, attrition_tamper, contract)

    source_tamper = deepcopy(manifests)
    excluded_row = next(
        row
        for row in source_tamper["trial-ocular-removed"]["rows"]
        if not row["condition_survivor"]
    )
    excluded_row["source_events_sha256"] = "f" * 64
    with pytest.raises(RealManifestGenerationError, match="trial event source reference drift"):
        validate_real_manifest_documents(inventory, source_tamper, contract)
