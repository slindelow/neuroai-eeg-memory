"""Invented-file tests for real preflight candidate derivation without outcomes."""

from pathlib import Path

import pytest

from peers.m1_preflight import select_preflight_roles
from peers.m1_preflight_execution import (
    PreflightExecutionError,
    build_real_candidate_inventory,
    execution_source_inventory,
    summarize_real_candidate_inventory,
)


SHA = "a" * 64


def _manifests(data_root: Path):
    subjects = ("sub-SYN001", "sub-SYN002", "sub-SYN003")
    manifests = {}
    bundle_sha256s = {}
    for condition_index, condition in enumerate(("ocular-removed", "ocular-kept")):
        cohort_name = f"cohort-{condition}"
        trial_name = f"trial-{condition}"
        manifests[cohort_name] = {
            "rows": [
                {
                    "subject_id": subject,
                    "eligible": True,
                    "eligibility_reason_code": "ELIGIBLE",
                }
                for subject in subjects
            ]
        }
        trial_rows = []
        for subject_index, subject in enumerate(subjects, start=1):
            derivative_id = f"derivatives/{subject}/{condition}.fif"
            derivative = data_root / derivative_id
            derivative.parent.mkdir(parents=True, exist_ok=True)
            derivative.write_bytes(
                bytes([condition_index + subject_index]) * (10 * subject_index + condition_index)
            )
            for trial in range(1, subject_index + condition_index + 1):
                trial_rows.append(
                    {
                        "subject_id": subject,
                        "session": 0,
                        "trial": trial,
                        "item_num": 100 * subject_index + trial,
                        "derivative_id": derivative_id,
                        "included": True,
                        "condition_survivor": True,
                        "exclusion_reason_code": "INCLUDED",
                    }
                )
        manifests[trial_name] = {"rows": trial_rows}
        bundle_sha256s[trial_name] = str(condition_index + 1) * 64
    return manifests, {"bundle_sha256s": bundle_sha256s}


def test_real_candidate_inventory_uses_common_survivors_and_unique_input_files(tmp_path):
    manifests, index = _manifests(tmp_path)
    inventory = build_real_candidate_inventory(manifests, index, data_root=tmp_path)
    assert [row["subject_id"] for row in inventory] == [
        "sub-SYN001",
        "sub-SYN002",
        "sub-SYN003",
    ]
    assert inventory[0]["eligible_trials_by_condition"] == {
        "ocular-removed": 1,
        "ocular-kept": 2,
    }
    assert inventory[0]["input_bytes_by_condition"] == {
        "ocular-removed": 10,
        "ocular-kept": 11,
    }
    summary = summarize_real_candidate_inventory(manifests, index, data_root=tmp_path)
    assert summary["outcome_access"] is False
    assert len(summary["candidate_inventory_sha256"]) == 64
    assert summary["selected_roles"] == select_preflight_roles(inventory)


def test_candidate_selection_is_determined_by_trials_then_bytes_then_subject(tmp_path):
    manifests, index = _manifests(tmp_path)
    inventory = build_real_candidate_inventory(manifests, index, data_root=tmp_path)
    roles = select_preflight_roles(inventory)
    assert roles == {
        "median_subject_id": "sub-SYN002",
        "maximum_subject_id": "sub-SYN003",
    }


def test_candidate_inventory_rejects_nonregular_source_and_manifest_drift(tmp_path):
    manifests, index = _manifests(tmp_path)
    source = tmp_path / manifests["trial-ocular-removed"]["rows"][0]["derivative_id"]
    source.unlink()
    with pytest.raises(PreflightExecutionError, match="regular file"):
        build_real_candidate_inventory(manifests, index, data_root=tmp_path)

    manifests, index = _manifests(tmp_path / "second")
    index["bundle_sha256s"].pop("trial-ocular-kept")
    with pytest.raises(PreflightExecutionError, match="digest"):
        build_real_candidate_inventory(manifests, index, data_root=tmp_path / "second")


def test_execution_source_inventory_binds_both_scopes_and_is_tamper_sensitive():
    m1_inventory = {
        "schema_id": "synthetic-m1",
        "files": [{"source_id": "derivatives/a.fif", "sha256": "a" * 64}],
    }
    behavioral = {
        "schema_id": "neuroai-m1-behavioral-reference-inventory-v1",
        "schema_version": "1.0",
        "population": "full_valid_local_ltpFR_behavioral_trial_table",
        "files": [{"source_id": "peers/a.tsv", "sha256": "b" * 64}],
        "trial_count": 100,
        "subject_count": 3,
    }
    document, digest = execution_source_inventory(m1_inventory, behavioral)
    assert document["m1_manifest_source_inventory_sha256"] != (
        document["behavioral_reference_inventory_sha256"]
    )
    assert len(digest) == 64
    changed = dict(behavioral, trial_count=101)
    changed_document, changed_digest = execution_source_inventory(m1_inventory, changed)
    assert changed_document["behavioral_reference_trial_count"] == 101
    assert changed_digest != digest
