"""Invented-data tests for the outcome-free real-source feature bridge."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from peers.m1_feature_generation import (
    RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT,
    build_raw_cache_metadata,
    combine_training_session_statistics,
    compact_feature_values,
    extract_locked_epoch_session,
    load_raw_spectral_session_bundle,
    raw_spectral_bundle_contract_sha256,
    summarize_log_power_session,
    write_raw_spectral_session_bundle_atomic,
)
from peers.m1_features import (
    FeatureTableValidationError,
    aggregate_normalized_log_power,
    apply_strict_inductive_normalizer,
    feature_schema_sha256,
    fit_strict_inductive_normalizer,
)
from peers.m1_manifests import load_json_unique


ROOT = Path(__file__).resolve().parents[1]
CHARTER = load_json_unique(ROOT / "docs" / "analysis-charter-m1-v1.json")
CONTRACT = load_json_unique(ROOT / "docs" / "m1-feature-table-contract.json")


def _channels():
    return [
        channel
        for roi in CHARTER["features"]["roi_order"]
        for channel in CHARTER["features"]["roi_channels"][roi]
    ]


def _keys(session: int, n_trials: int):
    return tuple(
        ("sub-SYN001", session, trial, 1000 + session * 100 + trial)
        for trial in range(1, n_trials + 1)
    )


def _summary(session: int, values: np.ndarray):
    return summarize_log_power_session(
        values,
        np.linspace(-1.0, 2.0, values.shape[-1], dtype=np.float64),
        _keys(session, values.shape[0]),
        subject_id="sub-SYN001",
        session=session,
        condition="ocular-removed",
        derivative_id=f"derivatives/sub-SYN001/session-{session}.fif",
        derivative_sha256=f"{session + 1}" * 64,
        charter=CHARTER,
    )


def _metadata(summary):
    frequencies = CHARTER["features"]["frequencies_hz"]
    return build_raw_cache_metadata(
        summary,
        created_at_utc="2026-08-17T22:00:00Z",
        bindings={
            "trial_manifest_sha256": "a" * 64,
            "source_inventory_sha256": "b" * 64,
            "charter_sha256": "c" * 64,
            "roi_map_sha256": "d" * 64,
            "feature_schema_sha256": feature_schema_sha256(CHARTER, CONTRACT),
            "uv_lock_sha256": "e" * 64,
            "package_versions_sha256": "f" * 64,
            "code_commit": "1" * 40,
            "dirty_diff_sha256": None,
        },
        generator_command="synthetic raw cache test",
        python_version="3.12-test",
        mne_version="1.test",
        numpy_version=np.__version__,
        frequencies_hz=frequencies,
        n_cycles=[value / 2.0 for value in frequencies],
        decim=CHARTER["features"]["decim"],
        parallel_worker_count=2,
        channel_order=_channels(),
        time_bin_ids=CHARTER["features"]["time_bin_ids"],
    )


def test_raw_cache_contract_exactly_matches_machine_document():
    assert CONTRACT["version"] == "1.5-draft"
    assert CONTRACT["raw_spectral_session_bundle"] == (
        RAW_SPECTRAL_SESSION_BUNDLE_CONTRACT
    )
    assert len(raw_spectral_bundle_contract_sha256(CONTRACT)) == 64


def test_compact_summary_is_equivalent_to_full_array_normalize_then_aggregate():
    rng = np.random.default_rng(20260817)
    first = rng.normal(size=(3, 40, 8, 31)).astype(np.float64)
    second = rng.normal(loc=0.3, scale=1.2, size=(4, 40, 8, 31)).astype(np.float64)
    summaries = [_summary(0, first), _summary(1, second)]

    compact_statistics = combine_training_session_statistics(
        summaries, fit_sessions=[0, 1], charter=CHARTER
    )
    direct_statistics = fit_strict_inductive_normalizer(
        np.concatenate([first, second], axis=0)
    )
    np.testing.assert_allclose(compact_statistics.mean, direct_statistics.mean, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        compact_statistics.raw_std, direct_statistics.raw_std, rtol=0, atol=2e-15
    )

    compact = compact_feature_values(summaries[1], compact_statistics, charter=CHARTER)
    direct = aggregate_normalized_log_power(
        apply_strict_inductive_normalizer(second, direct_statistics),
        summaries[1].time_axis,
        _channels(),
        CHARTER,
        CONTRACT,
    )
    np.testing.assert_allclose(compact, direct, rtol=0, atol=2e-15)


def test_held_session_statistics_cannot_influence_training_only_fit():
    rng = np.random.default_rng(44)
    training = _summary(0, rng.normal(size=(2, 40, 8, 31)).astype(np.float64))
    held = _summary(1, rng.normal(loc=50.0, size=(2, 40, 8, 31)).astype(np.float64))
    fit_with_held_available = combine_training_session_statistics(
        [training, held], fit_sessions=[0], charter=CHARTER
    )
    fit_training_only = combine_training_session_statistics(
        [training], fit_sessions=[0], charter=CHARTER
    )
    np.testing.assert_array_equal(fit_with_held_available.mean, fit_training_only.mean)
    np.testing.assert_array_equal(fit_with_held_available.raw_std, fit_training_only.raw_std)
    with pytest.raises(FeatureTableValidationError, match="fit_sessions"):
        combine_training_session_statistics(
            [training, held], fit_sessions=[0, 2], charter=CHARTER
        )


def test_raw_bundle_roundtrip_is_typed_authenticated_and_immutable(tmp_path):
    rng = np.random.default_rng(7)
    summary = _summary(0, rng.normal(size=(2, 40, 8, 31)).astype(np.float64))
    metadata = _metadata(summary)
    destination = tmp_path / "session-cache"
    digest = write_raw_spectral_session_bundle_atomic(
        destination, summary, metadata, CHARTER, CONTRACT
    )
    loaded, loaded_metadata = load_raw_spectral_session_bundle(
        destination,
        CHARTER,
        CONTRACT,
        expected_bundle_sha256=digest,
        expected_metadata_bindings={
            "derivative_id": metadata["derivative_id"],
            "trial_manifest_sha256": metadata["trial_manifest_sha256"],
            "source_inventory_sha256": metadata["source_inventory_sha256"],
            "derivative_sha256": metadata["derivative_sha256"],
            "charter_sha256": metadata["charter_sha256"],
            "roi_map_sha256": metadata["roi_map_sha256"],
            "feature_schema_sha256": metadata["feature_schema_sha256"],
            "uv_lock_sha256": metadata["uv_lock_sha256"],
            "package_versions_sha256": metadata["package_versions_sha256"],
            "code_commit": metadata["code_commit"],
            "dirty_diff_sha256": metadata["dirty_diff_sha256"],
        },
    )
    assert loaded_metadata == metadata
    assert loaded.trial_keys == summary.trial_keys
    np.testing.assert_array_equal(loaded.bin_means, summary.bin_means)
    np.testing.assert_array_equal(loaded.statistics_mean, summary.statistics_mean)
    np.testing.assert_array_equal(loaded.statistics_m2, summary.statistics_m2)
    np.testing.assert_array_equal(loaded.time_axis, summary.time_axis)
    with pytest.raises(FeatureTableValidationError, match="must not exist"):
        write_raw_spectral_session_bundle_atomic(
            destination, summary, metadata, CHARTER, CONTRACT
        )


def test_raw_bundle_rejects_external_identity_member_tamper_and_outcome_field(tmp_path):
    rng = np.random.default_rng(8)
    summary = _summary(0, rng.normal(size=(2, 40, 8, 31)).astype(np.float64))
    metadata = _metadata(summary)
    destination = tmp_path / "session-cache"
    digest = write_raw_spectral_session_bundle_atomic(
        destination, summary, metadata, CHARTER, CONTRACT
    )
    with pytest.raises(FeatureTableValidationError, match="external/sidecar"):
        load_raw_spectral_session_bundle(
            destination,
            CHARTER,
            CONTRACT,
            expected_bundle_sha256="0" * 64,
            expected_metadata_bindings={field: metadata[field] for field in CONTRACT["raw_spectral_session_bundle"]["external_binding_fields"]},
        )

    values_path = destination / "bin-means.npy"
    payload = bytearray(values_path.read_bytes())
    payload[-1] ^= 1
    values_path.write_bytes(bytes(payload))
    with pytest.raises(FeatureTableValidationError, match="file identity"):
        load_raw_spectral_session_bundle(
            destination,
            CHARTER,
            CONTRACT,
            expected_bundle_sha256=digest,
            expected_metadata_bindings={field: metadata[field] for field in CONTRACT["raw_spectral_session_bundle"]["external_binding_fields"]},
        )

    extra_metadata = _metadata(summary)
    extra_metadata["recalled"] = False
    with pytest.raises(FeatureTableValidationError, match="fields mismatch"):
        write_raw_spectral_session_bundle_atomic(
            tmp_path / "outcome-cache", summary, extra_metadata, CHARTER, CONTRACT
        )


def test_raw_cache_contract_drift_fails_closed():
    drifted = deepcopy(CONTRACT)
    drifted["raw_spectral_session_bundle"]["combination"][
        "held_session_statistics_allowed"
    ] = True
    with pytest.raises(FeatureTableValidationError, match="drifted"):
        raw_spectral_bundle_contract_sha256(drifted)


def test_locked_epoch_extractor_rehashes_source_and_uses_exact_survivor_rows(
    tmp_path, monkeypatch
):
    derivative_id = "derivatives/sub-SYN001/sub-SYN001_ses-0_ocular-removed_epo.fif"
    derivative = tmp_path / derivative_id
    derivative.parent.mkdir(parents=True)
    derivative.write_bytes(b"synthetic-fif-identity")
    derivative_sha256 = sha256(derivative.read_bytes()).hexdigest()
    rows = [
        {
            "subject_id": "sub-SYN001",
            "session": 0,
            "trial": trial,
            "item_num": 1000 + trial,
            "condition": "ocular-removed",
            "derivative_id": derivative_id,
            "derivative_sha256": derivative_sha256,
            "epoch_row": trial - 1,
            "included": True,
            "condition_survivor": True,
            "exclusion_reason_code": "INCLUDED",
        }
        for trial in (1, 2)
    ]
    power = np.exp(
        np.random.default_rng(9).normal(size=(2, 40, 8, 31)).astype(np.float64)
    )
    times = np.linspace(-1.0, 2.0, 31, dtype=np.float64)

    class FakeTFR:
        ch_names = _channels()

        def __init__(self):
            self.times = times

        def get_data(self):
            return power

    class FakeEpochs:
        def __init__(self):
            self.ch_names = _channels()
            self.metadata = pd.DataFrame(
                {
                    "subject_id": ["sub-SYN001", "sub-SYN001"],
                    "session": [0, 0],
                    "trial": [1, 2],
                    "item_num": [1001, 1002],
                }
            )

        def __len__(self):
            return 2

        def __getitem__(self, indices):
            assert list(indices) == [0, 1]
            return self

        def copy(self):
            return self

        def pick(self, channels):
            self.ch_names = list(channels)
            return self

        def compute_tfr(self, **kwargs):
            assert kwargs["method"] == "morlet"
            assert kwargs["decim"] == 4
            assert kwargs["n_jobs"] == 2
            return FakeTFR()

    import mne

    monkeypatch.setattr(mne, "read_epochs", lambda *args, **kwargs: FakeEpochs())
    summary = extract_locked_epoch_session(
        derivative,
        rows,
        data_root=tmp_path,
        charter=CHARTER,
        n_jobs=2,
    )
    assert summary.trial_keys == _keys(0, 2)
    assert summary.bin_means.shape == (2, 2, 40, 8)
    assert summary.statistics_observation_count == 62

    invalid = deepcopy(rows)
    invalid[0]["condition_survivor"] = False
    with pytest.raises(FeatureTableValidationError, match="condition-survivor"):
        extract_locked_epoch_session(
            derivative,
            invalid,
            data_root=tmp_path,
            charter=CHARTER,
            n_jobs=2,
        )
