"""Authenticated outcome-blind storage for validated M1 evaluation records.

The container inherits its record fields from the supplied D42-A evaluation contract. It stores
the deliberate absence of balanced accuracy, computes no metric, and permits no real publication
before the separate scientific, comprehension, and execution gates are satisfied.
"""

from collections.abc import Mapping
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from peers.m1_evaluation import (
    EvaluationConsumerValidationError,
    canonical_common_intersection_aggregate_bytes,
    canonical_common_intersection_subject_bytes,
    canonical_primary_aggregate_bytes,
    canonical_subject_evaluation_bytes,
    evaluation_contract_sha256,
    validate_common_intersection_aggregate_record,
    validate_common_intersection_subject_record,
    validate_primary_aggregate_record,
    validate_subject_evaluation_record,
)
from peers.m1_manifests import (
    ManifestValidationError,
    _fsync_directory,
    _rename_directory_noreplace,
    _write_fsynced,
    load_json_unique,
)


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

RESULT_RECORD_BUNDLE_CONTRACT = {
    "schema_id": "neuroai-m1-evaluation-result-record-bundle-v1",
    "schema_version": "1.0",
    "atomic_publication": "temporary_directory_fsync_then_atomic_noreplace_rename",
    "exact_files": ["record.json", "bundle.json", "bundle.sha256"],
    "record_kinds": {
        "subject_evaluation": {
            "contract_key": "subject_record",
            "schema_id": "neuroai-m1-subject-evaluation-v1",
        },
        "common_intersection_subject_evaluation": {
            "contract_key": "common_intersection_subject_record",
            "schema_id": "neuroai-m1-common-intersection-subject-evaluation-v1",
        },
        "primary_aggregate": {
            "contract_key": "primary_aggregate",
            "schema_id": "neuroai-m1-primary-aggregate-v1",
        },
        "common_intersection_aggregate": {
            "contract_key": "common_intersection_aggregate",
            "schema_id": "neuroai-m1-common-intersection-aggregate-v1",
        },
    },
    "record_fields_source": "resolved_evaluation_contract_exact_fields",
    "balanced_accuracy_policy": "D42_A_deliberately_omitted_no_threshold_or_fields",
    "record_format": "canonical_finite_JSON_object",
    "bundle_manifest_exact_fields": [
        "schema_id",
        "schema_version",
        "bundle_contract_sha256",
        "evaluation_contract_sha256",
        "record_kind",
        "record_schema_id",
        "record_schema_version",
        "record_json_sha256",
    ],
    "exact_regular_files_no_symlinks": True,
    "external_bundle_sha256_required_on_readback": True,
    "complete_logical_revalidation_required_on_readback": True,
    "real_publication_allowed_before_final_readiness": False,
    "failure_staging_policy": "retain_for_evidence_no_automatic_deletion",
}

RESULT_RECORD_BUNDLE_FILES = frozenset(RESULT_RECORD_BUNDLE_CONTRACT["exact_files"])


class EvaluationResultBundleValidationError(EvaluationConsumerValidationError):
    """A stored evaluation record violates its authenticated bundle contract."""


def _canonical_object_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationResultBundleValidationError(
            "result bundle object is not canonical finite JSON"
        ) from exc
    return (text + "\n").encode("utf-8")


def _bundle_contract_payload(
    evaluation_contract: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        declared = evaluation_contract["result_record_storage_bundle"]
    except (KeyError, TypeError) as exc:
        raise EvaluationResultBundleValidationError(
            "evaluation contract is missing the result-record storage bundle"
        ) from exc
    if declared != RESULT_RECORD_BUNDLE_CONTRACT:
        raise EvaluationResultBundleValidationError(
            "result-record storage bundle contract drifted from the implementation"
        )
    return dict(declared)


def result_record_bundle_contract_sha256(
    evaluation_contract: Mapping[str, Any],
) -> str:
    """Digest the exact D42-A-bound result-record storage layout."""
    return sha256(
        _canonical_object_bytes(_bundle_contract_payload(evaluation_contract))
    ).hexdigest()


def _record_contract(
    evaluation_contract: Mapping[str, Any], record_kind: str
) -> Mapping[str, Any]:
    bundle_contract = _bundle_contract_payload(evaluation_contract)
    kinds = bundle_contract["record_kinds"]
    if record_kind not in kinds:
        raise EvaluationResultBundleValidationError(
            f"unknown evaluation result record kind: {record_kind}"
        )
    declaration = kinds[record_kind]
    try:
        record_contract = evaluation_contract[declaration["contract_key"]]
    except (KeyError, TypeError) as exc:
        raise EvaluationResultBundleValidationError(
            f"evaluation contract is missing the {record_kind} record contract"
        ) from exc
    if not isinstance(record_contract, Mapping):
        raise EvaluationResultBundleValidationError(
            f"{record_kind} record contract must be an object"
        )
    if record_contract.get("schema_id") != declaration["schema_id"]:
        raise EvaluationResultBundleValidationError(
            f"{record_kind} schema identity drifted from the storage contract"
        )
    if not isinstance(record_contract.get("schema_version"), str) or not record_contract[
        "schema_version"
    ]:
        raise EvaluationResultBundleValidationError(
            f"{record_kind} schema version must be a non-empty string"
        )
    exact_fields = record_contract.get("exact_fields")
    if (
        not isinstance(exact_fields, list)
        or not exact_fields
        or len(exact_fields) != len(set(exact_fields))
        or not all(isinstance(field, str) and field for field in exact_fields)
    ):
        raise EvaluationResultBundleValidationError(
            f"{record_kind} exact fields are invalid"
        )
    return record_contract


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationResultBundleValidationError(
            f"{label} must be a regular file"
        )
    try:
        value = load_json_unique(path)
    except ManifestValidationError as exc:
        raise EvaluationResultBundleValidationError(str(exc)) from exc
    if not isinstance(value, dict):
        raise EvaluationResultBundleValidationError(
            f"{label} must contain one JSON object"
        )
    if path.read_bytes() != _canonical_object_bytes(value):
        raise EvaluationResultBundleValidationError(
            f"{label} is not canonical JSON bytes"
        )
    return value


def _write_result_record_bundle_atomic(
    bundle_path: Path,
    record: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    record_kind: str,
    record_bytes: bytes,
) -> str:
    bundle_contract = _bundle_contract_payload(evaluation_contract)
    record_contract = _record_contract(evaluation_contract, record_kind)
    if record_bytes != _canonical_object_bytes(record):
        raise EvaluationResultBundleValidationError(
            "validated result record bytes are not canonical"
        )
    if set(record) != set(record_contract["exact_fields"]):
        raise EvaluationResultBundleValidationError(
            f"{record_kind} fields do not match the evaluation contract"
        )

    destination = Path(bundle_path)
    parent = destination.parent
    if not parent.is_dir():
        raise EvaluationResultBundleValidationError(
            f"result-record bundle parent is not a directory: {parent}"
        )
    if os.path.lexists(destination):
        raise EvaluationResultBundleValidationError(
            f"result-record bundle already exists: {destination}"
        )

    record_digest = sha256(record_bytes).hexdigest()
    manifest = {
        "schema_id": bundle_contract["schema_id"],
        "schema_version": bundle_contract["schema_version"],
        "bundle_contract_sha256": result_record_bundle_contract_sha256(
            evaluation_contract
        ),
        "evaluation_contract_sha256": evaluation_contract_sha256(
            evaluation_contract
        ),
        "record_kind": record_kind,
        "record_schema_id": record_contract["schema_id"],
        "record_schema_version": record_contract["schema_version"],
        "record_json_sha256": record_digest,
    }
    manifest_bytes = _canonical_object_bytes(manifest)
    bundle_digest = sha256(manifest_bytes).hexdigest()
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
    try:
        _write_fsynced(temporary / "record.json", record_bytes)
        _write_fsynced(temporary / "bundle.json", manifest_bytes)
        _write_fsynced(
            temporary / "bundle.sha256", f"{bundle_digest}\n".encode("ascii")
        )
        _fsync_directory(temporary)
        if os.path.lexists(destination):
            raise EvaluationResultBundleValidationError(
                f"result-record bundle already exists: {destination}"
            )
        _rename_directory_noreplace(temporary, destination)
        _fsync_directory(parent)
    except Exception as exc:
        raise EvaluationResultBundleValidationError(
            "result-record bundle publication failed; staged evidence retained at "
            f"{temporary}: {exc}"
        ) from exc
    return bundle_digest


def _load_authenticated_result_record(
    bundle_path: Path,
    evaluation_contract: Mapping[str, Any],
    record_kind: str,
    *,
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    bundle_contract = _bundle_contract_payload(evaluation_contract)
    record_contract = _record_contract(evaluation_contract, record_kind)
    if not isinstance(expected_bundle_sha256, str) or not SHA256_RE.fullmatch(
        expected_bundle_sha256
    ):
        raise EvaluationResultBundleValidationError(
            "expected external result-record bundle digest must be a lowercase SHA-256"
        )
    bundle = Path(bundle_path)
    if not bundle.is_dir() or bundle.is_symlink():
        raise EvaluationResultBundleValidationError(
            f"result-record bundle is not a regular directory: {bundle}"
        )
    actual_files = {path.name for path in bundle.iterdir()}
    if actual_files != RESULT_RECORD_BUNDLE_FILES:
        raise EvaluationResultBundleValidationError(
            "result-record bundle does not contain the exact files required by the contract"
        )
    for name in RESULT_RECORD_BUNDLE_FILES:
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise EvaluationResultBundleValidationError(
                f"result-record bundle entry must be a regular file: {name}"
            )

    sidecar = (bundle / "bundle.sha256").read_bytes()
    if len(sidecar) != 65 or not sidecar.endswith(b"\n"):
        raise EvaluationResultBundleValidationError(
            "result-record bundle sidecar must be one lowercase SHA-256 plus newline"
        )
    try:
        sidecar_digest = sidecar[:-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise EvaluationResultBundleValidationError(
            "result-record bundle sidecar is not ASCII"
        ) from exc
    if not SHA256_RE.fullmatch(sidecar_digest):
        raise EvaluationResultBundleValidationError(
            "result-record bundle sidecar is not a lowercase SHA-256"
        )

    manifest_path = bundle / "bundle.json"
    manifest = _read_canonical_json(manifest_path, "result-record bundle manifest")
    bundle_digest = sha256(manifest_path.read_bytes()).hexdigest()
    if sidecar_digest != bundle_digest:
        raise EvaluationResultBundleValidationError(
            "result-record bundle sidecar does not match the canonical manifest"
        )
    if expected_bundle_sha256 != bundle_digest:
        raise EvaluationResultBundleValidationError(
            "external bundle digest does not match the canonical result-record manifest"
        )
    if set(manifest) != set(bundle_contract["bundle_manifest_exact_fields"]):
        raise EvaluationResultBundleValidationError(
            "result-record bundle manifest fields do not match the contract"
        )
    if (
        manifest["schema_id"] != bundle_contract["schema_id"]
        or manifest["schema_version"] != bundle_contract["schema_version"]
    ):
        raise EvaluationResultBundleValidationError(
            "result-record bundle schema identity mismatch"
        )
    if manifest["bundle_contract_sha256"] != result_record_bundle_contract_sha256(
        evaluation_contract
    ):
        raise EvaluationResultBundleValidationError(
            "result-record bundle contract digest mismatch"
        )
    if manifest["evaluation_contract_sha256"] != evaluation_contract_sha256(
        evaluation_contract
    ):
        raise EvaluationResultBundleValidationError(
            "result-record bundle evaluation-contract digest mismatch"
        )
    if manifest["record_kind"] != record_kind:
        raise EvaluationResultBundleValidationError(
            "result-record bundle record kind mismatch"
        )
    if (
        manifest["record_schema_id"] != record_contract["schema_id"]
        or manifest["record_schema_version"] != record_contract["schema_version"]
    ):
        raise EvaluationResultBundleValidationError(
            "result-record bundle record schema mismatch"
        )
    for field in (
        "bundle_contract_sha256",
        "evaluation_contract_sha256",
        "record_json_sha256",
    ):
        if not isinstance(manifest[field], str) or not SHA256_RE.fullmatch(
            manifest[field]
        ):
            raise EvaluationResultBundleValidationError(
                f"result-record bundle {field} is not a SHA-256"
            )

    record_path = bundle / "record.json"
    record = _read_canonical_json(record_path, "result-record bundle record")
    if sha256(record_path.read_bytes()).hexdigest() != manifest["record_json_sha256"]:
        raise EvaluationResultBundleValidationError(
            "result-record bundle record digest mismatch"
        )
    if set(record) != set(record_contract["exact_fields"]):
        raise EvaluationResultBundleValidationError(
            f"{record_kind} fields do not match the evaluation contract"
        )
    if (
        record.get("schema_id") != record_contract["schema_id"]
        or record.get("schema_version") != record_contract["schema_version"]
    ):
        raise EvaluationResultBundleValidationError(
            "result-record bundle record schema identity mismatch"
        )
    return record


def write_subject_evaluation_bundle_atomic(
    bundle_path: Path,
    record: Mapping[str, Any],
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    expected_evaluation_provenance: Mapping[str, Any],
) -> str:
    """Validate and atomically publish one condition-specific subject result."""
    _bundle_contract_payload(evaluation_contract)
    record_bytes = canonical_subject_evaluation_bytes(
        record,
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_bindings=expected_bindings,
        expected_evaluation_provenance=expected_evaluation_provenance,
    )
    return _write_result_record_bundle_atomic(
        bundle_path,
        record,
        evaluation_contract,
        "subject_evaluation",
        record_bytes,
    )


def load_subject_evaluation_bundle(
    bundle_path: Path,
    table: Mapping[str, Any],
    trial_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, str],
    expected_evaluation_provenance: Mapping[str, Any],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Authenticate and fully recompute one condition-specific subject result."""
    record = _load_authenticated_result_record(
        bundle_path,
        evaluation_contract,
        "subject_evaluation",
        expected_bundle_sha256=expected_bundle_sha256,
    )
    validate_subject_evaluation_record(
        record,
        table,
        trial_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_bindings=expected_bindings,
        expected_evaluation_provenance=expected_evaluation_provenance,
    )
    return record


def write_common_intersection_subject_evaluation_bundle_atomic(
    bundle_path: Path,
    record: Mapping[str, Any],
    table: Mapping[str, Any],
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    expected_evaluation_provenance: Mapping[str, Any],
) -> str:
    """Validate and publish one paired same-trial subject result."""
    _bundle_contract_payload(evaluation_contract)
    record_bytes = canonical_common_intersection_subject_bytes(
        record,
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
        expected_evaluation_provenance=expected_evaluation_provenance,
    )
    return _write_result_record_bundle_atomic(
        bundle_path,
        record,
        evaluation_contract,
        "common_intersection_subject_evaluation",
        record_bytes,
    )


def load_common_intersection_subject_evaluation_bundle(
    bundle_path: Path,
    table: Mapping[str, Any],
    ocular_removed_table: Mapping[str, Any],
    ocular_kept_table: Mapping[str, Any],
    ocular_removed_trial_manifest: Mapping[str, Any],
    ocular_kept_trial_manifest: Mapping[str, Any],
    intersection_manifest: Mapping[str, Any],
    charter: Mapping[str, Any],
    feature_contract: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
    prediction_contract: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_removed_bindings: Mapping[str, str],
    expected_kept_bindings: Mapping[str, str],
    expected_intersection_manifest_sha256: str,
    expected_prediction_table_sha256_by_condition: Mapping[str, str],
    expected_comparison_provenance: Mapping[str, Any],
    expected_evaluation_provenance: Mapping[str, Any],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Authenticate and recompute one paired same-trial subject result."""
    record = _load_authenticated_result_record(
        bundle_path,
        evaluation_contract,
        "common_intersection_subject_evaluation",
        expected_bundle_sha256=expected_bundle_sha256,
    )
    validate_common_intersection_subject_record(
        record,
        table,
        ocular_removed_table,
        ocular_kept_table,
        ocular_removed_trial_manifest,
        ocular_kept_trial_manifest,
        intersection_manifest,
        charter,
        feature_contract,
        manifest_contract,
        prediction_contract,
        evaluation_contract,
        expected_removed_bindings=expected_removed_bindings,
        expected_kept_bindings=expected_kept_bindings,
        expected_intersection_manifest_sha256=(
            expected_intersection_manifest_sha256
        ),
        expected_prediction_table_sha256_by_condition=(
            expected_prediction_table_sha256_by_condition
        ),
        expected_comparison_provenance=expected_comparison_provenance,
        expected_evaluation_provenance=expected_evaluation_provenance,
    )
    return record


def write_primary_aggregate_bundle_atomic(
    bundle_path: Path,
    record: Mapping[str, Any],
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> str:
    """Validate and publish the primary aggregate result record."""
    _bundle_contract_payload(evaluation_contract)
    record_bytes = canonical_primary_aggregate_bytes(
        record,
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        expected_aggregate_provenance=expected_aggregate_provenance,
    )
    return _write_result_record_bundle_atomic(
        bundle_path,
        record,
        evaluation_contract,
        "primary_aggregate",
        record_bytes,
    )


def load_primary_aggregate_bundle(
    bundle_path: Path,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Authenticate and recompute the primary aggregate result record."""
    record = _load_authenticated_result_record(
        bundle_path,
        evaluation_contract,
        "primary_aggregate",
        expected_bundle_sha256=expected_bundle_sha256,
    )
    validate_primary_aggregate_record(
        record,
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        expected_aggregate_provenance=expected_aggregate_provenance,
    )
    return record


def write_common_intersection_aggregate_bundle_atomic(
    bundle_path: Path,
    record: Mapping[str, Any],
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
) -> str:
    """Validate and publish the paired ocular aggregate result record."""
    _bundle_contract_payload(evaluation_contract)
    record_bytes = canonical_common_intersection_aggregate_bytes(
        record,
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        expected_aggregate_provenance=expected_aggregate_provenance,
    )
    return _write_result_record_bundle_atomic(
        bundle_path,
        record,
        evaluation_contract,
        "common_intersection_aggregate",
        record_bytes,
    )


def load_common_intersection_aggregate_bundle(
    bundle_path: Path,
    subject_records: Any,
    charter: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    *,
    expected_charter_sha256: str,
    expected_record_sha256_by_subject: Mapping[str, str],
    expected_aggregate_provenance: Mapping[str, Any],
    expected_bundle_sha256: str,
) -> dict[str, Any]:
    """Authenticate and recompute the paired ocular aggregate result record."""
    record = _load_authenticated_result_record(
        bundle_path,
        evaluation_contract,
        "common_intersection_aggregate",
        expected_bundle_sha256=expected_bundle_sha256,
    )
    validate_common_intersection_aggregate_record(
        record,
        subject_records,
        charter,
        evaluation_contract,
        expected_charter_sha256=expected_charter_sha256,
        expected_record_sha256_by_subject=expected_record_sha256_by_subject,
        expected_aggregate_provenance=expected_aggregate_provenance,
    )
    return record
