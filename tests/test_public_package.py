import csv
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_report_prioritizes_model_and_data_work() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    report = (ROOT / "docs" / "REPORT.md").read_text(encoding="utf-8")

    for required in (
        "Converting an experiment into a prediction table",
        "Building the behavioral baseline first",
        "Designing an interpretable EEG representation",
        "Preventing normalization leakage",
        "Building the three-model comparison",
        "Building reproducibility into the model",
    ):
        assert required in report

    assert "The result was secondary to the main challenge" in readme
    assert "not fully session-cross-fitted" in readme
    assert "does not establish equivalence" in readme


def test_public_narrative_has_no_private_paths_or_non_personal_identity() -> None:
    checked = [
        ROOT / "README.md",
        ROOT / "GETTING_STARTED.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    prohibited = (
        str(Path.home()),
        str(Path("/", "Volumes")) + "/",
    )
    for path in checked:
        text = path.read_text(encoding="utf-8")
        for value in prohibited:
            assert value not in text, (path, value)


def test_sofia_voice_public_narrative_avoids_en_and_em_dashes() -> None:
    checked = [
        ROOT / "README.md",
        ROOT / "GETTING_STARTED.md",
        ROOT / "docs" / "REPORT.md",
        ROOT / "docs" / "RESEARCH_DECISIONS.md",
        ROOT / "docs" / "FAILURES_AND_FIXES.md",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert "–" not in text, path
        assert "—" not in text, path


def test_public_summary_matches_authenticated_primary_result() -> None:
    summary_path = ROOT / "results" / "summary.csv"
    provenance = json.loads(
        (ROOT / "results" / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["schema_version"] == "neuroai_public_result_provenance_v1"
    assert provenance["scientific_classification"] == "exploratory"
    assert provenance["summary_csv_sha256"] == sha256(summary_path.read_bytes()).hexdigest()
    assert provenance["frozen_scientific_commit"] == (
        "62b99f790ec1cb7888f8d0899f816b47779498a3"
    )
    assert provenance["verified_run_index_sha256"] == (
        "a4d1e14637bd91b91e1e5dc7087cf51729ca90f1e2d306103091df03fcf38c75"
    )
    assert provenance["independent_verifier_receipt_sha256"] == (
        "6bf86b6117308213a002fcedec1bdf45a538db2a89cf965165b885fe92b1f402"
    )
    assert provenance["clean_reproduction_equality_receipt_sha256"] == (
        "c557d33e55f3fad318af17d005b730b50b18679959a12787145bb762f0ab6c5d"
    )
    assert provenance["primary_aggregate_record_sha256"] == (
        "e9c1fc58bc065c0f9d446436ae86df30fe4319f8aa0f2a224265f5d85c6eae57"
    )
    assert provenance["condition_sensitivity_record_sha256"] == (
        "5dddfa08e75aba63896c1f546326a3d4f3086c4282556e3c4ff527fae8cd6e7f"
    )

    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    primary = next(
        row
        for row in rows
        if row["measure"] == "mean_delta_auc_m3_minus_b3"
        and row["condition"] == "ocular_removed"
    )

    assert float(primary["estimate"]) == -0.0043799736
    assert float(primary["interval_low"]) == -0.0121122622
    assert float(primary["interval_high"]) == 0.0032501280
    assert primary["n_participants"] == "38"
    assert primary["status"] == "no_detectable_increment_imprecise"
