import csv
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
    with (ROOT / "results" / "summary.csv").open(newline="", encoding="utf-8") as handle:
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
