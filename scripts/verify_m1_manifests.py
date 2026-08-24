"""Verify a real M1 manifest run against every local source without model evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from peers.m1_manifest_generation import verify_real_manifest_run_against_sources


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RUN = DEFAULT_DATA_ROOT / "m1" / "manifests" / "v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--index-sha256", required=True)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else REPO_ROOT / args.run
    data_root = args.data_root if args.data_root.is_absolute() else REPO_ROOT / args.data_root
    inventory, manifests, index = verify_real_manifest_run_against_sources(
        repo_root=REPO_ROOT,
        data_root=data_root,
        directory=run,
        expected_index_sha256=args.index_sha256,
    )
    print(f"index_sha256={args.index_sha256}")
    print(f"code_commit={index['code_commit']}")
    print(f"source_files={len(inventory['files'])}")
    for condition in ("ocular-removed", "ocular-kept"):
        cohorts = manifests[f"cohort-{condition}"]["rows"]
        trials = manifests[f"trial-{condition}"]["rows"]
        print(f"{condition}_candidate_subjects={len(cohorts)}")
        print(f"{condition}_eligible_subjects={sum(row['eligible'] for row in cohorts)}")
        print(
            f"{condition}_surviving_trials="
            f"{sum(row['condition_survivor'] for row in trials)}"
        )
    print(
        "cross_condition_surviving_trials="
        f"{len(manifests['cross-condition-intersection']['rows'])}"
    )
    print("outcome_metrics_computed=false")
    print("source_reproduction=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
