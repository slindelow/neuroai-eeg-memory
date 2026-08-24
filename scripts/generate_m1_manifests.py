"""Generate the frozen outcome-blind M1 manifest run."""

from __future__ import annotations

import argparse
from pathlib import Path

from peers.m1_manifest_generation import generate_real_manifest_run


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_DESTINATION = DEFAULT_DATA_ROOT / "m1" / "manifests" / "v1"
GENERATOR_COMMAND = (
    "PYTHONPATH=src uv run python scripts/generate_m1_manifests.py "
    "--output data/m1/manifests/v1"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    destination = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    data_root = args.data_root if args.data_root.is_absolute() else REPO_ROOT / args.data_root
    digest = generate_real_manifest_run(
        repo_root=REPO_ROOT,
        data_root=data_root,
        destination=destination,
        generator_command=GENERATOR_COMMAND,
    )
    print(f"manifest_run={destination}")
    print(f"index_sha256={digest}")
    print("outcome_metrics_computed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
