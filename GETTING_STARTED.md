# Getting started

## 1. Install the environment

This project uses Python 3.12 and `uv`.

```bash
uv sync --locked --all-groups
uv run pytest -q
```

The portable suite uses synthetic fixtures and does not require raw EEG. Data-dependent tests
skip with an explicit reason until PEERS events are available.

## 2. Choose a data location

By default, scripts use `data/` inside the repository. To keep the dataset elsewhere, set:

```bash
export NEUROAI_DATA_ROOT=/absolute/path/to/neuroai-data
```

The expected layout begins as follows:

```text
$NEUROAI_DATA_ROOT/
  peers/
    participants.tsv
    sub-LTP063/
      ses-0/
        sub-LTP063_ses-0_task-ltpFR_events.tsv
        sub-LTP063_ses-0_task-ltpFR_eeg.edf
```

Raw EEG, derived epochs, participant predictions, and authenticated execution bundles are
gitignored and are not distributed in this repository.

## 3. Download a small event sample

Start with behavioral events because they are small and enough to validate the trial labels:

```bash
./scripts/download_peers_events.sh 5 3
PYTHONPATH=src uv run python scripts/check_event_alignment.py
PYTHONPATH=src uv run python scripts/run_baselines.py
```

## 4. Build a small EEG sample

PEERS EEG files are large. Start with one participant and a small number of sessions:

```bash
SESSIONS="0 1 2" ./scripts/download_peers_eeg.sh sub-LTP063
PYTHONPATH=src uv run python scripts/build_one_session.py sub-LTP063 0
```

The full dataset builder is resumable:

```bash
PYTHONPATH=src uv run python scripts/build_dataset.py 5 3
```

The builder deletes each downloaded EDF after verified epoch files are produced to control disk
use. Set `KEEP_EDF=1` if you want to preserve the source EDFs.

## 5. Inspect the model contracts

The scientific choices are represented in machine-readable files:

- `docs/analysis-charter-m1-v1.json`
- `docs/roi-map-m1-v1.json`
- `docs/m1-manifest-contracts.json`
- `docs/m1-feature-table-contract.json`
- `docs/m1-prediction-table-contract.json`
- `docs/m1-evaluation-contract.json`

These contracts fix the feature order, split rules, learner inputs, estimand, seeds, schemas, and
provenance fields used by the implementation and tests.

## 6. Generate outcome-blind manifests

After the required event and epoch sources are present:

```bash
PYTHONPATH=src uv run python scripts/generate_m1_manifests.py \
  --data-root "$NEUROAI_DATA_ROOT" \
  --output "$NEUROAI_DATA_ROOT/m1/manifests/v1"
```

The manifest stage determines the eligible participants, surviving trial keys, and session folds
without computing model performance.

## 7. Inspect the full model graph

The public source includes the complete feature, learner, stacking, evaluation, artifact, and
orchestration modules. The original full run also required personal human-capacity and teach-back
receipts. Those private execution records and participant-level outputs are intentionally not
included in this companion repository.

The synthetic integration tests exercise the same model and artifact contracts without those
private receipts.

## 8. Read the project

- [Full research report](docs/REPORT.md)
- [Research decisions](docs/RESEARCH_DECISIONS.md)
- [Failures and fixes](docs/FAILURES_AND_FIXES.md)
- [Analysis Charter](docs/analysis-charter-m1-v1.md)
- [References](docs/references.md)
