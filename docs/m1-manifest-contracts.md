# M1 Manifest and Validator Contracts

Status: **FROZEN PRE-RESULT SPECIFICATION + LOCKED, SOURCE-REPRODUCED REAL N2 RUN**
Machine-readable companion: `docs/m1-manifest-contracts.json` (`1.8-frozen`; individual manifest
schema versions remain independently binding)

## Purpose

These contracts make the future M1/B3/M3 comparison auditable before any model performance is
available. They define how to identify eligible subject-condition units, align trial rows, assign
session folds, and prove cross-condition comparability. They enforce the resolved D26-A readiness
boundary and the resolved D29-A analysis window. The ROI map is already frozen in
`docs/roi-map-m1-v1.{md,json}`; validators bind and check it but do not choose or alter it.

“Outcome-blind” here means that inclusion, joins, and folds cannot depend on M1/M3 predictions,
scores, coefficients, or which analysis looks favorable. The binary recall label is permitted
only for the prespecified two-class feasibility check required to define ROC-AUC. No other
label-dependent threshold or exclusion is allowed.

Actual manifests could be generated only after the complete Analysis Charter was approved, taught
back, and hash-bound. That gate passed before the locked N2 run. This document remains the frozen
schema contract; the separate tracked receipt is the eligibility and source-reproduction record.

The large immutable real run is stored under the gitignored personal data root at
`data/m1/manifests/v1`; a small tracked receipt records its external index digest. F53 documents
why the original `findings/` destination was replaced before any run was published. Storage
location does not alter manifest content, cohort rules, or the source-reproduction requirement.

## Implemented outcome-blind structural layer

`src/peers/m1_manifests.py` implements the outcome-blind contract; synthetic unit tests remain in
`tests/test_m1_manifests.py`. `src/peers/m1_manifest_generation.py` and the two scripts under
`scripts/` apply and independently reproduce it against the personal data. The shared layer:

- rejects duplicate JSON keys, non-finite values, unknown fields, wrong types, forbidden
  performance/model fields, invalid hashes, and false schema identifiers;
- produces deterministic canonical bytes and SHA-256 values independent of input row order while
  rejecting a stored manifest whose rows are not already canonical;
- rejects duplicate manifest keys and duplicate included derivative/epoch identities;
- enforces the frozen highest-session outer split, inner-session disjointness/completeness,
  two-class feasibility, and count-unit-safe multi-stage attrition conservation on synthetic rows;
- validates the frozen ROI map against the charter and rejects mismatched order/membership,
  cardinality, overlap, or missing-channel policy;
- compares externally supplied charter/ROI/code/inventory bindings and an expected sidecar digest;
- writes and reads synthetic immutable two-file manifest bundles through one OS-native atomic
  no-replace directory rename, rejecting a competing empty destination, unsupported
  platform/filesystem, partial, extra, noncanonical, overwritten, or digest-mismatched artifacts;
- recomputes canonical event/trial key-set equality, the exact two-condition survivor intersection,
  complete-row and behavioral-event provenance hashes, and paired B3/M1/M3 outer-test prediction
  keys on synthetic inputs;
- binds inner-fold fit/validation key hashes to the trial manifest and proves, on synthetic inputs,
  that B3 and M1 each fit only the declared other training sessions and predict every
  outer-training survivor exactly once;
- binds D44-C's single outer-training-wide M1 `C` selection to the exact grid, selected value,
  metric, tie rule, sessions, key digests, and inner-fold-manifest digest; explicitly records that
  predicted-session labels may influence selection while the outer test does not; and rejects the
  stronger fully session-cross-fitted claim;
- rejects a Boolean human-gate bypass and requires a closed, final-charter-bound receipt of all
  nine Sofia-owned protocol responses, their source/digest, and a completed advisor review; and
- requires the exact strict-inductive D26-A mode with no held-session/label access or second M1
  scaler, the resolved D29-A bounds, resolved D42/D44 decision IDs, an exact frozen status, a
  recomputed canonical charter digest, and an accepted teach-back receipt before real generation;
  and
- requires the charter and manifest contract to contain D47's exact bounded-preflight/full-run
  capacity policy. Preflight cannot report outcome metrics, and a later validated budget receipt
  plus Sofia's capacity go/no-go are required before the full locked run; and
- treats D42-C as a valid Sofia choice but not an executable value under the current contract:
  manifest readiness stops until its required cohort/threshold re-plan is versioned [D49/F37].

The type schema uses `dirty_diff_sha256 = null` for clean code and requires a lowercase SHA-256
when code is dirty. Exact schema IDs and versions are binding, not descriptive labels.

The synthetic layer alone proves only decision-independent comparison logic on invented rows. The
locked N2 run adds real event/FIF extraction, real cohort/trial/fold/intersection/attrition counts,
immutable publication, and independent exact regeneration from all 376 bound sources. Its index
SHA-256 is `3c4e6fe36258e3154b3812d14fd1835075ef0385324513c941b26beb9beb3712`;
the tracked human- and machine-readable receipts are in `findings/m1/`. This still does **not**
prove real spectral feature recomputation, learner consumption, prediction serialization,
probability recomputation, or evaluation/report consumption. The companion feature contract
supplies the exact inner/outer D26 transform-provenance validator; those real N4+ artifacts remain
future evidence.
The readiness guard detects D26/D29/D42-A/D44-C drift or stale pending-gate labels, a non-final
status, a charter-digest mismatch, and invalid transcript/receipt bindings. The frozen N1 artifacts
satisfy that human-gate boundary, and the N2/N3 source-reproduced run closes AC-04. AC-05/06 remain
open because real feature, fitting, prediction, and leakage-audit evidence does not yet exist.

## Common envelope and canonicalization

Every manifest is UTF-8 JSON with one top-level object containing:

- `schema_id`, `schema_version`, `manifest_kind`, and `created_at_utc`;
- `charter_id`, `charter_version`, and the final `charter_sha256`;
- `roi_map_id` and `roi_map_sha256`;
- `code_commit` plus `dirty_diff_sha256` when the code state is not clean;
- `generator_command`, `source_inventory_sha256`, and `rows`.

Exact envelope/row types, schema IDs/versions, fixed conditions, reason codes, and externally
checked binding fields live in the machine-readable companion. A clean repository records
`dirty_diff_sha256: null`; a dirty repository records the SHA-256 of the exact dirty patch.

Rows are sorted by the manifest-specific key below. Hashing uses canonical JSON with recursively
sorted object keys, preserved array order, no insignificant whitespace, and a final newline. The
SHA-256 is written to a sidecar file to avoid a self-referential digest. Timestamps and absolute
machine paths are recorded as provenance but excluded from row identity.

### Immutable atomic bundle

Each stored manifest is a new directory containing exactly `manifest.json` and `manifest.sha256`.
The writer validates and canonicalizes in memory, writes and fsyncs both files in a temporary
sibling directory, fsyncs that directory, and publishes it with one native exclusive rename:
`renamex_np(RENAME_EXCL)` on Darwin or `renameat2(RENAME_NOREPLACE)` on Linux. Existing
destinations—including empty directories created during publication—are never overwritten.
Unsupported runtimes or filesystems fail closed rather than falling back to ordinary rename.
Readers reject missing/extra entries,
directories or symbolic links in place of regular files, noncanonical bytes, malformed sidecars,
digest mismatches, schema drift, or external-binding drift.

The single exclusive rename prevents a cooperating reader from observing a published half-pair
and closes F20's time-of-check/time-of-use overwrite race on the tested local macOS filesystem.
It is not a claim of durability across every filesystem, network filesystem, hardware, or power
failure mode. Hidden temporary directories are not valid bundles and a future real run must still
reconcile every manifest against its sources.

### Protocol teach-back transcript and receipt

`docs/protocol-teach-back-m1-v1.md` is the pre-result worksheet and advisor rubric. The authoritative
capture uses transcript schema `neuroai-m1-protocol-teach-back-transcript-v1` version `1.0` and
receipt schema `neuroai-m1-protocol-teach-back-v1` version `1.1`. The transcript preserves exact
message text/hashes, roles, topics, kinds, dispositions, ordering, reply links, correction trails,
and Sofia's explicit capture approval. The receipt selects the accepted Sofia-owned message IDs
for every topic and binds the charter ID/version/SHA-256, decision-log SHA-256, source task,
transcript digest, approval message ID, canonical response digest, record/review times, per-topic
verdicts, overall verdict, and open revision requests.

The validator rejects missing/extra topics or fields, placeholder or unreviewably short text,
stale charter/decision-log/transcript binding, message-text drift, invalid or forward reply links,
advisor or superseded response selection, response text not exactly derived from the selected
messages, absent explicit Sofia approval, a changed response digest, a reviewer listed as the
response author, a missing scientific-advisor role, a review that predates the response, any topic
not marked `pass`, a non-passing overall verdict, any open revision request, and a bare Boolean in
place of the receipt. This proves only schema
completeness, provenance, and recorded acceptance. It cannot prove genuine comprehension; the
advisor must evaluate Sofia's explanations conversationally and may not write them for her.

Validators reject unknown fields, duplicate keys, non-finite numbers, wrong types, missing
source hashes, unsorted rows, or a digest mismatch. Validation never repairs, imputes, or silently
drops a row.

## 1. Cohort manifest

One row per `(subject_id, condition)`.

Required fields:

- `subject_id`, `condition` (`ocular-removed` or `ocular-kept`);
- `sessions_present`, `sessions_technically_eligible`, `outer_train_sessions`, and
  `outer_test_session`;
- `eligible` and exactly one `eligibility_reason_code`;
- per-session counts for studied rows, matched epoch rows, class 0, class 1, and exclusions;
- derivative/event/metadata source SHA-256 values and channel-schema identifier.

Allowed eligibility reason codes are fixed and technical: `ELIGIBLE`, `TOO_FEW_SESSIONS`,
`MISSING_DERIVATIVE`, `UNREADABLE_METADATA`, `CHANNEL_SCHEMA_MISMATCH`, `NONFINITE_INPUT`,
`TRIAL_EPOCH_KEY_MISMATCH`, `SINGLE_CLASS_OUTER_TRAIN`, `SINGLE_CLASS_OUTER_TEST`, or
`SINGLE_CLASS_INNER_FOLD`. Free-text reasons are forbidden.

For an eligible row, the outer test session equals the highest technically eligible session;
all lower technically eligible sessions are outer training sessions; there are at least three
eligible sessions; and every declared fit/score partition has both recall classes.

## 2. Trial manifest

One row per `(subject_id, session, trial, item_num, condition)` after strict event-to-epoch
alignment. The first four fields are the charter's canonical trial key; condition is appended
only because the manifest stores processing-specific survivors.

Required fields:

- `subject_id`, `session`, `trial`, `item_num`, `condition`;
- `serial_position`, `list_number`, `onset_s`, and boolean `recalled`;
- source events path identifier and SHA-256; derivative identifier and SHA-256; `epoch_row`;
- `included`, one `exclusion_reason_code`, and `condition_survivor`;
- `outer_role` (`train`, `test`, or `excluded`).

Practice lists and negative-sample study events never enter included rows. Each included event
matches exactly one epoch row. No array position, data-frame index, or current sort order can act
as an implicit join key.

## 3. Inner-fold manifest

One row per `(subject_id, condition, inner_fold_id)`.

Required fields:

- `subject_id`, `condition`, `inner_fold_id`;
- `fit_sessions`, `validation_session`, and `outer_test_session`;
- fit/validation class counts and canonical-key SHA-256 values;
- `eligible` and the fixed feasibility reason code.

Each outer-training session appears as validation exactly once. Its validation keys are disjoint
from its fit keys. The outer test session appears in neither set. The union of a fold's fit and
validation keys is the complete outer-training key set, and every eligible partition contains
both classes.

## 4. Cross-condition intersection manifest

One row per common canonical trial key, without condition in the identity.

Required fields:

- `subject_id`, `session`, `trial`, `item_num`;
- ocular-removed and ocular-kept trial-row SHA-256 values;
- shared label and event-provenance SHA-256;
- each condition's epoch-row identifier.

The validator recomputes the exact set intersection. A label/provenance disagreement or a row
present in only one condition cannot be included. Condition-specific results do not use this
intersection; only direct processing-condition comparisons do.

## 5. Attrition manifest

Schema v1.2 uses one row per `(condition, flow_id, stage_index)`. Each row records `count_unit`,
the fixed stage name, `input_count`, an `excluded_by_reason` integer map, and `survivor_count`.
Within a stage, input minus the sum of named exclusions equals survivors; within a flow, each
stage input equals the preceding stage survivor count. Zero-count reasons are omitted, unknown
reasons fail, and no catch-all `other` category exists.

Count units are never mixed:

- `trial_analysis` counts trials through behavioral filtering, condition technical survival,
  exact event–epoch alignment, and eligible-subject retention. At alignment, the explicit
  `AUTOREJECT_DROPPED` reason records planned D14 preprocessing exclusions; the separate
  `TRIAL_EPOCH_KEY_MISMATCH` reason is reserved for metadata that cannot be reconciled to exactly
  one behavioral presentation [F51];
- `subject_condition_eligibility` counts subject-condition units through technical session,
  minimum-session, outer-class, and inner-fold-class gates; and
- `cross_condition_sensitivity` counts trials and binds its input to the final condition-specific
  trial survivor count before removing keys absent from the other processing condition.

Every condition must contain every declared flow and stage. Each stage's ordered fixed reason list
also defines first-applicable precedence, so a unit is excluded once rather than double-counted.
Version 1.1 fixed F18's unit/reason conservation problem. Version 1.2 fixes F51's missing
AutoReject vocabulary before any real attrition manifest exists; it does not change the D14/D23
survivor rule.

## Forbidden content

Pre-fit manifests may not contain predictions, probabilities, AUC, IDI, NRI, Brier score,
balanced accuracy, coefficients, chosen hyperparameters, model ranks, favorable/adverse labels,
or any field derived from held-out M1/M3 performance. Feature values also belong in versioned
caches, not manifests.

## Validator test specification

All tests use synthetic fixtures unless a read-only inventory check is explicitly named. They
must not fit a real-data model or calculate held-out M1/M3 performance.

1. **Canonical determinism:** permuting input-file or input-row order produces byte-identical
   canonical rows and the same SHA-256.
2. **Duplicate key rejection:** duplicate study events, recall matches, epoch rows, or manifest
   identities fail with the offending key.
3. **Strict alignment:** missing/extra event or epoch rows fail; row reordering alone does not
   change alignment.
4. **Outer split:** the validator accepts only the highest eligible session as test and proves
   train/test key disjointness and completeness.
5. **Inner split and first-level model provenance:** every outer-training session is validation
   once; B3/M1 fit and predicted key sets match the declared fit/validation sessions; stored key
   hashes recompute; every training survivor is predicted exactly once; outer-test keys are absent.
   D26 transform fit/apply scope is separately enforced by the feature-provenance validator.
6. **Class feasibility:** a synthetic single-class train, validation, or test partition receives
   its exact fixed reason code; no alternate subject/session is substituted.
7. **Cross-condition intersection:** the stored keys equal a recomputed exact intersection and
   disagreeing labels/provenance fail.
8. **Attrition conservation:** both conditions contain every fixed trial, subject-condition, and
   cross-condition stage; named exclusions sum within stage; within-unit stages chain; the
   cross-condition input binds to condition survivors; unit mixing, unexplained loss, or double
   counting fails.
9. **Schema closure:** unknown fields and every forbidden performance/model field fail.
10. **Atomic bundle and tamper detection:** invalid input publishes no final bundle; a deterministic
    competing-empty-destination race preserves the competitor and removes the temporary bundle;
    unsupported platforms fail closed; overwrite, partial/extra/non-file entries, noncanonical
    bytes, or a changed source hash, row value, charter/code binding, or sidecar digest invalidates
    readback.
11. **Pending-gate fail-closed:** D26/D29 contract drift, unresolved or undeclared D42/D44 values,
    stale gate labels, a non-final status, a canonical charter-digest mismatch, or a missing,
    mismatched, overlapping, or invalid ROI-map binding prevents real manifest generation even
    though this contract document itself validates.
12. **Execution-gate fail-closed:** removing or weakening outcome-blind preflight selection,
    measurement fields, time/disk/memory/retry/resume formulas, the preflight receipt, Sofia's
    capacity approval, the current-run cutoff source, or the `IN_PROGRESS` failure rule prevents
    readiness.
13. **Protocol teach-back transcript and receipt:** a Boolean cannot satisfy the human gate;
    missing/extra or placeholder topics, invalid message hashes/order/replies, advisor or
    superseded authorship, absent Sofia approval, stale charter/transcript binding, response
    rewriting, review-time inversion, non-passing topic/overall verdicts, or open revisions fail.
    A complete synthetic transcript/receipt passes structurally without claiming that software can
    measure comprehension.
14. **Future paired-prediction gate:** before scoring, B3, M1, and M3 prediction keys must equal
    the locked condition-specific survivor keys exactly; the scoring function rejects filtering.
15. **Baseline reference exclusion:** a synthetic perturbation of every outcome from a target
    subject leaves that subject's word-memorability values unchanged, including the smoothing
    prior. F17's three focused synthetic regressions now pass; this remains a required precondition
    for B3 refitting and must stay green after future evaluator changes.

## Stop/replan gate

Stop before manifest generation if the D26-A or D29-A contract drifts; if the frozen ROI map fails
channel/schema validation; if the execution-gate policy drifts; if event/epoch identity is ambiguous;
or if technical eligibility would need a performance-based exception. Record the failure and
revise the charter or contract with Sofia before inspecting any model result.
