# M1 v1 evaluation-consumer contract

Status: **D42-A/D44-C-BOUND PRE-FINAL-CHARTER SYNTHETIC SUBJECT + PRIMARY + PAIRED COMMON-INTERSECTION AGGREGATE CONSUMERS WITH AUTHENTICATED RECORD STORAGE — no real metric or outcome calculated**

Machine-readable companion: `docs/m1-evaluation-contract.json` (`1.3-draft`)

## Purpose

D34 says that every result must come from one validated paired B3/M1/M3 outer-test table. The
existing metric utilities accepted arbitrary arrays, so a future caller could accidentally reorder,
filter, or substitute labels or probabilities after table validation. This contract closes that
boundary for one subject and condition: validate the complete table first, then pass every stored
canonical row to the fixed metric functions and bind the record back to the table digest.

This layer does not change the resolved D26-A/D29-A/D42-A/D44-C contracts, inspect a real outcome,
or compute balanced accuracy. It defines the synthetic-only paired aggregate and
discordance modifier needed to compare ocular conditions on exactly the same subjects, plus an
authenticated physical container for each validated result-record kind.

## F23 and D35: table-bound subject metrics

F23 records that correct standalone metric formulas are insufficient evidence of correct metric
inputs. D35 therefore requires:

- the complete prediction-table validator and independent external bindings to pass first;
- labels and B3/M1/M3 probabilities to be extracted from every canonical row without filtering;
- the primary subject comparison to remain `M3 − B3` on identical rows;
- B3, M1, and M3 AUC and Brier scores plus M3-versus-B3 delta AUC, IDI, continuous NRI, and paired
  DeLong statistics;
- exact trial/class counts and the prediction-table SHA-256 in a closed record;
- a SHA-256 of this complete evaluation contract; and
- independently supplied evaluator code, dirty-diff, package, and command provenance.

Validation recomputes the complete subject record from the validated table. A self-reported digest
or plausible metric value is not accepted.

## Balanced accuracy is deliberately not computed

Sofia selected D42-A: omit balanced accuracy. The selecting message was exactly `A`; it supplied no
Sofia-authored rationale, so the advisor recommendation remains separate; her later explanation is
preserved in the accepted teach-back transcript and receipt. The machine-readable contract marks balanced accuracy
`OMITTED_BY_D42_A_NOT_COMPUTED`. It defines no threshold, predicted-label field, metric function,
or result field. Reintroducing balanced accuracy requires a new pre-result charter version and
cannot be motivated by an observed result.

Brier score has no threshold and is therefore safe to bind now. All implemented metrics were
already named by the charter and D25; this contract changes their input provenance, not their
scientific role. Secondary metrics still cannot rescue a non-positive primary result.

## F24 and D36: exact primary aggregation

D35 ends with one validated record per subject and condition. F24 records that correct subject
records still do not prove the headline estimand: pooling trials, weighting subjects by held-out
trial count, resampling trials, or assigning a category from the point estimate could all change
the reported conclusion.

D36 therefore implements only the primary ocular-removed aggregate and requires:

- an externally supplied exact subject-to-record-digest map with no missing, extra, duplicate, or
  stale subject record;
- a single charter, evaluation-contract, and complete evaluator-provenance binding across the
  complete subject set;
- the unweighted mean of per-subject `AUC(M3) − AUC(refit B3)`, never pooled or trial-weighted;
- 2,000 subject-level bootstrap resamples from `numpy.random.default_rng(0)`;
- the 2.5th and 97.5th percentiles using NumPy's explicit `linear` method;
- `positive_increment` only when the lower bound is above zero, `adverse_increment` only when the
  upper bound is below zero, and `no_detectable_increment__imprecise` otherwise; and
- closed false flags for secondary-metric rescue and equivalence claims.

The aggregate refuses ocular-kept records. This is intentional: the charter permits ocular-kept
only as sensitivity evidence. Ocular discordance is never approximated from the condition-specific
D35 records.

## F26 and D38: paired common-intersection subject metrics

D37 proves that both conditions' probabilities are aligned to the same held-out trials, but a
metric caller could still bypass that table or calculate each condition after different filtering.
D38 therefore requires one paired record per subject and common outer-test session that:

- fully revalidates and hashes the D37 table plus both source tables and the intersection manifest;
- passes every stored common row to the frozen D25/D35 metric set once for ocular-removed and once
  for ocular-kept;
- binds both metric sets, class/trial counts, source fold identities, evaluation-contract digest,
  and evaluator provenance in one closed record;
- fails when the common intersection loses either recall class, even if both condition-specific
  source tables remain individually feasible; and
- forbids balanced accuracy, interval categories, aggregate fields, and `condition_discordant`.

Complete recomputation validates every stored field. D38 creates the exact subject-level input for
D43; it does not define the eligible real subject set or calculate a real interval.

## F30 and D43: paired common-intersection aggregation

Valid paired subject records still did not force the sensitivity comparison to use the complete
same subject set, equal subject weights, or one paired bootstrap. F30 records that independently
aggregating conditions could produce different uncertainty purely from different resampled
subjects and could let a sensitivity result drift into the primary conclusion.

D43 therefore requires:

- an externally supplied exact subject-to-paired-record-digest map with no missing, extra,
  duplicate, stale, charter-drifted, contract-drifted, or mixed-provenance record;
- one canonically ordered common subject set for both conditions and equal weight per subject;
- the mean subject `AUC(M3) − AUC(refit B3)` separately for ocular-removed and ocular-kept;
- one seed-0 matrix of 2,000 subject-bootstrap indices applied to both condition columns;
- the same interval-only categories as D36 for each condition;
- `condition_discordant` only when the two point estimates have strictly opposite signs or their
  interval categories differ, with exact zero never treated as an opposite sign; and
- closed flags that sensitivity cannot change the primary category, create a positive path, imply
  equivalence, or include balanced accuracy.

The complete record is recomputed before canonical serialization or SHA-256 binding. D43 reports
whether the sensitivity disagrees; it does not average conditions, replace D36's primary result,
or make ocular correction part of the headline estimand.

## F44 and D53: authenticated physical result-record storage

The four logical result records above were canonical and digestible, but their contract selected
no physical representation. F44 records that a future evidence chain could therefore retain a
logical digest while reloading an extra, stale, substituted, noncanonical, or incompletely bound
file. D53 closes only that technical gap.

Each condition-specific subject record, paired common-intersection subject record, primary
aggregate, and paired common-intersection aggregate is stored as its own immutable directory with
exactly `record.json`, `bundle.json`, and `bundle.sha256`. The manifest binds the storage-contract
digest, complete evaluation-contract digest, record kind, record schema identity, and canonical
record digest. Readback requires an independently supplied bundle digest, rejects extra or
symlinked members and duplicate-key/noncanonical/nonfinite JSON, and then calls the record-kind's
existing complete logical validator. Aggregate reload therefore still requires the exact bound
subject-record set; subject reload still requires the authenticated source prediction/comparison
evidence.

The format inherits the exact fields and schema from the supplied evaluation contract and now
binds D42-A's deliberate absence. The storage policy contains no threshold or balanced-accuracy
field, while the complete evaluation-contract digest changes if any record schema drifts. The N1
human/frozen-charter prerequisite is satisfied; real publication remains forbidden until frozen
real source artifacts and the later execution gates pass.

Canonical JSON is used rather than a typed array because each artifact is a small heterogeneous
closed record whose scientific type/field constraints are already recomputed by its logical
validator. A binary matrix, pickle, database, or columnar dependency would add another schema
without improving numeric identity or recovery. Publication reuses the verified fsync and atomic
no-replace filesystem boundary; existing destinations are never overwritten and failed staging
evidence is retained.

## Verification ladder

| Level | Evidence | Claim | Status |
| --- | --- | --- | --- |
| Formula unit tests | Invented arrays | Existing AUC/IDI/NRI/DeLong formulas behave as specified | Existing |
| Subject consumer | Invented manifest/table/contracts | Validation precedes metrics; every row is consumed; output is closed, recomputable, and digest-bound | Implemented |
| Real integration | Frozen real artifacts after all human/hash gates | Stored probabilities and evaluator provenance reproduce every subject record | Deferred |
| Primary aggregate | Invented validated subject-record set | Exact cohort digests, unweighted mean, 2,000-resample seed-0 bootstrap, and interval categories | Implemented synthetic-only |
| Ocular comparison table | Invented condition tables plus intersection manifest | Exact same-trial probabilities and source provenance, with no metric | Implemented synthetic-only [D37/F25] |
| Ocular subject consumer | Invented D37 table and full source evidence | Every common row produces both closed subject metric sets; one-class intersections fail | Implemented synthetic-only [D38/F26] |
| Ocular aggregates | Invented exact common-intersection subject-record set | Equal subject weights, paired bootstrap indices, both interval categories, and the discordance modifier use one subject set | Implemented synthetic-only [D43/F30] |
| Result-record storage | Invented records plus complete source evidence | External/member authentication, exact canonical JSON, immutable atomic publication, and full logical recomputation for all four record kinds | Implemented synthetic-only [D53/F44] |

## What synthetic success does not prove

The tests do not prove that a real probability came from its named model, that normalization used
permitted rows, that a real prediction bundle was read without dtype loss, that the complete cohort
is present, that an evaluator process used the recorded code/environment, or that real comparison,
subject, and aggregate records use the required common trial and subject intersections. Those
require independent source-to-result execution, authenticated real result-record bundle readback,
and proof that downstream reporting consumes only those validated bundles after frozen real inputs
and provenance validate.

No real label, feature, prediction, coefficient, score, metric, or held-out outcome is loaded by
this contract.
