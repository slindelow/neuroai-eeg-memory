# Analysis Charter M1 v1

Status: **FROZEN PRE-RESULT — D26-A/D29-A/D42-A/D44-C and Sofia's teach-back recorded**
Version: 1.3-frozen
Decision owner: Sofia
Technical custodian: the active NeuroAI orchestrator
Primary machine-readable companion: `docs/analysis-charter-m1-v1.json`

This charter defines the smallest interpretable EEG model that can answer NeuroAI's central
question. It is written before any held-out M1 or M3 outcome is computed or inspected. A null,
adverse, imprecise, or condition-discordant result is acceptable.

The resolved fields below are locked against outcome-driven change. Sofia completed the protocol
teach-back in her own words and approved the complete verbatim capture on 2026-08-17. A new
scientific choice after this freeze creates a new
charter version and invalidates all dependent manifests, caches, predictions, and results.

## 1. Scientific purpose and claim ceiling

**Question.** Does spectral EEG recorded while a person studies a word add held-session
predictive information about that word's later immediate free recall beyond what the word and
its serial position already predict?

**Scientific value.** The intended contribution is the incremental comparison, leakage controls,
transparent feature model, reproducibility, failure record, and justified uncertainty—not a high
accuracy number.

**Exploratory status.** R4 used recall labels from the 40-subject EEG cohort and influenced the
2 Hz lower bound and two-bin design. This charter therefore governs exploratory model development,
not untouched confirmation. Holding out sessions prevents train/test contamination but does not
undo feature-family development on labeled cohort-level evidence.

**Permitted claim ceiling.** “In this prespecified exploratory processing pipeline, study-period
EEG did/did not add held-session predictive information about immediate free recall beyond word
and serial-position predictors.”

**Forbidden claims.** No confirmatory, causal, cortical-source, neural-mechanism, clinical,
real-time, memory-formation, new-participant-generalization, or publication-ready claim. Immediate
free recall is a behavioral outcome involving both encoding and retrieval, not a direct measure of
whether a memory was formed [D5].

## 2. Data, unit of analysis, and outcome

- Dataset: personal local PEERS `ltpFR` derivatives only; no `ltpFR2` or `VFFR` mixing [D3].
- Unit of prediction: one studied-word presentation within one participant and session.
- Binary outcome: `recalled = true` only when a `REC_WORD` event with `item_num > 0` matches the
  studied presentation on `subject_id`, `session`, `trial`, and `item_num` [D5–D7].
- Practice lists (`trial <= 0`) and studied-word presentations with `sample < 0` are excluded in
  the shared trial-table builder [F16]. Final free recall and recognition do not relabel trials.
- Canonical trial key: `(subject_id, session, trial, item_num)`. `serial_position`, `onset_s`,
  condition, source file, and epoch row are provenance columns, never implicit join keys.
- Every manifest and prediction table must have unique canonical keys. Joins are one-to-one and
  fail on duplicates, missing rows, extras, or order-dependent alignment.

## 3. Conditions, cohort, and paired rows

- Primary EEG condition: `ocular-removed`.
- Sensitivity condition: `ocular-kept`.
- Condition-specific technical eligibility is determined before fitting from derivative presence,
  readable metadata, finite samples, channel/schema compatibility, and fold/class feasibility.
- A subject-condition requires at least three complete sessions: the highest eligible session is
  the outer test session and at least two earlier sessions form the outer training set.
- Each outer train/test partition and each inner train/validation fold must contain both recall
  classes. Failure excludes that subject-condition before model fitting and records attrition; no
  replacement, outcome-performance threshold, or named-subject exception is permitted.
- B3, M1, and M3 are fitted and scored on identical condition-specific survivor rows and folds.
  B3 is refit; the 78-subject historical B3 score is context, not the comparator score [D23].
- Direct ocular-condition comparisons use only the locked cross-condition intersection of
  canonical keys. Condition-specific results retain their own cohort/attrition tables [D15].
- Current file inventory is descriptive, not a frozen cohort: 232 derivative files cover 40
  subject directories; 38 currently contain three sessions in both conditions and two contain one
  session in both. The validator, not this count, determines the final outcome-blind manifest.

## 4. Split and cross-fitting contract

### Outer evaluation

For each eligible subject-condition, hold out the highest numbered eligible session. All earlier
eligible sessions are outer training data. The outer test session is never used for model fitting,
hyperparameter selection, threshold selection, calibration, feature selection, or stopping.

### Inner model selection

- Inner folds leave one outer-training session out at a time; there is no random trial split.
- Selection score is the unweighted mean validation ROC-AUC across feasible inner sessions.
- Hyperparameter ties are resolved deterministically in favor of the smallest `C` (strongest L2
  regularization). If scores remain tied, use the earliest value in the declared grid.
- After selection, refit the first-level learner on every outer-training row and predict the outer
  test rows exactly once.

### Cross-fitted stack

- M3 receives two first-level probabilities: B3 and M1.
- Every M3 meta-training row receives B3 and M1 probabilities from models and learned transforms
  that did not fit that row. The cross-fit unit is session.
- The inner fold ID, first-level fit-key hashes, selected hyperparameters, transform fit-key hashes,
  and prediction provenance are stored per meta-training row.
- Outer-test B3 and M1 probabilities come from first-level models refit on all outer-training rows.
  The M3 meta-model is fit only on cross-fitted outer-training predictions and then applied once to
  those two outer-test probabilities.
- No in-sample first-level prediction may enter M3 training. Missing cross-fit predictions, row-key
  mismatch, or a fit hash containing the predicted row fails the subject-condition.
- The decision-independent validator recomputes B3/M1 model fit and predicted key sets from the
  trial and inner-fold manifests, checks their stored hashes, and requires exact-once coverage of
  outer-training survivors. D26 now forbids held-session feature statistics in the primary path;
  exact inner/outer transform fit and apply keys require separate provenance validation before real
  feature generation.
- **D44-C is resolved with qualified wording:** each M1 meta-prediction's coefficients and learned
  transform exclude the predicted session, but one M1 `C` is selected from the complete outer-
  training set's inner-session validation scores. The predicted session's labels may therefore
  influence that shared `C`. The exact permitted claim is “session-out coefficients and transforms
  conditional on an outer-training-selected `C`”; these predictions may not be called fully
  session-cross-fitted [D44/D55/F31/F47].
- The D44-C provenance record binds the complete candidate grid, selected `C`, selection metric,
  tie rule, outer-training fit/validation sessions and key digests, exact inner-fold-manifest
  digest, predicted sessions included in selection validation, and explicit outer-test exclusion.
  The outer test session enters neither hyperparameter selection nor coefficient/transform fitting.

## 5. Feature contract

### Locked fields

- Source: immutable cached `mne.Epochs` derivatives; no preprocessing change under this charter.
- Spectral estimator: MNE Morlet time-frequency power, `n_cycles = frequency / 2`, `decim = 4`,
  matching the verified R4 implementation.
- Frequencies: eight values from `numpy.logspace(log10(2), log10(100), 8)`, ascending. The exact
  vector and full-precision serialization live in the JSON companion.
- Transform before normalization: `log10(power + 1e-30)`.
- Aggregation: arithmetic mean of normalized log power across all samples in the declared time bin
  and all channels in the declared ROI. No label enters feature extraction.
- Feature order: time-bin order, then declared ROI order, then ascending frequency.
- Byte-stable axis IDs: time bins are ordinal `t0`, `t1`; frequencies are ordinal `f00` through
  `f07`. These IDs do not encode numeric windows/frequencies: the exact Hz vector and frozen D29-A
  boundaries remain separate fields in the feature-schema digest [D33/D41].
- Feature name: `eeg__time-{time_bin_id}__roi-{roi_id}__freq-{frequency_id}`.
- Output: one `float64`, finite value per canonical trial key and feature; no implicit index column.
- Missing required channel, duplicate channel, non-finite value, mismatched sampling/time vector,
  or feature-count/order mismatch fails that subject-condition. No outcome-dependent imputation.

### Resolved analysis window — D29

- Sofia approved D29-A: use 300–1600 ms after word onset with the exact midpoint at 950 ms.
- `t0` is `[0.300, 0.950)` seconds and `t1` is `[0.950, 1.600]` seconds. Any sample exactly at
  950 ms belongs only to `t1`; the implementation uses exact comparisons with no hidden tolerance.
- Advisor rationale, not Sofia-authored teach-back: this has the closest PEERS-family item-level
  spectral precedent, excludes the earliest sensory/evoked interval, limits flexibility and
  dimensionality, and avoids the outcome-sensitive 800 ms boundary [D29].
- Limitation: meaningful activity may exist in 0–300 and 1600–2000 ms. Primary M1 does not test
  those intervals, and the chosen window must not be described as the stimulus duration. R4's
  prior labeled influence keeps v1 exploratory.

### Resolved normalization — D26

- Primary mode: strict inductive normalization. Compute `log10(power + 1e-30)`, then fit a mean and
  population standard deviation (`ddof=0`) over trial and time axes separately for each of the 40
  declared ROI channels and each frequency using only the model's permitted training sessions.
- Inner selection: each inner-training session set fits its own transform; its validation session
  is transformed with those statistics unchanged.
- Outer refit: all outer-training sessions fit one transform; the held-out highest session is
  transformed with those statistics unchanged.
- A zero training standard deviation is replaced by one before centering, matching the intended
  stable no-division behavior. Recall labels never enter the transform, and no validation/test
  feature distribution contributes to its statistics.
- The outcome-blind numerical implementation binds the immutable mean, raw standard deviation,
  effective scale, and zero-cell mask by typed array hashes and records the zero-cell count and
  time samples per fit trial. Synthetic tests prove unchanged application and target-batch
  invariance; real source-array recomputation remains required [D40/F28].
- There is no second scaler on the aggregated 128-feature EEG block. Optional D26-B whole-held-
  session batch normalization remains a prespecified transductive sensitivity only if the preflight
  compute bound permits it; it can never replace or rescue the primary strict-inductive result.
- Sofia's reason: “the primary question should be whether a model trained on past sessions can
  predict recall in a genuinely unseen session without using that session's EEG distribution.”

### Frozen scalp-ROI map — D30

The ROI axis is fixed to eight non-overlapping five-channel sensor-space regions in this order:
`LAI`, `LAS`, `LPI`, `LPS`, `RAI`, `RAS`, `RPI`, `RPS`. Exact membership and the source-to-montage
translation are binding in `docs/roi-map-m1-v1.{md,json}` [D30]. The map uses an exact published
five-electrode 10–10 template [Maxfield 2012] translated through EGI's measured approximate
HydroCel equivalents [Luu & Ferree 2005]. Header-only inspection verified all 40 required channels
in all 232 current derivatives.

This is a project-specific HydroCel implementation of the eight-region family used in relevant
EEG work [Weidemann 2009; Katerman 2022], not an exact replication of those papers or of the
six-region Long et al. map [Long 2014]. It is sensor-space dimensionality reduction, not cortical
localization. Exact published HydroCel membership could not be established; EGI's 10–10
correspondences are approximate and the project lacks participant-specific digitization. These
limitations must appear with any spatial interpretation.

The expected feature count is exactly `2 × 8 × 8 = 128` under the frozen D29-A boundaries.
No held-out result may be used to add/remove a frequency, time bin, ROI, channel, or aggregation
rule.

## 6. Learner contracts

### B3 — behavioral comparator

- Inputs: the existing B3 blocks only—15 one-hot serial-position indicators plus list number,
  session number, word length, word frequency, and leave-one-subject-out word memorability [D18–D20].
- Word memorability is estimated from the full valid local `ltpFR` behavioral reference table,
  then joined onto the condition-specific survivor rows. For a target subject, the word numerator,
  denominator, **and smoothing grand mean** exclude every outcome from that subject; smoothing is
  10 pseudo-observations at that other-subject grand mean. F17 corrected the former global-prior
  implementation; synthetic regression tests now prove that changing every target-subject outcome
  leaves that subject's feature vector unchanged and that a missing comparison subject fails closed.
  No B3 result was recomputed during that correction. The historical B3 score remains context only.
- Pipeline: `StandardScaler` fitted on the permitted training rows, then binary
  `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, fit_intercept=true,
  class_weight=null)`.
- B3 is refit within every outer/inner fold on the exact M1/M3 survivor rows.

### M1 — interpretable EEG model

- Inputs: only the frozen M1 feature vector from section 5.
- Learner: binary L2 logistic regression with intercept, `solver="lbfgs"`, `max_iter=2000`,
  `class_weight=null`, and no calibration.
- Candidate `C` grid: `[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`.
- Selection: inner leave-one-training-session-out mean ROC-AUC with the deterministic tie rule in
  section 4. No feature selection, PCA, learned spatial filter, or post-outcome grid expansion.
- The resolved D26 strict-inductive transform is the only normalization of the EEG feature block;
  no additional scaler is applied after ROI/time-bin aggregation.

### M3 — incremental stack

- Inputs: exactly two probabilities, `prob_b3` and `prob_m1`.
- Learner: binary `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
  fit_intercept=true, class_weight=null)` with no probability calibration and no extra scaler.
- Under D44-C, M3 trains only on the qualified session-out B3/M1 probabilities described in
  section 4. Outer-test probabilities come only from first-level outer-training refits, and the
  outer test remains absent from every selection, transform, and model fit.
- M3 may not receive raw behavioral or EEG features. This keeps the added-value question explicit
  and prevents the larger EEG block from dominating by feature count [D24].

## 7. Determinism, fit provenance, and tie-breaking

- Global analysis seed: `0`.
- Subject bootstrap: 2,000 resamples, `numpy.random.default_rng(0)`.
- Any sign-flip/permutation result: 10,000 draws with seed `0`, unless the metric is retained only
  for historical reproduction and its existing 1,000-draw contract is explicitly named.
- Session folds are deterministic and unshuffled.
- Every learned model/transform records: learner type, parameters, software versions, fit canonical
  keys or their SHA-256, fit sessions, inner/outer fold IDs, selected hyperparameter, and seed.
- Convergence warning, non-finite coefficient/probability, or different result from the same frozen
  inputs is a failure, not a warning to suppress.

## 8. Estimands, metrics, and bounded result language

- Primary estimand: unweighted mean across eligible subjects of
  `AUC(M3) − AUC(refit B3)` on each subject's ocular-removed outer-test rows.
- Primary uncertainty: 95% percentile interval from 2,000 subject-level bootstrap resamples.
- Primary path is fixed to ocular-removed. Ocular-kept cannot rescue a non-positive primary result.
- Secondary/descriptive: B3, M1, and M3 per-subject AUC; IDI; continuous NRI; DeLong comparison on
  paired rows; and Brier score. D42-A deliberately omits balanced accuracy: no scientifically
  relevant operating point was specified, no threshold is fit or applied, and no balanced-
  accuracy field may appear in a result record. Reintroducing it requires a new pre-result charter
  version and cannot rescue the primary result [D42/D54]. These retained metrics cannot create an
  alternative positive claim [D25].
- Bounded category:
  - `positive_increment` only if the primary 95% interval is wholly above zero;
  - `adverse_increment` only if it is wholly below zero;
  - `no_detectable_increment__imprecise` if it includes zero—this is not an equivalence claim;
  - `condition_discordant` is an additional sensitivity flag when the ocular-kept point estimate
    has the opposite sign from ocular-removed, or the two conditions receive different interval-
    based categories above; it never changes the primary category.
- No smallest-effect bound has been justified, so the charter does not permit the word “equivalent”
  or a strong statistical-null claim.

## 9. Manifests, caches, and prediction tables

- Freeze versioned cohort, trial, outer-fold, inner-fold, and cross-condition-intersection manifests
  before model fitting. Record row counts and attrition at every join.
- Attrition schema v1.1 keeps trial and subject-condition counts in separate fixed flows, sums only
  named stage-specific reasons, requires within-unit stage continuity for both conditions, and
  binds cross-condition input to the final condition-specific trial survivors. F18 records why the
  former one-row-per-reason draft could not prove multi-reason conservation.
- Cache identity includes: derivative path and SHA-256; epoch metadata hash; code commit; dirty diff
  hash if any; charter hash; feature-schema version and SHA-256; condition; trial-manifest hash;
  fold-manifest hash; `uv.lock` hash; Python/package versions; and output-schema version.
- Manifest bundles publish as an immutable two-file sibling directory (`manifest.json` plus
  `manifest.sha256`) only after validation/fsync, using one OS-native atomic no-replace directory
  rename; overwrite—including a competing empty directory—partial or extra entries, noncanonical
  bytes, unsupported exclusive-rename semantics, or digest drift fail. Cache writes use the
  analogous `temporary entry → validation → rename` rule. No best-effort reuse is permitted.
- One canonical condition-specific outer-test prediction table contains the trial key, label,
  `prob_b3`, `prob_m1`, `prob_m3`, condition, subject, outer fold, all model/transform fit hashes,
  charter/manifests/code/environment hashes, and no duplicated row.
- Every metric is derived from this table; no metric-specific filtering.
- The synthetic closed prediction-table contract in
  `docs/m1-prediction-table-contract.{md,json}` binds each paired row to its trial-manifest label,
  exact fold/model/transform provenance, learner contracts, and finite probabilities while
  excluding metrics and result fields [D34/F22]. D51 adds a synthetic typed atomic prediction
  bundle with canonical metadata/row records, a little-endian B3/M1/M3 probability matrix,
  external and member authentication, and complete logical readback [D51/F40]. The synthetic subject-level consumer in
  `docs/m1-evaluation-contract.{md,json}` validates that table first, passes every canonical row to
  the frozen rank/continuous metrics and Brier scores, and binds a closed recomputable record to
  table plus evaluator provenance [D35/F23]. Real artifact/probability readback and evaluator
  integration remain required after charter freeze. Its synthetic primary aggregate also fixes the
  exact ocular-removed subject set, unweighted mean, seeded percentile bootstrap, bounded category,
  and no-secondary-rescue rule [D36/F24]. D53 adds one authenticated atomic
  canonical-JSON container for each condition-specific subject, paired common-intersection
  subject, primary aggregate, and paired aggregate record. Its field set is inherited from the
  supplied evaluation contract; D54 now binds the resolved D42-A absence while readback reruns
  each complete logical validator [D53/D54/F44]. A separate metric-free table now binds each shared
  outer-test key, label, event, both conditions' six B3/M1/M3 probabilities, and both source-table
  digests to the exact cross-condition intersection [D37/F25]. D52 adds its typed atomic bundle,
  external/member authentication, and complete source/intersection recomputation after readback
  [D52/F42]. A paired synthetic consumer now
  revalidates that table, passes every common row to both conditions' frozen metrics, rejects a
  one-class intersection, and binds one closed recomputable subject record without categories or
  discordance [D38/F26].
- The synthetic logical feature-table contract in `docs/m1-feature-table-contract.{md,json}` fixes
  byte-stable axes, exact partition-key matching, finite 128-value rows, and provenance bindings.
  D26-A now adds exact per-inner-fold and outer-refit transform fit/apply session and key provenance,
  statistic axes/shape, no-held-distribution access, and canonical digests. Its synthetic numerical
  kernel now proves exact `float64` log transform, training-only population statistics, unchanged
  application, zero handling, and typed statistic evidence. D50 adds a synthetic typed atomic
  feature bundle whose little-endian `float64` matrix, canonical keys/metadata, member hashes,
  externally supplied bundle digest, and reconstructed logical table all fail closed on drift.
  Real extraction, source-array recomputation, bundle publication/readback, and learner consumption
  remain deferred until after the now-validated N2 real manifests and their provenance bindings
  [D33/D39/D40/D50/F21/F27/F28/F38].
- The synthetic structural validator in `src/peers/m1_manifests.py` enforces the decision-independent
  schema, canonicalization, provenance-binding, ROI, and session-split rules, enforces the resolved
  D26-A/D29-A/D42-A/D44-C readiness boundaries, including the qualified claim and selection-
  provenance semantics, and accepts the frozen charter only when its bound transcript and receipt
  validate. It also recomputes the canonical final-charter digest rather than
  accepting an echoed hash-shaped value. Synthetic tests recompute exact key-set alignment, cross-condition survivor
  intersection plus row/event provenance, paired B3/M1/M3 outer-test prediction keys, and
  session-disjoint B3/M1 cross-fit model keys plus fold hashes. Synthetic attrition fixtures also
  prove count-unit separation, fixed reason maps, stage conservation, all-condition coverage, and
  cross-flow binding. Synthetic bundle tests prove atomic publication, race-safe no-replace
  immutability, exact file membership, canonical bytes, and digest readback on the tested local
  filesystem. The separate real generator and independent verifier have now published and exactly
  regenerated the immutable N2 cohort/trial/fold/intersection/attrition run from 376 bound sources;
  real normalization-statistic recomputation, spectral extraction, prediction/model artifact readback,
  probability recomputation, real prediction/comparison-bundle-to-evaluator integration,
  authenticated result-record-bundle readback and bundle-to-report consumption, and
  real common-intersection condition-aggregate/ocular-discordance evidence remain required after
  the charter is frozen. Their synthetic contracts are implemented. The N1 human gate and N2/N3
  manifest/source-reproduction boundary are closed; N4 real feature/cache and bounded preflight
  evidence are next.

## 10. Preflight and stop rules

- Before a full run, order subjects technically eligible in both conditions by total eligible
  trial count, then input bytes, then subject ID—all without outcome access. Select the lower
  median and maximum workload subjects, distinct when possible, and run both ocular conditions
  without computing or interpreting outcome performance.
- Record the exact machine-charter measurement inventory: stage/total wall time, peak resident
  memory, input and cache bytes, disk before/after, workers, longest stage/no-progress intervals,
  retry count, scaled full-run forecast, projected remaining cache, and a host-safe memory floor.
- Before the full locked run, require an authenticated preflight budget receipt and Sofia's
  capacity go/no-go. Sofia records a current-run cutoff after seeing the forecast. The effective
  deadline is the earlier of that cutoff and `1.3 ×` the scaled forecast; each subject is bounded
  at `2 ×` the maximum-subject preflight time.
- Abort after `2 ×` the longest preflight stage without progress, below the recorded safe memory
  floor, or below `1.2 ×` projected remaining cache bytes plus 5 GiB of disk headroom. Resume only
  from atomically completed, hash-valid entries.
- One retry is permitted only for a classified transient technical failure. No retry changes the
  cohort, feature family, hyperparameter grid, condition, metric, or seed.
- Stop on charter/manifest drift, stale cache, duplicate/missing keys, fold contamination,
  transform/model fit-scope violation, preprocessing change, unsafe resources, repeated failure,
  unsupported claim, or Sofia fatigue/capacity failure.
- Record a scientifically meaningful failure before fixing it. Preserve verified atomic outputs and
  leave the project `IN_PROGRESS`; never weaken the protocol to meet a date.

The exact object is duplicated in the machine charter and manifest contract and enforced by the
readiness validator [D47/F34]. Its policy is frozen before real generation; its measured values and
Sofia's capacity receipt necessarily remain later full-run gates.

`docs/m1-preflight-contract.{md,json}` now provides the synthetic executable shape of those later
gates [D48/F36]. It externally binds the complete candidate inventory; recomputes the lower-median
and maximum subjects; requires both conditions, closed unscored pipeline stages, and atomic
hash-valid caches; scales wall time and remaining cache by the larger observed per-trial or
per-input-byte ratio; and binds Sofia's later `go`/`no_go`, reason, current-run cutoff, and the
derived full-run budget. The available-memory floor may be operator-set but cannot be below
observed peak RSS. The N2 real manifests and external bindings now exist, but this remains synthetic
policy evidence until the real bounded preflight receipt is produced. A future runner must
continuously enforce the returned limits.

## 11. Presentation relevance

Phase 1 must produce evidence that later application materials can explain without rewriting the
science:

1. the behavioral B3 bar and why chance is the wrong comparator;
2. the held-session and cross-fitting diagram;
3. the interpretable M1 feature map (time × frequency × ROI);
4. the paired per-subject incremental result with uncertainty;
5. the ocular sensitivity and attrition record;
6. R4's exploratory double-use disclosure, leakage failures, and limitations.

Presentation materials may improve legibility but may not change categories, omit adverse/null
evidence, infer neural sources, or supply Sofia's interpretation.

## 12. Teach-back resolution and remaining execution gates

The resolved D26-A, D29-A, D42-A, and D44-C fields are frozen against outcome-driven change.
Sofia's nine-topic teach-back passed after Sofia-owned corrections to topics 3 and 4, and she
explicitly approved the complete verbatim capture. The exact transcript and receipt are stored in
`docs/protocol-teach-back-transcript-m1-v1.json` and
`docs/protocol-teach-back-receipt-m1-v1.json`; the worksheet and advisor rubric remain in
`docs/protocol-teach-back-m1-v1.md`.

The receipt binds the canonical frozen-charter SHA-256, the post-D26/D29/D42/D44/D56 decision-log
SHA-256, the complete transcript SHA-256, the exact capture-approval message, and the accepted
Sofia-owned message IDs for all nine topics. This proves recorded provenance and structural
acceptance, not genuine comprehension by software; the conversational advisor review is the
substantive evidence. Sofia's one-character D42 and D44 choice messages remain distinct from her
later explanations, and advisor reasoning is not attributed to her.

D30's exact ROI map is already sourced, machine-readable, and frozen as a pre-result technical
contract. Reopening it requires a new charter/map version and invalidates all dependent evidence.

The approved teach-back receipt preceded the now-locked and source-reproduced outcome-blind
manifest run. Real feature/cache plumbing and the bounded unscored preflight are next. Do not fit
the full M1/M3 pipeline or inspect held-out outcomes until all later execution and capacity gates
pass.
