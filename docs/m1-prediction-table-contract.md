# M1 Paired Prediction-Table Contract

Status: **PRE-FINAL-CHARTER SYNTHETIC VALIDATORS + TYPED BUNDLES — no model fitted, real prediction generated, or metric calculated**

Machine-readable companion: `docs/m1-prediction-table-contract.json` (`1.3-draft`)

## Purpose

NeuroAI's incremental claim depends on comparing B3, M1, and M3 on the same held-out trials. A
key-set check alone cannot prove that each key carries the correct recall label, probabilities from
the named fold/model artifacts, or the intended learner and transform provenance. This contract
defines one canonical subject-condition outer-test table from which every later metric must be
derived.

It enforces but does not change the resolved D26-A/D29-A/D42-A/D44-C contracts. It does not fit a learner,
generate a real probability, calculate a metric, or inspect a real outcome. D51
selects the physical bundle for each condition-specific table; D52 does the same for the already
frozen common-intersection comparison table. The teach-back and frozen-charter prerequisite is
satisfied; real publication still requires frozen real manifests, feature/model provenance, and
later execution gates. D55 now binds the qualified D44-C selection scope that may feed M3 training
rows.

## F22 and D34: close the paired evidence unit

F22 records that `validate_prediction_key_sets` proved only the presence of B3/M1/M3 keys. It did
not bind the values attached to those keys. D34 therefore makes the complete paired row—not three
separate arrays—the auditable evidence unit.

Each logical table represents one subject, ocular condition, and deterministic outer fold. It
binds:

- the exact outer-test survivor keys and recall labels from the frozen trial manifest;
- finite `float64`-shaped `prob_b3`, `prob_m1`, and `prob_m3` values in `[0, 1]`;
- the final charter, ROI map, trial/fold manifests, feature schema/table, and cross-fit provenance;
- code, dirty diff if any, `uv.lock`, Python, and package-version identities;
- exact B3, M1, and M3 learner-contract hashes;
- model artifact, selected-hyperparameter, transform-chain, transform-fit-key, model-fit-key, and
  prediction-key hashes;
- the exact outer-training sessions and global seed;
- M3's upstream cross-fit provenance; and
- a per-row hash link to each model's closed provenance object.

B3 and M1 fit-key hashes must equal all outer-training survivor keys. M3's fit-key hash covers the
same keys as cross-fitted meta-training rows, and its upstream hash must equal the table's external
cross-fit provenance. All three prediction-key hashes equal the outer-test survivor keys.

Under D44-C, “cross-fitted meta-training rows” is qualified: each M1 coefficient and learned
transform excludes the predicted session, but one `C` is selected from all outer-training inner
folds and may therefore depend on the predicted session's labels. The table binds that canonical
selection-provenance digest, requires M3's upstream digest to match it, and forbids describing the
M1 meta-predictions as fully session-cross-fitted [D55/F47].

The learner hash covers the declared learner, split, cross-fitting, and seed contracts. M1 also
binds the resolved feature contract, so D26 or D29 drift prevents a prediction table from
validating. Transform hashes remain opaque here, but their permitted inner/outer fit/apply key
semantics are now validated by the D26 feature-provenance contract.

## Closed row schema

Each row contains only:

- canonical trial key: `subject_id`, `session`, `trial`, `item_num`;
- `condition` and deterministic `outer_fold_id`;
- trial-manifest `recalled` label;
- the three paired probabilities; and
- B3/M1/M3 provenance-object digests.

Unknown fields are rejected. In particular, AUC, delta AUC, result categories, coefficients, or
other summaries may not be stored in the prediction table. Every metric must later consume the
same validated canonical table without metric-specific filtering.

## D51 and F40: authenticated typed prediction bundle

The logical digest alone did not prove what typed probability matrix a later evaluator would
reload. D51 stores one condition-specific table as an immutable directory containing:

- canonical `metadata.json` with every table field except `rows`;
- canonical `row-records.json` with every row field except the three probabilities, preserving the
  trial identity, recall label, fold, and model-provenance links;
- C-contiguous little-endian `float64` `probabilities.npy` in exact B3, M1, M3 column order;
- canonical `bundle.json` with the contract, logical-table, and member digests; and
- `bundle.sha256`, which must also match an independently supplied expected digest.

Publication fsyncs each member and the staging directory before an atomic no-replace rename;
existing destinations are not overwritten and failed staging evidence is retained. Readback
rejects symlinks, missing or extra members, noncanonical JSON, unsafe or trailing-byte arrays,
wrong dtype/order/shape/range, member drift, contract drift, and wrong external identity. It then
reconstructs the full logical table, reruns the existing key/label/fold/provenance validator, and
checks the original logical digest. NumPy's authoritative `.npy` guidance supports the single
typed array and `allow_pickle=False` boundary [NumPy save; NumPy I/O guide]. This technical choice
does not select an evaluation threshold, M1 hyperparameter scope, learner, estimand, or result.

## F25 and D37: common-intersection ocular comparison evidence

The two D34 tables are deliberately condition-specific. They can therefore be individually valid
while summarizing different surviving trials—the confound already recorded in F5. F25 records that
condition-specific D35 subject records cannot establish a direct ocular comparison.

D37 adds a second, still metric-free logical table for one subject and one common outer-test
session. It first validates both complete D34 tables and the complete D31 cross-condition survivor
intersection. It then requires independently expected digests for both source tables and the
intersection manifest and copies only the exact common outer-test rows. Each row binds:

- the common canonical key, recall label, and behavioral-event provenance;
- the complete ocular-removed and ocular-kept source prediction-row digests; and
- both conditions' B3, M1, and M3 probabilities.

Different subjects, different outer-test sessions, reversed condition roles, empty intersections,
missing/extra rows, label or probability drift, stale source links, and metric/result fields fail
closed. This table makes a later same-trial sensitivity calculation possible; it does not perform
that calculation or assign `condition_discordant`.

## D52 and F42: authenticated typed common-intersection bundle

The D37 logical digest did not prove which typed six-column probability matrix a later ocular
sensitivity evaluator reloaded. D52 stores each metric-free same-trial table with the same atomic,
externally authenticated five-member layout as D51, specialized to:

- canonical comparison metadata, including both source-table digests and the intersection digest;
- canonical non-probability row records with the common key/label, event provenance, and both
  source-row digests; and
- a C-contiguous little-endian `float64` matrix ordered ocular-removed B3/M1/M3, then ocular-kept
  B3/M1/M3.

Readback reconstructs the complete table and invokes the existing complete recomputation against
both source prediction tables, both trial manifests, and the exact intersection manifest. It also
requires the external bundle digest and rejects the same unsafe-array, trailing-byte, symlink,
extra-member, member/hash/schema, label, probability, and logical-digest drift classes as D51.
This closes physical identity for the same-trial evidence unit without calculating a metric or
altering D42-A/D44-C. The authoritative NumPy sources already recorded for D50/D51 support the `.npy`
and `allow_pickle=False` container boundary [NumPy save; NumPy I/O guide].

## Test strategy

| Layer | Current evidence | Acceptance target | Status |
| --- | --- | --- | --- |
| Unit/contract | Invented train/test rows and probabilities | Closed fields, types, ranges, exact model set, and deterministic digests | Implemented |
| Identity integration | Synthetic trial manifest | Exact keys, labels, outer fold/session, fit/prediction key hashes, and canonical order | Implemented |
| Cross-document | Resolved synthetic charter | Exact B3/M1/M3 stack; learner hash changes on learner, split, cross-fit, seed, D26, or D29 drift | Implemented |
| Model-artifact integration | Frozen real fits | Recompute learner, fit-key, transform, hyperparameter, and artifact hashes | Deferred |
| Probability recomputation | Frozen fit artifacts and inputs | Reproduce selected row probabilities to tolerance fixed before outcomes | Deferred |
| Condition-specific storage/readback | Invented tables and temporary directories | Atomic no-replace publication, exact dtype/order/shape, canonical metadata/row records, external and member digests, full logical reconstruction, and tamper rejection | Implemented synthetic-only [D51/F40] |
| Ocular comparison table and storage | Two invented condition tables plus invented intersection and temporary directories | Exact common outer-test rows, paired probabilities, source-row/table digests, no metrics, typed atomic publication, external/member authentication, and complete recomputation after readback | Implemented synthetic-only [D37/D52/F25/F42] |
| Evaluation consumer | Invented validated canonical table | Subject rank/continuous metrics and Brier scores use every identical row with no extra filtering | Implemented synthetic-only [D35/F23] |
| Real evaluation integration | Frozen canonical artifact bundles | Stored subject records fully recompute under independent evaluator provenance | Deferred |

Synthetic failures cover missing/extra/duplicate keys, label drift, integer/non-finite/out-of-range
probabilities, wrong fold, model-set or fit-role drift, wrong fit/prediction keys or sessions, stale
learner/artifact/cross-fit provenance, broken row links, common-intersection/source digest drift,
condition-role drift, and metric/result fields.

## What this evidence cannot prove

Passing this validator and synthetic bundle readback does not prove that any future probability
came from the named model, that M1 normalization used D26-permitted rows, that M3 meta-training
inputs were genuinely out of fold, or that a real evaluation process consumed only these bundles.
Those claims require real artifact readback, selected-row probability recomputation, the existing
cross-fit validator, and real evaluator provenance/readback after frozen real inputs and manifests
validate. No real label, prediction, model, coefficient, score, or metric was used here.
