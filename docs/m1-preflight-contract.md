# M1 preflight, capacity, and full-run authorization contract

Status: **FIRST REAL RECEIPT RETAINED AS ENGINEERING EVIDENCE; REPLACEMENT FINAL-CODE PREFLIGHT REQUIRED — no capacity receipt or run authorization**

Machine-readable companion: `docs/m1-preflight-contract.json` (`1.2-draft`)

## Purpose

The preflight exists to answer one operational question before the locked cohort runs: can the
prespecified analysis finish within measured time, memory, disk, and Sofia-capacity limits without
changing the scientific protocol or inspecting partial results?

This contract turns D47's policy into executable evidence. It does not alter resolved D42-A or
D44-C, generate a real manifest or feature, fit a real model, score a prediction, establish
machine safety, or prove that a human-authorship statement is genuine.

## When a real receipt may exist

A real preflight receipt is prohibited until all of the following are independently true:

1. the Analysis Charter is `frozen_pre_result` with D42 and D44 resolved;
2. no human gate remains, so Sofia's protocol teach-back receipt already exists;
3. the supplied charter digest recomputes from the exact charter;
4. the code commit, dirty-diff state, environment lock, source inventory, protocol receipt, and
   both-condition trial manifests have externally derived hashes; and
5. the complete candidate inventory is derived from technical eligibility without outcome access.

The frozen charter, accepted teach-back, and N2 real manifests satisfy those prerequisites. The
first real unscored receipt is retained under its exact code identity, but F59 prevents it from
authorizing the later completed publisher. A replacement preflight must bind the final clean
runner/verifier commit before Sofia is asked for capacity. D42-A is resolved and deliberately adds
no threshold or balanced-accuracy metric. D44-C is resolved with qualified meta-prediction wording.
D49's D42-C stop remains a regression boundary for a future versioned re-plan, not an active branch.
Passing synthetic tests is not authorization to run the full cohort.

## Outcome-blind candidate selection

The receipt stores every subject technically eligible in both ocular conditions, with condition-
specific eligible trial counts, input bytes, and trial-manifest hashes. Input bytes are the sum of
logical file sizes of the unique locked epoch derivatives actually read for that subject-condition;
a derivative is counted once even though it supplies many trials. The validator requires
exact equality to an independently supplied inventory and recomputes:

- workload order: total eligible trials, total input bytes, then subject ID, all ascending;
- median role: lower-median index `floor((n−1)/2)`; and
- maximum role: the last workload row, distinct from the median whenever more than one candidate
  makes that possible.

The candidate array itself uses subject-ID order for deterministic hashing. A caller cannot select
a convenient subject by editing both the role and the receipt because the external inventory and
selection rule are validated separately.

The locked N2 run now yields 38 common eligible participants and candidate-inventory SHA-256
`fc4702ab998a5db310f4971b2ca9d410f839bfd26c9d5e40d440b4ce83b16527`. The frozen ordering selects
lower-median `sub-LTP063` (1,222 trials; 459,195,209 input bytes) and maximum `sub-LTP104` (1,503
trials; 564,754,535 input bytes). These are workload facts only; no recall rate, feature,
prediction, or metric entered the ordering [D58].

## Full behavioral reference binding

B3's word-memorability feature is computed from every valid local ltpFR behavioral presentation,
not only the 40 participants with cached EEG derivatives. The runner inventories and hashes every
contributing events TSV, aligned trial count, and subject count, then combines that digest with
the authenticated N2 source-inventory digest in a separate execution-source identity. Raw caches,
unscored execution evidence, and the preflight receipt bind that composite identity. The target
participant is still excluded from the word numerator, denominator, and smoothing grand mean.
This closes a provenance gap without changing B3 [D59].
The outcome-blind inventory validation finds 264 local event files, 80 participants, and 67,392
aligned presentations; its canonical SHA-256 is
`4fc85c1ef824bd1295a19d801afe2992a0f54d176d4578b24bf1b663ec4b5df8`. These are source and
denominator facts, not recall-rate or held-model results.

## Closed measurement evidence

Every distinct selected subject must have one row for each of `ocular-removed` and `ocular-kept`.
The only permitted stages are:

1. `load_validate_inputs`;
2. `build_validate_feature_tables`;
3. `fit_predict_without_scoring`; and
4. `publish_readback_artifacts`.

Each row records exact stage and total wall time, peak resident memory, input and cache bytes, disk
before/after, workers, longest stage/no-progress interval, retries, and the cache artifact-set
digest. Cache entries must be explicitly atomic and hash-valid. The input bytes must reconcile to
the candidate inventory; stage maxima, role membership, condition coverage, retry limit, row order,
and the measurement digest are recomputed.

The stage name `fit_predict_without_scoring` is intentional: the preflight may exercise the frozen
pipeline only after the charter gates close, but it may not compute, expose, or interpret a held-
session metric. The receipt's closed schema and false outcome-access attestation reject metric
fields; the attestation remains recorded evidence, not proof of what an external process truly read.
The retained preflight artifact stores raw-cache hashes, fitted-model hashes, selected `C`, and
hashes of the held prediction arrays, but not held probability values, labels, or any held score
[D60].

## Forecast formulas

The wall-time forecast is deliberately conservative and reproducible. Within each condition it
computes the largest observed seconds-per-trial and seconds-per-input-byte ratios across the
selected subjects, scales both to the complete candidate workload, takes the larger estimate, and
sums the two conditions. Wall seconds are rounded to six decimal places.

The remaining-cache forecast applies the analogous maximum cache-per-trial/cache-per-input-byte
rule only to unmeasured subjects, rounds each condition upward, and sums them. Selected-subject
caches are subtracted only because every measured row proves an atomic, hash-valid artifact set;
invalid cache means the receipt fails rather than assuming reuse.

The receipt also recomputes total measured time, maximum-subject two-condition time, maximum peak
resident memory, longest stage/no-progress intervals, measured run count, and worker count. The
operator-recorded available-memory abort floor must be at least the maximum observed peak resident
memory. This lower bound prevents a nonsensical receipt; it is not a universal hardware-safety
claim and the operator may set a larger floor.

## Sofia capacity receipt

After seeing the validated forecast, Sofia records exactly one human-owned capacity receipt bound
to the preflight receipt's canonical SHA-256. It contains her `go` or `no_go`, her reason in her own
words, the source task/message binding, and—only for `go`—a current-run UTC cutoff later than the
decision time.

A `no_go` receipt is valid evidence and must use a null cutoff, but it never authorizes execution.
The validator can prove structure, digest equality, and recorded provenance; it cannot prove
authorship or genuine informed consent.

## Derived full-run budget

The capacity receipt acknowledges one exact budget derived from the preflight:

- per-subject timeout: `2 ×` maximum-subject two-condition wall time;
- full-run timeout: `1.3 ×` scaled wall-time forecast;
- effective deadline: earlier of run-start plus full-run timeout and Sofia's cutoff;
- no-progress abort: `2 ×` longest observed stage;
- disk floor: `ceil(1.2 × projected remaining cache bytes + 5 GiB)`;
- available-memory floor: the receipt-bound operator value, no lower than observed peak RSS;
- workers: exactly the measured preflight value;
- retry limit: one classified transient retry; and
- resume: atomically completed, hash-valid entries only.

`authorize_full_locked_run` revalidates the preflight and capacity receipts plus current disk,
available memory, workers, start time, and cutoff before returning this budget. It cannot change a
scientific decision, weaken the protocol, or permit partial-outcome inspection. A future runner
must still enforce the returned deadline/floors continuously and record any failure before replan.

## Verification and claim limit

All current fixtures are invented. Synthetic tests cover exact contract drift, external candidate
binding, role recomputation, both-condition completeness, stage and digest drift, cache validity,
retry/forecast tampering, draft-charter rejection, Sofia receipt provenance, `no_go`, resource
shortfalls, worker drift, expired cutoff, and canonical tamper sensitivity.

These tests establish only that the specified evidence would fail closed. They do not establish a
real cohort, runtime, memory footprint, cache size, capacity decision, authorization, continuous
monitor, or held-out result.

## Stop/replan rule

Remain `IN_PROGRESS` and do not inspect a partial outcome if any external binding, selection,
measurement, cache, forecast, capacity, cutoff, disk, memory, worker, retry, or contract check
fails. Do not alter the cohort, features, folds, learners, conditions, metrics, seeds, or claim
ceiling to fit a budget. Re-plan the operational schedule with Sofia or preserve the failure.
