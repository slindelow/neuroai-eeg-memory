# M1 Logical Feature-Table Contract

Status: **LOCKED N2 INPUTS + REAL-SOURCE BRIDGE CONTRACT — no real EEG features generated yet**
Machine-readable companion: `docs/m1-feature-table-contract.json` (`1.5-draft`)

## Purpose

This contract defines what a future M1 feature table must prove before a learner can consume it:
the exact trial identity, exact feature-axis order, finite numeric shape, and bindings to the
charter, ROI map, manifests, transform provenance, code, and environment. It prevents a plausible
matrix from being accepted merely because it has 128 columns.

It enforces Sofia's resolved D26-A normalization, D29-A time-boundary, D42-A omission, and D44-C
decision identifiers. D50 now adds a
technical storage and readback format without changing either scientific choice. It does not read
a real epoch, compute a real feature, fit a model, or inspect an outcome. The teach-back and frozen-
charter prerequisite and frozen-real-manifest prerequisites are satisfied. D42-A and D44-C do not
change the already tested feature arithmetic or bundle mechanics. D57 adds the bounded real-source
bridge required before the first real feature is published.

## F21 and D33: byte-stable axis identity

The original charter's feature-name template interpolated a raw floating-point frequency without
specifying its serialization and referenced a `time_bin_id` that was never defined. Two correct
implementations could therefore produce different column names for the same numeric axis, or the
same informal “early/late” name after a boundary changed.

D33 assigns ordinal identifiers only:

- time bins: `t0`, `t1` in declared temporal order;
- frequencies: `f00` through `f07` in the frozen ascending-frequency order;
- ROIs: the frozen D30 identifiers in their declared order;
- names: `eeg__time-{time_bin_id}__roi-{roi_id}__freq-{frequency_id}`.

The identifiers are not scientific values. Exact frequencies, time boundaries, normalization,
ROI membership, estimator, aggregation, dtype, and missing-channel policy are separately included
in `feature_schema_sha256`. Changing any such field changes the schema digest even when the ordinal
names remain stable.

Rejected alternatives were platform-default float formatting, rounded frequency labels, and
semantic labels such as “early” and “late.” Default formatting is not a cross-implementation
contract; rounding can collide or conceal schema drift; semantic labels can imply an
interpretation not established by an ordinal bin.

## Logical table

One logical table covers one subject, ocular condition, and strict-inductive partition. Its exact
top-level fields are machine-bound in the JSON companion. The essential rules are:

- `partition_id` identifies the caller-supplied normalization/fold context without defining it;
- `expected_key_set_sha256` binds the exact externally supplied partition keys;
- charter, ROI, trial-manifest, fold-manifest, source-inventory, transform-provenance, `uv.lock`,
  package-version, code-commit, and dirty-diff identities are explicit;
- `feature_schema_sha256` recomputes from the resolved charter rather than trusting the table;
- `feature_names` must equal the exact 128-name Cartesian order `time × ROI × frequency`;
- each row contains exactly the canonical four-field trial key and 128 finite Python `float`
  values, corresponding to the charter's logical `float64` contract;
- rows match the expected partition key set exactly once and are stored in canonical key order;
- no recall label, probability, coefficient, score, or unknown field is accepted.

The logical canonical JSON digest is a synthetic identity primitive, not a decision about the
eventual on-disk matrix format.

## D26-A normalization-transform provenance

The resolved primary transform is fit separately for every eligible inner fold and once for the
outer refit. Each record binds its exact fit/apply sessions and trial-key hashes, trial counts,
trial/time reduction axes, preserved channel/frequency axes, `ddof=0`, zero-standard-deviation
policy, and hashes of the fitted mean, raw-standard-deviation, effective-scale, and zero-mask
arrays. It also records the zero-cell count and time samples per fit trial, and explicitly records
`label_access=false` and `held_session_distribution_access=false`.

The inner record fits only the fold's training sessions and applies unchanged to its validation
session. The outer record fits all outer-training survivors and applies unchanged to the held
highest session. Statistics have shape `40 × 8`: the 40 declared ROI channels in frozen order by
the eight frozen frequencies. Selecting only these channels before independent per-channel
normalization is mathematically equivalent to normalizing all channels and discarding the
non-ROI channels afterward.

The validator recomputes session and key scopes from synthetic trial/fold manifests, binds the
charter/ROI/manifest/schema/environment identities, and canonicalizes records in inner-fold order
followed by outer refit. It never stores real statistic values. D26-B remains only an optional,
separately prespecified sensitivity and is not accepted by this primary contract.

The outcome-blind numerical kernel now implements exact `float64` `log10(power + 1e-30)`,
population-statistic fitting over the supplied training trial/time axes, immutable fitted arrays,
zero-standard-deviation replacement, and unchanged application. Its array digests include shape
and dtype before fixed-order bytes. Synthetic target-batch invariance proves that changing or
splitting an application batch cannot change the fitted transform or another row's normalized
value [D40/F28].

## D29-A aggregation contract

The outcome-blind aggregation kernel maps the frozen D29-A window without a hidden tolerance:

- `t0 = [0.300, 0.950)` seconds;
- `t1 = [0.950, 1.600]` seconds; and
- a sample exactly at 950 ms belongs to `t1` only.

The supplied time vector must be one-dimensional, finite, strictly increasing, cover the complete
declared window, and yield at least one sample in each bin. An exact midpoint sample is not
required. The supplied channel names must contain the frozen 40-channel union exactly once;
aggregation reorders them to the D30 ROI/channel sequence before taking an arithmetic mean over
each five-channel ROI and selected time samples. Output order is exactly `time × ROI × frequency`,
giving 128 finite `float64` values. Evidence records the exact time vector, masks, per-bin sample
counts, schema digest, and aggregation-contract digest [D41/F29].

## D50 typed atomic cache bundle

F38 found that canonical logical-table bytes did not prove the identity of a future numerical
cache or its readback. D50 separates identity from efficient storage while making them mutually
verifiable. One immutable bundle contains exactly:

- `metadata.json`: every logical-table field except rows, in canonical JSON;
- `trial-keys.json`: the exact canonical four-field row keys;
- `values.npy`: a C-contiguous `n_rows × 128` little-endian `float64` matrix;
- `bundle.json`: hashes for all three members, the logical-table digest, exact shape/type fields,
  and the bundle-contract digest; and
- `bundle.sha256`: the canonical bundle-manifest digest, which must also be supplied independently
  by the caller during readback.

NumPy documents `.npy` as its standard single-array binary format and recommends disabling pickle
for security and portability when object arrays are unnecessary [NumPy save; NumPy I/O guide].
The writer therefore calls `numpy.save(..., allow_pickle=False)` and the reader calls
`numpy.load(..., allow_pickle=False)`. Readback rejects trailing bytes, unsafe/object arrays,
wrong byte order/dtype/shape, nonfinite values, extra or symlinked members, noncanonical JSON,
member/directory digest drift, and a self-consistent bundle that lacks the independently supplied
digest. It reconstructs the complete logical table and reruns the existing exact key, axis, and
provenance validator before returning it.

Publication fsyncs a temporary directory and every member, then uses the already verified
platform-specific atomic no-replace rename. A failed staged directory is retained for evidence
under the project's no-deletion boundary. Passing synthetic round-trip tests does not authorize a
real bundle: frozen real manifests and source-to-feature provenance are still required.

## D57 raw single-session spectral cache

The real extractor must not recompute an approximately gigabyte-scale Morlet array for every inner
and outer fold, and it must not normalize a session before the permitted training-session set is
known. The outcome-free bridge therefore stores one immutable cache per subject, session, and
ocular condition with:

- canonical trial keys for exactly the locked manifest epoch rows, with no recall label;
- per-trial, per-time-bin, per-channel, per-frequency means of raw
  `log10(Morlet power + 1e-30)` before D26 normalization;
- the full decimated epoch time axis; and
- per-channel/frequency session mean, centered sum of squares (`M2`), and trial-time count computed
  over every trial and every decimated epoch time sample.

Training-session statistics are combined in ascending session order with the Chan mean/`M2`
formula. Population variance is combined `M2 / count`; a zero standard deviation is replaced by
one. No validation or held-session sufficient statistic enters a fitted transform. The reduction
is algebraically identical to the frozen full-array operation because, for each channel and
frequency, averaging `(x - mean) / scale` over a time bin can be performed from the raw bin mean,
and the subsequent ROI mean is also linear. Synthetic tests must compare the compact path with the
existing full-array D26/D29 kernels before any real publication.

The bundle contains typed little-endian `float64` NPY members, canonical metadata and keys, member
hashes, an externally required bundle digest, and the same atomic no-replace publication boundary.
The extractor rehashes the source derivative and requires exact manifest epoch rows before Morlet
computation. Labels, probabilities, coefficients, hyperparameters, metrics, and result categories
are forbidden. This is a technical cache design, not a new scientific feature or normalization
choice [D57/F54].

## Test strategy

| Layer | Current evidence | Acceptance target | Status |
| --- | --- | --- | --- |
| Unit/contract | Invented two-row tables | Pending gates fail; exact axes, fields, dtype, finite values, bindings, keys, order, and digests enforced | Implemented |
| Cross-document | Charter and JSON contract | Axis IDs, order, template, and count identical | Implemented |
| D26 transform provenance | Invented trial/fold manifests and transform hashes | Exact inner/outer fit/apply sessions, keys, axes, shape, access flags, bindings, and digest | Implemented |
| D26 numerical kernel | Invented float64 trial × channel × frequency × time arrays | Exact log transform, training-only population statistics, zero handling, unchanged batch-invariant application, and typed evidence fields | Implemented |
| D29 aggregation | Invented time axes and normalized arrays | Exact interval endpoints, full-window coverage, ROI/channel reordering, arithmetic mean, 128-value order, and digests | Implemented |
| Raw real-source bridge | Invented full arrays and mocked exact epoch sources | Compact/full-array equivalence, training-session-only statistic merge, exact locked rows, source rehash, typed atomic readback, external bindings, and tamper rejection | Implemented outcome-blind; real preflight evidence pending [D57/F54/F55] |
| Cache publication/readback | Invented two-row tables and temporary directories | Typed no-pickle matrix, canonical metadata/keys, atomic no-replace publication, external digest, member hashes, exact reconstruction, and tamper rejection | Implemented synthetic-only [D50/F38] |
| Integration | Frozen real manifests plus D29-A | Derive correct partition keys, sample masks, channel order, and transform provenance for every table | Deferred |
| Data recomputation | Real epoch derivatives | Recompute selected cells and hashes from immutable sources | Deferred |
| Model boundary | Frozen M1 consumer | Learner receives only validated table rows for its permitted fit/score partition | Deferred |

Synthetic coverage targets the business-critical leakage and identity boundary rather than trivial
getters. Failures include missing/extra/duplicate keys, labels or unknown fields, non-float or
non-finite values, wrong length or axis order, stale provenance, noncanonical rows, D26 drift,
held-session or label access. Synthetic pending-D29 fixtures remain only to prove fail-closed
behavior if a future charter regresses to a placeholder.

## What this evidence cannot prove

Passing the synthetic validator, numerical suite, and bundle round-trip does not establish that a
future caller fed the named epoch samples or permitted manifest rows into the tested kernel, that
the stored values match recomputation from the named sources, that the target filesystem is safe,
or that a learner consumed the validated artifact. Those require integration and source-
recomputation evidence after the real manifests and their frozen-charter bindings validate. No real
feature or outcome has been used.
