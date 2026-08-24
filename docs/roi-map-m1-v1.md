# M1 v1 scalp-ROI map

Status: **FROZEN PRE-RESULT TECHNICAL CONTRACT**

Map ID: `neuroai-m1-roi-v1`

Machine-readable companion: `docs/roi-map-m1-v1.json`

## Purpose

This map turns M1's spatial feature axis into an exact, inspectable contract. It preserves the
eight bilateral scalp-region factors used in the relevant EEG literature—left/right ×
anterior/posterior × inferior/superior—while matching the `GSN-HydroCel-129` montage attached to
NeuroAI's actual epoch files. These are scalp summaries for prediction and visualization, not
cortical-source labels.

## Why this is a translation, not an exact replication

The closest memory sources do not supply an exact channel list that can be copied into this
project. Weidemann, Mollison, and Kahana publish eight shaded ROIs on an older 128-channel GSN
figure, but no machine-readable membership; Long, Burke, and Kahana publish six task-motivated
scalp regions without a numbered membership table; and Katerman et al. reuse eight regions on a
128-channel BioSemi cap rather than NeuroAI's HydroCel montage [Weidemann 2009; Long 2014;
Katerman 2022]. The older GSN drawing also mixes numeric sensors and approximate 10–10 labels, so
silently treating it as a HydroCel sensor list would be false precision.

The chosen translation therefore uses:

1. the exact, symmetric five-electrode 10–10 membership published for each of the same eight
   scalp factors by Maxfield et al. [Maxfield 2012];
2. EGI's own measured table of approximate 10–10 equivalents for the adult 128-channel HydroCel
   GSN to translate each named site to an `E*` channel [Luu & Ferree 2005]; and
3. an outcome-blind read of every current epoch header to confirm that all translated channels
   exist in the actual derivatives.

This is a reproducible **NeuroAI HydroCel implementation of the published eight-factor scalp
scheme**. It must never be described as the exact Weidemann, Long, or Katerman sensor map.

## Frozen order and membership

Feature order uses the ROI order in this table. Within each ROI, the named 10–10 order is retained;
channel averaging is arithmetic and therefore does not otherwise depend on channel order.

| Order | ROI | Meaning | Published 10–10 template | NeuroAI HydroCel-129 channels |
| ---: | --- | --- | --- | --- |
| 1 | `LAI` | left anterior inferior | FP1, F7, F5, FT7, FC5 | E22, E33, E27, E34, E28 |
| 2 | `LAS` | left anterior superior | AF3, F3, F1, FC3, FC1 | E23, E24, E19, E29, E13 |
| 3 | `LPI` | left posterior inferior | TP7, CP5, P7, P5, O1 | E46, E47, E58, E51, E70 |
| 4 | `LPS` | left posterior superior | CP3, CP1, P3, P1, PO3 | E42, E37, E52, E60, E67 |
| 5 | `RAI` | right anterior inferior | FP2, F8, F6, FT8, FC6 | E9, E122, E123, E116, E117 |
| 6 | `RAS` | right anterior superior | AF4, F4, F2, FC4, FC2 | E3, E124, E4, E111, E112 |
| 7 | `RPI` | right posterior inferior | TP8, CP6, P8, P6, O2 | E102, E98, E96, E97, E83 |
| 8 | `RPS` | right posterior superior | CP4, CP2, P4, P2, PO4 | E93, E87, E92, E85, E77 |

No channel appears in more than one ROI. Forty of the 125 EEG channels in each current derivative
enter the ROI features; the remaining channels do not enter M1 v1.

## Outcome-blind actual-state verification

On 2026-08-16, header-only inspection found 232 `*_epo.fif` derivatives, one unique ordered
125-channel schema, eight ROIs of five unique channels, and zero derivatives missing any of the
40 required channels. The four absent HydroCel labels are E8, E25, E126, and E127; none is required
by this map. The reference is stored as `Cz` after the project's documented E129 alias.

This inventory is a snapshot, not permission to trust future data silently. The future manifest
validator must recheck the full channel schema and bind the ROI-map SHA-256 before feature
extraction. A missing, duplicated, renamed, or extra-required channel fails the affected
subject-condition; there is no nearest-neighbor substitution or outcome-dependent repair.

## Residual uncertainty and claim limits

- EGI calls the 10–10 correspondences approximate. Its technical note reports that ventral and
  posterior equivalents can be 1–2.5 cm from their named 10–10 sites [Luu & Ferree 2005].
- NeuroAI uses the MNE standard HydroCel template attached during preprocessing, not individual
  digitized sensor locations. Cap placement and head anatomy vary across participants.
- The exact five-electrode template was published in a primary ERP language experiment, not a
  subsequent-memory prediction study. Its role here is reproducible spatial factorization; the
  PEERS papers support the eight-region family and memory relevance, not this exact membership.
- Averaging improves interpretability and reduces dimension, but can blur focal or oppositely
  signed effects within a patch. No held-out performance was used to choose the number, names,
  membership, or order of the ROIs.
- Scalp ROIs must be described as sensor-space regions. They cannot establish a cortical source,
  lobe-specific mechanism, or anatomical localization.

## Alternatives rejected

- **Copy the older GSN figure as if it were HydroCel.** Rejected because the paper provides no
  compatible machine-readable list and its numeric/10–10 labeling does not translate one-to-one.
- **Use Long et al.'s six regions.** Scientifically relevant, but exact numbered membership is not
  published and the omission of posterior-superior regions was motivated by that paper's
  intracranial comparison rather than M1's general predictive feature map.
- **Use all 125 channels or data-driven spatial filters.** More dimensional and less interpretable;
  it would also abandon the pre-result ROI design and increase overfitting risk.
- **Invent coordinate thresholds or tune clusters.** Rejected because no task-relevant primary or
  authoritative source fixes those thresholds for this montage, and tuning would create another
  outcome-sensitive degree of freedom.

## Stop/replan rule

Stop before feature extraction if the JSON companion and this table disagree, any required channel
is absent, the epoch montage is not the frozen HydroCel schema, or an implementation attempts to
change membership/order after seeing performance. A scientifically justified replacement requires
a new map version, a new charter version, invalidation of dependent manifests/caches/results, and
Sofia's approval before any new held-out outcome is inspected.
