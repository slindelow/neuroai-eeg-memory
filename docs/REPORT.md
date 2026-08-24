# NeuroAI research report

## Building a leakage-safe EEG memory prediction pipeline from real data

## Project focus

The most important outcome of NeuroAI was the analysis system itself. I wanted to understand what
it takes to turn real EEG recordings and behavioral event logs into a defensible prediction study.
That required much more than fitting a classifier. I had to define the outcome at the level of one
studied word, build a fair non-EEG comparator, engineer an interpretable EEG representation,
prevent several forms of leakage, compare the models on identical trials, and preserve enough
provenance to reproduce the result.

My role was to own the research question, scientific decisions, protocol teach-backs, and
interpretation. I used AI coding tools to assist with implementation, tests, and documentation,
then reviewed the scientific boundaries and traced them to the relevant source, contract, or test.

The scientific question was whether EEG recorded while a participant studied a word added
held-session predictive information about later immediate free recall beyond what the word and its
experimental position already predicted.

The project used the Penn Electrophysiology of Encoding and Retrieval Study, or PEERS, available
through [OpenNeuro ds004395](https://doi.org/10.18112/openneuro.ds004395.v2.0.0). PEERS combines
multi-session scalp EEG with a free-recall task in which participants study lists of words and
then try to recall them.

![Model pipeline](assets/neuroai-methods-pipeline.png)

## 1. Converting an experiment into a prediction table

The prediction target was not a participant-level memory score. One row represented one studied
word presentation. Its binary label indicated whether that same item was produced during the
immediate free-recall period for its list.

Each row was identified by four fields:

- participant;
- session;
- list;
- item position.

This key was used throughout the event table, EEG features, fold manifests, predictions, and
evaluation records. A strict key prevented a common research-software failure in which arrays are
joined by row order even though preprocessing has removed or reordered trials.

The event audit also separated immediate list recall from recognition, final free recall, practice
events, and intrusions. This mattered because a technically valid classifier can still answer the
wrong scientific question if the label is attached to the wrong event type.

Implementation: [`src/peers/events.py`](../src/peers/events.py)

## 2. Building the behavioral baseline first

An EEG-only model was not enough to answer the project question. Recall is already predictable
from the structure of the task and the properties of the studied word. If the EEG model were only
compared with chance, it could appear useful while adding nothing beyond those existing signals.

The final behavioral baseline, B3, used:

- serial position in the 16-word list;
- session and list context;
- word length;
- word frequency; and
- leave-one-participant-out word memorability.

![Recall by serial position](assets/serial_position_curve.png)

The serial-position curve shows why this baseline was necessary. Recall probability varied across
the list, so the experimental design itself contained predictive information.

Word memorability required a separate leakage control. A naive memorability score calculated from
the full cohort would include the target participant's own recall outcome inside their input
feature. B3 therefore removed the target participant from the word's recalled count, total count,
and smoothing prior.

B3 used L2-regularized logistic regression. Logistic regression made the baseline transparent,
fast enough for nested session-level fitting, and easy to refit on the exact trials that survived
the EEG pipeline.

Implementation: [`src/peers/baselines.py`](../src/peers/baselines.py)

## 3. Preprocessing real EEG

The EEG pipeline downloaded selected PEERS EDF files, identified study events, created word-locked
epochs, repaired or rejected problematic channels and trials, and wrote separate ocular-removed
and ocular-kept conditions.

Real EEG introduced practical problems that are absent from a clean tutorial dataset. Files were
large, preprocessing could fail after several minutes, and the two ocular-processing conditions
did not always retain the same trials. The build therefore became resumable at the session level.
Each completed epoch file could be validated and reused after an interruption.

The direct condition comparison used the exact common trial intersection. Without that rule, a
difference between ocular conditions could reflect different word presentations rather than the
preprocessing choice.

Implementation:

- [`scripts/build_one_session.py`](../scripts/build_one_session.py)
- [`scripts/build_dataset.py`](../scripts/build_dataset.py)
- [`src/peers/eeg.py`](../src/peers/eeg.py)

## 4. Designing an interpretable EEG representation

The M1 model summarized EEG spectral power rather than learning directly from the full time series.
Each trial produced 128 features:

`2 time bins x 8 sensor-space regions x 8 frequencies`

The analysis window covered 300 to 1600 ms after word onset and was split exactly at 950 ms. The
window followed the closest item-level PEERS-family spectral precedent while excluding the earliest
sensory interval and limiting the model's dimensionality. It omitted 0 to 300 ms and 1600 to
2000 ms, which may also contain useful information.

The eight frequency values were log-spaced from 2 to 100 Hz. The HydroCel-129 recording montage
was reduced to eight symmetric sensor-space groups. These groups were a reproducible
dimensionality-reduction translation, not an exact replication of published sensor membership and
not a claim about cortical sources.

![Exploratory spectral subsequent-memory contrast](assets/spectral_sme_ocular-removed.png)

The exploratory spectral analysis used recalled-minus-forgotten power to understand the data and
help define M1. Because labels from the same cohort influenced part of that design, the final
analysis was classified as exploratory. The held-session evaluation protects the prediction
boundary, but it does not turn same-cohort feature development into confirmation.

Primary methodological sources included item-level PEERS spectral work by
[Weidemann and Kahana](https://doi.org/10.1037/xlm0000966), subsequent-memory spectral analyses by
[Long and colleagues](https://doi.org/10.1016/j.neuroimage.2013.08.052), and PEERS free-recall
retrieval spectral work by
[Katerman and colleagues](https://doi.org/10.1016/j.neuroimage.2021.118748).

Implementation:

- [`src/peers/m1_feature_generation.py`](../src/peers/m1_feature_generation.py)
- [`src/peers/m1_features.py`](../src/peers/m1_features.py)
- [`docs/roi-map-m1-v1.md`](roi-map-m1-v1.md)

## 5. Preventing normalization leakage

Spectral power differs across channels, frequencies, participants, and sessions. M1 therefore
needed normalization, but normalization can leak information if the held session helps determine
its own scale.

The primary D26-A rule was strictly inductive. For each channel-frequency combination, the mean
and population standard deviation were calculated from the permitted training sessions and then
applied unchanged to the validation or held session. The held session's unlabeled EEG distribution
was not used.

This design matched the intended claim. A model trained on earlier sessions should be evaluated on
a genuinely unseen session without adapting to the complete new-session batch. Whole-session
unlabeled normalization remained a separately labeled transductive sensitivity option, not the
primary pipeline.

The implementation records fit keys, apply keys, partition hashes, and zero-variance handling for
every transform. Tests intentionally insert held-session keys and confirm that validation fails.

## 6. Evaluating later-session generalization

The split was organized within each participant. The highest eligible session was reserved for
final evaluation, while earlier sessions formed the outer-training set.

Within the outer-training sessions, leave-one-session-out folds served two purposes. They selected
M1's regularization value and generated the session-out predictions later used to train M3.

![Held-session evaluation boundary](assets/neuroai-leakage-controls.png)

The regularization parameter `C` controls how strongly logistic regression restrains its
coefficients. Candidate values were compared by mean inner-session ROC-AUC. Ties selected the
smallest `C`, then the declared grid order, which prevented an arbitrary favorable choice.

![Selected M1 regularization values](assets/regularization-selection.png)

Under the selected D44-C design, the predicted inner session was excluded from the coefficient and
normalization fits used on it. Its labels could still influence the shared `C` selected across the
outer-training folds. The resulting inputs were therefore described as session-out, not fully
session-cross-fitted.

Implementation:

- [`src/peers/m1_model_execution.py`](../src/peers/m1_model_execution.py)
- [`src/peers/m1_predictions.py`](../src/peers/m1_predictions.py)
- [`src/peers/m1_manifests.py`](../src/peers/m1_manifests.py)

## 7. Building the three-model comparison

### B3

B3 estimated recall from behavioral and lexical features without EEG. It was refitted on the exact
M1 survivor trials so the comparison could not be explained by different trial sets.

### M1

M1 used the 128 normalized spectral features in an L2-regularized logistic regression. It produced
one probability of later recall for each held-session word presentation.

### M3

M3 used only two inputs, the session-out B3 probability and the session-out M1 probability. It did
not concatenate the raw behavioral and EEG feature blocks.

Raw concatenation could learn interactions between the two feature sets, but the larger EEG block
could dominate the model and obscure the incremental question. The two-probability stack made the
comparison explicit and kept the small dataset from supporting a needlessly flexible model.

Training M3 on ordinary fitted-row B3 and M1 predictions would have created another leakage path.
Those inputs can be unrealistically confident because the base models already saw the same labeled
rows. M3 therefore received session-out probabilities for its training rows and one-time outer
held-session probabilities for evaluation.

## 8. Fixing the estimand before results

The primary quantity was calculated separately for each participant:

`held-session AUC(M3) - held-session AUC(refitted B3)`

The project then took the unweighted mean of those 38 participant differences. Each participant
therefore contributed equally, rather than allowing participants with more surviving trials to
dominate the result.

Uncertainty was calculated with 2,000 participant-level bootstrap resamples using a fixed seed.
The primary condition was ocular-removed EEG. Ocular-kept EEG and the exact common-trial comparison
were sensitivity evidence.

Balanced accuracy was not included because it requires a probability threshold. The project had
no intervention, cost rule, or scientific reason that justified one threshold over another. Adding
one after seeing the result would have introduced another flexible analysis path.

Implementation: [`src/peers/m1_evaluation.py`](../src/peers/m1_evaluation.py)

## 9. Building reproducibility into the model

The project used machine-readable contracts for the charter, scalp regions, manifests, feature
tables, predictions, evaluation records, and preflight rules. These contracts fixed column order,
data types, feature names, learner inputs, seeds, hashes, and provenance fields.

Large outputs were written atomically and reopened through validators before being accepted. The
run index bound the model artifacts to the scientific charter, input inventory, code state, and
environment lock. A separate verifier loaded the stored files from disk and recomputed their
relationships instead of trusting console output.

The full analysis was then reproduced from a fresh root. The first and reproduced runs matched on
trial keys, labels, features, probabilities, selected models, metrics, participant records, and
aggregate results.

The public [aggregate provenance record](../results/provenance.json) preserves the frozen code,
verified run, reproduction comparison, and source result digests without exposing participant
data or local execution paths.

Implementation:

- [`src/peers/m1_artifact_execution.py`](../src/peers/m1_artifact_execution.py)
- [`src/peers/m1_result_storage.py`](../src/peers/m1_result_storage.py)
- [`src/peers/m1_locked_orchestration.py`](../src/peers/m1_locked_orchestration.py)
- [`src/peers/m1_full_execution.py`](../src/peers/m1_full_execution.py)

## 10. What the fitted models showed

![Held-session model AUC distributions](assets/model-auc-comparison.png)

The mean held-session AUCs were 0.6245 for B3, 0.5197 for M1, and 0.6202 for M3 in the primary
ocular-removed condition. These values describe how the individual models behaved, but they were
not the primary test.

The primary mean participant difference between M3 and refitted B3 was -0.0044. The 95%
participant-bootstrap interval ranged from -0.0121 to 0.0033, which produced the prespecified no
detectable increment and imprecise category.

![Participant-level incremental comparison](assets/primary-increment.png)

The exact common-trial ocular analysis was discordant. Ocular-removed remained no-detectable and
imprecise, while ocular-kept was adverse.

![Ocular-processing sensitivity](assets/condition-sensitivity.png)

The result does not establish equivalence or prove that EEG contains no useful memory-related
information. It shows that this selected spectral representation did not add detectable
held-session predictive information beyond B3 in this exploratory pipeline.

## 11. Failures that changed the project

The final pipeline was shaped by failures rather than produced in one clean pass.

- An earlier dataset was rejected after the task structure made the intended encoding analysis
  invalid.
- Naive word memorability contained the target participant's own outcome, so it was replaced with
  a target-excluded calculation that also removed the participant from the smoothing prior.
- Ocular conditions retained different trials, so the direct comparison was changed to an exact
  common intersection.
- Normalization scope was ambiguous, so the primary question was tied explicitly to training-only
  statistics.
- In-sample base predictions made M3 too easy to train, so its inputs became session-out.
- Several full reproductions reached operational limits. Partial outcomes remained sealed, and the
  capacity plan changed without changing the science.
- A comparator initially treated run-specific provenance pointers as scientific differences. The
  comparison was corrected while keeping all scientific payloads exact-required.

The [failures and fixes record](FAILURES_AND_FIXES.md) explains why each issue mattered.

## 12. What I learned

The main learning was that model validity depends on the entire path from experimental event to
evaluation record. The classifier was one part of that path. Label construction, trial identity,
normalization scope, cross-fitting, paired comparison, uncertainty, and reproducibility determined
whether the final number answered the intended question.

Working with real EEG also made the tradeoff between interpretability and information loss
concrete. Regional spectral averages created a model I could inspect and defend, but they may have
discarded informative spatial or temporal structure. A more complex model could represent more of
the signal, but it would require more data and a new prespecified development and confirmation
cycle.

The project also changed how I think about a useful result. A positive EEG effect was not required
for the work to succeed. The research value came from building a test that remained interpretable
when the increment was weak, uncertain, or preprocessing-dependent.

## 13. Limitations and next study

This project evaluated later-session generalization within participants, not transfer to unseen
people. The EEG features omitted the first 300 ms and final 400 ms of the stored epoch, compressed
the montage into approximate sensor-space regions, and summarized the signal as spectral power.
The same cohort's labeled exploratory findings influenced part of M1's design.

A confirmatory next study would freeze the current pipeline before opening labels from an untouched
participant cohort. Wider windows, alternative spatial representations, connectivity features,
or neural networks would be treated as a new exploratory development phase. Any new model would
then require its own untouched confirmation boundary.

## Reproduce and inspect

The public repository includes the model code, machine-readable contracts, synthetic tests,
de-identified aggregate summary, and figures. Raw EEG, participant-level predictions, and local
execution receipts are not redistributed.

```bash
uv sync --locked --all-groups
uv run pytest -q
```

See [GETTING_STARTED.md](../GETTING_STARTED.md) for the data workflow and
[RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md) for the main design alternatives.
