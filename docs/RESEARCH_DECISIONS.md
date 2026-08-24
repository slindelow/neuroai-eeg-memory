# Research decisions

These decisions were fixed before the held-session M1 and M3 result was opened. The purpose was to
reduce flexible analysis choices and make every model comparison reproducible.

| Decision | Selected design | Main alternative | Reason |
| --- | --- | --- | --- |
| Prediction target | Recall of each studied word during its list's immediate free-recall period | Participant-level memory score | Preserves the item-level relationship between one word, one EEG epoch, and one outcome |
| Comparator | B3 behavioral and lexical baseline | Chance or M1 alone | Tests whether EEG adds information beyond known task and word effects |
| EEG window | 300 to 1600 ms, split at 950 ms | Full 0 to 2000 ms window or outcome-sensitive 800 ms split | Closest PEERS-family item-level precedent, bounded dimensionality, and no reuse of the labeled 800 ms boundary |
| Spatial representation | Eight reproducible sensor-space groups | Individual channels or source localization | Reduces dimensionality without making unsupported cortical claims |
| Normalization | Training-session mean and population standard deviation applied unchanged | Whole-held-session unlabeled batch normalization | Matches genuinely unseen-session prediction without adapting to the held EEG distribution |
| Outer split | Highest eligible session held out within each participant | Random trial split | Tests later-session generalization and blocks nearby trials from crossing the boundary |
| M1 model | L2 logistic regression over 128 spectral features | Neural network or nonlinear ensemble | Interpretable, bounded, and appropriate for the available participant-session structure |
| M3 model | Logistic regression over session-out B3 and M1 probabilities | Raw feature concatenation | Keeps the incremental EEG comparison explicit and prevents the larger EEG block from dominating |
| Primary condition | Ocular-removed EEG | Ocular-kept EEG | Fixed primary preprocessing choice with ocular-kept retained as sensitivity evidence |
| Primary estimand | Mean participant AUC(M3) minus AUC(refitted B3) | Pooled-trial AUC or M1 versus chance | Gives each participant equal weight and directly answers the incremental question |
| Uncertainty | 2,000 participant bootstrap resamples, seed 0 | Trial bootstrap | Matches the participant-level estimand and preserves within-participant trial dependence |
| Balanced accuracy | Not computed | Fixed or learned threshold | No scientific decision rule justified a probability threshold |

## Exploratory classification

Recall-labeled exploratory evidence from the same 40-participant EEG cohort influenced M1's lower
frequency bound and two-bin design. The held-session result was protected from direct tuning, but
the project remains exploratory because the feature design was not developed on an untouched
cohort.

## Rejected claim paths

The analysis was not allowed to claim:

- causality between EEG and recall;
- a neural mechanism of memory encoding;
- cortical localization from sensor-space groups;
- equivalence or proof of no useful EEG information;
- confirmation;
- generalization to unseen participants; or
- a positive primary result based on a secondary metric or sensitivity condition.
