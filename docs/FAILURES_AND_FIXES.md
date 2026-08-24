# Failures and fixes

The project keeps failures because they explain why the final pipeline has its current controls.

| Failure | Scientific risk | Fix |
| --- | --- | --- |
| An earlier demonstration dataset did not support the intended encoding question | A polished model could answer a different or confounded question | Reject the dataset and move to PEERS |
| Initial recall-label assumptions mixed event types | The model could predict the wrong memory outcome | Audit event semantics and bind each label to participant, session, list, and item position |
| Naive word memorability included the target participant | The answer could enter the behavioral input | Remove the target participant from the numerator, denominator, and smoothing prior |
| Ocular conditions retained different trials | A condition difference could be caused by trial composition | Compare conditions on the exact common participant and trial intersection |
| EEG normalization scope was ambiguous | The held session could influence its own representation | Fit channel-frequency statistics on permitted training sessions only |
| M3 could receive fitted-row base predictions | The stack could train on unrealistically easy inputs | Generate session-out B3 and M1 probabilities for every M3 training row |
| Hyperparameter dependence was initially described too strongly | Session-out predictions could be mislabeled fully cross-fitted | Record the D44-C qualification that the shared `C` can depend on all outer-training labels |
| Several full runs reached operational ceilings | Partial outcomes could influence replanning | Seal partial outputs, change capacity only, and rerun with unchanged science |
| The reproduction comparator flagged expected run metadata | Execution timestamps could be mistaken for scientific disagreement | Normalize only run-specific pointers while requiring exact scientific equality |
| Public release of the original repository exposed unrelated historical metadata | A personal project could become associated with non-personal history | Preserve the authoritative repository privately and publish this clean companion |

## Failure policy

A held-session key in model-fitting or normalization provenance invalidates the affected artifact.
The correct response is to stop, record the leak, regenerate from a verified boundary, and rerun
the comparison. A null, adverse, imprecise, or discordant result is not a pipeline failure.
