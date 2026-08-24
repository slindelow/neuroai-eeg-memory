# References

Every external claim in this project cites a key from this file. Keys are `[Author Year]`.

**Only verified entries appear here.** Where an author list or page range has not been
checked against the source itself, it is marked `(unverified detail)` rather than guessed.
A fabricated citation is worse than a missing one.

---

## Methods we use

**[Jas 2017]** Jas, M., Engemann, D.A., Bekhti, Y., Raimondo, F., Gramfort, A.
*Autoreject: Automated artifact rejection for MEG and EEG data.* NeuroImage 159:417–429, 2017.
→ Governs **D14** (per-channel cross-validated thresholds with per-epoch interpolation).
(Author list and volume unverified against the source PDF; DOI to be confirmed.)

**[Winkler 2015]** *On the influence of high-pass filtering on ICA-based artifact reduction
in EEG-ERP.* https://pubmed.ncbi.nlm.nih.gov/26737196/
→ Governs **D13**. Finding: high-pass between **1–2 Hz** consistently gave the best
signal-to-noise, single-trial classification accuracy, and proportion of near-dipolar
components. Also supports the pattern of fitting ICA on a high-passed copy and applying the
weights to less-filtered data, preserving low-frequency information.

**[Klug 2021]** Klug, M., Gramann, K. *Identifying key factors for improving ICA-based
decomposition of EEG data in mobile and stationary experiments.* European Journal of
Neuroscience, 2021. https://onlinelibrary.wiley.com/doi/10.1111/ejn.14992
→ Refines **D13**. Finding: 0.5 Hz is acceptable at 64 channels, but **higher channel counts
need a higher cutoff, up to 1.25 Hz**. This dataset has 125 EEG channels, placing our 1 Hz
choice at the appropriate end of the range rather than at a default.

**[Appelhoff 2019]** Appelhoff, S. et al. *MNE-BIDS: Organizing electrophysiological data
into the BIDS format and facilitating their analysis.* Journal of Open Source Software
4:1896, 2019. https://doi.org/10.21105/joss.01896 → cited by the dataset itself.

**[Pernet 2019]** Pernet, C.R., Appelhoff, S., Gorgolewski, K.J., Flandin, G., Phillips, C.,
Delorme, A., Oostenveld, R. *EEG-BIDS, an extension to the brain imaging data structure for
electroencephalography.* Scientific Data 6:103, 2019.
https://doi.org/10.1038/s41597-019-0104-8 → cited by the dataset itself.

**[Scikit-learn leakage]** *Common pitfalls and recommended practices: data leakage.*
scikit-learn 1.9 documentation.
https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
→ Authoritative implementation guidance for the strict D26 option. Any statistic learned by
a preprocessing transform, including a normalization mean and variance, is fitted on training
data and then applied unchanged to held-out data. The documentation explicitly treats inclusion
of test observations in the normalization average as test information influencing the model.

**[AdaBN 2018]** Li, Y., Wang, N., Shi, J., Hou, X., Liu, J. *Adaptive Batch Normalization for
practical domain adaptation.* Pattern Recognition 80:109–117, 2018.
https://doi.org/10.1016/j.patcog.2018.03.005
→ Primary methods evidence for the transductive D26 option. Replacing source-distribution
normalization statistics with statistics from the unlabeled target domain is an explicit domain-
adaptation procedure. This supports describing held-session normalization as batch adaptation,
not as ordinary inductive prediction.

**[Simmons 2011]** Simmons, J.P., Nelson, L.D., Simonsohn, U. *False-positive psychology:
Undisclosed flexibility in data collection and analysis allows presenting anything as
significant.* Psychological Science 22(11):1359–1366, 2011.
https://doi.org/10.1177/0956797611417632
→ Primary evidence for D29's fail-closed rule: analysis-window and split flexibility must be
resolved before prediction outcomes, especially because labeled R4 evidence already showed that
the visual boundary can change the apparent result.

**[Luu & Ferree 2005]** Luu, P., Ferree, T. *Determination of the HydroCel Geodesic Sensor
Nets' Average Electrode Positions and Their 10–10 International Equivalents.* Electrical
Geodesics, Inc. Technical Note, revised 24 October 2005.
https://www.egi.com/images/HydroCelGSN_10-10.pdf
→ Authoritative manufacturer evidence for D30. EGI digitized adult 128-channel HydroCel nets and
10–10 sites, registered and averaged them, then selected approximate HydroCel equivalents using
surface arc distance. Table 1 supplies every 10–10-to-`E*` mapping used by NeuroAI's ROI map. The
note explicitly calls the mapping approximate: frontal sites are generally closer, while temporal,
ventral, and posterior equivalents can be 1–2.5 cm away. This supports an exact reproducible
channel contract, not a claim of exact anatomical correspondence.

**[Maxfield 2012]** Maxfield, N.D., Pizon-Moore, A.A., Frisch, S.A., Constantine, J.L.
*Exploring semantic and phonological picture-word priming in adults who stutter using
event-related potentials.* Clinical Neurophysiology 123(6):1131–1146, 2012.
https://doi.org/10.1016/j.clinph.2011.10.003 (PMID 22055837; PMC3277650).
→ Primary experimental source for D30's machine-readable eight-factor 10–10 template. Its analysis
section defines exactly five electrodes in each left/right × anterior/posterior × inferior/superior
ROI and states that other electrodes are excluded. This is spatial-method evidence from an ERP
language experiment, not evidence about subsequent memory; NeuroAI uses only the exact symmetric
sensor grouping and translates it through [Luu & Ferree 2005].

**[Weidemann 2009]** Weidemann, C.T., Mollison, M.V., Kahana, M.J.
*Electrophysiological correlates of high-level perception during spatial navigation.*
Psychonomic Bulletin & Review 16(2):313–319, 2009.
https://doi.org/10.3758/PBR.16.2.313 (PMID 19293100; PMC2704578).
→ Primary source for the eight GSN scalp-region family later reused by [Katerman 2022]. The paper
defines LAI/LAS/LPI/LPS and their right homologues as shaded regions on a 128-channel GSN figure,
but does not provide machine-readable membership. Because the figure uses an older GSN drawing
with mixed numeric and 10–10 labels, it cannot be claimed as an exact HydroCel-129 list.

---

## Dataset

**[PEERS]** Kahana, M.J., Rudoler, J.H., Lohnas, L.J., Healey, K., Aka, A., Broitman, A.,
Crutchley, E., Crutchley, P., Alm, K.H., Katerman, B.S., Miller, N.E., Kuhn, J.R., Li, Y.,
Long, N.M., Miller, J., Paron, M.D., Pazdera, J.K., Pedisich, I., Weidemann, C.T.
*Penn Electrophysiology of Encoding and Retrieval Study (PEERS).* OpenNeuro ds004395, CC0.
https://openneuro.org/datasets/ds004395 · doi:10.18112/openneuro.ds004395.v2.0.0
(Author list taken from the dataset's own `dataset_description.json` — verified.)

**[ds008496]** *Brain Connectivity Based on EEG of Student Learning Processes Using Gen-AI.*
OpenNeuro ds008496, CC0. doi:10.18112/openneuro.ds008496.v1.0.1
→ The dataset rejected in **D2**. `Authors` field is empty and no paper is linked, which is
part of why its methods could not be checked.

---

## Analysis of the same data by the originating lab

**[Katerman 2022]** Katerman, B.S., Li, Y., Pazdera, J.K., Keane, C., Kahana, M.J.
*EEG biomarkers of free recall.* NeuroImage 246:118748, 2022.
https://doi.org/10.1016/j.neuroimage.2021.118748
(Authors, journal, volume and article number verified by extracting the PDF directly.)
→ Two things read from the source, both corroborating our independent choices:
- Bad channels identified by **log-variance |z| > 3** (plus a Hurst-exponent criterion we do
  not use), separately per partition. Independently matches our **D10** statistic.
- Spectral features: multitaper power **4–128 Hz**, log-transformed, electrodes grouped into
  regions of interest; frequencies spaced every 2 Hz over 4–26 Hz and every 6 Hz over
  26–128 Hz. A precedent for **M1** feature design.
- Eight scalp ROIs use the same left/right × anterior/posterior × inferior/superior family as
  [Weidemann 2009], but the recordings used a 128-channel **BioSemi** system. The paper's map is
  relevant to the regional factorization, not an exact membership source for HydroCel-129.
- Note this paper concerns *retrieval*, not encoding, so it is not a benchmark for our AUC.

**[EEG decoders 2024]** *EEG decoders track memory dynamics.* Nature Communications, 2024.
https://www.nature.com/articles/s41467-024-46926-0
→ Verified full text; same source as [Li 2024] below. Uses ltpFR2 (PEERS4), with 88 of 98
participants meeting the prespecified recall-rate eligibility rule. Encoding classifiers use
leave-one-session-out evaluation. Within each session, power is log-transformed and z-scored
across events separately for every electrode-frequency pair. This is the closest published
decoder to our project, but its normalization uses the unlabeled held-session feature
distribution and therefore estimates a batch-adapted rather than one-trial-at-a-time target.

---

## Evaluating incremental value

**[Brodersen 2010]** Brodersen, K.H., Ong, C.S., Stephan, K.E., Buhmann, J.M.
*The balanced accuracy and its posterior distribution.* 20th International Conference on Pattern
Recognition:3121–3124, 2010. https://doi.org/10.1109/ICPR.2010.764
→ Primary methods source defining binary balanced accuracy as the mean of sensitivity and
specificity. It motivates the metric for class-imbalanced held-out classification but starts from
a confusion matrix, so a continuous prediction must first be converted to a class label at a
specified operating threshold. For D42-A, that dependency supports deliberate omission rather
than silently inventing an operating point that the project question does not define.

**[Youden 1950]** Youden, W.J. *Index for rating diagnostic tests.* Cancer 3(1):32–35, 1950.
https://doi.org/10.1002/1097-0142(1950)3:1%3C32::AID-CNCR2820030106%3E3.0.CO;2-3
→ Primary source for `J = sensitivity + specificity − 1`. Because balanced accuracy is
`(sensitivity + specificity) / 2`, maximizing balanced accuracy over thresholds is mathematically
equivalent to maximizing Youden's J; it is an estimated operating-point choice, not part of AUC.

**[Fluss 2005]** Fluss, R., Faraggi, D., Reiser, B. *Estimation of the Youden Index and its
associated cutoff point.* Biometrical Journal 47(4):458–472, 2005.
https://doi.org/10.1002/bimj.200410135
→ Primary simulation study comparing procedures for estimating both Youden's J and its associated
cutoff. It found the empirical procedure performed worst overall for J and that cutoff-estimation
performance varied by distribution/scenario. This supports treating a fitted threshold as another
estimated, provenance-bearing model choice rather than a harmless display default. D42-A avoids
adding that otherwise unnecessary fitted choice to exploratory v1.

**[Brier 1950]** Brier, G.W. *Verification of forecasts expressed in terms of probability.*
Monthly Weather Review 78(1):1–3, 1950.
https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2
→ Primary source for the quadratic score on probabilistic forecasts. NeuroAI's binary Brier score
evaluates the held-out probabilities directly and therefore needs no class-label threshold. It
remains the prespecified probability-accuracy evidence after D42-A omits balanced accuracy.

**[AJE 2012]** *Interpreting Incremental Value of Markers Added to Risk Prediction Models.*
American Journal of Epidemiology 176(6):473, 2012.
https://academic.oup.com/aje/article/176/6/473/118184
→ Governs the revision to **D17**. Findings: the increase in AUC **depends on the strength of
the baseline model**, and AUC has been criticised for insensitivity when the baseline
performs well; being rank-based, its power to detect improvement is low. Recommends
reporting ΔAUC, integrated discrimination improvement (IDI) and continuous net
reclassification improvement (NRI) together, as they carry complementary information.
(Author list unverified.)

**[Added value 2018]** *Quantifying the added value of new biomarkers: how and how not.*
Diagnostic and Prognostic Research, 2018.
https://link.springer.com/article/10.1186/s41512-018-0037-2
→ Cautions that IDI and NRI can indicate improvement where none exists — so they supplement
ΔAUC rather than replace it.

---

## Combining feature blocks

**[Wolpert 1992]** Wolpert, D.H. *Stacked generalization.* Neural Networks 5(2):241–259,
1992. https://doi.org/10.1016/S0893-6080(05)80023-1
→ Primary foundation for D44. The second-level generalizer receives guesses made for portions of
the learning set not used to teach the corresponding base generalizer. This supports out-of-fold
base predictions as the stack-training representation; it does not specify NeuroAI's
hyperparameter-selection scope.

**[van der Laan 2007]** van der Laan, M.J., Polley, E.C., Hubbard, A.E. *Super Learner.*
Statistical Applications in Genetics and Molecular Biology 6(1), 2007.
https://doi.org/10.2202/1544-6115.1309
→ Primary ensemble-method evidence for D44. The combination rule is learned from V-fold
cross-validated predictions of candidate learning algorithms. NeuroAI cites this for the
cross-validated meta-input principle, not for a claim that its current provenance is sufficient.

**[Cawley & Talbot 2010]** Cawley, G.C., Talbot, N.L.C. *On over-fitting in model selection and
subsequent selection bias in performance evaluation.* Journal of Machine Learning Research
11:2079–2107, 2010. https://jmlr.org/papers/v11/cawley10a.html
→ Primary model-selection evidence for D44. Optimizing a finite-sample cross-validation criterion
can itself overfit; selection must therefore be treated as part of the learned procedure rather
than ignored when judging independence or evaluation evidence.
→ D44-C application: NeuroAI therefore records the shared outer-training-selected M1 `C` as a
selection dependence and limits the claim to session-out coefficients/transforms conditional on
that selected `C`. This exact wording is a project-specific inference, not language from the paper.

**[Scikit-learn stacking]** *StackingClassifier.* scikit-learn 1.9 documentation.
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
→ Authoritative implementation guidance for D44. The final estimator is trained on
cross-validated base-estimator predictions, while the production base estimators are fit on all
training rows. The `prefit` warning makes the in-sample overfitting risk explicit. This is an
implementation contract, not primary scientific literature.

**[Multimodal 2020]** *Integrating across neuroimaging modalities boosts prediction accuracy
of cognitive ability.* PLOS Computational Biology.
https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008347
→ Supports stacking over concatenation: adding modalities raises overfitting risk, and
second-level models over per-modality predictions handle heterogeneous blocks better.

**[Stacking brain dynamics]** *Stacking models of brain dynamics to improve prediction of
subject traits in fMRI.* Imaging Neuroscience (MIT Press).
https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00267/123872/
→ Same conclusion in a different modality: first-level predictions per feature type,
combined by a regularised second-level model.

---

## Claims resolved 2026-08-09 (verified by literature search)

### U1 — epoch window "−200 to +1600 ms is ERP field standard" → **REFUTED as stated, [C] precedent exists**

No methods authority prescribes an epoch length. [Keil 2014] says only *"The time range used to
segment the data should be reported."* [ERP CORE] uses **−200 to +800** for all five
stimulus-locked components, and Luck's own recommended pipeline uses −200 to +800.

The +1600 figure is a **Computational Memory Lab house window**, and it originated as a
*stimulus duration*: in ltpFR2 each word was on screen for 1600 ms. [Broitman 2025] states
plainly that it was inherited — *"We selected this encoding epoch to match several previous
scalp EEG studies."*

**However, precedent on our exact paradigm does exist.** [Long 2014] used 0–1600 ms on a
16-list × 16-word scalp sample, and [Broitman 2025] used 0–1600 ms on ltpFR. So the window is
defensible as *precedent in this literature*, never as *ERP standard*.

**Open item:** ltpFR's actual stimulus duration could not be verified from the data
(`events.tsv` `duration` is `n/a`, no task JSON). If it is 3000 ms rather than 1600 ms, then
the inherited window covers only the first half of each presentation, and extending the
analysis window is an unexplored and defensible option.

### U2 — baseline "−200 to 0 ms is standard" → **[C] dominant convention, explicitly not prescribed — and see the interaction below**

[Keil 2014], verbatim: *"The choice of baseline period is up to the investigators and should be
appropriate to the experimental design… A justification for the time window and baseline period
should be included."* The guidelines require justification; "it's standard" is precisely what
they ask you to replace. It is the default in [ERP CORE], Luck's pipeline, MNE-Python and
ERPLAB, so "dominant convention" holds; "recommended by an authority" does not.

**The material finding — baseline correction interacts with high-pass filtering.** Three
independent sources warn against the combination this pipeline uses (0.5 Hz high-pass *and*
200 ms baseline correction):

- [Tanner 2015]: *"high-pass filters with cutoffs at 0.3 Hz and above produced artifactual
  effects of opposite polarity before the true effect."*
- [Maess 2016]: *"we recommend avoiding baseline correction in combination with high-pass
  filtering."*
- [Delorme 2023]: *"if the data is high-pass filtered at or above 0.5 Hz, subtracting mean
  baseline activity should be omitted for event-related analyses… especially when the baseline
  is shorter than 500 ms."*

**Scope limit, and why this is survivable:** these warnings concern **ERP amplitude**
measurement. This project analyses **spectral power** (D16). Subtracting a per-epoch constant
alters only the DC component, to which Morlet wavelets at ≥2 Hz do not respond, so the baked-in
baseline correction is close to inert for our primary analysis. The warnings would bite if we
returned to ERP measurement, and are recorded here so that we do not.

### U3 — "average reference is standard for high-density EEG / EGI nets" → **CONTESTED as stated; the ordering claim is [C] SUPPORTED**

Splits into three claims that verify very differently.

**Coverage requirement — [C] SUPPORTED.** [Junghöfer 1999] names the bias: *"this approximation
is valid only with accurate spatial sampling… If electrodes are concentrated in one region of
the surface, such as just on the scalp, then the average is biased toward that region."* The
"Polar Average Reference Effect" most affects electrodes at the *centre* of the covered region
— i.e. near the vertex, i.e. near Cz.

**"Standard/recommended" — NOT SUPPORTED.** No methods authority found recommends it. [Keil
2014]'s referencing section is a *reporting* requirement only. Luck (2022) declines to endorse
it: *"it often provides a very poor approximation of the absolute voltage."*

**The EGI-specific clause cuts against us.** EGI's own documentation puts the adequacy bar at
**256** channels: *"To achieve a spatial sampling with 2 cm inter-electrode distances, about 256
recording channels are required."* We have 129. The best support comes from a hostile witness —
[Yao 2019], who advocate REST instead, concede: *"AR may be an acceptable approximation if…
the montage is with good coverage, such as wider than a semi-head surface as the EGI system with
enough density such as > 128 channels."*

**The ordering claim — [C] STRONGLY SUPPORTED, near-verbatim, on matching hardware.**
[Debnath 2020], a peer-reviewed pipeline validated on a **128-channel EGI net referenced to
Cz**: *"channel interpolation must be performed prior to the computation of the average
reference so that there is not a biased weighting of specific scalp locations in the estimate
of the average reference."* And [Bigdely-Shamlo 2015]: *"Noisy channels can irrecoverably
contaminate the signal when preprocessing applies average referencing prior to bad channel
detection… The average can be highly skewed by a single outlier."*

**Reviewer objections to anticipate:** REST ([Yao 2019]: *"we do not recommend using AR if the
REST is available"*); the zero-integral assumption being false on a non-spherical head
([Yao 2017]); current source density as a reference-free alternative; and EGI's own 256-channel
bar. The defensible framing is that average reference is an approximation with known,
named bias, chosen for interpretability and comparability — not that it is "standard".

---

## Sources added 2026-08-09

**[Keil 2014]** Keil, A., Debener, S., Gratton, G., Junghöfer, M., Kappenman, E.S., Luck, S.J.,
Luu, P., Miller, G.A., Yee, C.M. *Committee report: Publication guidelines and recommendations
for studies using electroencephalography and magnetoencephalography.* Psychophysiology
51(1):1–21, 2014. doi:10.1111/psyp.12147 — the governing guidelines document.

**[ERP CORE]** Kappenman, E.S. et al. *ERP CORE: An open resource for human event-related
potential research.* NeuroImage 225:117465, 2021. (DOI not verified.) Epoch −200/+800 and
baseline −200/0 for all five stimulus-locked components.

**[Luck 2022]** Luck, S.J. *Applied Event-Related Potential Data Analysis.* LibreTexts, 2022.
doi:10.18115/D5QG92 — open access, quotes verbatim-checkable.

**[Junghöfer 1999]** Junghöfer, M., Elbert, T., Tucker, D.M., Braun, C. *The polar average
reference effect: a bias in estimating the head surface integral in EEG recording.* Clinical
Neurophysiology 110(6):1149–1155, 1999. doi:10.1016/s1388-2457(99)00044-9

**[Debnath 2020]** Debnath, R., Buzzell, G.A., Morales, S., Bowers, M.E., Leach, S.C., Fox,
N.A. *The Maryland analysis of developmental EEG (MADE) pipeline.* Psychophysiology 57:e13580,
2020. doi:10.1111/psyp.13580 — validated on 128-ch EGI, Cz-referenced. Governs D9's ordering.

**[Bigdely-Shamlo 2015]** Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K.-M., Robbins, K.A.
*The PREP pipeline: Standardized preprocessing for large-scale EEG analysis.* Frontiers in
Neuroinformatics 9:16, 2015. doi:10.3389/fninf.2015.00016

**[Yao 2019]** Yao, D., Qin, Y., Hu, S., Dong, L., Bringas Vega, M.L., Valdés Sosa, P.A.
*Which reference should we use for EEG and ERP practice?* Brain Topography 32(4):530–549, 2019.
doi:10.1007/s10548-019-00707-x

**[Yao 2017]** Yao, D. *Is the surface potential integral of a dipole in a volume conductor
always zero? A cloud over the average reference of EEG and ERP.* Brain Topography
30(2):161–171, 2017. doi:10.1007/s10548-016-0543-x

**[Tanner 2015]** Tanner, D., Morgan-Short, K., Luck, S.J. *How inappropriate high-pass filters
can produce artifactual effects and incorrect conclusions in ERP studies of language and
cognition.* Psychophysiology 52(8):997–1009, 2015. doi:10.1111/psyp.12437

**[Maess 2016]** Maess, B., Schröger, E., Widmann, A. *High-pass filters and baseline
correction in M/EEG analysis. Commentary on: "How inappropriate high-pass filters can produce
artefacts…"* Journal of Neuroscience Methods 266:164–165, 2016.
doi:10.1016/j.jneumeth.2015.12.003

**[Delorme 2023]** Delorme, A. *EEG is better left alone.* Scientific Reports 13(1):2372, 2023.
doi:10.1038/s41598-023-27528-0

### Subsequent-memory precedent on this exact dataset family

**[Long 2014]** Long, N.M., Burke, J.F., Kahana, M.J. *Subsequent memory effect in intracranial
and scalp EEG.* NeuroImage 84:488–494, 2014. doi:10.1016/j.neuroimage.2013.08.052
(PMID 24012858, PMC3849113). 16 lists × 16 words — **matches our paradigm**. Analysis window
0–1600 ms. Bands: theta 3–8, alpha 10–14, beta 16–26, low gamma 28–42, high gamma 44–100 Hz.
The scalp experiment used a 129-sensor GSN and averaged six bilateral regions (anterior superior,
anterior inferior, posterior inferior) selected to parallel three intracranial cortical regions.
The article's figure does not publish numbered sensor membership and deliberately omits a
posterior-superior region, so it supports task relevance but not an exact eight-ROI channel map.

**[Weidemann 2021]** Weidemann, C.T., Kahana, M.J. *Neural Measures of Subsequent Memory
Reflect Endogenous Variability in Cognitive Function.* JEP:LMC 47(4):641–651, 2021.
doi:10.1037/xlm0000966. Window truncated to **300–1600 ms**, chosen before analysis.

**[Li 2024]** Li, Y., Pazdera, J.K., Kahana, M.J. *EEG decoders track memory dynamics.* Nature
Communications 15:2981, 2024. PMC10998865. ltpFR2, 88 subjects, ~22 sessions,
leave-one-session-out, L2 logistic regression on frequency × electrode.
**Encoding AUC 0.59 (SE 0.004) resampled; 0.60 (SE 0.004) all events.** Within each session,
the authors log-transform power and z-score across events at each electrode-frequency pair.
Project inference: because evaluation holds out sessions, this published feature construction
uses the unlabeled feature distribution of each held session. It is valid as a prespecified
batch-adapted target, but it is not strict inductive inference for a single unseen trial.

**[Broitman 2025]** Broitman, A.W., Healey, M.K., Kahana, M.J. *Neural biomarkers of
age-related memory change.* Psychology and Aging, advance online. doi:10.1037/pag0000876.
**ltpFR — our paradigm.** 188 subjects, 7 sessions, leave-one-session-out.
**Encoding AUC .54**, t(178) = 15, p < .0001. Reports **no gamma SME** (F < .1, p = .79) but a
significant **negative alpha SME**.

---

### ERP subsequent-memory reviews — effect latencies, not pipeline conventions

**Key methodological caveat, verified across five reviews:** none states a recommended
*preprocessing* epoch or baseline for word-encoding SME studies. Reviews report the **latency
ranges over which effects were observed**, not analysis-pipeline conventions. So "the standard
epoch for word-encoding SME is X" remains **[U]**; what is citable **[C]** is the effect
latency and the sustained characterisation.

**[Rugg 2002]** Rugg, M.D., Otten, L.J., Henson, R.N. *The neural basis of episodic memory:
evidence from functional neuroimaging.* Phil. Trans. R. Soc. Lond. B 357:1097–1110, 2002.
→ The citable "sustained" claim: *"the ERP subsequent memory effect for semantically studied
items took the form of a sustained increase in positivity for remembered words."* Windows used:
0–350 and 550–1000 ms.

**[Friedman 2000]** Friedman, D., Johnson, R. Jr. *Event-Related Potential (ERP) Studies of
Memory Encoding and Retrieval: A Selective Review.* Microscopy Research and Technique
51:6–28, 2000. → *"items that were subsequently recognized (Hits) elicited greater positivities
in the interval between 400–1,100 ms."* Measurement windows 500–800 and 810–1100 ms. Also
documents a *"long-lasting, positive-going slow wave (1,000–2,000 ms)."*

**[Paller 2002]** Paller, K.A., Wagner, A.D. *Observing the transformation of experience into
memory.* Trends in Cognitive Sciences 6(2), 2002, 93–102 (end page unverified).
→ *"Dm effects – enhanced posterior positivity at a latency of 400–800 ms for subsequently
remembered words."* **Negative finding:** this paper does **not** describe the Dm as sustained;
do not cite it for that.

**[Wagner 1999]** Wagner, A.D., Koutstaal, W., Schacter, D.L. *When encoding yields
remembering.* Phil. Trans. R. Soc. Lond. B 354:1307–1324, 1999. → SMEs are *"often associated
with the P300 but also extending beyond it."* Documents both early (175–225 ms) and late
(400–900 ms) components.

**[Friedman 2006]** Friedman, D. *Memory Encoding and Retrieval in the Aging Brain.* Clinical
EEG and Neuroscience 37(4), 2006 (pages unverified). → *"SMEs... onset at roughly 400 ms and
lasted about 600 ms."* Uses early (400–800 ms) and late (1200–1400 ms) windows.

**[Forester 2020]** Forester, G., Kroneisen, M., Erdfelder, E., Kamp, S.-M. *Survival
processing modulates the neurocognitive mechanisms of episodic encoding.* Cognitive, Affective,
& Behavioral Neuroscience 20(4):717–729, 2020. doi:10.3758/s13415-020-00798-1
→ **The closest published match to our design** — single visual words, free recall,
subsequently-recalled vs unrecalled. *"The EEG was segmented from 500 ms before word onset...
to 3,000 ms after word onset."* Analysis windows: P300 **500–700 ms** and frontal slow wave
**1,000–2,000 ms**. Direct precedent for a window extending well past 1600 ms on a free-recall
word task.

**[Kamp 2017]** Kamp, S.-M., Bader, R., Mecklinger, A. *ERP Subsequent Memory Effects Differ
between Inter-Item and Unitization Encoding Tasks.* Frontiers in Human Neuroscience 11:30,
2017. doi:10.3389/fnhum.2017.00030 → Epoch −400 to 2000 ms; windows 300–600 ms and 600–2000 ms.

**Consequence for us:** our −1000/+2000 ms epoch comfortably contains every window used in this
literature, from 400–800 ms through 1000–2000 ms. The *analysis* window is therefore a live,
citable choice rather than a constraint — and [Forester 2020] shows a free-recall word study
looking out to 3000 ms.

---

### Authoritative filesystem API sources — evidence-artifact publication

**[Apple rename(2)]** Apple, *Mac OS X Manual Page For rename(2)*.
https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/rename.2.html
Ordinary `rename` removes a compatible existing destination. This supports F20's diagnosis that
check-then-rename does not enforce immutable destination identity.

**[Apple exclusive rename]** Apple Developer Documentation,
*volumeSupportsExclusiveRenaming*.
https://developer.apple.com/documentation/foundation/urlresourcevalues/volumesupportsexclusiverenaming
The property reports whether the volume supports the `RENAME_EXCL` option of `renamex_np(2)`.
The installed macOS SDK additionally declares `renamex_np` and defines `RENAME_EXCL` as
`0x00000004` in `usr/include/sys/stdio.h`.

**[Linux renameat2]** Linux man-pages project, *rename(2): renameat2*.
https://man7.org/linux/man-pages/man2/renameat2.2.html
`RENAME_NOREPLACE` forbids overwriting `newpath`, returns `EEXIST` when it exists, and requires
underlying-filesystem support. This supports the Linux implementation and fail-closed policy.

These sources define technical publication semantics, not scientific evidence. They support the
atomic no-replace boundary reused by D50–D53; the measured project evidence is the synthetic F20
current-filesystem probe and the record-specific regression suites.

### Authoritative NumPy I/O sources — typed feature artifacts

**[NumPy save]** NumPy Developers, *numpy.save — NumPy Manual*.
https://numpy.org/doc/stable/reference/generated/numpy.save.html
Defines `numpy.save` as writing one array in `.npy` format and documents `allow_pickle`; disabling
pickle avoids its code-execution and cross-installation portability risks.

**[NumPy I/O guide]** NumPy Developers, *Reading and writing files — NumPy User Guide*.
https://numpy.org/doc/stable/user/how-to-io.html
Identifies `numpy.save`/`numpy.load` for `.npy` files, supports memory mapping for large arrays,
and recommends `allow_pickle=False` for security and portability when objects are unnecessary.

These sources support D50's feature-container, D51's prediction-container, and D52's same-trial
comparison-container choices, not a scientific or performance claim.

---

## Claims still unverified

| # | Claim | Where it bites | Status |
|---|---|---|---|
| U4 | Immediate free recall is the standard SME label | D5 | supported by all four Kahana-lab papers using it; no explicit methodological statement found |
| U5 | Group SME effects typically reported at 20–30 subjects | D3 | **refuted in spirit** — [Broitman 2025] used 188, [Li 2024] 88. Our 40 is modest for this literature |
| U6 | A blink is 100–200 µV; the SME is 1–2 µV | motivated D13, D14 | needs citation |
| U7 | Clean EEG epochs run 50–100 µV peak-to-peak | judging AutoReject output | needs citation |
| U8 | Scalp gamma is heavily muscle-contaminated | D12 caveat | needs citation |
| U9 | Primacy reflects rehearsal; recency an active buffer | interpreting the curve | textbook, needs source |
| U10 | Visual information reaches V1 in ~60–80 ms | justified the baseline window | needs citation |
| U11 | Recognition exceeds recall because it supplies the cue | D5 reasoning | textbook, needs source |
| U12 | ltpFR stimulus presentation duration | affects whether 1600 ms covers the full on-screen presentation | **partly resolved by R6**: onset spacing is measured at about 4 s and rules out next-trial overlap, but exact on-screen duration remains unverified |

---

## Process note: a fabricated source was caught

One agent's WebFetch against a raw PDF returned **entirely invented content** — a non-existent
paper, fabricated sample size, fabricated AUC. It was caught by re-extracting the PDF locally
and finding none of it present.

**Consequence for this project:** any claim sourced from a PDF summarisation must be verified
against locally extracted text before entering this file. This is the failure mode the
citation discipline exists to catch, and it occurred on the first serious attempt to use it.
