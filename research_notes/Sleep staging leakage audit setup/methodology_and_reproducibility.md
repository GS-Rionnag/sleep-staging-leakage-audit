# Methodology and Reproducibility Playbook for a Sleep-Staging Leakage Audit Repo

Scope: general ML leakage methodology, pre-registration, statistical reporting, reporting guidelines, reproducible-repo and CI practice. Sleep-specific literature and model choices are out of scope (another researcher covers them). Research date: 2026-11-24. Every finding has a URL. Items marked **[UNVERIFIED]** come from prior knowledge or were not fetched in this session, so check them before relying on them.

Session note: physionet.org returned HTTP 502 on two fetches during this research (https://physionet.org/content/sleep-edfx/1.0.0/ and https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt). This bears on CI design: the upstream data host can be briefly unavailable, so CI should cache the data and retry downloads.

---

## Q1. Leakage taxonomies (Kapoor & Narayanan 2023, REFORMS 2024, EEG-specific) and which types apply to epoch-wise splitting

### Takeaway
Kapoor & Narayanan's 8-type taxonomy fits this audit well. A random epoch-wise split is a textbook case of **L3.2 (non-independence between train and test samples)**. It is made worse by temporal autocorrelation between neighbouring 30-s epochs (related to **L3.1**), and it yields a test set not drawn from the distribution of interest, which is unseen people (**L3.3**). The design must keep L1.2 (preprocessing), L1.3 (feature selection) and test-set tuning clean in *both* arms, so the measured gap comes from the split rule alone. EEG-specific work (Brookshire et al. 2024; Del Pup et al. 2025) confirms that segment-based holdout is common and inflates estimates.

### Cited Findings
**Kapoor & Narayanan (Patterns 2023), taxonomy and model info sheets**
- The paper is "Leakage and the reproducibility crisis in machine-learning-based science", Patterns 2023. Full text: https://www.cell.com/patterns/pdfExtended/S2666-3899(23)00159-9. Preprint: https://arxiv.org/abs/2207.07048. Project page: [reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/)
- The arXiv abstract says: "we find 17 fields where errors have been found, collectively affecting 329 papers" and "a fine-grained taxonomy of 8 types of leakage that range from textbook errors to open research problems." — [arXiv 2207.07048](https://arxiv.org/abs/2207.07048)
- The project page's running tally is now "41 papers across 30 fields", which together affect 648 papers. The counts have grown since publication, so cite the version you use. — [Project page](https://reproducible.cs.princeton.edu/)
- The 8 types, from Section 2.4 of the preprint — [arXiv HTML 2207.07048v1](https://arxiv.org/html/2207.07048v1):
  - **L1.1 No test set**: "Using the same dataset for training as well as testing the model is a text-book example of overfitting"
  - **L1.2 Pre-processing on training and test set**: "Using the entire dataset for any pre-processing steps such as imputation or over/under sampling."
  - **L1.3 Feature selection on training and test set**: "Feature selection on the entire dataset results in using information about which feature performs well on the test set"
  - **L1.4 Duplicates in datasets**: "the same data could exist in the training as well as test set."
  - **L2 Model uses features that are not legitimate**: "If the model has access to features that should not be legitimately available for use in the modeling exercise"
  - **L3.1 Temporal leakage**: "the test set should not contain any data from a date before the training set."
  - **L3.2 Nonindependence between train and test samples**: "Nonindependence between train and test samples constitutes leakage". The paper adds: "In the extreme (but unfortunately common) case, train and test samples come from the same people or units", and gives the example of a histopathology study with "different observations of the same patient" in both sets.
  - **L3.3 Sampling bias in test distribution**: "Sampling bias in the choice of test dataset can lead to data leakage."
- **Model info sheets** are a template for documenting why a claim is free of leakage, inspired by Mitchell et al.'s model cards. A .docx template is on the project page — [Project page](https://reproducible.cs.princeton.edu/). The sheet asks for three arguments that map to the taxonomy — [arXiv HTML](https://arxiv.org/html/2207.07048v1):
  - (L1) clean train-test separation: the test set does not interact with training data at any stage
  - (L2) each feature is legitimate
  - (L3) the test set is drawn from the distribution of scientific interest, without selection or sampling bias
- The authors caution that model info sheets cannot prevent false claims, that checking them needs a computationally reproducible analysis, and that they do not cover reproducibility problems other than leakage. — [Project page](https://reproducible.cs.princeton.edu/)
- Civil-war case study: four prominent political-science papers claimed complex ML beat logistic regression. Leakage came from imputing train and test together (Muchlinski et al.), reusing an imputed dataset incorrectly (Colaresi & Mahmood; Wang), and target proxies (Kaufman et al.). After correction, "complex models such as AdaBoost and Random Forests don't perform substantively better than decades-old Logistic Regression." Reproduction code is a CodeOcean capsule: https://codeocean.com/capsule/6282482/tree/v1 — [Project page](https://reproducible.cs.princeton.edu/)

**REFORMS (Kapoor et al., Science Advances 2024)**
- REFORMS has 32 items in 8 sections: Study design; Computational reproducibility; Data quality; Data preprocessing; Modeling; Data leakage; Metrics and uncertainty quantification; Generalizability. It came from a consensus of 19 researchers. DOI 10.1126/sciadv.adk3452. The checklist (checklist.docx) and guidelines (guidelines.docx) are downloadable. — [reforms.cs.princeton.edu](https://reforms.cs.princeton.edu/); full text on PMC: [PMC11092361](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361)
- Leakage section: "We ask authors to justify that their study does not suffer from the main sources of leakage." — [REFORMS site](https://reforms.cs.princeton.edu/)
- One item asks authors to justify that preprocessing and modeling steps use only training data. Item 6c asks authors to justify that each feature or input is legitimate and does not cause leakage (search-result summary of the paper). — [PMC11092361](https://pmc.ncbi.nlm.nih.gov/articles/PMC11092361)
- Metrics and uncertainty: authors should "report the metrics and uncertainty estimates used in enough detail" for readers to judge them. — [REFORMS site](https://reforms.cs.princeton.edu/)
- On the Data preprocessing section, the REFORMS site cites Vandewiele et al.: oversampling before splitting caused widespread errors in pre-term birth detection. — [REFORMS site](https://reforms.cs.princeton.edu/)

**EEG and biosignal-specific evidence**
- Brookshire et al., "Data leakage in deep learning studies of translational EEG", Frontiers in Neuroscience, 2024:
  - Segment-based holdout puts data from individual subjects in both training and test sets. "All segments from a given subject must be assigned to only a single partition."
  - On Alzheimer's and epilepsy EEG datasets, performance on unseen subjects was "strongly overestimated" under segment-based holdout.
  - Most translational DNN-EEG studies they surveyed use segment-based holdout.
  - Sources: [PMC11099244](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244); [Frontiers](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1373515/full); [medRxiv preprint](https://medrxiv.org/content/10.1101/2024.01.16.24301366v1.full.pdf)
- Del Pup et al., "The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject analysis: a preliminary study" (arXiv 2505.13021, May 2025):
  - Compares five CV settings across three cross-subject EEG tasks and four architectures, training more than 100,000 models.
  - Concludes that subject-based CV matters, and that nested leave-N-subjects-out (N-LNSO) is more reliable than non-nested schemes. Non-nested schemes are "prone to data leakage and favor larger models overfitting to validation data."
  - Code: https://github.com/MedMaxLab/eegpartition
  - The abstract gives no single inflation figure.
  - Source: [arXiv 2505.13021](https://arxiv.org/abs/2505.13021)

**scikit-learn guidance relevant to epoch-wise splits**
- Group splitting: "An example would be when there is medical data collected from multiple patients, with multiple samples taken from each patient... the patient id for each sample will be its group identifier." GroupKFold "makes it possible to detect this kind of overfitting situations." — [sklearn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- Time series: "Time series data is characterized by the correlation between observations that are near in time (autocorrelation)... KFold and ShuffleSplit assume the samples are independent and identically distributed, and would result in unreasonable correlation between training and testing instances (yielding poor estimates of generalization error)." — [sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)
- Shuffling non-i.i.d. data "will likely lead to a model that is overfit and an inflated validation score: it will be tested on samples that are artificially similar (close in time) to training samples." — [sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)
- Preprocessing "should be learnt from a training set and applied to held-out data". Pipeline provides this under CV. — [sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)
- Tuning on the test set: "knowledge about the test set can 'leak' into the model and evaluation metrics no longer report on generalization performance." — [sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)
- Roberts et al. 2017 (Ecography 40(8):913–929, DOI 10.1111/ecog.02881) on blocked CV:
  - Dependence structures in the data persist into model residuals and make random CV over-optimistic.
  - Block CV was "nearly always more appropriate than random cross-validation" when the goal is predicting to new data, and they recommend blocking wherever dependence exists, even if residual correlation is not visible.
  - Caveat: blocking can restrict predictor ranges and overestimate error.
  - Sources: [PDF](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf); [Dryad data](https://datadryad.org/dataset/doi:10.5061/dryad.737gk)

### Inferences
- **Map each leakage type onto this audit:**
  - **L3.2 (primary target of the audit):** the leaky arm deliberately commits L3.2. Epochs from the same subject, and from the same night, end up in both train and test.
  - **L3.1 / L1.4-like near-duplicates:**
    - Adjacent 30-s epochs are strongly autocorrelated, so a random epoch split gives each test epoch near-identical neighbours in training. sklearn's warning about "artificially similar (close in time)" samples applies directly.
    - If any model takes a context window of neighbouring epochs as input, test inputs literally overlap training inputs. That is effectively a duplicate (L1.4), and the leaky gap for sequence models will likely be larger than for single-epoch models. This is worth stating as a secondary hypothesis.
  - **L3.3:** the scientific claim is about new people. An epoch-wise test set is drawn from already-seen people, so it is the wrong distribution.
  - **L1.2:** typical traps include global z-scoring, filtering parameters or class rebalancing/oversampling fit on all data, or label smoothing across recordings.
    - Per-recording normalization that uses only that recording's own signal is available at inference time and is acceptable in both arms.
    - A scaler fit on the pooled dataset before splitting is not acceptable.
    - Put all fitted transforms inside an sklearn Pipeline, or a fold-local equivalent for torch.
  - **L1.3:** any feature selection or channel selection must run inside the training fold.
  - **Test-set tuning:** hyperparameters must be tuned on an inner validation split that is itself grouped by subject (nested), following Del Pup et al.'s N-LNSO finding. Otherwise the "honest" arm also leaks.
  - **L2:** declare any feature that could proxy the label, such as epoch index or time-since-lights-off, subject/recording ID, or recording-level metadata. Either exclude it or justify it in the model info sheet.
- **Design principle:** change *only* the split rule between arms. Keep the models, hyperparameter search budget, preprocessing and seeds identical. The delta then isolates L3.2 + L3.1.
- **Optional intermediate arm:** a "record-wise but subject-mixed" split holds out nights, so the same subject's other night can sit in train. It shows a dose-response: epoch-wise > night-wise > subject-wise. This is cheap to add and very informative. Sleep-EDF Sleep Cassette has two nights for most subjects (see Gaps).
- **Repo hygiene:** fill out a Kapoor & Narayanan model info sheet as a markdown file in the repo, e.g. `docs/model_info_sheet.md`. Also add a REFORMS checklist crosswalk (`docs/reforms_checklist.md`) with Yes/No/NA and a pointer to code or README for each item. Both are cheap and credible.

### Gaps
- I could not retrieve the verbatim text of all 32 REFORMS items, in particular whether one item explicitly covers dependence between samples (the REFORMS site summary does not mention it). The checklist .docx on [reforms.cs.princeton.edu](https://reforms.cs.princeton.edu/) should be read directly.
- I found no dedicated, formally published "EEG leakage checklist". The closest are Brookshire et al. 2024 and Del Pup et al. 2025. Sleep-specific leakage literature was deliberately not researched.
- Del Pup et al.'s quantitative inflation numbers were not in the abstract. They would need the PDF or the repo.
- **[UNVERIFIED]** Several Sleep-EDF details could not be confirmed because the PhysioNet page returned 502:
  - that Sleep Cassette subjects mostly have two nights;
  - that the SC filename scheme `SC4ssN…` encodes subject `ss` and night `N`;
  - the ~78-subject count.

  These matter for choosing the grouping key (subject, not recording). The dataset-focused researcher should confirm them.

---

## Q2. Pre-registration for ML/computational studies (lightweight templates, what to state, examples)

### Takeaway
The best-fitting template is Hofman et al. 2023, "Pre-registration for Predictive Modeling". It is a two-phase form: Phase A before training/validation, Phase B before touching the test set. It maps almost one-to-one onto a README "pre-registration" section. Timestamp it with a git tag/commit, and optionally freeze it on OSF or with a Zenodo release before the final run. General OSF templates are social-science oriented. NeurIPS ran pre-registration workshops in 2020–2021, but no standard ML registry exists.

### Cited Findings
- Hofman, Chatzimparmpas, Sharma, Watts & Hullman (2023), "Pre-registration for Predictive Modeling", arXiv:2311.18807, introduces "a lightweight pre-registration template". — [arXiv abs](https://arxiv.org/abs/2311.18807)
- Template, Section 4 / Table 1 — [arXiv HTML](https://arxiv.org/html/2311.18807):
  - **Phase A (before training and validation):**
    - A.1 Research question
    - A.2 Dependent variable ("What is the main outcome of interest and how is it measured?")
    - A.3 Independent variable (features)
    - A.4 Training data ("How was the training data constructed?")
    - A.5 Test data ("How will the test dataset be constructed?")
    - A.6 Data transformations
    - A.7 Metrics
    - A.8 Baselines
    - A.9 Anything else
  - **Phase B (before testing):**
    - B.1 Prediction method(s)
    - B.2 Model training (cross-validation, random seeds, hyperparameters, computational budget)
    - B.3 "Have you in any way previously accessed the test data?"
    - B.4 Secondary/exploratory analyses
    - B.5 Plan adjustments ("Are there any changes in your process as compared to your previous answers that you'd like to report?")
- Hofman et al. on test-set use — [arXiv HTML](https://arxiv.org/html/2311.18807):
  - "The test set, however, is meant to be used once and only once."
  - Continuing to optimize after test evaluation ("over-hyping") is a form of leakage.
  - In interviews, half the participants described strategies for re-accessing the test set.
- Limitations the same paper states (Section 6.1) — [arXiv HTML](https://arxiv.org/html/2311.18807):
  - Pre-registration is "not a panacea".
  - It does not fix a poor proxy metric, an unrepresentative dataset or distribution shift.
  - It does not fix publication bias.
  - It "does not equate to eliminating all data-dependent decision-making."
- NeurIPS 2020 workshop "The pre-registration experiment: an alternative publication model for machine learning research" — [NeurIPS 2020](https://neurips.cc/virtual/2020/workshop/16158):
  - A pre-registered paper is submitted for peer review "without any experimental results".
  - The organizers cite an earlier small computer-vision trial (Henriques et al., 2019).
  - The 2021 follow-up ran a full pre-registration review cycle — [NeurIPS 2021](https://neurips.cc/virtual/2021/workshop/21885).
- The OSF "Preregistration Template" wiki is a general (social-science) form: research questions, hypotheses, timing of registration. It has no ML-specific items. — [OSF template](https://osf.io/x4gzt/wiki/Preregistration%20Template)
- Example of a pre-registered ML negative result (seen only as a search-result title, not read): "Pre-registered tests of solid-state-physics-inspired LLM compression: a cluster-level negative result at small-language-model scale". **[UNVERIFIED content]** — [arXiv 2609.34292](https://arxiv.org/pdf/2609.34292)
- AsPredicted (aspredicted.org) is a short 8–9-question pre-registration form that gives a frozen timestamped PDF. **[UNVERIFIED: not fetched this session]** — https://aspredicted.org/

### Inferences
- **Recommended README "Pre-registration" section** (written and committed *before* the final full run). Map it to Hofman A/B:
  1. **Hypotheses (A.1):**
     - H1: for every model, the subject-wise score is lower than the epoch-wise score.
     - H2: the gap is larger for models that use temporal context.
     - State the expected direction and, if willing, a rough magnitude.
  2. **Outcome and primary metric (A.2/A.7):**
     - Name ONE primary metric (e.g., macro-F1, or Cohen's kappa over 5 stages).
     - Name the aggregation unit: subject-level mean of per-subject metrics vs pooled-epoch metric. Pick one as primary and report the other as secondary.
     - List secondary metrics.
  3. **Split rules (A.4/A.5):**
     - Leaky arm: random epoch-wise K-fold with a fixed K.
     - Honest arm: GroupKFold or StratifiedGroupKFold over subject ID, with K fixed (or LOSO).
     - Grouping key = subject ID, not recording/night ID.
     - Inner validation also grouped by subject.
     - Exact seeds list (e.g., 5 seeds fixed in advance).
  4. **Data inclusion/exclusion rules (A.6):**
     - Which subset (e.g., SC only) and which channels.
     - How unscored/movement epochs and wake trimming are handled.
     - What happens to recordings that fail to parse. Decide in advance: exclude and report, never silently drop.
  5. **Models and budget (B.1/B.2):** the list of models and the hyperparameter grid/budget, identical across arms.
  6. **Analysis plan:**
     - Paired per-subject differences (honest minus leaky).
     - Subject-cluster bootstrap 95% CI with the number of resamples fixed.
     - Seeds averaged within subject (see Q3).
  7. **Success/failure criteria:**
     - Examples: "The audit claim 'epoch-wise splitting inflates performance' is supported if the 95% subject-bootstrap CI of the mean paired difference excludes 0 for ≥ X of N models."
     - "It is refuted/inconclusive if the CI includes 0 for all models."
     - Also define *pipeline* failure criteria: e.g., the subject-overlap test fails, CI smoke test fails, or a seed run differs by more than tolerance on rerun.
  8. **Stopping rule:** the final run is executed once, on tag `prereg-v1`. Any re-run after seeing results is logged under "Deviations" (Hofman B.5) with a reason.
  9. **Test-data access statement (B.3).**
  10. **Exploratory analyses (B.4):** labeled as exploratory.
- **Timestamp:** commit the section, then create an annotated git tag (e.g. `prereg-v1`) whose commit hash is quoted in the results section. The README itself should say "results below generated from commit X, which is a descendant of prereg tag Y".
  - Optionally upload the section to OSF or make a Zenodo GitHub release for a third-party timestamp. GitHub commit dates alone can be rewritten, so an external timestamp is stronger. This is general knowledge about git, not sourced here.
- **Honesty about the "pilot" phase:** the student will need small-subset runs to debug the pipeline. State in the pre-registration that development runs used only the CI subset (e.g., N subjects), and that full-data results were not viewed before registration. This mirrors Hofman B.3.

### Gaps
- I did not find a public GitHub README that functions as a pre-registration for an ML evaluation. No concrete repo example could be cited.
- I did not fetch AsPredicted's or OSF's current form lists, or any OSF "secondary data" pre-registration template. Check these if a formal registration is wanted.

---

## Q3. Statistical reporting for a leaky-vs-honest comparison (subject as unit, seeds, paired differences, CV-test cautions, small n)

### Takeaway
Treat the **subject** as the independent unit. Compute a per-subject metric under each protocol, average over seeds *within* subject, then take paired differences (honest minus leaky) per subject and model. Report a subject-level (cluster) bootstrap percentile CI of the mean difference, plus the full per-subject distribution. Do not treat folds or seeds as independent samples:
- Naive CV standard errors are too small (Bates, Hastie & Tibshirani; Varoquaux 2018).
- Resampled t-tests across folds need the Nadeau–Bengio correction.

With 20–78 subjects, expect wide intervals, around ±10% at n≈100 per Varoquaux. Frame results as estimates with uncertainty, not p-value verdicts.

### Cited Findings
- **Bates, Hastie & Tibshirani**, "Cross-validation: what does it estimate and how well does it do it?", JASA, DOI 10.1080/01621459.2023.2197686. Findings:
  - CV estimates the *average* error of models fit on other training sets from the same population, not the error of the specific fitted model.
  - Because each point is used for both training and testing, fold accuracies are correlated, so the naive variance estimate is too small and naive CIs are too narrow.
  - In a sparse logistic regression example, nominal 90% intervals had 31% miscoverage.
  - They propose **nested CV** to estimate variance; it gives roughly correct coverage in many examples. In one example, intervals needed widening by about 1.6×.
  - For simple data splitting, do not refit on the combined data if you want valid intervals.
  - Sources: [arXiv 2104.00673](https://arxiv.org/abs/2104.00673); [NSF PAR copy](https://par.nsf.gov/servlets/purl/10509473)
- **Varoquaux 2018**, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 180(A):68–77, DOI 10.1016/j.neuroimage.2017.06.061:
  - Typical neuroimaging sample sizes give CV error bars of about ±10% at 100 samples.
  - "The standard error across folds strongly underestimates them."
  - Sources: [arXiv 1706.07581](https://arxiv.org/pdf/1706.07581); [ar5iv](https://ar5iv.labs.arxiv.org/html/1706.07581)
- **Nadeau & Bengio 2003** (Machine Learning 52:239–281) proposed the corrected resampled t-test. The plain resampled t-test is too liberal (inflated Type I error) because it underestimates variance, as Dietterich (1998) found. The correction inflates the variance to account for overlap between training sets. — [correctR vignette](https://search.r-project.org/CRAN/packages/correctR/vignettes/correctR.html)
  - **[UNVERIFIED formula, from prior knowledge]** The commonly used corrected variance is (1/J + n_test/n_train)·σ̂² for J resamples.
- **Bouckaert & Frank 2004** (PAKDD) introduced *replicability*: a test has low replicability if its outcome depends strongly on the particular random partitioning. 5×2cv, resampling and 10-fold CV t-tests showed low replicability, and they favoured the corrected resampled t-test. — [Demšar 2006 JMLR survey](https://jmlr2020.csail.mit.edu/papers/volume7/demsar06a/demsar06a.pdf); [Bouckaert ICML 2004](https://icml.cc/Conferences/2004/proceedings/papers/61.pdf)
- A Python implementation of the corrected tests exists: `correctipy`. It currently supports two-model comparisons. — [correctipy GitHub](https://github.com/hendersontrent/correctipy)
- **Patient/subject-cluster bootstrap examples** (recent arXiv preprints, not peer reviewed; illustrate common practice, not an authoritative guideline):
  - A mammography paper resamples *patients* and keeps all their images, using 2,000 resamples and percentile CIs. For model comparisons it computes the metric difference within each resample. It notes that an image-level bootstrap breaks within-patient correlation and understates uncertainty. — [arXiv 2607.02394](https://arxiv.org/pdf/2607.02394)
  - A CT forecasting paper (38 trajectories, 26 patients) uses 5,000 patient-level replicates. It states that the number of independent test units is set by patients, not by repeated training seeds, and evaluates paired methods on the same resampled patients. — [arXiv 2609.21197](https://arxiv.org/pdf/2609.21197)
  - A COPD paper uses a paired, class-stratified bootstrap with 20,000 resamples and reports the chance baseline alongside intervals. — [arXiv 2608.21599](https://arxiv.org/pdf/2608.21599)
- Bootstrapping the CV estimate itself is possible, but validity depends on stability conditions and large-sample approximations. — [arXiv 2307.00260](https://arxiv.org/html/2307.00260v1)
- Nested leave-N-subjects-out gave more reliable estimates than non-nested schemes in EEG. — [arXiv 2505.13021](https://arxiv.org/abs/2505.13021)

### Inferences
**Recommended analysis (scaled for a solo student):**
1. **Per-subject scoring in both arms.**
   - Under the honest (subject-wise) arm, each subject is in exactly one test fold. Compute the metric over that subject's epochs.
   - Under the leaky (epoch-wise) arm, each epoch is tested exactly once across K folds. Collect the out-of-fold predictions and compute the same metric per subject over that subject's epochs.
   - Each subject then has two numbers per model per seed. This is what enables a **paired** design.
2. **Seeds:**
   - Seeds vary both model initialization and fold assignment. Fix e.g. 3–5 seeds in the pre-registration.
   - Average the metric over seeds *within* each subject before bootstrapping. Seeds are not independent observations of generalization, as arXiv 2609.21197 states.
   - Separately report the seed-to-seed spread, e.g. the range of the subject-mean across seeds, as a stability/replicability indicator in the spirit of Bouckaert & Frank.
3. **Primary estimate:** the mean over subjects of d_s = metric_honest(s) − metric_leaky(s), per model, with a **subject-cluster bootstrap** 95% percentile CI. Resample subjects with replacement (e.g. 10,000 times) and recompute the mean d. Use the same resampled subjects for both arms, so the comparison stays paired.
4. **Secondary:**
   - Report the CI of each arm's mean separately.
   - Report the fraction of subjects with d_s < 0.
   - Optionally run a Wilcoxon signed-rank test on d_s across subjects. Subjects, not folds, are the independent units, so a standard paired test is defensible here.
   - Do **not** run t-tests across folds. If you must compare *models* using fold scores, use the Nadeau–Bengio corrected resampled t-test (correctipy).
5. **Caveat to state in the limitations:**
   - The bootstrap treats the trained models as fixed and resamples test subjects only. It ignores training-set variability, which Bates et al. show matters.
   - The intervals therefore describe uncertainty over *which subjects were tested*, not full generalization uncertainty.
   - Nested CV (Bates et al.) is the principled fix. Mention it as out of scope if compute is too limited.
6. **Small n:**
   - With about 20 subjects, use LOSO or 5-fold GroupKFold, and expect intervals of several points. Varoquaux's ±10% at n=100 suggests even wider at n=20.
   - With 78 subjects, 5- or 10-fold StratifiedGroupKFold is reasonable.
   - Always plot every subject's point. Avoid "significance" language and report effect sizes with CIs.
7. **Pooled vs per-subject metrics:** pooled-epoch macro-F1/kappa and the mean of per-subject metrics can differ. Pre-register which one is primary, and report both.

### Gaps
- I found no authoritative methodological paper specifically prescribing the subject-cluster bootstrap for ML evaluation. The cited examples are 2026 arXiv preprints, and the practice should be justified as standard cluster-bootstrap logic.
- I did not verify the exact Nadeau–Bengio correction formula from the original paper (it is marked unverified above).
- The full text of Bates et al. was not read. The 31% miscoverage and 1.6× widening figures come from search-result summaries of the NSF PAR copy.

---

## Q4. Reporting guidelines (TRIPOD+AI, CLAIM, model cards, datasheets): what fits a non-clinical audit

### Takeaway
The best fits are ML-science guidelines: **REFORMS** (32 items) and a **Kapoor–Narayanan model info sheet**. **TRIPOD+AI** (27 items) is for clinical prediction-model studies; use it selectively as a crosswalk for data source, participants, sample size, performance with uncertainty, and code/data availability. **CLAIM 2024** is imaging-specific and mostly not applicable. **Datasheets for datasets** is useful for documenting which Sleep-EDF subset and labels are used. **Model cards** are optional; most items concern deployed models.

### Cited Findings
- **TRIPOD+AI:**
  - Collins et al., BMJ 2024;385:e078378 (16 Apr 2024), DOI 10.1136/bmj-2023-078378.
  - It is a 27-item checklist plus an expanded checklist and a 13-item TRIPOD+AI for Abstracts.
  - It applies "irrespective of whether regression modelling or machine learning methods have been used", and supersedes TRIPOD 2015.
  - Open access, CC BY 4.0.
  - Sources: [PMC11025451](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/); [Oxford ORA](https://ora.ox.ac.uk/objects/uuid:414e5fb8-5575-4b67-b94d-a0833a7cc58a)
- **CLAIM 2024 Update:**
  - Tejani et al., Radiology: AI, May 2024, DOI 10.1148/ryai.240300.
  - 44 items (up from 42 in 2020), each answered Yes/No/NA.
  - Prefers "external testing" over "external validation" and the term "reference standard".
  - Explicitly not a scoring system, and not extended to imaging biomarkers.
  - Sources: [PMC11304031](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031); [EQUATOR](https://www.equator-network.org/reporting-guidelines/checklist-for-artificial-intelligence-in-medical-imaging-claim-a-guide-for-authors-and-reviewers/)
- **REFORMS** is designed to apply across fields that use ML methods. — [REFORMS](https://reforms.cs.princeton.edu/)
- **Model info sheets** are inspired by Mitchell et al.'s model cards and focus on justifying the absence of leakage. — [reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/)
- **Model cards:** Mitchell et al., "Model Cards for Model Reporting", FAT* 2019. — [arXiv 1810.03993](https://arxiv.org/abs/1810.03993) **[UNVERIFIED: URL and content from prior knowledge, not fetched this session]**
- **Datasheets for datasets:** Gebru et al., "Datasheets for Datasets" (CACM 2021). — [arXiv 1803.09010](https://arxiv.org/abs/1803.09010) **[UNVERIFIED: not fetched this session]**

### Inferences
- **For a non-clinical audit, adopt:**
  - (1) the REFORMS checklist (as a Markdown table in `docs/`);
  - (2) a model info sheet focused on L1–L3;
  - (3) a short "data card" based on datasheets questions: source and version (sleep-edfx 1.0.0), license, collection context, subset used, label scheme, known biases such as age range and population, and preprocessing.
- **TRIPOD+AI crosswalk:** borrow its framing on participants/data sources, sample size justification, handling of missing data, performance measures *with uncertainty*, and open science. Explicitly state "not a clinical prediction model; no clinical claims" so readers don't expect calibration, decision-curve analysis or intended-use sections.
- **CLAIM:** borrow only the terminology. Say "held-out subjects" or "internal testing", avoid "external validation" (there is no external dataset), and call the scorer hypnograms the "reference standard".
- **Model cards:** a single combined `MODEL_CARD.md` section per model family is enough. Most items (intended use, deployment, ethical considerations) are lightweight for an audit, but "evaluation data" and "quantitative analyses" (disaggregated per-subject results) are relevant.

### Gaps
- The exact TRIPOD+AI item wording was not retrieved. The crosswalk above is based on the general structure, so the student should map specific item numbers from the BMJ article.
- I did not fetch the model cards or datasheets papers.

---

## Q5. Reproducible Python repo practices (late 2026)

### Takeaway
Use **uv** with a committed, universal `uv.lock`. Provide a single entry point (`uv run python -m sleepaudit run --config configs/full.yaml`, optionally wrapped by `just`/`make`) and YAML configs for "full" vs "ci-subset". Add:
- a download script that verifies **SHA256** against PhysioNet's published `SHA256SUMS.txt` and caches files locally;
- centralized seeding;
- pytest tests, including an assertion that no subject appears in both the train and test indices of any honest fold;
- results written as CSV/JSON plus figures, with run metadata (commit hash, lockfile hash, seeds, package versions).

### Cited Findings
- **uv lockfile:**
  - "uv's lockfile (`uv.lock`) is created with a universal resolution and is portable across platforms."
  - "By default, the universal resolver attempts to solve for all platforms and Python versions". `environments` can narrow this, and `required-environments` can force wheels for specific platforms.
  - Source: [uv resolution docs](https://docs.astral.sh/uv/concepts/resolution/)
- **`--locked` vs `--frozen`:**
  - `--locked`: "If the lockfile is not up-to-date, uv will raise an error instead of updating the lockfile."
  - `--frozen` uses the lockfile without checking it.
  - `uv run` locks and syncs automatically.
  - `uv export --format requirements.txt` (also `pylock.toml`, CycloneDX) exports the lock.
  - Source: [uv sync docs](https://docs.astral.sh/uv/concepts/projects/sync/)
- **Grouped splitting in scikit-learn:**
  - GroupKFold ensures "the same group is not represented in both testing and training sets".
  - StratifiedGroupKFold preserves class distribution "while keeping each group within a single split", though "it might produce imbalanced splits even if perfect stratification is possible".
  - With `shuffle=True`, GroupKFold "attempts to place an equal number of distinct groups in each fold (but does not account for group sizes)".
  - Source: [sklearn CV guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- **Randomness in scikit-learn:**
  - The `random_state` default `None` means "the shuffling will be different every time `KFold(..., shuffle=True)` is iterated".
  - "To get identical results for each split, set `random_state` to an integer."
  - "By default no shuffling occurs, including for the (stratified) K fold cross-validation performed by specifying `cv=some_integer`... Keep in mind that train_test_split still returns a random split."
  - Source: [sklearn CV guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- **Sleep-EDF Expanded download:**
  - The PhysioNet page lists `wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/` plus Python-package and AWS options.
  - The 1.0.0 file index lists `SHA256SUMS.txt` (38.9 KB, 2019-02-20).
  - The 153 SC* files come from a 1987–1991 study of age effects on sleep in healthy people aged 25–101, and there are 44 ST files. 197 whole-night recordings in total.
  - These details are from search-result summaries, because the page itself returned HTTP 502 during this session.
  - Sources: [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/); [archive index](https://archive.physionet.org/pn6/sleep-edfx/)
- **PyTorch reproducibility:**
  - The PyTorch randomness note (now at https://docs.pytorch.org/docs/stable/notes/randomness.html, redirected from pytorch.org) could not be fetched in readable form this session.
  - **[UNVERIFIED, from prior knowledge]** Standard steps:
    - `torch.manual_seed`, `random.seed`, `np.random.seed`;
    - `torch.use_deterministic_algorithms(True)`;
    - `torch.backends.cudnn.benchmark = False`;
    - `CUBLAS_WORKSPACE_CONFIG=:4096:8` on CUDA;
    - DataLoader `worker_init_fn` + `generator`.
  - The page also warns that completely reproducible results are not guaranteed across PyTorch releases, platforms, or CPU vs GPU.

### Inferences
- **Layout (suggested):**
  - `pyproject.toml` + `uv.lock` (committed) + `.python-version`.
  - `src/sleepaudit/` with modules:
    - `data/download.py` (manifest + SHA256 verify + retry)
    - `data/epochs.py`
    - `splits.py` (the ONLY place splits are made; returns fold indices + group IDs)
    - `models/`
    - `evaluate.py` (per-subject metrics)
    - `stats.py` (bootstrap)
    - `report.py` (tables/figures)
    - `__main__.py`
  - `configs/full.yaml`, `configs/ci.yaml`
  - `tests/`
  - `results/` (committed final CSV/figures from the pre-registered run)
  - `docs/` (prereg, model info sheet, REFORMS crosswalk, data card)
- **Single command:** `uv run python -m sleepaudit all --config configs/full.yaml`. The `all` command runs download → verify → preprocess (cached) → train/eval both arms × models × seeds → stats → report.
  - A `justfile` or `Makefile` can alias it. `just` works natively on Windows, while `make` usually doesn't without extra tooling (general knowledge, unverified).
- **Data download:**
  - Download only the files listed in a manifest (subset for CI, full SC for the final run).
  - Verify each file against PhysioNet's `SHA256SUMS.txt`, and commit a copy of the relevant lines into the repo so verification does not depend on PhysioNet being up.
  - Use resumable downloads with retries and backoff, given the 502s seen during this research.
  - Store data under a configurable `DATA_DIR`, defaulting outside the repo, and git-ignore it.
- **Caching:** cache preprocessed epoch arrays (e.g. `.npz`/parquet), keyed by a hash of (raw file SHA256 + preprocessing config + code version). This makes reruns cheap and the cache invalidates when preprocessing changes.
- **Seeds:** use one `set_seed(seed)` function for `random`, `numpy` (prefer passing `np.random.default_rng(seed)` objects explicitly) and torch. Pass `random_state=seed` explicitly to every sklearn splitter and estimator. Record the seeds in the output metadata.
- **Run metadata:** each run writes `results/<run_id>/meta.json` with git commit, dirty flag, `uv.lock` SHA256, Python/OS, package versions, config, seeds, the data manifest hash, and timestamps.
- **Tests (pytest):**
  - (a) **No subject overlap:** for every honest fold, including the inner validation folds, `set(groups[train]) & set(groups[test]) == set()`.
  - (b) **Leaky-arm sanity check:** it *does* have overlap, which shows the leaky arm is actually leaky.
  - (c) Every epoch is tested exactly once per seed in each arm.
  - (d) The grouping key is the subject ID parsed from filenames, not the recording ID. Use a fixture with two nights from the same subject.
  - (e) Fitted preprocessing sees only training indices, e.g. a spy transformer that records the indices it was fit on.
  - (f) Determinism: the same seed gives identical fold indices, and metrics match within tolerance.
  - (g) Checksum verification fails on a corrupted file.
  - (h) The stats module bootstrap returns a CI that contains the point estimate, and handles n=2.
  - Use tiny synthetic EDF-like arrays for unit tests, so most tests need no download.
- **Torch determinism on CPU-only CI** is easier than on GPU. Assert tolerance-based equality, not bitwise equality, across OSes.

### Gaps
- I could not confirm the current contents and format of `SHA256SUMS.txt` (the fetch returned 502). The "hash filename" line format is assumed from typical PhysioNet practice. **[UNVERIFIED]**
- Whether MNE's built-in `mne.datasets.sleep_physionet` fetcher verifies hashes, and which subjects it covers, was not checked. **[UNVERIFIED]**
- conda-lock and pip-tools were not researched in depth. uv appears sufficient for a solo project.

---

## Q6. GitHub Actions specifics (limits, caching EDF files, setup-uv, smoke test, artifacts, Windows vs Linux)

### Takeaway
For a public repo, standard GitHub-hosted runners are free. `ubuntu-latest` has 4 vCPU / 16 GB RAM / 14 GB SSD. Jobs are capped at 6 h. `actions/cache` allows 10 GB per repo, with entries evicted after 7 days without access, oldest-access first. A small subset of a few EDF recordings (tens of MB each) fits easily. Use `astral-sh/setup-uv` with `enable-cache: true` and `uv sync --locked`, cache the EDF subset keyed on the manifest hash, and upload results as artifacts. A weekly scheduled run keeps the data cache warm.

### Cited Findings
- **Job and workflow limits:**
  - Job execution time on GitHub-hosted runners: 6 hours per job, after which the job is terminated and fails.
  - Workflow run time: 35 days.
  - Free plan concurrency: 20 jobs (5 macOS).
  - Cache: 10 GB per repository.
  - Artifact storage: 500 MB for GitHub Free and 1 GB for Pro, shared with Packages.
  - Source: [GitHub Actions limits](https://docs.github.com/en/actions/reference/limits)
- **Free usage:** "Standard GitHub-hosted runners in public repositories" are listed as free usage that "does not reduce the included compute usage". — [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
- **Public-repo runner specs:**

  | Label | CPU | RAM | SSD | Arch |
  |---|---|---|---|---|
  | `ubuntu-latest` | 4 | 16 GB | 14 GB | x64 |
  | `windows-latest` | 4 | 16 GB | 14 GB | x64 |
  | `macos-latest` | 3 (M1) | 7 GB | 14 GB | arm64 |

  The page links `ubuntu-latest` to the Ubuntu 24.04 image, and also lists ubuntu-26.04 and 22.04 labels. "The `-latest` runner images are the latest stable images that GitHub provides". — [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- **Cache eviction and scope:**
  - Caches not accessed in "over 7 days" are removed.
  - At the size limit, GitHub deletes "caches in order of last access date, from oldest to most recent".
  - Frequent eviction can cause thrashing.
  - User-owned repos can raise the limit, but storage above 10 GB is billed.
  - Runs can restore caches from the current branch or the default branch, and PRs can also restore from the base branch. Sibling-branch caches are inaccessible.
  - Source: [Dependency caching reference](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
- **uv in GitHub Actions** — [uv GitHub integration guide](https://docs.astral.sh/uv/guides/integration/github/):
  - Pin `astral-sh/setup-uv` to a commit SHA (example: `@c771a70e6277c0a99b617c7a806ffedaca235ff9 # v9.0.0`), and pin the uv `version`.
  - Use `enable-cache: true`.
  - Run `uv sync --locked --all-extras --dev` then `uv run pytest tests`.
  - Prune with `uv cache prune --ci`.
  - `setup-uv`'s `python-version` input, or `UV_PYTHON`, overrides the pinned Python in a matrix. `actions/setup-python` with `python-version-file` also works.

### Inferences
- **Workflow sketch:**
  - Triggers: `on: [push, pull_request, workflow_dispatch]`, plus a `schedule:` cron once a week to keep the data cache from hitting the 7-day eviction.
  - Two jobs:
    - (1) `test`: unit tests on `ubuntu-latest`. Optionally add a `windows-latest` matrix leg for unit tests only, to catch Windows/Linux differences cheaply.
    - (2) `smoke`: download or restore a small subset (e.g. 3–6 subjects, 2 nights each where available, so the subject-vs-night grouping logic is exercised), then run `uv run python -m sleepaudit all --config configs/ci.yaml` with 1–2 seeds, 2-fold splits and tiny models/epochs.
  - Set `timeout-minutes: 30` so a hang can't burn the 6 h limit.
- **Data cache step:** `actions/cache` with `path: ~/sleepedf_subset` and `key: sleepedf-${{ hashFiles('configs/ci_manifest.txt') }}`.
  - On a miss, run the download script, which verifies SHA256 against the committed checksums.
  - Cache creation should happen on a default-branch run (push to main or schedule), so PR runs can restore it. Per the branch-scoping rule, a cache created only on a PR branch isn't shared with other branches.
- **Disk:** the 14 GB SSD rules out the full ~8 GB dataset plus derived arrays plus environment in CI. Keep CI to the subset, and run the full analysis locally.
- **Artifacts:**
  - Upload `results/<run_id>/` (CSV, JSON metadata, PNG/SVG figures) with `actions/upload-artifact` and a short `retention-days` (e.g. 14) to stay within storage.
  - Never upload EDF data as artifacts.
  - Whether the 500 MB artifact quota applies to public repos is unclear from the limits page; keep artifacts small regardless.
- **Optional CI checks:**
  - A "results-consistency" check that the CI subset's leaky score > honest score. This is a weak but useful regression signal; mark it `continue-on-error` or use a tolerance, because tiny subsets are noisy.
  - Lint with `ruff`.
  - A check that `uv.lock` is up to date (`uv lock --check` or `uv sync --locked`, which already errors if it is stale).
- **Windows-dev vs Linux-CI pitfalls (general engineering knowledge, not sourced this session — [UNVERIFIED]):**
  - **Paths:** use `pathlib` everywhere and never hard-code `\`.
  - **Filename case:** Linux is case-sensitive and Windows is not. Test filename parsing of e.g. `SC4001E0-PSG.edf` / `SC4001EC-Hypnogram.edf` on Linux.
  - **Line endings:** add a `.gitattributes`, e.g. `* text=auto` and `*.edf binary`, so CRLF conversion doesn't alter checksum or manifest files or committed CSVs.
  - **Multiprocessing:** Windows uses `spawn`, so all parallel code needs an `if __name__ == "__main__":` guard and picklable functions. Linux's default start method differs and changed in recent Python versions.
  - **BLAS and floats:** BLAS/thread nondeterminism and OS-specific float differences mean results should be compared with tolerances. Set `OMP_NUM_THREADS`/`MKL_NUM_THREADS` if exact reproducibility matters.
  - **Shell:** shell scripts and `make` aren't native on Windows, so the Python module entry point is the portable single command.
  - **Caches:** the uv cache and data directory default to different locations per OS, so make `DATA_DIR` configurable via env var.

### Gaps
- Artifact retention defaults (commonly 90 days for public repos) and whether the 500 MB artifact storage quota applies to public repositories were not confirmed. **[UNVERIFIED]**
- The current major version of `actions/upload-artifact` / `actions/cache` (v4 at my knowledge cutoff) was not verified for Nov 2026. **[UNVERIFIED]**
- The `setup-uv` v9.0.0 SHA and uv version `0.13.9` are as shown in the uv docs. Check for newer releases at implementation time.

---

## Q7. Presenting results honestly (figures, negative-result framing, limitations, example audit repos)

### Takeaway
Show both protocols side by side for each model as **paired per-subject points connected by lines** (slopegraph / paired dot plot), with a bootstrap CI for the mean difference, and a table listing every subject. Frame the subject-wise number as the headline and the epoch-wise number as the "optimistic artifact". Report null or small gaps as results, not failures. Write the limitations to cover:
- what the CIs cover (test-subject sampling only);
- dataset representativeness;
- no external dataset;
- seeds/compute budget;
- deviations from the pre-registration.

### Cited Findings
- Kapoor & Narayanan's civil-war reproduction is the canonical "audit" example. Corrected models showed no substantive advantage of complex ML over logistic regression, and code is on CodeOcean: https://codeocean.com/capsule/6282482/tree/v1 — [reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/)
- EEG partitioning audit with public code: Del Pup et al. trained over 100,000 models across 5 CV settings. Code: https://github.com/MedMaxLab/eegpartition — [arXiv 2505.13021](https://arxiv.org/abs/2505.13021)
- Brookshire et al. 2024 demonstrated segment-based vs subject-based holdout on two EEG datasets, and paired it with a literature survey showing most studies use segment-based holdout. — [PMC11099244](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244)
- The NeurIPS pre-registration workshop motivates publishing negative results, which are "often omitted". — [NeurIPS 2020 workshop](https://neurips.cc/virtual/2020/workshop/16158)
- Varoquaux recommends data sharing/pooling and evaluating on multiple studies as remedies for small-sample error bars, which supports listing "single dataset" as a limitation. — [arXiv 1706.07581](https://arxiv.org/pdf/1706.07581)
- Hofman et al. list what pre-registration cannot fix (poor proxy metrics, unrepresentative data, distribution shift), which is a ready-made limitations scaffold. — [arXiv HTML 2311.18807](https://arxiv.org/html/2311.18807)
- Bates et al. show that CV estimates the average performance over training sets, not the deployed model's error. That is a key interpretive caveat for the results text. — [arXiv 2104.00673](https://arxiv.org/abs/2104.00673)
- The COPD-screening example reports the chance-level baseline (class prevalence) alongside intervals. This is a good practice for imbalanced sleep stages. — [arXiv 2608.21599](https://arxiv.org/pdf/2608.21599)

### Inferences
- **Figure 1 (headline):**
  - One panel per model: x-axis = {epoch-wise (leaky), subject-wise (honest)}, y-axis = primary metric.
  - One faint line per subject connecting its two scores, so a "slope" means inflation.
  - Overlay the bold mean, with the subject-bootstrap 95% CI as error bars.
  - Add a horizontal dashed line for a trivial baseline (majority class, or a per-stage-prior classifier).
- **Figure 2:** forest plot of the mean paired difference (honest − leaky) with 95% CI for each model. Optionally add an intermediate "night-wise" arm. A zero line makes null results visible.
- **Figure 3 (optional):** per-subject honest scores sorted, to show subject heterogeneity (which subjects are hard). Per-class F1 under both arms shows *where* leakage inflates.
- **Table:** rows = models; columns:
  - leaky mean [95% CI];
  - honest mean [95% CI];
  - difference [95% CI];
  - % subjects with lower honest score;
  - seed range.

  Put the full per-subject × seed × arm CSV in `results/`.
- **Negative-result framing:**
  - Pre-commit to the interpretation (Q2 criteria).
  - If the gap is small for some models, report it plainly: "for model X, the epoch-wise split did not materially inflate kappa (Δ = … [CI])".
  - Discuss why that might be, e.g. models without temporal context and strong per-recording normalization.
  - Avoid "fails to beat" language and "state-of-the-art" comparisons. The audit's object is the evaluation protocol, not the models.
- **Limitations section (checklist):**
  1. Single dataset and population. Healthy adults for SC (per the dataset description), so no external testing.
  2. CIs reflect test-subject sampling, not training-set variability. No nested-CV variance estimate (Bates et al.).
  3. Limited seeds and hyperparameter budget, identical across arms by design. Larger budgets might change the gap.
  4. Only a subset of models and channels.
  5. Deviations from the pre-registration, with reasons.
  6. Development runs on a CI subset may have informed engineering choices.
  7. Reference-standard (scorer) noise caps achievable agreement.
  8. Results are not clinical performance claims.
- **Transparency extras:**
  - Link the exact commit/tag for the reported run.
  - Include `meta.json`.
  - Provide a "how to reproduce this table" command.
  - Include the model info sheet and the REFORMS crosswalk.

### Gaps
- I did not find a specific published "leakage-audit figure" design guideline. The figure designs above are recommendations derived from the paired, per-subject analysis design, not from a cited standard.
- I did not inspect the eegpartition repo or the CodeOcean capsule to describe their structure or figures. The report writer should present them only as "public code exists".
