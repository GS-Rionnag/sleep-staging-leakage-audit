# REFORMS checklist

REFORMS (Kapoor et al., 2024, *Science Advances*; <https://reforms.cs.princeton.edu/>) has 32 items in 8 modules. The item wording below is paraphrased; the numbering follows the published checklist.

"Prereg" means [`preregistration.md`](preregistration.md). The other documents are [`data_card.md`](data_card.md), [`model_info_sheet.md`](model_info_sheet.md) and the [README](../README.md).

The results-dependent items (2c, 7a–7c, 8a, 8b) are addressed in design here and completed with the full runs (plan Task 13).

## Module 1: Study goals

| # | Item | Status |
|---|---|---|
| 1a | Population or distribution the claim is about | Addressed: prereg A1, A3; model info sheet §3. Healthy adults in the Sleep-EDF SC cohort, scored per held-out subject. |
| 1b | Why that population | Addressed: prereg A3; data card "What is used". It is the most-cited open sleep-staging benchmark, and SC-78 matches the one controlled measurement of the gap (Köksal 2026). |
| 1c | Why ML methods are used | Addressed: README "Why this matters"; prereg A1. The audit measures how ML evaluation choices inflate scores, so ML models are the object of study. |

## Module 2: Computational reproducibility

| # | Item | Status |
|---|---|---|
| 2a | Dataset description with a link or DOI | Addressed: data card "Source"; committed manifest `data/manifests/sc_files.csv` with SHA256 per file. |
| 2b | Code with a link or version identifier | Addressed: this repository. The protocol is frozen at tag `prereg-v1`, and each run's `meta.json` records the commit and a dirty flag. |
| 2c | Computing infrastructure | Addressed: prereg B1. `meta.json` records host, platform, CPU count, GPU and every package version; `uv.lock` is hashed. |
| 2d | README with instructions to reproduce | Addressed: README "Quickstart". |
| 2e | One reproduction script for all results | Addressed: `uv run python -m sleepaudit all --config configs/sc78.yaml` (and `sc20.yaml`) downloads, verifies, builds features, evaluates and writes every table and figure. CI runs the same command on a 4-subject fixture. |

## Module 3: Data quality

| # | Item | Status |
|---|---|---|
| 3a | Data sources, collection time, ground-truth process and documentation | Addressed: data card "Source", "What is used", "Labels". SC study 1987–1991; one R&K scorer per night. |
| 3b | Sampling frame | Addressed: data card "What is used". Healthy volunteers aged 25–101 recorded at home, with no sleep-related medication. |
| 3c | Why the dataset suits the task | Addressed: prereg A3; data card. It is the standard benchmark, every subject has two nights (which the `night` arm needs), and labels are public. |
| 3d | Outcome variable, definition and descriptive statistics by class | Addressed: data card class-count table; prereg A2, A4. |
| 3e | Sample size and outcome frequencies | Addressed: data card table (78 / 20 subjects; 195,479 / 42,308 epochs; counts per class). |
| 3f | Missing data, by class | Addressed: data card "Labels". Unscored and movement epochs are dropped (302 inside the trimmed `sc78` range); no feature value is missing for a kept epoch. |
| 3g | Why the sampling frame represents the claimed population | Addressed: model info sheet §3. The claim is restricted to the SC cohort; age gaps and healthy-only sampling are listed as limits. |

## Module 4: Data preprocessing

| # | Item | Status |
|---|---|---|
| 4a | Excluded samples and why | Addressed: prereg A4; data card. ST is excluded (different population and hardware); unscored and movement epochs are dropped; wake beyond ±30 min of sleep is trimmed (field convention). |
| 4b | Impossible or corrupt samples | Addressed: checksum-verified downloads (`test_checksum_mismatch_leaves_no_file`); labels past the signal end are clipped; a recording with no sleep epochs raises `ValueError` (`test_trim_all_wake_raises`); unknown annotation strings raise. |
| 4c | All transformations from raw data to model input | Addressed: prereg A4–A6 (label map → grid → clip → drop → trim → per-recording features → in-fold scaler). |

## Module 5: Modelling

| # | Item | Status |
|---|---|---|
| 5a | Every model trained | Addressed: prereg A6 table. |
| 5b | Why these model types | Addressed: prereg A6, A8; spec. A floor, a single-epoch model, a context model and a smoothed-feature model, to test whether feature design changes the leak (H3), plus a never-trained control. |
| 5c | Evaluation method and splits | Addressed: prereg A7 (five arms, K, seeds), B1 (out-of-fold). |
| 5d | How the reported model is chosen | Addressed: no model selection. Every model × arm is reported. |
| 5e | Hyperparameter tuning | Addressed: none. All hyperparameters are fixed in advance (prereg A6). |
| 5f | Appropriate baselines | Addressed: prereg A8. Majority class (κ = 0), zero-shot pretrained YASA, and literature reference points. |

## Module 6: Data leakage

| # | Item | Status |
|---|---|---|
| 6a | Pre-processing and modelling use only training data | Addressed: model info sheet §1 (L1.2, L1.3). The spy-scaler test confirms it; the `subject_prefit` arm measures the leak it guards against. |
| 6b | Dependencies or duplicates between train and test | Addressed: model info sheet §1 (L3.2). Subject-grouped folds; a run-time check before training; tests that use two nights per subject and one-night subjects. |
| 6c | Each feature is legitimate | Addressed: model info sheet §2, including the label-informed trimming caveat for `time_norm` / `time_hour`. |

## Module 7: Metrics and uncertainty

| # | Item | Status |
|---|---|---|
| 7a | All metrics, and why the primary one suits the task | Addressed: prereg A2, B4. Per-subject κ is primary because per-subject macro-F1 is unstable for subjects with few N3 epochs. Accuracy, macro-F1, per-class F1, ECE and pooled metrics are secondary. |
| 7b | Uncertainty estimates and how they are computed | Addressed: prereg A9. Paired subject bootstrap, 10,000 resamples, 95% percentile interval, seeds averaged within subject first. Seeds are reported as a stability range. |
| 7c | Statistical tests and their assumptions | Addressed: prereg A9. Decisions rest on the bootstrap intervals. A Wilcoxon signed-rank p on per-subject differences is reported as a secondary statistic (symmetric differences assumed; not computed under 6 subjects). No tests across folds, because fold estimates are correlated. |

## Module 8: Generalisability and limitations

| # | Item | Status |
|---|---|---|
| 8a | Evidence of generalisability beyond the data | N/A by design: no outside dataset is used, and the audit does not claim generalisation beyond held-out SC subjects. `sc20` vs `sc78` is reported as an internal comparison of cohort mix. |
| 8b | Contexts where the findings are not expected to hold | Addressed: data card "Known limitations"; model info sheet §3; README limitations (completed with the results). Ages 35–49, sleep disorders, other montages, AASM scoring, deep models, untrimmed nights. |
