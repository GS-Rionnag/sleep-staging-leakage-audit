# Pre-registration: sleep-staging leakage audit

- **Frozen:** 24 November 2026, at the commit tagged `prereg-v1` (annotated tag).
- **Template:** Hofman et al. (2023) two-phase pre-registration for predictive modelling ([arXiv 2311.18807](https://arxiv.org/html/2311.18807)). Phase A is fixed before any model is trained on the full data; Phase B before any full-data result is looked at. Both are frozen together here, because no full-data model has been trained yet.
- **Rule:** the full-data results (`sc78`, `sc20`) are produced once, from a commit that descends from `prereg-v1`. The code enforces this: `evaluate` and `all` refuse an `sc20` or `sc78` config unless the tag is an ancestor of `HEAD`. The only override is `--allow-without-prereg`, and `meta.json` records its use.
- **Scope:** this is a study of evaluation methods. It is not a medical tool, and nothing here supports clinical use.

Names in `code font` are the exact identifiers used in the code (`splits.ARMS`, `models.MODELS`, column names in `results/`).

---

## Phase A: before training

### A1. Research question

How much does the way a sleep-staging model is cross-validated inflate its score on Sleep-EDF Sleep Cassette? Which part of the inflation comes from neighbouring 30-s epochs, which from the same night, and which from the same person? Does the answer depend on whether the features carry information from neighbouring epochs?

### A2. Primary outcome

**Per-subject Cohen's κ** over the five stages (labels `[0, 1, 2, 3, 4]` = W, N1, N2, N3, REM), computed from out-of-fold predictions.

For each dataset, model, arm and seed, every epoch is predicted exactly once. κ is then computed separately over each subject's epochs. A κ that is undefined (truth and prediction both constant) stays NaN; the smallest subject has 824 epochs, so this is not expected.

The unit of analysis is the **subject**: not the fold, the epoch or the seed.

### A3. Data

- **Source:** Sleep-EDF Expanded 1.0.0, Sleep Cassette (SC) files only. Sleep Telemetry (ST) files are never downloaded. Details are in [`data_card.md`](data_card.md).
- **Primary dataset `sc78`:** all 153 SC recordings from 78 subjects (IDs 0–82; IDs 39, 68, 69, 78 and 79 do not exist). K = 10.
- **Secondary dataset `sc20`:** subjects 0–19 of the same files (39 recordings). K = 20, which makes the `subject` arm leave-one-subject-out.
- **Integrity:** each file is checked against the SHA256 in the committed manifest `data/manifests/sc_files.csv`, built from PhysioNet's `SHA256SUMS.txt`.

### A4. Preprocessing (identical for every arm)

1. **Label map:** `Sleep stage W`→0, `1`→1, `2`→2, `3` and `4`→3 (N3), `R`→4. `Sleep stage ?`, `Movement time`, and epochs no annotation covers → −1.
2. **Epoch grid:** epoch *i* covers [30*i*, 30*i*+30) s from the PSG start. An annotation fills epochs `ceil(onset/30)` to `floor(end/30)`. Labels are clipped to the PSG length, read from the EDF header.
3. **Drop and trim, in DeepSleepNet order:** first drop the −1 epochs. Then, on the remaining sequence, keep from 60 epochs before the first non-W epoch to 60 epochs after the last one (inclusive, clipped to the bounds).
4. **Count check:** the kept epochs must give `sc20` W 8,285 (±1), N1 2,804, N2 17,799, N3 5,703, REM 7,717, total 42,308 (±1). `sc78` must give W 65,951 (±10), N1 21,522, N2 69,132, N3 13,039, REM 25,835, total 195,479 (±10). During development all of these matched exactly.

### A5. Features (computed per recording, before any split; nothing in this step is fitted)

- **`bp_*` (5):** relative band power of `EEG Fpz-Cz` at 100 Hz. Welch PSD with `nperseg=256` on each 3,000-sample epoch; bands delta 0.5–4.5, theta 4.5–8.5, alpha 8.5–11.5, sigma 11.5–15.5, beta 15.5–30 Hz (the MNE sleep tutorial's bands). The five values sum to 1.
- **`bpctx_*` (20):** the `bp_*` values of the 2 kept epochs on each side of each epoch, within the same recording, in kept-epoch order. At the edges the nearest epoch is repeated.
- **`yasa_*` (116):** YASA 0.8.1 `SleepStaging(raw, eeg_name="EEG Fpz-Cz", eog_name="EOG horizontal").get_features()`, with no EMG. The raw signal is neither filtered nor z-scored first.
  - YASA runs on the trimmed segment `[first kept epoch, last kept epoch + 1) × 30 s`, cropped from the raw PSG, and rows are taken at the kept epochs.
  - The columns include YASA's smoothed and per-night robust-scaled variants (`*_c7min_norm`, a ±7-epoch centred window; `*_p2min_norm`, a 4-epoch trailing window), plus `time_hour` and `time_norm`.
- **`yasa_proba_{W,N1,N2,N3,REM}` (5):** the probabilities from YASA's pretrained classifier on the same segment.

### A6. Models (fixed hyperparameters, no tuning, no early stopping, no inner validation)

| Model (`MODELS`) | Inputs | Estimator |
|---|---|---|
| `majority` | none used | `DummyClassifier("most_frequent")`, fitted per fold |
| `bandpower_lr` | `bp_*` (5) | `StandardScaler` → `LogisticRegression(class_weight="balanced", max_iter=1000)`, other settings at scikit-learn 1.9.1 defaults (C = 1.0, L2, lbfgs) |
| `bandpower_ctx_lr` | `bp_*` + `bpctx_*` (25) | the same pipeline |
| `yasa_lgbm` | `yasa_*` minus `yasa_proba_*` (116) | `LGBMClassifier(n_estimators=500, max_depth=5, num_leaves=90, colsample_bytree=0.6, class_weight={W:1, N1:2.2, N2:1, N3:1.2, REM:1.4}, random_state=seed)`, LightGBM 4.7.0, other settings at defaults |
| `yasa_pretrained` | `yasa_proba_*` | YASA 0.8.1's pretrained classifier (trained on NSRR cohorts only); never trained in the audit |

The trained core models are `bandpower_lr`, `bandpower_ctx_lr` and `yasa_lgbm`. `majority` is the floor (κ = 0 by construction). `yasa_pretrained` is the split-invariance control.

### A7. Split arms (`splits.ARMS`; `splits.py` is the only place folds are made)

The same K, seeds, features, models and hyperparameters are used in every arm. Only the split rule changes.

| Arm | Folds | What can still leak into training |
|---|---|---|
| `epoch` | `KFold(K, shuffle=True, random_state=seed)` over epochs | neighbouring epochs, the same night, the same subject |
| `block` | `GroupKFold(K, shuffle=True, random_state=seed)` over 60-epoch (30-min) blocks per recording (`epoch_no // 60`). Then every training epoch of a test recording within 8 epochs of a test epoch is dropped, with distance counted in kept-epoch order. | the same night and subject, but no epoch whose features overlap a test epoch |
| `night` | `GroupKFold(K, shuffle=True, random_state=seed)` with recordings as groups | the subject's other night, when it lands in a training fold |
| `subject` (honest) | `GroupKFold(K, shuffle=True, random_state=seed)` with subjects as groups | nothing |
| `subject_prefit` | the `subject` folds, but the `StandardScaler` is fitted once on all epochs before the split | test-set scaling statistics (Kapoor and Narayanan's pre-split normalisation leak) |

- **Seeds:** `[0, 1, 2, 3, 4]`. A seed sets the fold assignment and LightGBM's `random_state`.
- **Assignment:** folds are assigned at random by group, never by contiguous subject-ID ranges.
- **`subject_prefit` scope:** this arm exists only for models with a scaler (`bandpower_lr`, `bandpower_ctx_lr`). For the other models it would be identical to `subject` and is skipped.

### A8. Baselines

`majority` (κ = 0) is drawn as a dashed line in every slopegraph. `yasa_pretrained` is a zero-shot reference that never saw Sleep-EDF. Literature reference points are listed in the spec (`reports/Sleep staging leakage audit setup.md`).

### A9. Hypotheses, expected sizes and decision rules

All contrasts are paired over subjects:

- Average each subject's score over the 5 seeds first.
- Compute one difference per subject.
- Resample subjects with replacement 10,000 times (bootstrap RNG seed 0) and take the 2.5th and 97.5th percentiles of the resampled means: the "interval" below.
- Drop subjects with a NaN in either side pairwise, and report how many were dropped.

`summary.csv` holds H1; `contrasts.csv` holds H2–H4.

**H1 (primary). Every trained model scores lower in `subject` than in `epoch`.**
- **Statistic:** d = κ(`subject`) − κ(`epoch`) per subject; the `summary.csv` row with `honest_arm = subject` and `leaky_arm = epoch`.
- **H1 holds for a model** if the interval for mean d lies entirely below 0 (`hi < 0`). An interval entirely above 0 contradicts H1 for that model.
- **The headline claim stands** if H1 holds for at least 2 of the 3 trained core models on `sc78`. If every interval includes 0, the audit reports a null result and says so; it is not dropped.
- **Expected size:** for the feature models, 5–10 accuracy points and 0.07–0.12 κ (mean d between −0.12 and −0.07). These figures are an inference from Köksal (2026) and informal replications, not a measurement of this pipeline.

**H2. The gap shrinks in the order `epoch` ≥ `block` ≥ `night` ≥ `subject`.**
- **Statistic:** for each trained model, three adjacent contrasts: `block - epoch`, `night - block` and `subject - night` (κ, rows `hypothesis = H2`).
- **Supported for a model** if all three mean differences are ≤ 0 and no interval lies entirely above 0.
- **Contradicted** if any interval lies entirely above 0.
- **Not supported** otherwise.
- Each step whose interval lies entirely below 0 is reported as resolved; the others as unresolved.
- **Expected size:** `night` vs `subject` about 1–2 accuracy points (an inference from personalisation studies). The `block` arm has no published estimate.

**H3. The gap is larger for `bandpower_ctx_lr` and `yasa_lgbm` than for `bandpower_lr`.**
- **Statistic:** for each of the two models, its per-subject gap κ(`subject`) − κ(`epoch`) minus `bandpower_lr`'s gap for the same subject (rows `hypothesis = H3`).
- **Supported for a model** if the interval lies entirely below 0, meaning a more negative gap and more inflation. Reported separately for each model.

**H4. The leaky gain concentrates in stable epochs.**
- **Strata:** each epoch is assigned to a stratum by its distance in kept epochs to the nearest stage change in the reference labels: `0-1`, `2-3`, `4`, `stable` (≥ 5).
- **Statistic:** for each trained model, per-subject accuracy in the `stable` stratum minus accuracy in the `0-1` stratum, each taken as `epoch` arm minus `subject` arm (rows `hypothesis = H4`, metric `accuracy`).
- **Supported for a model** if the interval lies entirely above 0. Accuracy is used rather than κ because κ within a stratum is badly conditioned: stable stretches are dominated by one or two stages.

**Multiplicity.** H1 is the single primary hypothesis, and its success rule is stated above. H2–H4 are secondary. They are reported with their intervals and no multiplicity correction, and are worded as such. The `sc20` analysis applies the same rules but is secondary, and does not change the headline decision.

**Other comparisons** in `summary.csv`, reported without a pre-set decision rule: `subject` vs `block`, `subject` vs `night`, and `subject` vs `subject_prefit`. The last is expected to be near zero: with 5–25 standardised features, a scaler fitted on all data leaks little.

### A10. Pipeline-failure criteria

If any of these fires, the run is a pipeline failure. Its headline numbers are not interpreted; the cause is fixed, logged in the deviations table, and the full run is repeated.

| Criterion | Unit or label test | Run-time check (`meta.json` → `pipeline_checks`) |
|---|---|---|
| An honest fold (`subject`, `subject_prefit`) shares a subject between train and test | `test_splits.py::test_subject_arm_no_overlap`, `test_unsorted_row_order_is_fine`, `test_checks.py::test_honest_fold_sharing_a_subject_raises` | `honest_folds_subject_disjoint`; `run_evaluation` stops before training |
| `yasa_pretrained` per-subject κ differs between arms (or seeds) | `test_yasa_model.py::test_pretrained_split_invariant`, `test_checks.py::test_split_invariance_check` | `yasa_pretrained_split_invariant`; `run_evaluation` exits non-zero after writing results |
| Kept-epoch counts outside the A4 tolerances | `test_labels.py::test_sc20_counts`, `test_sc78_counts`, `test_tutorial_recordings`, `test_checks.py::test_epoch_counts_match_published_sc20` | `epoch_counts_match_published`; `run_evaluation` stops before training |

In addition, `meta.json` must show `dirty: false` and a `commit` that descends from `prereg-v1`.

---

## Phase B: before testing

### B1. Methods and training details

- **Out-of-fold procedure:** for each dataset × model × arm × seed, fit a fresh estimator on each fold's training rows and predict its test rows. Every epoch is predicted exactly once per seed (tested: `test_every_row_tested_once`, `test_oof_covers_all_rows`).
- **Leakage guard:** a spy transformer test confirms the scaler is fitted only on training rows, except in `subject_prefit` (`test_spy_scaler_fit_rows`).
- **Machine:** all headline runs execute on one machine, the server `server` (Ubuntu 24.04, 32 cores, 62 GB RAM). LightGBM and BLAS differ between machines by a few thousandths of κ, so results are not mixed across machines.
- **Environment:** pinned by the committed `uv.lock` (`uv sync --locked`); Python 3.12, scikit-learn 1.9.1, LightGBM 4.7.0, YASA 0.8.1, MNE 1.13.2.
- **Command:** `uv run python -m sleepaudit all --config configs/sc78.yaml`, then `--config configs/sc20.yaml`.

### B2. Test-access statement

**Development used only subjects 0–3 plus label-only checks on all 153 hypnograms.**

- Full PSG signals were downloaded only for the 4-subject fixture: subjects 0–3, 8 recordings, 8,154 kept epochs.
- For every other recording, only the hypnogram and the first 4 KiB of the PSG (the EDF header, used for its length) were read. These were used to:
  - reproduce the published epoch counts;
  - check that PSG and hypnogram start times agree;
  - measure the split properties in B5.
- No model was fitted, and no feature computed, on any subject other than 0–3.

### B3. Outputs (under `results/<dataset>/`)

| File | Contents |
|---|---|
| `per_subject.csv` | one row per model × arm × seed × subject: κ, accuracy, macro-F1, ECE, number of epochs |
| `summary.csv` | H1 and the other `subject`-vs-leaky comparisons |
| `contrasts.csv` | H2–H4 |
| `pooled.csv` | pooled metrics per model × arm × seed |
| `confusion.csv` | row-normalised confusion matrices per model × arm |
| `strata.csv` | accuracy per model × arm × stratum |
| `nights.csv` | per-recording κ with night-level summaries |
| `figures/` | slopegraphs and the forest plot |
| `meta.json` | run metadata and `pipeline_checks` |
| `oof.parquet` | out-of-fold predictions (git-ignored) |

### B4. Secondary analyses (descriptive, no decision rule beyond A9)

- **Pooled metrics:** accuracy, κ and macro-F1, for comparison with papers that pool over folds. They are never mixed with per-subject numbers.
- **Per-class F1**, with attention to N1.
- **Row-normalised confusion matrices.**
- **Expected calibration error:** 15 equal-width bins on the maximum probability, per subject.
- **Stratum accuracy** by distance to the nearest stage change (`strata.csv`).
- **Night-level fragmentation:** each recording's transition share, N1 share, number of awakenings, WASO and age, joined with its κ per model and arm. The lowest and highest thirds of transition share and of N1 share are compared (`nights.csv`).
- **`sc20` vs `sc78`:** the same tables for both datasets. The age split between them (all of `sc20` is aged 25–34) tests whether a uniform cohort understates the subject leak; that inference is labelled as such.
- **Per-subject macro-F1:** taken over the stages present in that subject's truth or predictions, so an absent N3 is not scored as 0.

### B5. Known properties of the design, stated before the results

- **The `night` arm is diluted.** Folds are drawn over recordings, so some subjects have both nights in the same test fold and leak nothing in that arm. Measured on the labels: in `sc78`, 6, 6, 12, 8 and 9 of 75 two-night subjects for seeds 0–4. In `sc20`, 0, 0, 0, 0 and 1 of 19. The one-night subjects 13, 36 and 52 never leak in this arm. This biases the `night`–`subject` gap towards 0.
- **The `block` arm trains on slightly fewer epochs:** 87.6% of rows per fold on average in `sc78`, against 90.0% in the other arms; 93.7% against 95.0% in `sc20`. This is because of the buffer, and it biases the `block` score slightly downwards.
- **Training-set size is otherwise equal across arms** (same K). Individual `subject` and `night` folds vary in size because groups differ in size (minimum 87.9% in `sc78`).
- **`time_hour` and `time_norm` stay in `yasa_lgbm`'s inputs** because YASA's own classifier uses them. Since the segment is trimmed with the reference labels (±30 min of wake around sleep), position in the segment carries some label information. It does so equally in every arm, so it raises absolute scores, not the arm differences. The same holds for the field's standard trimming convention.
- **Per-night normalisation is not a leak.** YASA's robust scaling over the trimmed segment uses only that night's own signal, which would also be available when scoring a new person. In the `block` and `epoch` arms it does mix test-epoch statistics into training rows of the same night, which is part of what those arms measure.
- **No inner validation exists:** no model is tuned and none uses early stopping. The "no subject overlap in inner validation" requirement is therefore satisfied trivially.
- **YASA's pretrained classifier was pickled with scikit-learn 0.24.2** and is loaded under 1.9.1 with an `InconsistentVersionWarning`. On the fixture it agrees with the hypnogram at 0.85–0.86 accuracy, so it is used as is.
- **Intervals reflect subject sampling only.** They describe uncertainty about which subjects were tested, not variation in training sets, which only nested resampling would capture. Seeds are a stability range, not extra samples.

### B6. Not pre-registered (exploratory if run later)

The following are not part of this protocol. If any is run later, it will be reported as exploratory:

- wake-policy ablations (in-bed only, ±60 min);
- YASA on the untrimmed recording;
- feature-selection leaks other than the scaler;
- deep models (braindecode Chambon, USleep trained from scratch);
- any external reference.

Deep models would get a separate addendum before they run.

### B7. Decisions made before the freeze that differ from the spec's wording

| Decision | Reason |
|---|---|
| The block buffer is "distance ≤ 8 kept epochs" | It must exceed YASA's ±7-epoch window, and counting in kept-epoch order also covers the context features across dropped epochs |
| H1 needs the interval below 0, not just excluding 0 | An interval above 0 would contradict "scores lower" |
| H2–H4 have explicit statistics (`contrasts.csv`) | The spec states the hypotheses but not how to test them |
| Per-subject macro-F1 is over the stages present | Absent stages would otherwise score 0 and dominate older subjects' macro-F1 |
| `subject_prefit` is run only for models with a scaler | For the others it is identical to `subject` |

---

## Deviations log (after the freeze)

Every change after `prereg-v1` that affects the protocol, the code path of a full run, or the reading of a result is logged here with its commit, before the affected result is reported.

| Date | Commit | What changed | Why | Effect on results |
|---|---|---|---|---|
| 24 Nov 2026 | `b072949` | `report.slopegraph`: arm columns are keyed by arm name; forest-plot x-label shortened | After the full runs, the slopegraph means were found to be pooled across the two arms (both arm Series were named `kappa`). Found by inspecting the figure; the fix is figures only. A regression test was added, which failed before the fix | None on any number. Every CSV and `meta.json` is unchanged (`report` was re-run from the server's `per_subject.csv`; CSV differences were line endings only). Only `slopegraph_*.png` and `forest.png` changed. No model was retrained and no pipeline-failure criterion was involved |
