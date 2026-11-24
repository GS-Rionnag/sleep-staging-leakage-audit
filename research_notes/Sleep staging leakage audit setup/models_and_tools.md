# Models and tools for a Sleep-EDF leakage audit (YASA, U-Sleep, lightweight deep baselines, band-power + LR, sklearn group CV)

Research date: 2026-11-24. Versions were checked against live PyPI, docs and GitHub API pages on that date. Items marked **[UNVERIFIED]** come from prior knowledge or secondary snippets and were not confirmed against a primary source in this session.

---

## 1. YASA: version, `SleepStaging` API, classifier, training data, Sleep-EDF contamination, retraining, accuracy, install

### Takeaway
YASA 0.8.1 (released 2026-10-25, Python >= 3.11, BSD-3) ships a LightGBM classifier trained on 3,163 nights from seven NSRR cohorts (CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS). Sleep-EDF is not mentioned anywhere in the paper or the training repo, so the pre-trained model is **not contaminated by Sleep-EDF**. It is, however, a **cross-domain** evaluation: Sleep-EDF has Fpz-Cz EEG, horizontal EOG, a 1 Hz EMG envelope and R&K scoring, whereas YASA expects central-mastoid EEG, LOC-mastoid EOG, raw chin EMG and AASM. `get_features()` exposes the feature matrix, so the student can retrain any classifier under leaky and subject-wise splits. YASA's 7.5-min centred smoothing and per-night z-scoring will make epoch-wise random splits leak even more than raw features would.

### Cited Findings
**Version, packaging, dependencies**
- The latest YASA version is 0.8.1, released Oct 25, 2026. Earlier releases are 0.8.0 (Oct 2, 2026), 0.7.0 (Mar 8, 2026) and 0.6.5 (Jul 4, 2024). Requires-Python is >= 3.11 and the license is BSD-3-Clause — [PyPI yasa](https://pypi.org/project/yasa/)
- The GitHub repo `raphaelvallat/yasa` (default branch `master`) was last pushed 2026-10-25. It is not archived, uses BSD-3-Clause and has about 602 stars (GitHub API, queried 2026-11-24) — [GitHub raphaelvallat/yasa](https://github.com/raphaelvallat/yasa)
- The documentation has moved: raphaelvallat.com/yasa now 301-redirects to yasa-sleep.org — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)
- **v0.8.1 (Oct 2026):** fixed a `Pandas4Warning` in `yasa.SleepStaging` with pandas 3.1. There are no other SleepStaging changes — [YASA changelog](https://yasa-sleep.org/changelog.html)
- **v0.8.0 (Oct 2026):**
  - Removed `SleepStaging.predict_proba` (deprecated in 0.7). Probabilities now live in the `proba` attribute of the Hypnogram returned by `predict`.
  - Deleted the old 0.4.0 pre-trained classifiers. The most recent classifiers are always used.
  - Requires Python >= 3.11.
  - Minimum versions: numpy >= 2.2.2, scipy >= 1.15, pandas >= 2.3, matplotlib >= 3.10.1, mne >= 1.10, scikit-learn >= 1.6.1, lightgbm >= 4.6.
  - Removed `numba` and `tensorpac`.
  
  — [YASA changelog](https://yasa-sleep.org/changelog.html)
- **v0.7.0 (Mar 2026):**
  - Breaking change: `SleepStaging.predict()` returns a `Hypnogram` object instead of a string array. Use `.hypno` for string labels or `.as_int()` for integers.
  - mne, scikit-learn, lightgbm, antropy and tensorpac became optional. Automatic staging needs `pip install "yasa[full]"`.
  
  — [YASA changelog](https://yasa-sleep.org/changelog.html)
- None of the changelog entries for 0.7.0, 0.8.0 or 0.8.1 mention a new or retrained staging classifier — [YASA changelog](https://yasa-sleep.org/changelog.html)

**`yasa.SleepStaging` API (docs for v0.8.1)**
- **Constructor:** `SleepStaging(raw, eeg_name, *, eog_name=None, emg_name=None, metadata=None)`.
  - `raw` is an MNE `BaseRaw`.
  - `eeg_name` is required. A central electrode referenced to the mastoid is preferred (C4-M1, C3-M2), or one referenced to Fpz (C4-Fpz).
  - `eog_name` is optional. Left LOC referenced to a mastoid (e.g. E1-M2) or to Fpz is preferred.
  - `emg_name` is optional. A chin electrode is preferred.
  - `metadata` is an optional dict with `age` and `male`.
  
  — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)
- **Signal handling:** data are downsampled to 100 Hz automatically and converted to µV if necessary. The docs say "Do NOT transform (e.g. z-score) or filter the signal". Epochs are 30 s — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)
- **Features:**
  - Per epoch and channel: SD, IQR, skewness, kurtosis, zero-crossings, Hjorth mobility and complexity, absolute 0.4–30 Hz power, relative band powers (EEG and EOG only), power ratios, permutation entropy, and Higuchi and Petrosian fractal dimensions.
  - Smoothing: a 7.5-min centred triangular-weighted rolling average plus a 2-min past rolling average.
  - The smoothed features are then robust z-scored.
  
  — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)
- **Methods:**
  - `fit()` extracts features.
  - `get_features()` returns a copy of the feature DataFrame.
  - `predict([path_to_model])` returns per-epoch stages, with probabilities in `hyp.proba`.
  - `plot_predict_proba(...)` plots the probabilities.
  
  — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)
- The docs do not describe how channel names are matched, or what happens with names such as "EEG Fpz-Cz". They also do not specify the file format for `path_to_model` — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)
- **Warnings in the docs:**
  - Predictions should be checked by a trained scorer.
  - Classifiers were trained only on full nights, so nap data needs inspection.
  - Sleep disorders and suboptimal PSG systems or referencing need inspection.
  - "N1 is the least accurately detected stage", often confused with Wake or N2.
  
  — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)

**Classifier and training data (eLife 2021 paper)**
- **Training set:** 3,163 full-night PSGs (more than 31,000 h) from seven NSRR cohorts: CCSHS 414, CFS 586, CHAT 351, HomePAP 82, MESA 575, MrOS 565, SHHS 590 — [Vallat & Walker 2021, eLife (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Testing set 1 (NSRR holdout):** 585 nights. CCSHS 100, CFS 99, CHAT 100, MESA 97, MrOS 90, SHHS 99 — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Testing set 2 (Dreem Open Datasets):** 75 nights after exclusions, covering DOD-Healthy (25) and DOD-Obstructive. The paper's text says 55 OSA patients while its tables say n = 50, which is an internal inconsistency — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Sleep-EDF is not mentioned** anywhere in the paper — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- The training repo lists CCSHS, CFS, CHAT, HomePAP, MESA, MrOS and SHHS for training, NSRR for testing set 1 and DREEM for testing set 2. Sleep-EDF is not mentioned — [GitHub raphaelvallat/yasa_classifier](https://github.com/raphaelvallat/yasa_classifier)
- **Classifier:** LightGBM with 500 estimators, max depth 5, 90 max leaves and a 60% feature fraction.
  - Hyperparameters were chosen by a 3-fold cross-validated search.
  - Class weights: N1 2.2, N2 1, N3 1.2, REM 1.4, Wake 1.
  
  — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Preprocessing and features (paper):**
  - Signals downsampled to 100 Hz and band-pass filtered 0.40–30 Hz.
  - Welch periodograms with 5-s Hamming windows.
  - Features smoothed (7.5-min centred triangular and 2-min past), then z-scored per night.
  - Raw features and normalised time-in-night are also included.
  - Inputs: one central EEG, one left EOG, one chin EMG, plus optional age and sex.
  
  — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Training pipeline:** randomise train/test nights, extract features (`feature_extraction/`), then train and export (`02_create_classifiers.ipynb`). A `gridsearch/` folder covers hyperparameters, class weights and smoothing windows — [yasa_classifier](https://github.com/raphaelvallat/yasa_classifier)

**Reported accuracy and weaknesses**
- **Testing set 1:** median accuracy 87.46%, Cohen's kappa 0.819, MCC 0.821 — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **DOD:** DOD-Healthy accuracy 86.6% (kappa 0.801). DOD-Obstructive accuracy 84.3% (kappa 0.765) — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **N1 is the weakest stage.**
  - Testing set 1: sensitivity 45.4%, median F1 0.432.
  - DOD: N1 F1 52.0 in healthy subjects and 39.17 in OSA patients.
  - N3 F1 is 0.835 on testing set 1. REM, N2 and Wake all have F1 >= 0.86.
  
  — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Fragmented sleep:**
  - The two strongest predictors of per-night accuracy were %N1 and the number of stage transitions, together about 40% of relative importance.
  - Accuracy was 69.15% near transitions versus 94.08% in stable epochs.
  - Accuracy correlated negatively with AHI (r = −0.29) and with age (r = −0.128).
  
  — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- **Speed:** a full night at 100 Hz is processed "in less than 5 s on a rudimentary, basic consumer-level laptop" — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)

**Sleep-EDF signal facts that affect YASA input**
- **Sleep-EDF SC channels:** EEG Fpz-Cz and Pz-Oz, horizontal EOG, submental EMG, event marker, and often respiration and temperature. EEG and EOG are sampled at 100 Hz.
- **Submental EMG in SC files is a 1 Hz RMS envelope**, not raw EMG.
- **ST files:** EEG, EOG and EMG at 100 Hz.
- **Scoring:** hypnograms were scored with R&K 1968, using labels W, R, 1, 2, 3, 4, M and ?.
- **License:** ODC-By v1.0.

— [PhysioNet Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)

### Inferences
- **Contamination:**
  - The YASA pre-trained model is clean with respect to Sleep-EDF. Its training and validation data came only from NSRR cohorts; Sleep-EDF, which comes from separate Dutch and PhysioNet studies, is absent.
  - Applying it to Sleep-EDF is a zero-shot, cross-montage and cross-scoring-manual evaluation, not an in-distribution one. Expect lower accuracy than the paper's 87%.
  - Pre-trained YASA has no "split" at all, so under both leaky and subject-wise splits it should score the same on the same test epochs. That makes it a useful invariant check for the audit.
- **Channel choice on Sleep-EDF SC:**
  - Use `eeg_name="EEG Fpz-Cz"`, because the docs accept Fpz-referenced montages. This is still a montage mismatch with C4-M1.
  - `eog_name="EOG horizontal"` is a mismatch with LOC-mastoid.
  - Do **not** pass the SC "EMG submental" channel. It is a 1 Hz RMS envelope that YASA would upsample to 100 Hz, so its features would be meaningless. ST files do have 100 Hz EMG.
  - Channel names are passed verbatim from `raw.ch_names`. No automatic renaming is documented.
- **Label mapping:** merge R&K stages 3 and 4 into N3, and drop M and ? epochs. Most studies also trim long wake periods (MNE and braindecode use ±30 min). Whatever trimming is used must be applied identically across models.
- **Retraining:**
  - Call `SleepStaging(...).get_features()` per night, concatenate the frames with `subject_id` and `night` columns, and train LightGBM or LR with `GroupKFold`. This gives "YASA features + custom classifier" under both split regimes.
  - Because features are temporally smoothed over ±3.75 min and z-scored per night, an epoch-wise random split places near-duplicate neighbours in train and test. The leaky-split inflation should therefore be **larger** for YASA features than for raw per-epoch band powers. This is a good, testable hypothesis for the audit.
  - Using the paper's LightGBM hyperparameters and class weights for the retrain is reasonable and documented.
- **Install on Windows:** `pip install "yasa[full]"` in a Python 3.11–3.13 venv should work with pip wheels, since lightgbm >= 4.6 and mne >= 1.10 publish Windows wheels. **[UNVERIFIED]**: Windows wheel availability for every dependency was not checked.

### Gaps
- Whether `antropy` is still a required dependency for `SleepStaging` in 0.8.x. The 0.7.0 notes list it as optional and the 0.8.0 minimums omit it.
- The exact on-disk format expected by `predict(path_to_model=...)`. Historically this was a joblib-dumped LGBMClassifier whose feature names match `get_features()` columns **[UNVERIFIED]**.
- No published accuracy of pre-trained YASA on Sleep-EDF was found. SLEEPYLAND reports YASA only on CCSHS, MESA and DOD (see Section 6).
- How YASA reacts when `metadata` is omitted, and whether separate classifier files exist for EEG-only, EEG+EOG and EEG+EOG+EMG. Neither was confirmed in the 0.8.1 docs.

---

## 2. U-Sleep: code, weights, license, training datasets (Sleep-EDF contamination), hosted API, CPU feasibility, TF constraints, PyTorch reimplementations, training cost

### Takeaway
**The public U-Sleep model was trained on Sleep-EDF.** Table 1 of the U-Sleep paper puts SEDF-SC (78 subjects, 153 PSGs) and SEDF-ST (22 subjects, 44 PSGs) in the "Train/Test" group: at least 75% of each went into training, and only about 15% (23 SC and 8 ST records) was held out. Any evaluation of the pre-trained U-Sleep on Sleep-EDF is therefore contaminated unless restricted to the paper's exact test records, and the deployed webserver model may not even match those.

Practical access is also limited:
- The U-Time repo (MIT) does not distribute weights.
- The hosted API needs an access request and was unreachable (connection refused) when checked on 2026-11-24.
- The code depends on TF >= 2.8 and the archived `tensorflow-addons`.

Inference is cheap on CPU (seconds per night, per the paper). For an honest U-Sleep number, the student should train braindecode's PyTorch `USleep` from scratch under subject-wise folds; braindecode ships no pretrained weights. Alternatively, the SLEEPYLAND/SOMNUS weights treat SEDF as out-of-domain (Section 6).

### Cited Findings
**Training datasets and contamination**
- **"Train/Test" group (13 datasets):** ABC, CCSHS, CFS, CHAT, DCSM, HPAP, MESA, MROS, PHYS, **SEDF-SC (78 subjects / 153 records)**, **SEDF-ST (22 / 44)**, SHHS and SOF.
  - Each was split into ≥75% training, ≤10% validation and ≤15% testing.
- **"Hold-out" group (8 datasets):** ISRUC-SG1/2/3, MASS-C1/C3, SVUH, DOD-H and DOD-O. These were used for evaluation only.

— [Perslev et al. 2021, npj Digit Med (PMC8050216)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- **U-Sleep test-subset F1 (majority vote):**
  - SEDF-SC (23 test records): Wake 0.93, N1 0.57, N2 0.86, N3 0.71, REM 0.88, mean 0.79.
  - SEDF-ST (8 test records): mean 0.76.
  
  — [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- The paper does not say whether the model behind the webserver was retrained on all data, including the test splits — [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- An independent 2025 benchmark (SLEEPYLAND) also marks SEDF-SC and SEDF-ST as **in-domain** (✓) for published U-Sleep results — [Dei Rossi et al., SLEEPYLAND, arXiv 2506.08574v2](https://arxiv.org/html/2506.08574v2)

**Model, inputs, compute**
- **Inputs:** any single EEG plus any single EOG at standard positions, resampled to 128 Hz with polyphase filtering. EMG is not used — [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- **Training setup:** one NVIDIA Titan X (12 GiB), batch size 64, Adam with learning rate 1e-7, stopped after 100 epochs without validation improvement. The paper gives no total wall-clock time — [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- **Inference:** a full night "takes seconds on a laptop CPU" and under 1 s on a GPU — [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- **U-Time README compute notes:**
  - Recommends 4+ CPU cores, 8+ GiB RAM and 1 CUDA GPU for training from scratch.
  - The authors used 8 cores, 1 GPU and 40 GiB RAM.
  - A full reproduction needs about 4 TiB of storage.
  - The demo (6 Sleep-EDF SC + 6 DCSM studies) takes about 30 min including downloads, and CPU training "may take up to 30 minutes".
  - Full default training "may take very long". The README suggests raising the learning rate from 1e-7 to 1e-6.
  
  — [GitHub perslev/U-Time](https://github.com/perslev/U-Time)

**Code, weights, license, maintenance**
- **U-Time license and status:** MIT. Last commit 2026-06-24 (merge of PR #79). Not archived, about 322 stars (GitHub API, 2026-11-24) — [GitHub perslev/U-Time](https://github.com/perslev/U-Time)
- **README on weights:** "If you are looking to use our pre-trained U-Sleep model … please refer to https://sleep.ai.ku.dk". The README also says U-Sleep "is also available under research and commercial licenses on the youSleep BETA platform (yousleep.ai)". It gives no download link or license for the weights themselves — [GitHub perslev/U-Time](https://github.com/perslev/U-Time)
- **Version divergence:** the README says the repo "has been significantly extended since [the paper] and may gradually diverge" from the published U-Sleep — [GitHub perslev/U-Time](https://github.com/perslev/U-Time)
- **`requirements.txt` pins:** `tensorflow>=2.8.0`, `tensorflow-addons>=0.16.1`, `mne>=0.24.1`, `psg_utils>=0.1.6` and others. The README says GPU use needs TF v2.8.0, CUDA 11.2 and cuDNN 8.1 — [U-Time requirements.txt](https://github.com/perslev/U-Time/blob/main/requirements.txt); [U-Time README](https://github.com/perslev/U-Time)
- **`tensorflow/addons` is archived** on GitHub (archived=true, last push 2025-09-04) — [GitHub tensorflow/addons](https://github.com/tensorflow/addons)
- **PR #79 (merged 2026-06-24):** adds dynamic `data_per_prediction` for ONNX export "with the same interface as the shipped U-Sleep v1 model". It mentions exporting "v2 and single-channel models in `yousleep-ai/usleep`" and verifying ".h5 checkpoints". That repo returned 404 to the GitHub API, so it is likely private — [U-Time PR #79](https://github.com/perslev/U-Time/pull/79)

**Hosted API**
- **Access:** must be requested at sleep.ai.ku.dk/docs/api/access. Users log in and generate an API token, which was valid for 12 h at the time of writing — [GitHub perslev/U-Sleep-API-Python-Bindings](https://github.com/perslev/U-Sleep-API-Python-Bindings)
- **Description:** the README calls it "a free and highly accurate sleep staging algorithm", "publicly available for anyone interested in clinical sleep research" — [U-Sleep-API-Python-Bindings](https://github.com/perslev/U-Sleep-API-Python-Bindings)
- **Bindings:** MIT-licensed, `pip install usleep-api`. Example command: `usleep-api ./my_psg.edf ./hypnogram.tsv --anonymize`. The only model name shown is "U-Sleep v1.0" — [U-Sleep-API-Python-Bindings](https://github.com/perslev/U-Sleep-API-Python-Bindings)
- **Paper on the webserver:** "freely available" for non-commercial use. Users upload anonymised or public-domain EDF or HDF5 files — [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)
- **Reachability:** fetching https://sleep.ai.ku.dk/ and https://sleep.science.ku.dk/docs/api/overview on 2026-11-24 both failed with `ECONNREFUSED 130.225.251.190:443`. The terms of service could not be read (direct fetch attempt, this session).

**PyTorch reimplementation (braindecode)**
- **`braindecode.models.USleep` parameters:** `n_chans`, `sfreq`, `depth=12`, `n_time_filters=5`, `complexity_factor=1.67` (the paper uses 2), `with_skip_connection`, `n_outputs=5`, `input_window_seconds`, `time_conv_size_s=9/128`, `ensure_odd_conv_size`, `chs_info` and `n_times` — [braindecode USleep docs (v2.0.0 page)](https://braindecode.org/stable/generated/braindecode.models.USleep.html)
- **Pretrained weights:** the docs list no official pretrained USleep weights. There is only a generic Hugging Face Hub integration (`pip install braindecode[hub]`, `from_pretrained`, `push_to_hub`) with placeholder repo names — [braindecode USleep docs](https://braindecode.org/stable/generated/braindecode.models.USleep.html)
- The `braindecode/USleep` Hugging Face repo is architecture-only: "**No pretrained weights are distributed here.**" It is BSD-3-Clause, and fine-tuned checkpoints "inherit the licence of that checkpoint and its training corpus" — [HF braindecode/USleep README](https://huggingface.co/braindecode/USleep/blob/main/README.md)
- **braindecode release:** latest is 2.1.0 (released 2026-11-23), Requires-Python >= 3.11, BSD-3-Clause. Some components carry CC BY-NC and other licenses — [PyPI braindecode](https://pypi.org/project/braindecode/)
- braindecode has a U-Sleep sleep-staging example, `plot_sleep_staging_usleep.html` — [braindecode Chambon example page (links)](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)

### Inferences
- **Contamination verdict:** using the official U-Sleep v1 (webserver/API or any leaked checkpoint) on Sleep-EDF and reporting it next to honest models would be **contaminated**. About 75% of SEDF-SC/ST nights were training data, and the deployed model's exact training set is undocumented.
  - The only defensible uses are: (a) report it as an explicit "contaminated upper reference" with a warning; or (b) restrict it to the paper's 23 SC / 8 ST test records, if their IDs can be recovered (they are not in hand, see Gaps).
  - Recommendation for the audit: **exclude the pre-trained U-Sleep from the headline comparison**, or present it only as a worked example of training-set contamination. That example is itself on-topic for a leakage audit.
- **CPU and free-compute feasibility:**
  - Inference: feasible on a laptop CPU (seconds per night), but only with weights, which are not openly downloadable.
  - Training from scratch on Sleep-EDF-only (153 SC nights) with braindecode `USleep` on CPU: likely many hours per fold.
  - On a free Colab or Kaggle T4/P100 GPU: plausibly 1–3 h per fold for a few dozen epochs **[UNVERIFIED estimate, no benchmark found]**. A full k-fold subject-wise CV (e.g. 5 folds) fits within Kaggle's weekly GPU quota but probably not in one Colab session. Reproducing the 21-dataset model (about 4 TiB) is infeasible on free compute.
- **TF and Windows:** U-Time's stack (TF >= 2.8, archived `tensorflow-addons`) is fragile on Python 3.11+ and Windows. Native-Windows TF GPU support ended after TF 2.10 **[UNVERIFIED]**. Use WSL2, Colab, or the braindecode PyTorch port instead.

### Gaps
- U-Sleep v1 weight license and the webserver Terms of Service could not be read; the site was unreachable.
- Whether the deployed U-Sleep v1 was retrained on all data, including the SEDF test split.
- The list of the 23 SEDF-SC and 8 SEDF-ST test record IDs used in the paper (possibly in the paper's supplementary material or the U-Time split files). Not found.
- Whether `yousleep-ai/usleep` (ONNX v1/v2 checkpoints) is public. It returned 404 to the GitHub API.
- The latest `utime` PyPI version and date; the PyPI page failed to load.
- No measured braindecode `USleep` training times on Colab or Kaggle free GPUs were found.
- NIDRA (PyPI, MIT) reportedly ships U-Sleep weights "re-trained by Rossi et al.". This came from a search snippet only; the page failed to load, so its training datasets are unknown **[UNVERIFIED]** — [PyPI NIDRA](https://pypi.org/project/NIDRA/)

---

## 3. Lightweight deep baselines (braindecode Chambon2018, Eldele2021/AttnSleep, TinySleepNet) on CPU or free Colab

### Takeaway
braindecode is the practical route. It is maintained (v2.1.0, 2026-11-23), PyTorch-based, ships Sleep-EDF loaders and has worked sleep-staging examples. `SleepStagerChambon2018` trains on CPU at about 1.3–1.4 s per epoch on a 2-subject toy setup. The original TinySleepNet (TF 1.13, Py 3.6) and AttnSleep (PyTorch 1.4, Py 3.7) repos are unmaintained since 2023 and pin obsolete stacks; use braindecode's reimplementations instead. Their published Sleep-EDF numbers (around 84–85% accuracy, MF1 around 78–80) come from subject-wise k-fold CV and are a reasonable reference, but they were not re-verified from primary sources here.

### Cited Findings
- **braindecode Chambon2018 example (data and preprocessing):**
  - Sleep Physionet subjects 0 and 1 (recording 2), with `crop_wake_mins=30`.
  - EEG Fpz-Cz and Pz-Oz at 100 Hz, scaled to µV, 30 Hz low-pass.
  - Stages 3 and 4 merged.
  - Per-window channel-wise z-score.
  
  — [braindecode Chambon2018 example](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)
- **Same example (split and model):** a subject-wise split (subject 0 train, subject 1 validation). `SleepStagerChambon2018` is wrapped in `TimeDistributed` over 3 consecutive windows. Adam with lr 1e-3, batch 32, class-weighted cross-entropy — [braindecode Chambon2018 example](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)
- **Same example (results and runtime):**
  - The gallery run used CPU, about 1.3–1.4 s per training epoch and about 15.8 s total for 10 epochs.
  - Validation balanced accuracy was about 0.45, accuracy about 0.62, and macro-F1 0.48.
  - An offline 100-epoch run reached 64.2% balanced accuracy on the held-out recording, loaded as a pretrained checkpoint from the HF Hub.
  
  — [braindecode Chambon2018 example](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)
- braindecode also has an Eldele2021 (AttnSleep) example, `plot_sleep_staging_eldele2021.html` — [braindecode Chambon example page (links)](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)
- **TinySleepNet repo:**
  - TensorFlow 1.13.1, CUDA 10.0, Python 3.6.
  - Uses a 20-fold run command.
  - Apache-2.0, with a note "For academic and non-commercial use only".
  - Last push 2023-08-23 (GitHub API).
  
  — [GitHub akaraspt/tinysleepnet](https://github.com/akaraspt/tinysleepnet)
- **AttnSleep repo:**
  - PyTorch 1.4, Python 3.7.
  - Datasets: Sleep-EDF-20, Sleep-EDF-78 and SHHS, with K-fold set in `config.json`.
  - MIT per the GitHub API. Last push 2023-09-26.
  - The README reports no numbers.
  
  — [GitHub emadeldeen24/AttnSleep](https://github.com/emadeldeen24/AttnSleep)
- **Comparable published Sleep-EDF numbers (different protocols, not directly comparable):**
  - LWSleepNet: 86.6% accuracy / MF1 79.2 on Sleep-EDF-20; 81.5% / 74.3 on Sleep-EDF-78.
  - ADG-SleepNet: 87.1% / 84.0 on Sleep-EDF-20; 85.1% / 81.1 on Sleep-EDF-78.
  
  These come from search-result snippets with protocols not checked **[UNVERIFIED primary source]** — [WebSearch results incl. MDPI Symmetry 17(9):1461](https://www.mdpi.com/2073-8994/17/9/1461)

### Inferences
- **CPU budget estimate:** Chambon2018 is small, at about 1.4 s per epoch for 2 subjects on CPU. Scaling to Sleep-EDF-20 (about 39 nights) suggests roughly 20× the data, so about 30 s per epoch × 30–50 epochs × 5 folds ≈ 1–2 h on a laptop CPU **[estimate]**. That is feasible without a GPU.
- **Heavier models:** AttnSleep and USleep in braindecode are larger and probably need a free Colab or Kaggle GPU for full Sleep-EDF-78 CV.
- **Contamination of the Chambon checkpoint:** braindecode's pretrained Chambon checkpoint on the HF Hub was trained on Sleep-EDF subject 0. Reusing it on Sleep-EDF would contaminate results, so train from scratch inside folds.
- **Inherent subject-wise protocol:** TinySleepNet and AttnSleep use k-fold CV with folds equal to subjects (20-fold for Sleep-EDF-20), which is effectively subject-wise. Their numbers should be compared against the "honest" arm, not the leaky arm.

### Gaps
- Primary-source numbers for TinySleepNet were not verified. A secondary table reports 85.4% accuracy / MF1 80.5 / κ 0.80 on Sleep-EDF-20 **[UNVERIFIED]**.
- Primary-source AttnSleep Sleep-EDF-20 and -78 numbers were not verified; the README has none.
- Chambon 2018's own published Sleep-EDF numbers were not checked.
- No published wall-clock training times for these models on Colab or Kaggle free tiers were found.

---

## 4. Band-power + logistic regression baseline: features, MNE tutorial, expected subject-wise accuracy

### Takeaway
MNE's official sleep tutorial (MNE 1.13.2) defines the canonical minimal baseline:
- Features: multitaper PSD over 0.5–30 Hz, normalised to relative power, averaged in 5 bands (δ 0.5–4.5, θ 4.5–8.5, α 8.5–11.5, σ 11.5–15.5, β 15.5–30 Hz) per EEG channel.
- Classifier: a scikit-learn `Pipeline(FunctionTransformer → RandomForest)`.
- Result: about 0.69 accuracy (macro-F1 0.57) when training on one subject and testing on another.

Swapping RandomForest for `StandardScaler → LogisticRegression` and adding YASA-style features (Hjorth, entropy, neighbour-epoch context) is a natural extension. Context features themselves are a leakage amplifier under epoch-wise splits.

### Cited Findings
- **MNE tutorial data:** Sleep Physionet via `mne.datasets.sleep_physionet.age.fetch_data()`. "Alice" (train) and "Bob" (test), recording 1. Labels reduced to W, 1, 2, 3/4, R. Wake trimmed to 30 min before and after sleep. 30-s epochs: 841 training, 1,103 test — [MNE sleep tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)
- **MNE tutorial features (`eeg_power_band`):** multitaper PSD over 0.5–30 Hz, normalised by total power, then mean relative power in delta 0.5–4.5, theta 4.5–8.5, alpha 8.5–11.5, sigma 11.5–15.5 and beta 15.5–30 Hz. That gives 5 × n_EEG-channel features — [MNE sleep tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)
- **MNE tutorial classifier and results:**
  - `make_pipeline(FunctionTransformer(eeg_power_band), RandomForestClassifier(n_estimators=100, random_state=241))`.
  - Accuracy 0.687. Per-class F1: Wake 0.64, N1 0.24, N2 0.86, N3 0.86, REM 0.26. Macro-F1 0.57.
  
  — [MNE sleep tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)
- **YASA's richer hand-crafted feature set:** SD, IQR, skewness, kurtosis, zero-crossings, Hjorth mobility and complexity, absolute and relative band powers, power ratios, permutation entropy, and Higuchi and Petrosian fractal dimension, with 7.5-min centred and 2-min past smoothing — [YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html); [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)

### Inferences
- **Recommended baseline ladder:**
  1. `DummyClassifier(strategy="most_frequent")`, which predicts N2, or W if wake is untrimmed.
  2. Relative band powers (5 bands × Fpz-Cz [+ Pz-Oz]) + `StandardScaler` + `LogisticRegression(class_weight="balanced", max_iter=...)`.
  3. Option 2 + log-power, spectral entropy and Hjorth parameters.
  4. Option 3 + context features (lagged or lead neighbour epochs, rolling means).
  5. YASA `get_features()` + LightGBM.
  
  Comparing options 3 and 4 under both split regimes isolates how much context features inflate leaky-split scores.
- **Majority baseline sensitivity:** the majority-class baseline is very sensitive to wake trimming. Untrimmed Sleep-EDF SC files contain many hours of wake, so the majority class becomes W. Fix and report the trimming rule.
- **Expected subject-wise performance:** the MNE single-train-subject result (0.69 accuracy) is a floor. With many training subjects, simple band-power LR should land roughly in the 0.70–0.80 accuracy range subject-wise **[UNVERIFIED expectation, no primary benchmark found]**.

### Gaps
- No primary source was found reporting band-power + logistic regression accuracy on Sleep-EDF under subject-wise CV with many subjects.
- The MNE tutorial's exact current code (e.g. `compute_psd(method="multitaper")` arguments) was summarised, not copied verbatim.

---

## 5. scikit-learn tools for leakage-safe evaluation (version, splitters, pitfalls)

### Takeaway
scikit-learn 1.9.1 is current. Use `GroupKFold`, `StratifiedGroupKFold` or `LeaveOneGroupOut` with `groups=subject_id`, not night or epoch. Wrap all scaling and feature selection in a `Pipeline` so they are refit per fold.

Two things changed or are subtle:
- Since 1.6, `GroupKFold` has `shuffle` and `random_state`. The default `shuffle=False` is deterministic.
- `StratifiedGroupKFold` "shuffle" only permutes groups with similar class distributions, and stratification degrades towards GroupKFold when groups are large. Sleep nights are large groups.

### Cited Findings
- **`GroupKFold` (sklearn 1.9.1 docs):** signature `GroupKFold(n_splits=5, *, shuffle=False, random_state=None)`.
  - `shuffle` and `random_state` were added in version 1.6. The `n_splits` default changed from 3 to 5 in 0.22.
  - Each group appears exactly once in a test fold, and there must be at least `n_splits` distinct groups.
  - Folds are approximately balanced in sample count. `random_state` has no effect unless `shuffle=True`.
  
  — [sklearn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- **sklearn 1.6 changelog:** "*Enhancement* `GroupKFold` now has the ability to shuffle groups into different folds when `shuffle=True`" (#28519). 1.6.1 also fixed `cross_validate`, `cross_val_predict` and `cross_val_score` to accept `params=None` under metadata routing (#30451) — [sklearn v1.6 changelog](https://scikit-learn.org/stable/whats_new/v1.6.html)
- **`StratifiedGroupKFold`:** `(n_splits=5, shuffle=False, random_state=None)`. Per the docs:
  - "This implementation can only shuffle groups that have approximately the same y class distribution, no global shuffle will be performed."
  - When a few groups contain many samples, stratification may not be possible and behaviour approaches `GroupKFold`.
  
  — [sklearn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- `LeaveOneGroupOut` is "for splitting the data according to explicit, domain-specific grouping of the dataset" — [sklearn GroupKFold (related classes)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- In Sleep-EDF SC, several subjects have two nights (153 records from 78 subjects), so night-level splits can still leak subject identity — [PMC8050216 Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/); [PhysioNet Sleep-EDF](https://physionet.org/content/sleep-edfx/1.0.0/)

### Inferences
- **Group by subject:** set `groups` to the **subject ID** (in SC filenames, the 2-digit subject number in `SC4ssN...` **[UNVERIFIED filename convention]**), not the recording ID. A "record-wise" split is an intermediate leak between epoch-wise and subject-wise, and could be a useful third arm.
- **Pitfalls to encode as tests in CI [UNVERIFIED, from general sklearn knowledge; confirm against the sklearn "Common pitfalls" page]:**
  - (a) Passing `cv=5` (an int) to `cross_val_score` silently uses (Stratified)KFold and ignores `groups`. Always pass a splitter object *and* `groups=`.
  - (b) Calling `GroupKFold` without `groups` raises an error. Good, but `KFold` with `groups` silently ignores them.
  - (c) Fitting `StandardScaler` or feature selection on all data before CV leaks test statistics. Use `Pipeline`.
  - (d) In `GridSearchCV` inside an outer group CV, pass `groups` to the inner `fit` (or use metadata routing). Otherwise the inner CV is epoch-wise, a nested-CV leak.
  - (e) Per-night normalisation (as in YASA) is fine because it uses only that night's own data. Per-dataset normalisation fitted on all subjects is not.
  - (f) Before 1.6, `GroupKFold` was deterministic with no shuffle. Results with `shuffle=True` are not reproducible across sklearn < 1.6.
- **Unit test idea:** assert `set(groups[train]) & set(groups[test]) == set()` for every fold of every "honest" splitter, and assert a non-empty intersection for the "leaky" one.

### Gaps
- Release date of sklearn 1.9.1 and any 1.7–1.9 changes to group splitters were not checked (only the 1.6 changelog was read).
- The exact pre-1.6 `GroupKFold` balancing algorithm (greedy, largest-group-first) was not confirmed from the docs **[UNVERIFIED]**.
- The sklearn "Common pitfalls and recommended practices" page was not fetched.

---

## 6. Existing repos and papers comparing YASA, U-Sleep and simple baselines on Sleep-EDF with subject-disjoint splits; cross-model contamination summary

### Takeaway
No repo or paper was found that compares YASA, U-Sleep and simple baselines on Sleep-EDF under both leaky epoch-wise and subject-wise splits, so the planned audit appears novel in that specific framing.

The closest works are:
- Brookshire et al. 2024 (Frontiers in Neuroscience): segment-based holdout strongly overestimates performance on new subjects in translational DNN-EEG, but the datasets were Alzheimer's and seizure EEG, not sleep.
- SLEEPYLAND (Dei Rossi et al. 2025): a "fair evaluation" framework that releases weights trained with SEDF-SC/ST as out-of-domain. It explicitly flags published U-Sleep's SEDF results as in-domain.

### Cited Findings
- **Brookshire et al. 2024:** "performance on previously-unseen subjects is strongly overestimated when models are trained using segment-based holdout". This was shown on an Alzheimer's EEG dataset and a seizure EEG dataset. A literature review found most translational DNN-EEG studies use segment-based holdout — [Brookshire et al. 2024, Front Neurosci (PMC11099244)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/)
- **SLEEPYLAND (arXiv 2506.08574v2, Jun 2025) models and data:**
  - Trains and releases U-Sleep, DeepResNet and SleepTransformer, combined as the SOMNUS ensemble.
  - Lists SEDF-SC (153 PSGs) and SEDF-ST (44 PSGs) as **out-of-domain** for SOMNUS. Table 3 marks them **in-domain** (✓) for the published U-Sleep and SleepTransformer results.
  
  — [SLEEPYLAND arXiv](https://arxiv.org/html/2506.08574v2)
- **SLEEPYLAND results:**
  - SOMNUS macro-F1: SEDF-SC 74.0 ± 8.1, SEDF-ST 75.6 ± 7.6.
  - Published U-Sleep macro-F1 (in-domain): SEDF-SC 0.79, SEDF-ST 0.76.
  - YASA has no SEDF result in the paper (only CCSHS, MESA, DOD-H, DOD-O).
  
  — [SLEEPYLAND arXiv](https://arxiv.org/html/2506.08574v2)
- **SLEEPYLAND availability:** weights released via github.com/biomedical-signal-processing (the uSLEEPYLAND repo) and Docker Hub `bspsupsi/sleepyland`. Runs via Docker Compose on Linux, Mac or Windows. Trained on one NVIDIA L40S. License not stated in the portion read — [SLEEPYLAND arXiv](https://arxiv.org/html/2506.08574v2)
- **Bose et al. 2026:** evaluates DeepSleepNet, XSleepNet2 and others on SleepEDF-20 (20-fold) and SleepEDF-78 (10-fold), plus SHHS and iSLEEPS (stroke). It reports a cross-domain drop (SHHS→iSLEEPS 55.1% accuracy) but does **not** compare leaky versus subject-wise splits — [Bose et al., arXiv 2603.23582v2](https://arxiv.org/html/2603.23582v2)
- **MNE and braindecode official tutorials** both use subject-disjoint train/test (Alice/Bob, and subject 0/1) — [MNE sleep tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html); [braindecode Chambon example](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)

### Inferences

#### Cross-model contamination summary (for the audit's "threats to validity" table)

| Model | Pre-trained weights saw Sleep-EDF? | Evidence | Recommended use in the audit |
|---|---|---|---|
| Majority class | n/a (fit per fold) | — | Fit inside each fold; report under both splits |
| Band-power + LR | n/a (fit per fold) | — | Fit inside each fold via `Pipeline` |
| YASA pre-trained (LightGBM, 0.8.1) | **No.** NSRR only: CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS | [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/), [yasa_classifier](https://github.com/raphaelvallat/yasa_classifier) | Zero-shot, cross-montage reference (invariant to split) |
| YASA features + retrained clf | n/a | — | Main "realistic ML" arm under both splits |
| U-Sleep v1 (webserver/API) | **Yes.** SEDF-SC/ST ≥75% in training | [PMC8050216](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/), [SLEEPYLAND](https://arxiv.org/html/2506.08574v2) | Exclude from headline, or show as a contamination example |
| braindecode `USleep` / Chambon / AttnSleep from scratch | n/a (no weights shipped) | [HF braindecode/USleep](https://huggingface.co/braindecode/USleep/blob/main/README.md) | Train inside folds (GPU recommended for USleep) |
| braindecode Chambon HF checkpoint | **Yes** (trained on a Sleep-EDF subject) | [braindecode example](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html) | Do not reuse |
| SLEEPYLAND / SOMNUS weights | **No** (SEDF treated as OOD) | [SLEEPYLAND](https://arxiv.org/html/2506.08574v2) | Possible clean pre-trained deep reference if license permits **[license UNVERIFIED]** |

- **Novelty and framing:** the audit's novelty is the head-to-head of the *same* models under epoch-wise versus record-wise versus subject-wise splits on Sleep-EDF. Cite Brookshire et al. 2024 for the general phenomenon and SLEEPYLAND for the pre-training-overlap issue.

### Gaps
- A search snippet claimed a 2026 study found record-wise Macro-F1 about 9 pp higher than subject-wise on Sleep-EDF (and 7 pp on ISRUC). The paper the snippet seemed to point to (arXiv 2603.23582) does **not** contain this, and the true source was not identified. Treat it as unverified and do not cite it.
- No primary source quantifying the epoch-wise versus subject-wise gap specifically on Sleep-EDF was found.
- The SLEEPYLAND weights license and whether the Docker image runs CPU-only on Windows were not confirmed.
- No GitHub repo was found that benchmarks YASA pre-trained on Sleep-EDF.
