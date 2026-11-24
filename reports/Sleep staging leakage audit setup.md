# Split by Subject, Expect Ten-Point Inflation

The audit can start this week on a laptop. Use only the Sleep Cassette half of Sleep-EDF Expanded. Run the pre-registered headline analysis on all 78 subjects (Sleep-EDF-78: 153 recordings, 7.6 GB) and report Sleep-EDF-20 as a secondary slice. Sleep-EDF-20 is just subjects 0–19 of the same files, so one download and one preprocessing pass produce both. Preprocess the way the literature does: single-channel Fpz-Cz at 100 Hz, 30-s epochs, stages 3 and 4 merged into N3, movement and unscored epochs dropped, and wake trimmed to 30 minutes on each side of sleep. Check the pipeline by reproducing the published totals of **42,308 and 195,479 epochs**. Then run the same models under four split rules (random epoch-wise, blocked within-night, night-wise and subject-wise) and change nothing else between them. The core models are:

- a majority-class baseline;
- band power plus logistic regression;
- YASA features with a retrained LightGBM.

YASA's pre-trained classifier never saw Sleep-EDF, but it expects a different electrode setup. It is never trained during the audit, so its score cannot change with the split, which makes it a free pipeline check. **Pre-trained U-Sleep should not appear in the headline comparison.** It was trained on at least 75% of Sleep-EDF, its weights are not downloadable, and its hosted API refused connections on 24 November 2026. The only controlled, peer-reviewed measurement on this dataset puts the leak at about **0.10 macro-F1 and 0.11 κ**, and informal replications put it at 6–7 accuracy points. That is far smaller than the near-100% versus 55–85% gaps reported for EEG disease detection. The audit should therefore be designed and worded to detect a modest effect, with subjects (not folds or seeds) as the unit of analysis. Two points in the earlier portfolio research need correcting: arXiv 2611.08311 is a label-noise preprint, not a protocol reference, and U-Sleep is not a usable uncontaminated comparator.

## Sleep-EDF-78 carries the headline, and Sleep-EDF-20 comes free inside it

Both named subsets come from the Sleep Cassette (SC) study in PhysioNet's Sleep-EDF Expanded 1.0.0. Neither needs the Sleep Telemetry (ST) files.

- **Sleep-EDF-20 is SC subjects 0–19: 39 recordings and 1.96 GB of PSG.** PhysioNet still ships this list as `RECORDS-v1` from the 2013 release.
- **Sleep-EDF-78 is every SC file in the 2018 release: 78 subjects, 153 recordings, 7.60 GB.** Subject IDs run from 0 to 82, and IDs 39, 68, 69, 78 and 79 do not exist ([RECORDS-v1](https://physionet.org/files/sleep-edfx/1.0.0/RECORDS-v1); [RECORDS](https://physionet.org/files/sleep-edfx/1.0.0/RECORDS)).

Each subject has two consecutive day-night recordings, with three exceptions: subject 13 has night 1 only, and subjects 36 and 52 have night 2 only. Their other nights "were lost due to a failing cassette or laserdisk" ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)). The research team parsed all 153 SC hypnograms. Taking the 2018 SC files with subject ID below 20 reproduces the 2013 file list and its published epoch counts exactly ([sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)).

| | Sleep-EDF-20 | Sleep-EDF-78 |
|---|---|---|
| Subjects / recordings | 20 / 39 | 78 / 153 |
| PSG download | 1.96 GB | 7.60 GB |
| Age range (median) | 25–34 (28) | 25–101 (57) |
| Trimmed 30-s epochs | 42,308 | 195,479 |
| Class mix W / N1 / N2 / N3 / REM | 19.6 / 6.6 / 42.1 / 13.5 / 18.2% | 33.7 / 11.0 / 35.4 / 6.7 / 13.2% |
| Literature protocol | 20-fold, one subject per fold | 10-fold subject-wise |
| AttnSleep subject-wise accuracy | 84.5% | 81.3% |

Sources: epoch counts and AttnSleep accuracy from [AttnSleep](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf), with counts confirmed by [DeepSleepNet-Lite](https://arxiv.org/pdf/2108.10600); ages from [MNE age_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age_records.csv).

**What the audit measures favours SC-78 as the primary set.** The headline is a paired per-subject difference (honest minus leaky). The confidence interval around a mean over subjects narrows with the square root of their number, so 78 subjects give an interval about half as wide as 20. Varoquaux showed that cross-validation error bars are already around ±10% at 100 samples, and that the standard error across folds badly understates them ([Varoquaux 2018](https://arxiv.org/pdf/1706.07581)). Twenty subjects is a thin basis for detecting a modest effect.

Two further points favour SC-78:

- **Comparability.** The only controlled, peer-reviewed measurement of the gap on this dataset used SC-78 ([Köksal 2026](https://dergipark.org.tr/en/download/article-file/5192156)).
- **Cohort mix.** **All 20 SC-20 subjects are aged 25–34, while every subject with ID 20 or above is aged 50–101** ([MNE age_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age_records.csv)). A cohort of similar people plausibly understates how much a model gains from having already seen the test person. That is an inference, and comparing the two subsets tests it.

SC-20 has real advantages too. It underlies the most-cited leave-one-subject-out numbers, it downloads in a quarter of the time, and it keeps deep models feasible on a CPU. Because it nests inside SC-78, the student does not have to choose:

- pre-register SC-78 with 10-fold subject-wise CV as the primary analysis;
- pre-register SC-20 with 20-fold leave-one-subject-out as the secondary, literature-comparable analysis;
- run any stretch deep-model work on SC-20 only.

The age structure and the second study each set a trap:

- **Do not split by contiguous ID ranges.** No SC subject is aged 35–49, and every ID from 20 up is 50 or older. Training on IDs 0–59 and testing on 60–82 is therefore silently an age-shift experiment. Assign subjects to folds at random, or stratify by age band.
- **Leave ST out entirely.** Its 22 subjects had mild difficulty falling asleep and were recorded in hospital after temazepam or placebo, on different hardware with 100-Hz EMG and 10-s data records ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/); [sleep-telemetry files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/)). Humayun et al. found almost every Fpz-Cz feature distributed differently between SC and ST (p ≪ 0.001) ([arXiv 1904.10255](https://arxiv.org/pdf/1904.10255)). MNE's `temazepam.fetch_data` returns only the placebo nights in any case ([temazepam.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/temazepam.py)). Pooling ST would mix a domain shift into the leakage comparison.

### Preprocessing removes two-thirds of the epochs before any split

SC files are not single nights. **Each SC recording runs 17–24 hours (median 23 h) and starts in the afternoon. Sleep begins a median of 7.5 hours after the recording starts** ([sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)). PhysioNet describes them as "two PSGs of about 20 hours each" recorded at the subjects' homes ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)). Untrimmed, **68% of SC-20 epochs are wake**, so a classifier that always answers "wake" scores 68% accuracy under any split.

The field's standard convention comes from DeepSleepNet and was copied by TinySleepNet and AttnSleep ([DeepSleepNet prepare_physionet.py](https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py); [TinySleepNet prepare_sleepedf.py](https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py); [AttnSleep](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf)). It has three steps:

1. Map R&K stages 3 and 4 to N3.
2. Remove "Movement time" and "Sleep stage ?" epochs.
3. Keep only 60 epochs (30 minutes) before the first and after the last non-wake epoch of each recording.

This leaves **42,308 epochs for SC-20 (19.6% wake) and 195,479 for SC-78 (33.7% wake)**. The research team's own implementation matched every sleep-stage count exactly and matched wake to within 1 and 10 epochs. It also reproduced the MNE tutorial's 841 and 1,103 epochs for SC4001 and SC4011 ([MNE tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)). These totals make ideal assertions for unit tests.

Trimming uses only each recording's own labels, so it creates no leakage between splits. It does move the score. On single-channel EEG, XSleepNet2 scored **86.3% with ±30 minutes of wake versus 83.9% on in-bed data only** for SC-20, and 84.0% versus 80.3% on SC-78 ([XSleepNet](https://arxiv.org/abs/2007.05492)). Wake policy alone therefore swings accuracy by 2.4–3.7 points, about the size of the night-level leak the audit hopes to detect. Fix the policy in the pre-registration and test the alternatives only as an ablation.

The hypnograms have quirks the loader must handle:

- **150 of 153 SC hypnograms end in a "Sleep stage ?" block that runs past the end of the signal by up to 420 minutes.** Another 24 "?" blocks sit mid-recording ([sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)). DeepSleepNet's code includes an explicit "Remove extra index" step for labels that outrun the signal ([DeepSleepNet prepare_physionet.py](https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py)).
- Clip the labels to the PSG length, drop unscored epochs, then trim using the label array. Do not use the MNE tutorial's `annot[1]`/`annot[-2]` crop. It works only because SC hypnograms happen to start with a single wake block and end with a single "?" block.
- Braindecode's `crop_wake_mins=30` crops at the level of annotation blocks and does so before movement and unscored epochs are removed, so its counts can drift slightly from 42,308 and 195,479 ([braindecode sleep_physionet.py](https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/sleep_physionet.py)). Pick one implementation and test it against the published totals.
- **Read the subject and night from fixed character positions, not from hard-coded suffixes.** The subject is characters [3:5] and the night is character [5]. Pair each PSG with its hypnogram on the first seven characters. The seventh character varies (E, F or G), and the eighth character of a hypnogram name is one of 11 scorer codes, which can differ between a subject's two nights (SC4021EH, SC4022EJ) ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/); [sleep-cassette listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)).

**For the signal, use single-channel Fpz-Cz at 100 Hz, which gives 3,000 samples per 30-s epoch.** This is the default in DeepSleepNet, TinySleepNet, AttnSleep and DeepSleepNet-Lite ([DeepSleepNet-Lite](https://arxiv.org/pdf/2108.10600)). Pz-Oz and horizontal EOG share the 100-Hz rate and are easy optional extras, though adding EOG raised XSleepNet2's SC-20 accuracy by only 0.1 points ([XSleepNet](https://arxiv.org/abs/2007.05492)). Do not treat the "EMG submental" channel as EMG. In SC files it is an envelope (high-pass filtered, rectified, then low-pass filtered) sampled at 1 Hz ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)).

For downloading:

- **MNE.** MNE 1.13.2 (11 September 2026) provides `mne.datasets.sleep_physionet.age.fetch_data(subjects, recording=(1, 2), ...)`. It downloads each subject and night through pooch, checks each file against a bundled SHA1 hash, and caches under `PHYSIONET_SLEEP_PATH`, `MNE_DATA` or `~/mne_data` ([age.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age.py); [_utils.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/_utils.py)). Pass `on_missing="warn"` when requesting both nights of every subject. The default is actually `"raise"`, even though the docstring says "error".
- **Braindecode.** Braindecode 2.1.0's `SleepPhysionet` wraps the same fetcher and stores the subject and recording in each dataset's description, which makes grouped splits a one-liner ([PyPI braindecode](https://pypi.org/project/braindecode/)).
- **Fallback mirror.** MNE's base URL is an old "physiobank" path, and physionet.org returned HTTP 502 twice during this research. Keep the anonymous AWS mirror at `s3://physionet-open/sleep-edfx/1.0.0/` as a fallback; it served full files and byte ranges on 24 November 2026. Verify downloads against PhysioNet's `SHA256SUMS.txt` ([S3 RECORDS](https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/RECORDS); [SHA256SUMS.txt](https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt)).
- **Licence.** The data are under the Open Data Commons Attribution License v1.0, which requires citing Kemp et al. (2000) and the standard PhysioNet reference ([PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)).

## Four split arms change one thing at a time

Kapoor and Narayanan's taxonomy names what a random epoch-wise split does wrong. It is **"nonindependence between train and test samples" (their category L3.2)**, which in "the extreme (but unfortunately common) case" means train and test samples "come from the same people" ([Kapoor & Narayanan](https://arxiv.org/html/2207.07048v1)). In sleep data the same split makes a second, quieter error. Neighbouring 30-s epochs are near-duplicates, and scikit-learn's guide warns that shuffling such autocorrelated samples gives "an inflated validation score" because the model "will be tested on samples that are artificially similar (close in time) to training samples" ([scikit-learn CV guide](https://scikit-learn.org/stable/modules/cross_validation.html)). In translational EEG, Brookshire et al. found that splitting by segment "strongly overestimated" performance on unseen subjects, and that most such studies split that way ([Brookshire et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11099244/)).

The interesting question for this audit is not whether the leaky number is higher. It is which part of the leak supplies the inflation, and answering that needs more than two arms.

| Arm | What gets assigned to folds | What still leaks | Expected inflation over the honest arm |
|---|---|---|---|
| A. Random epoch-wise (leaky) | Individual 30-s epochs | Neighbouring epochs, the same night, the same subject | Largest; about 0.10 macro-F1 and κ in Köksal |
| B. Blocked within-night | Contiguous 30-minute blocks, with a buffer at block edges | The same night and the same subject, but not neighbouring epochs | Unmeasured; this arm is the novel part |
| C. Night-wise | Whole recordings | The test subject's other night | About 1–2 points (personalization studies) |
| D. Subject-wise (honest) | All of a subject's nights together | Nothing | Reference |

Arm B is the audit's one genuinely new idea. The research notes found no sleep-staging paper that separates leakage from neighbouring epochs from leakage from subject identity. Ecology's work on blocked cross-validation supports the method: Roberts et al. found block CV "nearly always more appropriate than random cross-validation" when observations depend on each other ([Roberts et al. 2017](https://www.biometrie.uni-freiburg.de/mitarbeiter/dormann/roberts-et-al-2017-ecography.pdf)). The buffer between blocks must be wider than the widest feature window. YASA smooths over ±3.75 minutes, so that means at least 8 epochs (an inference from YASA's design).

Arm C's expected size comes from personalization studies. On Sleep-EDF, fine-tuning a population model on the test subject's other night improved accuracy "on the order of two percentage points on average" ([Mikkelsen & De Vos](https://ar5iv.labs.arxiv.org/html/1801.02645)).

**Hold everything except the split rule fixed:** features, models, hyperparameters, seeds and the number of folds K. Using the same K in every arm keeps the training-set size the same.

- **SC-78: K = 10**, matching XSleepNet and SleepTransformer ([XSleepNet](https://arxiv.org/abs/2007.05492); [SleepTransformer](https://arxiv.org/abs/2105.11043)).
- **SC-20: K = 20**, which makes arm D the leave-one-subject-out protocol of DeepSleepNet and L-SeqSleepNet ([DeepSleepNet](https://arxiv.org/abs/1703.04046); [L-SeqSleepNet](https://arxiv.org/abs/2301.03441)).

In scikit-learn 1.9.1, arm D is `GroupKFold(n_splits=K, shuffle=True, random_state=seed)` with `groups=subject_id`. The `shuffle` option exists only from version 1.6 ([GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html); [sklearn 1.6 changelog](https://scikit-learn.org/stable/whats_new/v1.6.html)). `StratifiedGroupKFold` sounds better but behaves more and more like plain `GroupKFold` when groups are large, and a whole night is a large group ([StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)).

The published reference code agrees on the grouping key. AttnSleep groups files by `file_name[3:5]` ([AttnSleep util.py](https://github.com/emadeldeen24/AttnSleep/blob/main/utils/util.py)), and DeepSleepNet's loader holds out both nights of each test subject ([deepsleepnet data_loader.py](https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/data_loader.py)).

Four common mistakes are worth encoding as tests. They come from general scikit-learn behaviour that the notes did not confirm against a primary page:

- passing `cv=5` as an integer silently ignores the groups;
- `KFold` silently ignores a `groups` argument;
- a scaler fitted before the split leaks statistics from the test set;
- an inner `GridSearchCV` called without `groups` splits by epoch inside the training data.

**The cleanest way to avoid tuning on test data is not to tune.** Fix hyperparameters in advance from published values:

- logistic regression: scikit-learn's default regularization with `class_weight="balanced"`;
- LightGBM: YASA's published settings, which are 500 trees, depth 5, 90 leaves, a 60% feature fraction, and class weights of N1 2.2, N2 1, N3 1.2, REM 1.4 and wake 1 ([YASA eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)).

With fixed settings, no core model needs an inner tuning loop. Deep models still need early stopping, and that must use validation subjects held out inside each training fold: 4 subjects in XSleepNet and 7 in SleepTransformer on SC-78. DeepSleepNet had "no validation set" and tuned its architecture on the first fold of a different dataset; that is the pattern to avoid ([DeepSleepNet](https://arxiv.org/abs/1703.04046)). Del Pup et al. trained more than 100,000 EEG models and found schemes without nested validation "prone to data leakage" ([arXiv 2505.13021](https://arxiv.org/abs/2505.13021)).

The README also asks about normalization or feature selection fitted before the split (Kapoor's L1.2 and L1.3). Test this as a fifth condition: arm D with the scaler or feature selector fitted on all the data. Expect a small effect with only a few dozen features, and say plainly that it is unmeasured on sleep data. Two things are not leaks:

- Computing features for each epoch before splitting, because nothing is fitted.
- Per-night normalization such as YASA's robust z-score, because it uses only that night's own signal, which would also be available when scoring a new person.

Finally, score every arm per subject. In the leaky arm, collect out-of-fold predictions so that each epoch is predicted exactly once per seed, then compute the metric for each subject over that subject's epochs. Each subject then has one score per arm, model and seed, which is what makes a paired analysis possible.

## Pre-trained U-Sleep saw most of Sleep-EDF, while YASA saw none of it

"Contamination" here means a model's published weights were trained on some of the same Sleep-EDF recordings the audit tests on. It is a separate leakage channel from the split, and the README did not anticipate it. The model list sorts cleanly once each candidate is checked for prior exposure to Sleep-EDF.

| Model | Saw Sleep-EDF before the audit? | Compute | Role in the audit |
|---|---|---|---|
| Majority class (`DummyClassifier`) | No; fitted in each fold | Trivial | Floor; κ is 0 by construction |
| Relative band power (5 bands, Fpz-Cz) + logistic regression | No; fitted in each fold | Seconds on a laptop | Core single-epoch model |
| The same plus features from the 2 epochs on each side | No; fitted in each fold | Seconds | Tests whether context features amplify the leak |
| YASA 0.8.1 features + retrained LightGBM | No; fitted in each fold | Under 5 s per night to extract features | Core "realistic" model with smoothed features |
| YASA 0.8.1 pre-trained classifier | **No**; trained on NSRR cohorts only | Laptop | Clean zero-shot reference and split-invariance check |
| Braindecode `SleepStagerChambon2018`, trained from scratch | No; fitted in each fold | CPU, an estimated 1–2 h for 5 folds on SC-20 | Stretch deep model |
| Braindecode `USleep`, trained from scratch | No; no weights are shipped | GPU; no published timings | Stretch, SC-20 only |
| U-Sleep v1 pre-trained (webserver or API) | **Yes**; at least 75% of SEDF-SC and SEDF-ST | Weights not public | Exclude; use as a worked example of contamination |
| Braindecode's Chambon checkpoint on the Hugging Face Hub | **Yes**; trained on Sleep-EDF data | n/a | Do not use |
| SLEEPYLAND / SOMNUS | No; Sleep-EDF held out as out-of-domain | Docker | Optional clean deep reference; licence unverified |

**YASA is clean but expects a different setup.**

- **Version and install.** YASA 0.8.1 was released on 25 October 2026, needs Python 3.11 or newer, and installs its staging dependencies with `pip install "yasa[full]"`.
- **API changes.** Since version 0.7.0, `predict()` returns a Hypnogram object. Version 0.8.0 removed `predict_proba`; probabilities now sit in `hyp.proba` ([PyPI yasa](https://pypi.org/project/yasa/); [YASA changelog](https://yasa-sleep.org/changelog.html)).
- **Training data.** The classifier was trained on **3,163 nights from seven NSRR cohorts** (CCSHS, CFS, CHAT, HomePAP, MESA, MrOS and SHHS). Sleep-EDF appears nowhere in the paper or in the training repository ([YASA eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/); [yasa_classifier](https://github.com/raphaelvallat/yasa_classifier)).
- **Expected inputs.** YASA wants a central EEG referenced to the mastoid, an EOG from the left outer eye corner, raw chin EMG, and AASM scoring. Sleep-EDF offers Fpz-Cz, horizontal EOG, a 1-Hz EMG envelope and R&K scoring. Pass `eeg_name="EEG Fpz-Cz"` (the docs accept Fpz-referenced montages), pass `eog_name="EOG horizontal"`, and omit EMG. Do not filter or z-score the signal first, which the docs explicitly forbid ([YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)).
- **Input length.** Feed YASA the trimmed segment rather than the full 23-hour file. Its per-night z-scoring and time-in-night features assume a recording roughly one night long, and the docs warn the classifiers "were trained only on full nights". This is an inference; testing both inputs is cheap.
- **No published benchmark.** No YASA result on Sleep-EDF has been published, so the student's zero-shot number would be new. Expect it to fall below the paper's 87.5% median on its NSRR test nights.

Pre-trained YASA never trains during the audit, so its per-subject scores must be identical in every arm. That makes it an automatic test that the evaluation bookkeeping does not invent a gap.

**Retrained YASA features are where the leak should show most.** YASA smooths each feature with a 7.5-minute centred triangular window plus a 2-minute trailing window, then robust z-scores it per night ([YASA SleepStaging docs](https://yasa-sleep.org/generated/yasa.SleepStaging.html)). Each epoch's features therefore contain several minutes of its neighbours' signal. Under arm A, test epochs have near-copies in the training set, so the gap should be larger than for raw band power. That is a testable hypothesis.

The same paper is the source of the README's fragmentation question. Reviewers noted the algorithm "performs worse on N1 stage, older individuals and patients presenting sleep disorders (sleep fragmentation)", and the authors reported that removing the smoothing cost 3.6 macro-F1 points ([eLife peer review](https://doi.org/10.7554/eLife.70092.sa2); [YASA eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)).

**U-Sleep answers the README's free-compute question by making it moot.** Table 1 of the U-Sleep paper places **SEDF-SC (78 subjects, 153 recordings) and SEDF-ST (22 subjects, 44 recordings) in its "Train/Test" group**. Each dataset in that group was split into at least 75% training, at most 10% validation and at most 15% testing, which left only 23 SC recordings for testing ([Perslev et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)). An independent 2025 benchmark, SLEEPYLAND, also marks Sleep-EDF as in-domain for published U-Sleep results ([SLEEPYLAND](https://arxiv.org/html/2506.08574v2)). The paper does not say whether the model behind the webserver was retrained on all data, including those 23 test recordings.

Access is just as blocked:

- The U-Time repository is MIT-licensed but ships no weights. It points to sleep.ai.ku.dk instead ([U-Time](https://github.com/perslev/U-Time)).
- The hosted API requires an access request ([U-Sleep API bindings](https://github.com/perslev/U-Sleep-API-Python-Bindings)), and both of its hosts refused connections on 24 November 2026.
- The code requires TensorFlow 2.8 or newer plus `tensorflow-addons` ([requirements.txt](https://github.com/perslev/U-Time/blob/main/requirements.txt)), and `tensorflow-addons` is now archived ([tensorflow/addons](https://github.com/tensorflow/addons)).

Inference itself is cheap, "seconds on a laptop CPU" per night according to the paper, but the student cannot get the weights, and any score they produced would be contaminated.

That leaves two honest uses for U-Sleep:

- **Training from scratch inside the folds** with braindecode's PyTorch `USleep`. Braindecode ships no weights for it ([braindecode USleep](https://braindecode.org/stable/generated/braindecode.models.USleep.html); [HF braindecode/USleep](https://huggingface.co/braindecode/USleep/blob/main/README.md)). No measured training times on free GPUs were found; the research team's unverified estimate is 1–3 hours per fold on a free T4.
- **A short threats-to-validity section** showing how a widely used pre-trained model would quietly inflate a Sleep-EDF result.

The lighter deep models are more practical. Braindecode's Chambon example trained at **1.3–1.4 s per epoch on CPU** with two subjects ([braindecode Chambon example](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)). Scaling that to SC-20 suggests roughly 1–2 hours for 5 folds on a laptop, though that is an extrapolation, not a measurement. The original TinySleepNet and AttnSleep repositories pin TensorFlow 1.13 with Python 3.6, and PyTorch 1.4 with Python 3.7, and neither has been updated since 2023 ([TinySleepNet](https://github.com/akaraspt/tinysleepnet); [AttnSleep](https://github.com/emadeldeen24/AttnSleep)). Use braindecode's versions instead.

If the student wants a clean pre-trained deep reference, SLEEPYLAND's SOMNUS ensemble treated Sleep-EDF as out-of-domain and scored **74.0 ± 8.1 macro-F1 on SEDF-SC** ([SLEEPYLAND](https://arxiv.org/html/2506.08574v2)). Its licence and whether it runs CPU-only on Windows were not confirmed.

## Leakage should add about ten points, not the forty seen in disease detection

**The README's open question about the size of the gap now has an answer.** The cleanest measurement is Köksal (2026), a peer-reviewed study on SC-78 with Fpz-Cz, ±30-minute wake trimming, nights merged by subject, and a random forest on 17 handcrafted features ([Köksal 2026](https://dergipark.org.tr/en/download/article-file/5192156)). Median results across folds were:

| Split | Macro-F1 | κ |
|---|---|---|
| Random epoch-level 10-fold | **0.70** | **0.69** |
| Subject-wise 10-fold | **0.60** | **0.58** |
| Leave-one-subject-out | 0.54 | 0.58 |

The gap on a second dataset, ISRUC, was smaller: 0.71 versus 0.64 macro-F1. The leaky split also made the model overconfident, with an expected calibration error of 0.05 versus 0.02.

One terminology trap matters here. Köksal calls the random epoch split "record-wise", which is not the night-wise split of arm C. A claim seen in search results, that "record-wise" scores beat subject-wise by 9 points on Sleep-EDF, traces to this paper and means the epoch-wise leak.

Informal evidence points the same way. A non-peer-reviewed GitHub project used 10 SC subjects and a random forest with features from neighbouring epochs. It reached **0.892 accuracy with a random epoch split against 0.819 with subject-wise GroupKFold** ([shailesh-vishal19](https://github.com/shailesh-vishal19/sleep-stage-classification)). DeepSleepNet's 2017 table separated older Sleep-EDF studies by protocol: those whose training data included the test subjects' epochs reported 90.3–91.3% accuracy, while those with independent subjects reported 74.8–82.0%. Those studies also differed in features, channels and data subsets, so that comparison confounds leakage with everything else ([DeepSleepNet](https://arxiv.org/abs/1703.04046)).

Disease detection shows a very different picture. In the EEG study the README cites, splitting by sample pushed 10-fold accuracy for pathology tasks close to 100%, while leave-N-subjects-out medians were 85.4%, 58.1% and 54.7% for Parkinson's, Alzheimer's and a brain-computer-interface task ([Del Pup et al.](https://arxiv.org/html/2505.13021v1)).

Why the sleep gap is smaller is an inference, but a structural one:

- **The label varies within each subject.** In disease detection the label is constant per person, so recognising the person nearly gives away the answer. In sleep staging, recognising the person says little about which stage a given epoch is in.
- **Honest scores already sit near the human ceiling.** Human scorers agree with the majority score only **82.6% of the time overall and 63.0% on N1** ([Rosenberg & Van Hout 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3525994)), with κ about 0.76 under AASM rules ([Danker-Hopfe et al. 2009](https://pubmed.ncbi.nlm.nih.gov/19250176/)). That leaves little room for any leak to add.

**The working expectation is therefore:**

- **Feature models, arm A versus arm D:** about 5–10 accuracy points, 0.07–0.12 κ and 0.07–0.10 macro-F1.
- **Models with context features or smoothing:** a larger gap than single-epoch band power.
- **Arm C (night-wise):** about 1–2 points.

These are inferences from the studies above, not measured results for this pipeline. They are what the pre-registration should commit to.

Honest performance has well-documented reference points:

| Setting (Fpz-Cz, ±30 min wake, subject-wise unless noted) | Accuracy % | κ | Macro-F1 | N1 F1 |
|---|---|---|---|---|
| SC-20, deep models from scratch (SleepEEGNet to XSleepNet2 / L-SeqSleepNet) | 81.5–86.3 | 0.75–0.81 | 76.6–80.6 | 42.6–51.8 |
| SC-78, deep models from scratch (SleepEEGNet to XSleepNet2) | 80.0–84.0 | 0.73–0.78 | 73.6–77.9 | 40.4–49.9 |
| SC-78, TinySleepNet re-implementation (preprint) | 81.8 | n/r | 75.6 | n/r |
| SC-20, handcrafted features + sparse autoencoder (Tsinalis 2016) | 78.9 | n/r | 73.7 | 47.0 |
| SC-78, random forest on 17 features (fold medians) | n/r | 0.58 | 0.60 | n/r |
| 10 SC subjects, random forest + neighbouring-epoch context (not peer reviewed) | 81.9 | 0.75 | 76.7 | 44.4 |
| One training subject, band power + random forest (MNE tutorial) | 68.7 | n/r | 57 | 24 |
| Majority class (always N2) | 42.1 (SC-20) / 35.4 (SC-78) | 0 | n/r | 0 |

"n/r" means not reported by the source. Sources:

- SC-20 deep models: [DeepSleepNet](https://arxiv.org/abs/1703.04046); [XSleepNet](https://arxiv.org/abs/2007.05492); [L-SeqSleepNet](https://arxiv.org/abs/2301.03441)
- SC-78 deep models: [SleepTransformer](https://arxiv.org/abs/2105.11043)
- TinySleepNet re-implementation: [arXiv 2611.08311](https://arxiv.org/html/2611.08311v1)
- Random forest on 17 features: [Köksal 2026](https://dergipark.org.tr/en/download/article-file/5192156)
- 10-subject GitHub project: [shailesh-vishal19](https://github.com/shailesh-vishal19/sleep-stage-classification)
- One training subject: [MNE tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)
- Majority-class shares: [AttnSleep](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf)

**N1 is the weak stage everywhere, with F1 of 40–52 across honest deep models.** For a band-power model with many training subjects, the inferred target is **75–82% accuracy and κ of about 0.65–0.75 on SC-20, somewhat lower on SC-78**. Köksal's κ of 0.58 shows how far a sparse feature set, balanced class weights and fold medians can pull an honest baseline down.

Two cautions for reading results:

- **Treat very high scores as a warning.** A from-scratch single-channel model scoring above about **87% accuracy or κ 0.83** on Sleep-EDF suggests leakage or a non-standard subset (an inference from the table above). Pre-training on a large external dataset is the exception: L-SeqSleepNet reached 88.6% after pre-training on SHHS ([L-SeqSleepNet](https://arxiv.org/abs/2301.03441)).
- **Do not mix aggregation methods.** Metrics pooled across all folds (DeepSleepNet, XSleepNet) and per-subject means or medians (YASA, Köksal) are not interchangeable. Compare like with like.

## Subjects are the unit of evidence, and the README is the registry

**Treat the subject as the independent unit.** For each subject, model, arm and seed, compute the metric, then:

1. Average over seeds within each subject.
2. Take the paired difference d = honest − leaky for each subject.
3. Report the mean of d with a **subject-level bootstrap 95% percentile interval**: resample subjects with replacement 10,000 times, using the same resampled subjects for both arms.
4. Add the share of subjects whose honest score is lower, plus an optional Wilcoxon signed-rank test on d.

Never run t-tests across folds. Bates, Hastie and Tibshirani showed that naive cross-validation intervals are too narrow because fold estimates are correlated; in one of their examples, nominal 90% intervals missed 31% of the time ([Bates et al.](https://arxiv.org/abs/2104.00673)). If fold scores must be compared, use the Nadeau–Bengio corrected test, which is implemented in correctipy ([correctipy](https://github.com/hendersontrent/correctipy)). Recent patient-level bootstrap papers make the same point: the number of independent test units "is set by patients, not by repeated training seeds" ([arXiv 2609.21197](https://arxiv.org/pdf/2609.21197)). Seeds are reported as a stability range, not as extra samples. The limitations section should state that these intervals capture uncertainty about which subjects were tested, not variation in the training set, which only nested CV would capture.

**Make per-subject Cohen's κ the primary metric.** This is a judgment, made for these reasons:

- **Per-subject macro-F1 is fragile here.** N3 is only 6.7% of SC-78 epochs, so some older subjects will have very few N3 epochs, and their macro-F1 becomes unstable or undefined.
- **Per-subject κ is well estimated.** Even the smallest subject contributes 824 trimmed epochs ([sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)).

Report as secondary outcomes:

- pooled accuracy, κ and macro-F1, for comparison with DeepSleepNet and XSleepNet;
- per-class F1, with attention to N1;
- row-normalized confusion matrices;
- expected calibration error, because overconfidence is Köksal's second signature of leakage.

This follows Menghini et al.'s guidance for sleep-staging validation: per-subject metrics averaged across subjects, with SD and bootstrap confidence intervals ([Menghini et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7879416)).

**For the fragmentation analysis, use YASA's transition definition and extend it.** YASA calls an epoch "around a transition" if a human-scored stage change falls within the 3 minutes centred on it. Its accuracy was **69.15% around transitions against 94.08% in stable periods**, and the share of N1 and the share of transitions were the two strongest predictors of a night's accuracy, together about 40% of relative importance ([YASA eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)). Phan et al. instead binned errors by distance to the nearest transition and found more than 60% of shared errors within 4 epochs of one ([arXiv 2108.10211](https://arxiv.org/abs/2108.10211)).

Combining the two gives an analysis plan:

- **Epoch level:** stratify every model and arm by distance to the nearest stage change: at most 1 epoch, at most 3, at most 4, or stable.
- **Night level:** summarise each night by its share of transitions, share of N1, number of awakenings, wake after sleep onset, and the subject's age, and compare the lowest and highest thirds.

The new hypothesis is that the leaky arm's gain concentrates in stable epochs, where neighbours in the training set share the label. If that holds, the headline gap overstates the harm of leakage in fragmented sleep, where models are weakest anyway.

**Use Hofman et al.'s two-phase template for the pre-registration** ([Hofman et al. 2023](https://arxiv.org/html/2311.18807)):

- **Phase A, before training:** research question, outcome, features, how training and test data are built, transformations, metrics and baselines.
- **Phase B, before testing:** methods, training details, whether the test data were accessed, secondary analyses and plan changes.

The paper's rule is that the test set "is meant to be used once and only once". Write the section in the README, commit it, and create an annotated tag `prereg-v1` before any full-data model run. Optionally add a Zenodo or OSF snapshot as a timestamp that does not depend on git.

The section should state:

- **Hypotheses:**
  - H1: every trained model scores lower in arm D than in arm A.
  - H2: the gap shrinks from A to B to C to D.
  - H3: the gap is larger for smoothed or context features.
  - H4: the gain concentrates in stable epochs.
- **Expected sizes:** the gap magnitudes from the previous section.
- **Primary metric and level:** per-subject κ.
- **Dataset and split rules:** the datasets, K and five fixed seeds.
- **Fixed hyperparameters.**
- **Success criteria:** H1 holds for a model if its 95% interval for the mean d excludes zero. The headline claim stands if that holds for at least two of the three trained core models. If every interval includes zero, report a null result rather than dropping it.
- **Pipeline-failure criteria:** an honest fold shares a subject with its test set; pre-trained YASA scores differ between arms; or epoch counts drift from 42,308 or 195,479.
- **Test-access statement:** development used only a four-subject fixture, plus label-only checks on all hypnograms.
- **Deviations log.**

**Three documents make the audit legible to reviewers:**

- **A REFORMS checklist table** (32 items in 8 sections) ([REFORMS](https://reforms.cs.princeton.edu/)).
- **A Kapoor–Narayanan "model info sheet"** arguing clean train-test separation, legitimate features and the right test distribution ([reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/)).
- **A short data card** covering the source, licence, subset, labels and age bias.

Borrow from TRIPOD+AI only where it fits: data sources, uncertainty and open-science items ([TRIPOD+AI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11025451/)). Borrow CLAIM's vocabulary: say "held-out subjects" and "reference standard", and avoid "external validation", since no external dataset is used ([CLAIM 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11304031)).

For the headline figure, draw one panel per model with each subject's two scores joined by a faint line. Add the bold mean with its bootstrap interval and a dashed line for the majority-class baseline. Follow it with a forest plot of the paired gaps for each model. The full per-subject × seed × arm results go in `results/` as CSV files.

## Free GitHub runners and a laptop cover the whole core

**Pin the environment with uv.** Use uv with a committed `uv.lock`; its universal resolution is "portable across platforms", and `--locked` errors if the lockfile is stale ([uv resolution](https://docs.astral.sh/uv/concepts/resolution/); [uv sync](https://docs.astral.sh/uv/concepts/projects/sync/)). Give the repo one entry point: `uv run python -m sleepaudit all --config configs/full.yaml`, with a `configs/ci.yaml` twin for the small CI subset.

Inside `src/sleepaudit/`, the main modules are:

- a downloader that reads a manifest, verifies SHA256 against committed `SHA256SUMS.txt` lines, retries, and falls back to the S3 mirror;
- an epoching module;
- `splits.py`, the only place splits are made;
- model wrappers;
- per-subject evaluation;
- the bootstrap;
- the report generator.

Make the data directory configurable through an environment variable and keep it outside the repo. Cache preprocessed arrays under a hash of the raw-file checksums and the preprocessing config. Write a `meta.json` for every run with the commit, a flag for uncommitted changes, the lockfile hash, package versions, seeds and the manifest hash.

**Write tests that would catch a leak:**

- no subject appears in both the train and test sides of any honest fold, including inner validation;
- the leaky arm does share subjects between train and test;
- every epoch is tested exactly once per seed;
- grouping uses the subject, not the recording, checked on a fixture with two nights from one subject;
- a "spy" transformer records which rows it was fitted on;
- the same seed produces identical folds;
- a corrupted file fails its checksum;
- pre-trained YASA scores are identical across arms.

**A whole class of tests needs no signal data at all.** All 153 SC hypnograms total 0.68 MB, and each PSG's length can be read from its 4-KB EDF header ([sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)). The label parser, label clipping, trimming and the 42,308 / 195,479 epoch-count assertions can therefore run in seconds on every push.

**GitHub Actions is free for public repositories on standard runners** ([GitHub billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)). The relevant limits:

- `ubuntu-latest` provides 4 vCPU, 16 GB of RAM and a 14 GB SSD ([GitHub runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners));
- jobs stop after 6 hours, and the cache holds 10 GB per repository ([GitHub limits](https://docs.github.com/en/actions/reference/limits));
- cache entries are evicted after 7 days without access, and a pull request can read only its own branch's cache or the default or base branch's ([GitHub caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)).

A workable setup has two jobs:

- **A unit-test job** using `astral-sh/setup-uv` pinned to a commit with `enable-cache: true`, then `uv sync --locked` ([uv GitHub guide](https://docs.astral.sh/uv/guides/integration/github/)).
- **A smoke job** that restores a cached EDF fixture keyed on the manifest hash and runs `sleepaudit all` with 2 folds and 1–2 seeds, under `timeout-minutes: 30`.

Use subjects 0–3 with both nights, about 400 MB, as the fixture. It exercises the subject-versus-night grouping that a two-subject, one-night fixture (about 99.5 MB) cannot. Add a weekly scheduled run so the cache is not evicted, upload `results/` as short-lived artifacts, and never upload EDF files.

The full SC-78 run belongs on the laptop. Its 7.6 GB of raw data plus derived arrays and the environment is too tight for a 14 GB runner. An optional manually triggered workflow for SC-20 (1.96 GB) would fit within the cache and time limits for the classical models; that is an inference, not tested.

**Compute is not the bottleneck for the core.** YASA extracts features for a night in under 5 seconds on a basic laptop ([YASA eLife](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)), so all 153 nights take minutes. Logistic regression and LightGBM on 195,479 rows train in seconds to minutes. Free GPUs matter only for the deep stretch models, and their quotas are soft:

- Colab publishes no free-tier limits ([Colab FAQ](https://research.google.com/colaboratory/faq.html)).
- Kaggle's commonly quoted figure of about 30 GPU-hours per week could not be confirmed from its official page ([Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage)).

Development happens on Windows, so build in a few safeguards (general engineering practice, not checked against sources):

- use `pathlib` for all paths;
- add a `.gitattributes` file marking `*.edf` as binary so line-ending conversion cannot corrupt EDFs or checksum files;
- put an `if __name__ == "__main__":` guard around multiprocessing code;
- compare floating-point results across operating systems with a tolerance.

Whether YASA's dependencies all publish Windows wheels was not verified. The U-Time/TensorFlow stack is the part most likely to break on Windows, which is one more reason to leave it out.

## Seven claims from the earlier research need revising

The earlier portfolio research recommended this audit correctly, but several supporting details do not hold up against the new notes.

| Earlier material said | The new notes show | What changes |
|---|---|---|
| arXiv 2611.08311 is the example subject-wise protocol for Sleep-EDF (20-fold on SC-20, 10-fold on SC-78, validation inside the folds) | It is an LLM label-noise preprint (Salari & Kshirsagar, cs.CL, 5 Nov 2026). It runs one "adapted subject-wise 20-fold" CV on SC-78 only, cites no protocol source, and does not confirm that each subject's nights stay together ([arXiv 2611.08311](https://arxiv.org/html/2611.08311v1)) | Cite DeepSleepNet, XSleepNet and L-SeqSleepNet for SC-20, and XSleepNet and SleepTransformer for SC-78. Keep 2611.08311 only as a TinySleepNet reference point (81.8% accuracy, macro-F1 75.6) |
| U-Sleep is an optional comparator with public code; the only open question is free compute | Pre-trained U-Sleep was trained on at least 75% of SEDF-SC and SEDF-ST. Its weights are not distributed, the API is access-gated and was unreachable, and the code depends on archived `tensorflow-addons` ([Perslev et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8050216/)) | Drop it from the headline. Document it as a contamination example, and train braindecode `USleep` from scratch only as a stretch goal |
| The README's background frames the gap through EEG disease detection, where it was "large" | For sleep staging the gap measures about 0.10 macro-F1 and 0.11 κ, or 6–7 accuracy points ([Köksal 2026](https://dergipark.org.tr/en/download/article-file/5192156)) | Pre-register a modest effect and use 78 subjects to detect it |
| Sleep-EDF holds 197 "whole-night" recordings | SC recordings are 17–24-hour day-night recordings that are 68% wake untrimmed, and ST is a separate drug study | Wake trimming is mandatory, and only SC is used |
| The two-night structure and the filename grouping key were unverified because PhysioNet returned errors | Confirmed: 75 of 78 SC subjects have two nights, the subject is `name[3:5]`, and MNE's fetcher verifies SHA1 hashes ([RECORDS](https://physionet.org/files/sleep-edfx/1.0.0/RECORDS); [age.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age.py)) | Group by subject, and include two-night subjects in the CI fixture |
| YASA is the comparator whose weakness on fragmented sleep the audit analyses | Pre-trained YASA is clean but built for a different setup, and its score cannot change with the split. The leakage test applies to classifiers retrained on YASA features | YASA plays two roles, and the fragmentation analysis covers every model |
| The plan compares epoch-wise K-fold, leave-subject-out and nested validation | Night-wise and blocked within-night arms break the leak into its parts, and fixed published hyperparameters remove the need for nested tuning in the core models | Four arms plus a pipeline-leak condition |

## Ten steps from empty repo to tagged results

The order below front-loads everything that can be tested without model results, so the pre-registration is frozen before any full-data model runs. Week numbers are estimates in line with the README's unmeasured 2–4-week timeline.

| Step | Week (est.) | Build | Done when |
|---|---|---|---|
| 1 | 1 | Repo skeleton: uv project and lockfile, `src/sleepaudit`, `configs/ci.yaml` and `configs/full.yaml`, `.gitattributes`, ruff, pytest, and an Actions unit-test job | CI passes on Linux, plus a Windows leg for unit tests only |
| 2 | 1 | Label-only layer: fetch all 153 SC hypnograms (0.68 MB) and EDF headers; filename parser; label map; clip labels to PSG length; ±30-minute trim | Tests assert 42,308 (±1) and 195,479 (±10) epochs, 841 and 1,103 for SC4001 and SC4011, and 75 two-night subjects |
| 3 | 1 | Data layer: manifest with committed SHA256 lines, retrying downloader with S3 fallback, MNE/pooch cache; a CI fixture of subjects 0–3 with both nights, cached by manifest hash | The corrupted-file test fails as intended and the smoke job restores its cache |
| 4 | 1–2 | `splits.py`: arms A–D plus the pipeline-leak condition, with equal K and fixed seeds | Overlap tests pass: none in arm D (including inner splits), present in arm A, and each epoch tested once per seed |
| 5 | 2 | Majority-class and band-power + logistic regression models (with and without context features) as scikit-learn Pipelines; per-subject out-of-fold scoring | `sleepaudit all --config configs/ci.yaml` finishes in CI within 30 minutes |
| 6 | 2 | YASA: per-night features on the trimmed segment, then LightGBM with the paper's settings; pre-trained zero-shot run with Fpz-Cz and horizontal EOG, no EMG | Pre-trained YASA's per-subject scores are identical in every arm |
| 7 | 2 | Statistics and reporting: per-subject κ, seed averaging, paired subject-level bootstrap, transition strata, slopegraph and forest plot | Bootstrap edge-case tests pass and figures render from the fixture |
| 8 | 3 | Pre-registration (Hofman Phases A and B), model info sheet, REFORMS table and data card; tag `prereg-v1` | The tag is pushed before any full-data model run |
| 9 | 3 | Full local run: SC-78 primary (K = 10) and SC-20 secondary (K = 20), 5 seeds, with `meta.json` | Results CSVs and figures are committed from a commit descending from the tag |
| 10 | 3–4 | Error analysis by transition distance, fragmentation and age; limitations; deviations log; optional stretch (Chambon on SC-20 by CPU, `USleep` on a free GPU) | The README results table and limitations section are complete |

Steps 2 to 4 carry the most risk and the least visible output. That is why they come first: every later number depends on correct epoch counts and an honest split, and both can be proven without running a single model.

## Conclusion

Confirming that a random epoch split inflates Sleep-EDF scores adds little, because Köksal has already shown it on the same subset. The contribution available to a student is to break the inflation into parts: neighbouring epochs, the same night, and the same person. The blocked within-night arm, which no published sleep-staging study has run, does that. The same design tests a second idea: how much a leaky evaluation flatters a model depends on feature design as well as split choice. A small gap for single-epoch band power next to a large one for YASA's smoothed features would show that the features themselves carry leakage. That would be a reportable finding even if the headline gap turns out modest.

The research also widened what "leakage" has to cover. The README worried about splits; the more surprising risk turned out to be pre-trained weights that have already seen the test data. U-Sleep is the clearest example, and the same problem would affect any Sleep-EDF result built on a public checkpoint. Because the honest scores sit at human inter-rater agreement and the expected gap is modest, the realistic danger for this audit is a noisy, uninterpretable null result, not a dramatic one. A 78-subject paired design, a gap size committed to in advance, and a split-invariant control are what turn a small effect into a credible result.
