# Data leakage, evaluation protocols, metrics and expected performance in automatic sleep staging (Sleep-EDF focus)

Status tags used below: **[PR]** = peer-reviewed journal/conference; **[PP]** = preprint (arXiv/medRxiv, not peer-reviewed); **[NR]** = not reviewed (GitHub README/PR); **[arXiv copy of PR]** = arXiv version of a paper whose first page says it was published in a peer-reviewed venue. Research cut-off: 24 Nov 2026. Numbers taken from PDF tables were extracted with pdftotext and checked against the papers' running text where possible.

## Q1. Has anyone measured epoch-wise (random) vs record-wise vs subject-wise splits for sleep staging, and what gaps were reported? Which evaluation practices have been criticized?

### Takeaway
Yes, but only a few studies did it, and none is a large dedicated benchmark. The cleanest controlled measurement is Köksal (2026, peer-reviewed, Sleep-EDF-78 + ISRUC, random forest on handcrafted features). Its random epoch-level 10-fold split ("record-wise", RW) beat subject-wise 10-fold by about 0.10 macro-F1 and 0.11 kappa on Sleep-EDF. Older cross-paper comparisons (DeepSleepNet, 2017) show "non-independent" Sleep-EDF studies at about 90–91% accuracy versus 75–82% for subject-independent ones. Non-peer-reviewed GitHub demos report about 6–7 accuracy points. Two other protocol choices move numbers by amounts similar to a mild leak: wake trimming (±30 min of wake vs in-bed only shifts accuracy by 2.4–3.7 points in XSleepNet), and putting a subject's second night in training (personalization adds about 2 points).

### Cited Findings
**Controlled, same-pipeline measurements**
- Köksal, "Cross-Validation and Normalization in EEG Sleep Staging: Impacts on Generalization, Calibration, and Clinical Validity", Duzce Univ. J. Sci. Technol. 14(1):72–85, online 21 Jan 2026 (received 28 Aug 2025, revised 16 Oct 2025, accepted 23 Oct 2025 → journal review process) **[PR]**. Setup: Sleep-EDF Expanded Sleep Cassette with 78 subjects and 153 recordings; Fpz-Cz; long wake cut so that 30 min before/after sleep remain. Recordings from multiple nights were merged by subject ID, so the SW/LOSO splits keep both nights together. 40 handcrafted features were extracted per epoch, of which 17 were kept for SleepEDF. Classifier: Random Forest (300 trees, class_weight='balanced', no hyperparameter tuning). — [Köksal 2026 article page](https://dergipark.org.tr/en/pub/dubited/article/1773372); [full-text PDF](https://dergipark.org.tr/en/download/article-file/5192156)
- In that paper, "RW" is explicitly an **epoch-level random split**: "Without taking subject identity into account, RW divides epochs at random, which causes data leakage and inflated accuracy (optimism bias)". Protocols compared: D1 = RW 10-fold, D2 = subject-wise (SW) 10-fold, D4 = LOSO, plus normalization variants. — [Köksal 2026 PDF](https://dergipark.org.tr/en/download/article-file/5192156)
- Table 3 gives median fold-level results for Sleep-EDF:
  - D1 (RW, random epochs): macro-F1 0.70, κ 0.69, ECE 0.05, LogLoss 0.59
  - D2 (SW 10-fold): macro-F1 0.60, κ 0.58, ECE 0.02, LogLoss 0.76
  - D4 (LOSO): macro-F1 0.54, κ 0.58
  - ISRUC, D1 vs D2 vs D4: macro-F1 0.71 / 0.64 / 0.62; κ 0.68 / 0.58 / 0.61
  - The abstract describes the subject-wise drop as "9 and 7 percentage points" of macro-F1 (SleepEDF, ISRUC); the table's rounded medians imply 10 and 7.
  - The authors add that RW "impaired calibration through overconfident predictions".
  - Accuracy is not given in Table 3.
  — [Köksal 2026 PDF](https://dergipark.org.tr/en/download/article-file/5192156)
- A GitHub learning project **[NR]** used Sleep-EDF SC with 10 subjects (night 1 only, 9,913 epochs), wake cropped to 30 min, and Random Forest on band-power/Hjorth/entropy features with ±2-epoch context (within subject). Subject-wise GroupKFold gave accuracy 0.819, macro-F1 0.767, κ 0.752. A random epoch split gave 0.892 accuracy (accuracy only reported). N1 F1 was 0.444. A shuffled-label control was also run. — [shailesh-vishal19/sleep-stage-classification](https://github.com/shailesh-vishal19/sleep-stage-classification)
- A benchmark pull request **[NR]**, seen only as a search snippet and not verified, reports 0.775 subject-wise vs 0.832 random split on a 6-subject Sleep-EDF subset. It attributes the gap to autocorrelated consecutive epochs and shared subject identity. — [brain-researcher-benchmark PR #141](https://github.com/brain-researcher/brain-researcher-benchmark/pull/141)

**Cross-paper (confounded) evidence on Sleep-EDF**
- DeepSleepNet (Supratak et al., IEEE TNSRE 2017) **[arXiv copy of PR]** split prior Sleep-EDF work into "non-independent" methods, which "included parts of the test subjects' epochs in the training data", and "independent" ones. They state that "the practical evaluation scheme should not include any epochs from the test subjects. Also, it has been shown that the non-independent scheme resulted in an improvement of the performance [7]" (ref [7] = Tsinalis et al. 2016). Table IV numbers:
  - Non-independent: Hsu et al. (Fpz-Cz, 960 test epochs) 90.3% acc / MF1 76.5; Sharma et al. (Pz-Oz) 91.3% / 77 / κ 0.86; Hassan & Subasi (Pz-Oz) 90.8% / 80 / κ 0.85.
  - Independent: Tsinalis 2016 handcrafted/SSAE (Fpz-Cz) 78.9% / 73.7; Tsinalis CNN 74.8% / 69.8; DeepSleepNet Fpz-Cz 82.0% / 76.9 / κ 0.76; Pz-Oz 79.8% / 73.1 / κ 0.72.
  — [DeepSleepNet arXiv 1703.04046](https://arxiv.org/abs/1703.04046)
- Tsinalis et al. (Ann Biomed Eng 2016) **[PR]** criticized reliance on a single train/test split. They used cross-validation across subjects, "ensuring independence of training and testing data", and reached 78% overall accuracy (range 75–80%). — [Tsinalis 2016, PMC4837220](https://pmc.ncbi.nlm.nih.gov/articles/PMC4837220)
- Imtiaz & Rodriguez-Villegas (EMBC 2014) **[PR]** report in their abstract that "selecting different training and test subjects from the same database can alter the final performance results" (search-snippet level; full text not read). — [Imtiaz & Rodriguez-Villegas 2014](https://spiral.imperial.ac.uk/entities/publication/d5eb86ff-2970-4f54-9a9e-3539d5193653)
- Their 2015 toolbox paper **[PR]** says Sleep-EDF recordings "have been used in an inconsistent fashion". The toolbox standardizes data selection, hypnogram conversion and metrics. — [Imtiaz & Rodriguez-Villegas 2015 (Imperial repository)](https://spiral.imperial.ac.uk/handle/10044/1/29344); [sleep-edfx-toolbox on GitHub](https://github.com/anasimtiaz/sleep-edfx-toolbox)
- Phan & Mikkelsen's review (Physiol. Meas. 2022) **[arXiv copy of PR]** warns that its own Sleep-EDF/SHHS performance table "should not be used out-of-context", because "discrepancies in evaluation setup (e.g., data subsets, the number of channels, etc.) and modelling … renders such a comparison meaningless." — [Phan & Mikkelsen review, arXiv 2111.08446](https://arxiv.org/abs/2111.08446)
- L-SeqSleepNet's results table likewise flags prior Sleep-EDF results as "not compatible for a direct comparison … due to the discrepancies in data split, the number of channel used, the number of subject used". — [L-SeqSleepNet, arXiv 2301.03441](https://arxiv.org/abs/2301.03441)

**Two nights of one subject in different folds (record-wise leakage across nights)**
- Mikkelsen & De Vos, "Personalizing deep learning models for automatic sleep staging" **[PP; arXiv]**, used Sleep-EDF (19 subjects × 2 nights + 1 single-night subject; Fpz-Cz + EOG). Fine-tuning a population model on the test subject's *other* night improved second-night accuracy by "on the order of two percentage points on average". Gains were larger for subjects whose baseline accuracy was below about 0.80 (p = 0.0076 across per-subject averages). A model trained only on the same subject's first night was "significantly worse". — [arXiv 1801.02645 (ar5iv)](https://ar5iv.labs.arxiv.org/html/1801.02645)
- The Phan & Mikkelsen review summarizes this as: "the differences between algorithms trained on data just from the same subject, and data from both the same subject and many other subjects were minimal", and says subject-specific information raises performance. — [arXiv 2111.08446](https://arxiv.org/abs/2111.08446)
- Phan et al. (Physiol Meas 2020) report personalized single-night SeqSleepNet on Sleep-EDF-78 (in-bed data) at 79.6% accuracy, κ 0.706, MF1 73.0. XSleepNet lists the non-personalized in-bed Sleep-EDF-78 baselines in the same block: SeqSleepNet 79.0 / 0.708 / 74.6 and XSleepNet2 80.3 / 0.727 / 76.4. — [Phan et al. arXiv 2004.11349](https://arxiv.org/abs/2004.11349); [XSleepNet arXiv 2007.05492](https://arxiv.org/abs/2007.05492)
- Good practice now explicitly keeps nights together:
  - RobustSleepNet "split the data subject-wise to avoid data leakage", putting all nights of a subject in the same fold (search snippet). — [RobustSleepNet arXiv 2101.02452](https://arxiv.org/pdf/2101.02452)
  - Köksal merges nights by subject ID. — [Köksal 2026 PDF](https://dergipark.org.tr/en/download/article-file/5192156)
  - Sleep-EDF SC subjects have two nights except subject 13 in SC-20, and subjects 13, 36 and 52 in SC-78. — [XSleepNet](https://arxiv.org/abs/2007.05492); [SleepTransformer arXiv 2105.11043](https://arxiv.org/abs/2105.11043)

**Wake trimming changes the numbers (same model, same CV)**
- XSleepNet (IEEE TPAMI) **[arXiv copy of PR]** ran both conventions: "(1) only in-bed parts of the recordings were used as recommended in [54],[55]" (the Imtiaz & Rodriguez-Villegas 2014/2015 papers), and "(2) 30 minutes of data before and after in-bed parts", following DeepSleepNet. Single-channel EEG results (accuracy / κ / MF1):
  - SleepEDF-20: XSleepNet2 86.3 / 0.813 / 80.6 with ±30 min vs 83.9 / 0.771 / 78.7 in-bed only. SeqSleepNet baseline 85.2 / 0.798 / 78.4 vs 82.2 / 0.746 / 74.1.
  - SleepEDF-78: XSleepNet2 84.0 / 0.778 / 77.9 vs 80.3 / 0.727 / 76.4. SeqSleepNet 82.6 / 0.760 / 76.4 vs 79.0 / 0.708 / 74.6.
  — [XSleepNet arXiv 2007.05492](https://arxiv.org/abs/2007.05492)
- DeepSleepNet started the ±30 min convention: "We only included 30 minutes of such periods just before and after the sleep periods". — [arXiv 1703.04046](https://arxiv.org/abs/1703.04046)
- During YASA's revision, the authors stopped cropping nights to 15 minutes around sleep. — [YASA eLife, PMC8516415 (incl. author response)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)

**Tuning on test data / no validation set**
- DeepSleepNet: "There was no early stopping as there was no validation set in our evaluation scheme". Architecture, weight decay, learning rates and batch/sequence lengths were designed "with the first fold of the 31-fold cross-validation with the MASS dataset". Metrics were computed after combining "the predicted sleep stages from all folds". — [arXiv 1703.04046](https://arxiv.org/abs/1703.04046)
- Later protocols hold out validation *subjects* inside each training fold:
  - XSleepNet: 4 subjects on SleepEDF-78. — [arXiv 2007.05492](https://arxiv.org/abs/2007.05492)
  - SleepTransformer: 7 subjects on SleepEDF-78. — [arXiv 2105.11043](https://arxiv.org/abs/2105.11043)
  - L-SeqSleepNet: held-out validation subjects with LOSO, repeated 5 times. — [arXiv 2301.03441](https://arxiv.org/abs/2301.03441)

**General (non-sleep) framing**
- Saeb et al. (GigaScience 2017) **[PR]**: record-wise CV can overestimate the accuracy of clinical ML. Subject-wise CV mirrors "diagnosing newly recruited subjects". — [Saeb et al. 2017, PMC5441397](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5441397/)
- Counterpoint: Little et al. (GigaScience 2017) **[PR]** argue that subject-wise CV is not always valid, and that part of the discrepancy may come from subject-wise CV overestimating error. — [Little et al. 2017, PMC5441396](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5441396/)

### Inferences
- What to expect for a classical feature + LR/RF pipeline on Sleep-EDF (my inference from Köksal and the two GitHub demos):
  - With a random epoch-wise split, accuracy should rise by roughly **5–10 points**, κ by roughly **0.07–0.12**, and macro-F1 by roughly **0.07–0.10** compared with a subject-wise split.
  - The gap is probably larger when neighbour-epoch context features are used: in a random split, a test epoch's context window overlaps training epochs' raw data.
- A record-wise split that keeps each night intact but can put one subject's two nights in different folds should leak much less, probably about 1–2 accuracy points. This is inferred from the personalization results (about +2 points from fine-tuning on the subject's other night).
  - The audit could add this as a third arm, "night-wise", alongside random epoch-wise and subject-wise.
- The wake-trimming policy (in-bed only vs ±30 min vs untrimmed) moves accuracy and κ by about as much as the night-wise leak. The audit should fix and report it, or the leakage effect will be confounded with it.
  - Untrimmed Sleep-EDF SC recordings contain long daytime/evening wake periods, so untrimmed accuracy would be inflated further. This is my inference; I did not quantify it.
- The DeepSleepNet cross-paper gap of about 9–16 accuracy points mixes leakage with different features, channels and subsets. It is an upper-bound anecdote, not a measured leakage effect.

### Gaps
- I found no peer-reviewed study that runs **all three** protocols (random epoch-wise, night/record-wise, subject-wise) with **deep sequence models** on Sleep-EDF.
- Köksal's is the only controlled peer-reviewed measurement I found, and it covers RF only and does not report accuracy.
- I could not verify the experiment DeepSleepNet cites (Tsinalis 2016, ref [7]) showing that the non-independent scheme improves performance. The search snippets did not show it.
- I could not read Köksal's supplementary Tables S3–S5 (per-fold values, Mann-Whitney/Cliff's δ results).
- Imtiaz & Rodriguez-Villegas 2014/2015 full texts could not be fetched (the Imperial PDF returned 404). Their exact wake-selection and metric recommendations are known only second-hand.
- "Buriro et al., 2021", cited by Köksal on RW leakage, was not checked.

## Q2. What is arXiv 2611.08311 and what protocol does it describe?

### Takeaway
arXiv 2611.08311 is a **preprint** from 5 Nov 2026 on LLM-assisted label-noise filtering. It is not an evaluation-protocol paper. It reports a single "adapted subject-wise 20-fold cross-validation" on **Sleep-EDF-78** (SC, 153 PSGs). It does not describe 20-fold on Sleep-EDF-20 plus 10-fold on Sleep-EDF-78. It is a weak citation for "the standard subject-wise protocol". XSleepNet/SleepTransformer (10-fold on -78) and DeepSleepNet/XSleepNet (20-fold on -20) are better references.

### Cited Findings
- Bibliographic and method details:
  - Title: "LLM-Assisted Label Noise Correction for EEG-Based Sleep Stage Classification", by Elmira Salari and Shruti Kshirsagar.
  - Submitted 5 Nov 2026 (v1) under cs.CL **[PP]**.
  - Pipeline: train TinySleepNet; send epochs with softmax confidence < τ = 0.5 to LLaMA 3.1 8B, which labels them RELIABLE or SUSPICIOUS against AASM rules; remove SUSPICIOUS epochs and retrain.
  - Reported gain: accuracy 81.8 → 84.6%, MF1 75.6 → 77.4%.
  — [arXiv abs 2611.08311](https://arxiv.org/abs/2611.08311)
- Protocol details from the HTML full text:
  - Data: Sleep-EDF Expanded v1.0.0 **Sleep Cassette, 153 PSG recordings from 78 healthy subjects** aged 25–101; single-channel **Fpz-Cz** at 100 Hz; 30-s epochs following the TinySleepNet repository preprocessing; R&K labels mapped to 5 AASM classes.
  - Wake kept only within 30 min before the first and after the last non-wake epoch; MOVEMENT/UNKNOWN removed.
  - Split: "adapted subject-wise 20-fold cross-validation". The paper states that recordings of the same participant should be in the same partition, but it does not explicitly confirm the two nights were kept together and cites no protocol source.
  - Fold sizes (means): train 38,070, validation 3,938, test 2,211 epochs; 195,479 epochs in total. Use of the validation split and hyperparameters are not described.
  - Metrics: accuracy and MF1 only. No Cohen's κ, no absolute per-class F1 (only relative changes; N1 F1 "decreased slightly").
  - At τ = 0.7 performance fell to 80.5 / 74.8.
  — [arXiv HTML 2611.08311v1](https://arxiv.org/html/2611.08311v1)
- The baseline is a PyTorch re-implementation of TinySleepNet. The authors say it scores below the original paper because of framework and subset/configuration differences. — [arXiv HTML 2611.08311v1](https://arxiv.org/html/2611.08311v1)
- The TinySleepNet repository's run command iterates `--from_fold 0 --to_fold 19` (20 folds) and offers a `sleepedfx` option. The README does not state whether folds are subject-wise. — [TinySleepNet GitHub](https://github.com/akaraspt/tinysleepnet)

### Inferences
- 2611.08311 inherits the TinySleepNet codebase's 20-fold split rather than defining a protocol. Its TinySleepNet re-implementation baseline (81.8% / MF1 75.6 on -78) is a useful "honest" reference point for Sleep-EDF-78.
- For a citable protocol, use:
  - Sleep-EDF-20: 20-fold (one subject per fold) or LOSO subject-wise CV — [DeepSleepNet](https://arxiv.org/abs/1703.04046), [XSleepNet](https://arxiv.org/abs/2007.05492), [L-SeqSleepNet](https://arxiv.org/abs/2301.03441).
  - Sleep-EDF-78: 10-fold subject-wise CV with held-out validation subjects — [XSleepNet](https://arxiv.org/abs/2007.05492), [SleepTransformer](https://arxiv.org/abs/2105.11043).
- The paper's label-noise motivation reinforces Q4. Its LLM filter only touches training epochs, so it does not itself create test leakage. However, removing "suspicious" training labels with a test-fold-blind rule is fine only if τ was not tuned on test folds, and the paper does not say how τ was chosen (inference).

### Gaps
- Whether the 20 folds keep both nights of each SC-78 subject together is not explicitly confirmed.
- The exact fold assignment file is not published in the text.

## Q3. Typical honest (subject-wise) performance on Sleep-EDF-20 and Sleep-EDF-78, including per-class N1

### Takeaway
On Sleep-EDF-20 (Fpz-Cz, ±30 min wake, 20-fold/LOSO), published single-channel deep models trained from scratch cluster at **82–86% accuracy, κ 0.76–0.81, MF1 77–81**. On Sleep-EDF-78 (10-fold) they cluster at **80–84% accuracy, κ 0.73–0.78, MF1 73–78**. **N1 F1 is about 40–53** everywhere. Classical feature models evaluated subject-wise sit lower, at about 75–82% accuracy on -20 with MF1 about 0.70–0.77. One RF-only study on -78 reports κ 0.58 and MF1 0.60 (medians over folds).

### Cited Findings
**Sleep-EDF-20** (SC 2013 release, 20 subjects, 39 nights, Fpz-Cz, ±30 min wake, subject-wise). Values are accuracy % / κ / MF1, with per-class F1 as W/N1/N2/N3/REM.

| Model | Acc | κ | MF1 | Per-class F1 W/N1/N2/N3/REM | Source |
|---|---|---|---|---|---|
| DeepSleepNet (20-fold, pooled) | 82.0 | 0.76 | 76.9 | 84.7/46.6/85.9/84.8/82.4 | [DeepSleepNet Table IV](https://arxiv.org/abs/1703.04046) |
| SleepEEGNet | 81.5 | 0.75 | 76.6 | 89.4/44.4/84.7/84.6/79.6 | [L-SeqSleepNet Table III](https://arxiv.org/abs/2301.03441) |
| AttnSleep | 84.4 | 0.79 | 78.1 | 89.7/42.6/88.8/90.2/79.0 | [L-SeqSleepNet Table III](https://arxiv.org/abs/2301.03441) |
| TinySleepNet | 85.4 | 0.80 | 80.5 | 90.1/51.4/88.5/88.3/84.3 | [L-SeqSleepNet Table III](https://arxiv.org/abs/2301.03441) |
| SeqSleepNet (in the XSleepNet paper) | 85.2 | 0.798 | 78.4 | – | [XSleepNet Table 2](https://arxiv.org/abs/2007.05492) |
| XSleepNet2 (20-fold, pooled) | 86.3 | 0.813 | 80.6 | 92.2/51.8/88.0/86.8/83.9 | [XSleepNet](https://arxiv.org/abs/2007.05492); [L-SeqSleepNet](https://arxiv.org/abs/2301.03441) |
| L-SeqSleepNet, from scratch (LOSO, 5 reps) | 86.3±0.2 | 0.813 | 79.3 | 91.6/45.3/88.5/86.2/85.2 | [L-SeqSleepNet Table III](https://arxiv.org/abs/2301.03441) |
| L-SeqSleepNet, SHHS-pretrained | 88.6±0.1 | 0.845 | 82.9 | 94.1/53.3/89.7/88.4/88.9 | [L-SeqSleepNet Table III](https://arxiv.org/abs/2301.03441) |
| Tsinalis 2016 time-frequency + stacked sparse autoencoder (hand-designed features) | 78.9 | – | 73.7 | 71.6/47.0/84.6/84.0/81.4 | [DeepSleepNet Table IV](https://arxiv.org/abs/1703.04046) |
| U-Sleep (different split and channels; flagged as not comparable) | – | – | 79.0 | 93/57/86/71/88 | [L-SeqSleepNet](https://arxiv.org/abs/2301.03441); [SleepTransformer](https://arxiv.org/abs/2105.11043) |

- Per-class discrepancy: L-SeqSleepNet's table lists DeepSleepNet per-class F1 as 86.7/45.5/85.1/83.3/82.6, which differs from DeepSleepNet's own 84.7/46.6/85.9/84.8/82.4. Overall metrics match (82.0 / 0.76 / 76.9). — [L-SeqSleepNet](https://arxiv.org/abs/2301.03441); [DeepSleepNet](https://arxiv.org/abs/1703.04046)
- L-SeqSleepNet details:
  - Protocol: SleepEDF = SC subset, 20 subjects; "Conforming to majority of previous works in literature, we conducted leave-one-subject-out cross validation (LOSO CV) on SleepEDF". Validation subjects held out; experiments "repeated … 5 times". Recordings trimmed "from 30 minutes before to 30 minutes after its in-bed part".
  - The SleepEDF confusion matrices show N1 recall of only about 46–48%. N1 epochs go about 23% to N2, 16–17% to W and 13% to REM.
  — [arXiv 2301.03441](https://arxiv.org/abs/2301.03441)
- With EEG+EOG on SleepEDF-20 (±30 min), XSleepNet2 reaches 86.4 / 0.813 / 80.9 (EOG adds little on -20). — [XSleepNet](https://arxiv.org/abs/2007.05492)

**Sleep-EDF-78** (SC 2018 release, 78 subjects, 153 nights, Fpz-Cz, ±30 min wake, 10-fold subject-wise; SleepTransformer Table I, verified against its text).

| Model | Acc | κ | MF1 | Per-class F1 W/N1/N2/N3/REM | Source |
|---|---|---|---|---|---|
| SleepEEGNet | 80.0 | 0.730 | 73.6 | – | [SleepTransformer Table I](https://arxiv.org/abs/2105.11043) |
| AttnSleep | 81.3 | 0.74 | 75.1 | 92.0/42.0/85.0/82.1/74.1 | [SleepTransformer Table I](https://arxiv.org/abs/2105.11043) |
| SleepTransformer, from scratch | 81.4 | 0.743 | 74.3 | 91.7/40.4/84.3/77.9/77.2 | [SleepTransformer](https://arxiv.org/abs/2105.11043) |
| SeqSleepNet | 82.6 | 0.760 | 76.4 | 92.2/47.8/84.9/77.2/79.9 | [SleepTransformer](https://arxiv.org/abs/2105.11043) |
| XSleepNet2 | 84.0 | 0.778 | 77.9 | 93.3/49.9/86.0/78.7/81.8 | [SleepTransformer](https://arxiv.org/abs/2105.11043); [XSleepNet](https://arxiv.org/abs/2007.05492) |
| SleepTransformer, SHHS-pretrained | 84.9 | 0.789 | 78.8 | 93.5/48.5/86.5/80.9/84.6 | [SleepTransformer](https://arxiv.org/abs/2105.11043) |
| U-Sleep (not comparable) | – | – | 79.0 | 93/57/86/71/88 | [SleepTransformer](https://arxiv.org/abs/2105.11043) |
| TinySleepNet re-implementation (20-fold) **[PP]** | 81.8 | – | 75.6 | – | [arXiv 2611.08311](https://arxiv.org/html/2611.08311v1) |
| RF on 17 handcrafted features, SW 10-fold (fold medians) | – | 0.58 | 0.60 | – | [Köksal 2026](https://dergipark.org.tr/en/download/article-file/5192156) |
| RF on 17 handcrafted features, LOSO (fold medians) | – | 0.58 | 0.54 | – | [Köksal 2026](https://dergipark.org.tr/en/download/article-file/5192156) |

- SleepTransformer protocol: "We carried out 10-fold cross-validation on the SleepEDF-78 database as in prior works … In each iteration, seven subjects were left out from the training subjects as the validation set". The authors note it "overfitted this database quite easily". SHHS pretraining added +3.5 points accuracy, +0.046 κ and +4.5 MF1. — [arXiv 2105.11043](https://arxiv.org/abs/2105.11043)
- XSleepNet setup table: SleepEDF-20 uses 20-fold CV; SleepEDF-78 (listed as 79 subjects in its table) uses 10-fold CV with 4 validation subjects. "The outputs of a system across the cross-validation folds were pooled altogether to compute its performance metrics". — [arXiv 2007.05492](https://arxiv.org/abs/2007.05492)
- Classical baselines (other):
  - GitHub RF demo **[NR]**, 10 SC subjects, subject-wise GroupKFold: single-epoch RF 0.796 acc / MF1 0.731 / κ 0.722; RF with ±2 context 0.819 / 0.767 / 0.752; HistGradientBoosting with context 0.815 / 0.757 / 0.745. With LOSO, band-power RF with context gave 0.821±0.074 acc and κ 0.745±0.107 (mean±SD across subjects). — [shailesh-vishal19 repo](https://github.com/shailesh-vishal19/sleep-stage-classification)
  - In ear-EEG/cEEGrid datasets, L-SeqSleepNet's table lists a "Random Forest" baseline at κ 0.73 (ear-EEG) and 70.0% / κ 0.58 (cEEGrid). These are not Sleep-EDF numbers. — [arXiv 2301.03441](https://arxiv.org/abs/2301.03441)
- YASA (LightGBM on handcrafted features, trained on more than 3,000 NSRR nights) was **not evaluated on Sleep-EDF** in its eLife paper. Its independent test results:
  - NSRR holdout of 585 nights: median accuracy 87.46%, κ 0.819, MCC 0.821; N1 median F1 0.432 (sensitivity 45.4%).
  - DOD-Healthy: accuracy 86.6±6.2, MF1 78.5±9.4, N1 F1 52.0.
  - DOD-Obstructive: accuracy 84.30±7.85, MF1 73.96±10.79, N1 F1 39.17.
  — [YASA eLife, PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- Phan & Mikkelsen review: κ ≥ 0.81 ("almost perfect") is reached on mostly-healthy databases, including EDF-20, but performance is "still substandard" on pathological databases (ISRUC, CAP). — [arXiv 2111.08446](https://arxiv.org/abs/2111.08446)
- The AttnSleep (IEEE TNSRE 2021) README describes K-fold CV with the fold count in `config.json`. It does not say whether splits are subject-wise or whether both nights stay together. — [AttnSleep GitHub](https://github.com/emadeldeen24/AttnSleep)

### Inferences
- Plausible honest targets for the student's repo, single-channel Fpz-Cz with ±30 min wake:
  - Classical features + LR/RF: about **75–82% accuracy, κ about 0.65–0.75, MF1 about 0.68–0.77, N1 F1 about 0.30–0.47** on Sleep-EDF-20. Expect somewhat lower on -78, which has more subjects, older ages and more heterogeneity.
  - A reproduced DeepSleepNet/TinySleepNet: about 82–85% on -20 and about 80–83% on -78.
  - Results well above about 87% accuracy or κ 0.83 on Sleep-EDF with a from-scratch single-channel model are a red flag for leakage or a non-standard subset or wake policy.
- Köksal's RF numbers on -78 (κ 0.58) are much lower than the GitHub RF on 10 subjects (κ 0.75). Likely reasons (inference): no temporal-context features, only 17 features, balanced class weights, medians over folds rather than pooled metrics, and the older SC-78 cohort. This shows that aggregation choice and feature set swing "honest" baselines by more than 0.1 κ.
- Pooled-over-folds metrics (DeepSleepNet, XSleepNet) and per-subject means/medians (YASA, Köksal) are not interchangeable. Compare like with like.

### Gaps
- I could not open TinySleepNet's own Sleep-EDF-78 numbers (EMBC 2020, not on arXiv; README gives a figure only) or AttnSleep's original paper. The values above come from secondary tables in L-SeqSleepNet and SleepTransformer.
- I found no peer-reviewed Sleep-EDF-20 result for a plain logistic-regression or random-forest baseline with a subject-wise split and full per-class reporting.
- No YASA result on Sleep-EDF was found in the YASA paper.
- In the L-SeqSleepNet table, the "pretrained" vs "from scratch" assignment of the two rows was inferred from the row order and ⋆ convention used in other blocks; the superscript glyphs were lost in text extraction.

## Q4. Human inter-rater agreement ceiling

### Takeaway
Human scorers agree about **82–83%** overall, with **κ about 0.76** under AASM rules. N1 is the weak stage at about **63%** agreement, and N3 is about 67%. Agreement also falls at stage transitions. Published Sleep-EDF subject-wise results (82–86% accuracy, κ 0.76–0.81) are already at this level, which caps how much a leak can plausibly "improve" honest numbers.

### Cited Findings
- Rosenberg & Van Hout 2013 (J Clin Sleep Med, AASM inter-scorer reliability program) **[PR]**:
  - Data: more than 2,500 scorers, 9 record fragments × 200 epochs, 3,296,905 decisions.
  - Agreement with the majority score: **82.6% overall** (82.4% when records are weighted equally); W 84.1%; **N1 63.0%**; N2 85.2% (in the Discussion); **N3 67.4%**; REM highest.
  - N1 disagreements went to N2 (21.7%) and W (10.9%); N3 disagreements were almost all N2 (32.3%).
  - At W→N1, the last W epoch had 73.6% agreement, and 25.5% of scorers still scored the first N1 epoch as W.
  - N1 bouts averaged only 2.03 epochs. N2↔N3 disagreement stayed high regardless of distance from the transition.
  - Caveat: the reference is the majority vote, not an adjudicated gold standard.
  — [Rosenberg & Van Hout 2013, PMC3525994](https://pmc.ncbi.nlm.nih.gov/articles/PMC3525994)
- Danker-Hopfe et al. 2009 (J Sleep Res 18:74–84) **[PR]**: 72 SIESTA recordings. AASM gave 82.0% agreement, κ = 0.76; R&K gave 80.6%, κ = 0.68. — [PubMed 19250176](https://pubmed.ncbi.nlm.nih.gov/19250176/) (abstract via Europe PMC); [DOI](https://doi.org/10.1111/j.1365-2869.2008.00700.x)
- Phan & Mikkelsen review: "Many previous studies consistently reported the consensus among human scorers around a Cohen's kappa of 0.76". Consensus is "particularly poor" on N1, "epochs close to the boundary of two stages, and data from patients with fragmented sleep". Training on single-scorer labels transfers that scorer's subjectivity into the model. — [arXiv 2111.08446](https://arxiv.org/abs/2111.08446)
- YASA cites about 83% inter-scorer agreement (Rosenberg & Van Hout), about 90% intra-scorer agreement, and N1 agreement below 45% (Malhotra et al. 2013).
  - On DOD-Healthy, individual scorers reached 78.8–86.6% accuracy against the consensus of the other scorers; on DOD-Obstructive, 80.38–84.50%.
  - In the author response: unanimous human consensus occurred in 46.2% of stable epochs vs 11.5% of transition epochs.
  — [YASA eLife, PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- DeepSleepNet interprets κ 0.61–0.80 as "substantial" agreement. — [arXiv 1703.04046](https://arxiv.org/abs/1703.04046)

### Inferences
- Sleep-EDF has a single scorer per night (R&K). Honest models at about 82–86% accuracy are already at the human inter-rater level.
- A leaky random epoch split can push *above* the inter-rater ceiling (for example about 90–91% in the "non-independent" Sleep-EDF papers DeepSleepNet tabulated). It does this partly by learning the scorer's within-night labelling idiosyncrasies from neighbouring epochs. Exceeding inter-rater agreement on single-scorer data is a useful leak diagnostic for the audit (inference).

### Gaps
- Per-stage κ from Danker-Hopfe 2009 was not retrieved (paywalled full text).
- No inter-rater statistics specific to Sleep-EDF (single scorer) exist to my knowledge.

## Q5. Standard metrics and reporting conventions

### Takeaway
The de facto deep-learning convention on Sleep-EDF is to report:
- overall accuracy, Cohen's κ and macro-F1 (often also mean sensitivity and specificity)
- per-class F1 for W/N1/N2/N3/REM
- a confusion matrix
- metrics computed on predictions pooled across all CV folds

Clinical and validation guidance (Menghini et al. 2021, SLEEP) instead recommends per-subject metrics averaged across subjects, with SD and 95% CIs (bootstrap when samples are small or skewed), proportional confusion matrices, and Bland–Altman plots for derived sleep parameters. No consensus threshold for "good" exists.

### Cited Findings
- Deep-learning papers:
  - DeepSleepNet reports MF1, overall accuracy, Cohen's κ and per-class precision/recall/F1 from the confusion matrix, using pooled predictions from all folds. — [arXiv 1703.04046](https://arxiv.org/abs/1703.04046)
  - XSleepNet reports accuracy, MF1, κ, average sensitivity and average specificity "pooled altogether" across folds, with per-fold mean/SD in an appendix. — [arXiv 2007.05492](https://arxiv.org/abs/2007.05492)
  - SleepTransformer uses the same overall metrics plus "class-wise MF1". — [arXiv 2105.11043](https://arxiv.org/abs/2105.11043)
  - L-SeqSleepNet reports mean±SD over 5 repeated runs. — [arXiv 2301.03441](https://arxiv.org/abs/2301.03441)
- YASA computes agreement metrics "for each testing night separately" and reports **medians across nights**. It also reports stage-specific F1 medians and confusion matrices, and correlates per-night accuracy with covariates. — [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- Menghini, Cellini, Goldstone, Baker & de Zambotti, "A standardized framework for testing the performance of sleep-tracking technology: step-by-step guidelines and open-source code", SLEEP 44(2), 2021 **[PR]**:
  - Recommends discrepancy analysis, Bland–Altman plots (TST, SE, SOL, WASO; reference PSG on the x-axis; check proportional bias and heteroscedasticity; log-transform when needed) and epoch-by-epoch (EBE) analysis.
  - EBE metrics: accuracy, sensitivity, specificity, PPV, NPV, prevalence/bias indices, κ "with the corresponding CI", and PABAK for sleep/wake.
  - Per-subject matrices are averaged at group level ("The second modality is recommended"). EBE metrics "should be accompanied by information on their variability (SD) and uncertainty (CI)", with "nonparametric or bootstrap CI" for small or non-normal samples; the worked example uses a 10,000-replicate bootstrap.
  - Proportional confusion matrices with SD/CI are preferred.
  - "There is currently no consensus on threshold values defining a 'good' performance".
  — [Menghini et al. 2021, PMC7879416](https://pmc.ncbi.nlm.nih.gov/articles/PMC7879416)
- Köksal 2026 adds calibration metrics (ECE, LogLoss/NLL, Brier) next to MF1 and κ. It summarizes folds as medians [IQR] with **BCa bootstrap (10,000 resamples) 95% CIs**, and tests RW vs SW with Mann-Whitney U and Cliff's δ. — [Köksal 2026 PDF](https://dergipark.org.tr/en/download/article-file/5192156)
- In Menghini et al.'s framing, κ can show "high agreement but low kappa" because of class prevalence (sleep epochs dominate). — [PMC7879416](https://pmc.ncbi.nlm.nih.gov/articles/PMC7879416)
- The AASM Autoscoring Certification Program tests software against more than 140 PSG studies scored by the AASM Gold Standard Panel; certification lasts 3 years and the standards document is available on request. It has expanded from sleep staging to full PSG scoring. — [AASM Autoscoring Certification](https://aasm.org/about/industry-programs/autoscoring-certification/)
- Phan & Mikkelsen classify κ ≥ 0.81 as "almost perfect", citing the standard κ interpretation scale. — [arXiv 2111.08446](https://arxiv.org/abs/2111.08446)

### Inferences
- For a leakage audit, report both views:
  - **pooled** metrics (comparable with DeepSleepNet/XSleepNet tables)
  - **per-subject** accuracy, κ and MF1 with a **subject-level (cluster) bootstrap** 95% CI, per Menghini et al. and Köksal
- Report per-class F1 with special attention to N1, plus row-normalized confusion matrices.
- Report the *difference* between leaky and honest protocols with a paired, subject-level bootstrap CI. The quantity of interest is the gap, not either number alone.
- Calibration (ECE) is a cheap addition. Köksal shows leakage makes models overconfident, which is a second signature of leakage.

### Gaps
- I found no sleep-staging-specific ML reporting checklist (a TRIPOD-AI-style guideline for staging algorithms). The AASM certification standards are not public.
- The Imtiaz & Rodriguez-Villegas 2014 metric recommendations could not be read in full.

## Q6. Fragmentation and transition-stratified error analysis

### Takeaway
The best-documented approach is YASA's:
- per epoch, label an epoch "transition" if a human-scored stage change falls within ±1.5 min (a 3-min window), otherwise "stable";
- per night, regress accuracy on % stage transitions, % N1, AHI and age.

YASA found 69% accuracy around transitions vs 94% in stable periods. % N1 and % transitions were the top predictors of per-night accuracy. Phan et al. (pediatric) instead bin errors by distance in epochs to the nearest cross-stage transition, with a "rapid transition" class: 55–60%+ of shared errors are within 4 epochs of a transition.

### Cited Findings
- YASA (Vallat & Walker, eLife 2021) **[PR]**:
  - Definition: an epoch is "around a transition" if a human-scored stage transition occurs within the 3-min window centred on it (1.5 min before, 1.5 min after).
  - Accuracy: **69.15% ± 7.23 around transitions vs 94.08% ± 5.61 in stable periods** on 585 NSRR nights (p < 0.001); 68.95 ± 7.7 vs 91.2 ± 7.8 on DOD (n = 75).
  - Correlations: accuracy vs AHI r = −0.29; accuracy vs age r = −0.128; human-scored % transitions vs AHI r = 0.427.
  - A random forest predicting per-night accuracy ranked **% N1 and % transitions as the top two predictors (about 40% of relative importance)**. The authors call both "markers of sleep fragmentation".
  - WASO and number of awakenings were not analysed.
  — [YASA eLife, PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- YASA peer review and author response (eLife.70092.sa2; read via the PMC copy because elifesciences.org did not load):
  - Reviewers noted that the algorithm "performs worse on N1 stage, older individuals and patients presenting sleep disorders (sleep fragmentation)" and asked whether errors occur in stable or transition periods.
  - The authors added the transition analysis, the random-forest accuracy-predictor analysis, separate healthy and OSA results, and a Limitations section.
  - Reviewers also questioned the 5-min smoothing. The authors tested 49 window combinations and kept a 2-min past plus 7.5-min centred triangular rolling average; removing both cost 3.6 MF1 points.
  — [eLife peer review DOI](https://doi.org/10.7554/eLife.70092.sa2); [PMC8516415](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- Phan, Mertins & Baumert, "Pediatric Automatic Sleep Staging" (IEEE TBME 2022) **[arXiv copy of PR]**:
  - They plot "the distance of the misclassified epochs to their nearest transitions", with "rapid transition" meaning epochs immediately before *and* after a transition.
  - Findings: ">60% (single-channel EEG) and 55% (EEG·EOG) of the common errors were … at most 4 epochs away from their nearest cross-stage transitions"; about 20% were rapid-transition epochs.
  - Networks misclassified epochs *before* transitions more than those *after* at equal distance, a difference "not seen" in a similar adult analysis with SeqSleepNet.
  — [arXiv 2108.10211](https://arxiv.org/abs/2108.10211)
- SleepTransformer, on SleepEDF-78:
  - "Often, the low-confidence and transitioning epochs are misclassified"; Fig. 3(c) plots the share of transitioning epochs among low- vs high-confidence epochs.
  - When the lowest-confidence 20% of epochs are set aside, their accuracy is about 60%.
  — [arXiv 2105.11043](https://arxiv.org/abs/2105.11043)
- Rosenberg & Van Hout show that human agreement also degrades at W→N1, N1→N2 and N2→REM boundaries, and N2/N3 agreement stays low throughout. Part of the error at transitions is label ambiguity, not model failure. — [PMC3525994](https://pmc.ncbi.nlm.nih.gov/articles/PMC3525994)
- L-SeqSleepNet reports that long-context modelling mainly reduces "exceptionally high individual errors". It frames sleep as "a slow-transitioning neural process" with cycles of 90–110 min (180–220 epochs). — [arXiv 2301.03441](https://arxiv.org/abs/2301.03441)

### Inferences
- A concrete stratification scheme for the audit (my synthesis, not a single published standard):
  1. **Epoch level**:
     - compute the distance d (in epochs) to the nearest human-scored stage change;
     - report accuracy and MF1 for d = 0/1 (rapid transition), d ≤ 3 (YASA's ±1.5 min), d ≤ 4 (Phan), and stable (d > 4);
     - optionally split by transition type (W↔N1, N1↔N2, N2↔N3, N2↔REM).
  2. **Night level**, computed from the human hypnogram:
     - % transitions = stage changes / epochs, or stage shifts per hour of sleep
     - % N1
     - number of awakenings and WASO
     - age (Sleep-EDF-78 SC spans 25–101 years)
     - correlate per-night accuracy, κ and MF1 with each; report terciles (low, mid, high fragmentation) with subject-bootstrap CIs.
  3. **Leakage interaction**: compute the random-split minus subject-split gap separately in stable vs transition epochs. The prediction (inference) is that random-split leakage helps most in **stable** epochs, where neighbours in training share the label, and least at transitions. If so, the headline gap overstates real-world harm in fragmented sleep, and fragmented nights will expose the honest model's weakness.
- Restrict the transition window to what the hypnogram resolution supports. Sleep-EDF uses 30-s epochs with R&K labels mapped to AASM, so N1 bouts are very short (Rosenberg: mean 2.03 epochs).

### Gaps
- I found no paper that analyses error vs fragmentation **on Sleep-EDF** specifically, or that measures how leakage interacts with transitions.
- No sourced standard definition of a "sleep fragmentation index" (for example awakenings plus stage shifts per hour) was retrieved in this session. Cite a clinical definition separately if used.
- YASA did not analyse WASO or number of awakenings.

## Q7. Is leakage in sleep staging smaller than in EEG disease detection? Is temporal autocorrelation leakage discussed?

### Takeaway
No source compares the two directly. The measured sleep-staging gaps (about 0.10 macro-F1 and 0.11 κ in Köksal; about 6–7 accuracy points in GitHub demos) are moderate. The 2025 EEG disease-detection study is described in the brief as showing large gaps. A structural reason plausibly explains the difference, but it is inference. In disease detection the label is constant per subject, so subject identity almost gives the label away. In sleep staging the label changes within a subject, so subject leakage helps only through (i) subject-specific EEG signatures per stage and (ii) temporal autocorrelation: neighbouring 30-s epochs are near-duplicates and usually share the same stage. Temporal autocorrelation is named as a mechanism mainly in non-peer-reviewed sources and in discussions of sequence context. I found no paper that isolates it quantitatively.

### Cited Findings
- Köksal names subject-level leakage as the mechanism: "Because record-wise (RW) CV ignores subject identity, epochs from the same person may appear in both training and test folds, causing data leakage and artificially inflated performance". The measured inflation is moderate: SleepEDF macro-F1 0.70 vs 0.60, κ 0.69 vs 0.58. — [Köksal 2026 PDF](https://dergipark.org.tr/en/download/article-file/5192156)
- A GitHub benchmark PR **[NR, snippet only]** states that consecutive 30-s epochs from one night are highly autocorrelated and share subject identity, and reports a 0.06 accuracy gap. — [PR #141](https://github.com/brain-researcher/brain-researcher-benchmark/pull/141)
- The shailesh-vishal19 demo **[NR]** shows a 0.073 accuracy gap (0.892 vs 0.819) when ±2-epoch context features are combined with a random split. — [repo](https://github.com/shailesh-vishal19/sleep-stage-classification)
- Sleep-stage persistence and long-range dependence:
  - Sleep cycles last 90–110 min (180–220 epochs), and sequence models exploit dependence up to a whole cycle. — [L-SeqSleepNet](https://arxiv.org/abs/2301.03441)
  - DeepSleepNet models stage-transition rules, for example "if a subject is in sleep stage N2, continue to…". — [DeepSleepNet](https://arxiv.org/abs/1703.04046)
  - Accuracy is 94% in stable stretches vs 69% near transitions. — [YASA](https://pmc.ncbi.nlm.nih.gov/articles/PMC8516415/)
- Inter-subject variability is substantial. Personalization with same-subject data improves staging by only about 2 points on average (more for poorly-fit subjects). — [Mikkelsen & De Vos arXiv 1801.02645](https://ar5iv.labs.arxiv.org/html/1801.02645); [Phan & Mikkelsen review](https://arxiv.org/abs/2111.08446)
- Contrast study (not re-summarized): the 2025 EEG data-partition study on disease-detection tasks. — [arXiv 2505.13021](https://arxiv.org/abs/2505.13021)

### Inferences
- **Why the sleep-staging gap should be smaller than in disease detection** (inference):
  - The per-epoch label varies within each subject, so memorizing a subject's "fingerprint" does not reveal the stage.
  - Honest baselines are already near the human ceiling (82–86%; κ about 0.76–0.81), which leaves limited headroom. Leaky Sleep-EDF results top out around 90–92%.
  - The personalization literature suggests same-subject information is worth only about 2 points in a properly designed (whole-night) split.
- **Why the gap is not zero** (inference):
  - A random epoch split puts, for most test epochs, the epoch immediately before and after it into training. Those epochs have nearly identical spectra and the same label, so a feature model can act like a nearest-neighbour lookup within the night.
  - This "temporal-adjacency" leak is distinct from subject leakage. The audit can separate the two by adding a **blocked within-subject split** (contiguous chunks of each night, for example 30-min blocks, assigned to train/test, optionally with a gap or buffer of several epochs).
  - Comparing random epoch vs blocked within-night vs night-wise vs subject-wise splits would decompose the gap into adjacency, within-night and subject components. I found no published sleep-staging paper doing this decomposition, so it would be a novel contribution.
- For deep sequence models that take 20–200 epochs of input context, a random epoch split puts overlapping raw-signal windows in train and test. The leak should therefore be larger for sequence models than for single-epoch classical models (unmeasured).

### Gaps
- No peer-reviewed quantification of temporal-autocorrelation (adjacent-epoch) leakage in sleep staging, separate from subject leakage, was found.
- No direct, same-author comparison of the leakage gap in sleep staging vs EEG disease detection was found.
- Generic blocked-CV references for autocorrelated data (for example from ecology or time-series) were not retrieved in this session.
