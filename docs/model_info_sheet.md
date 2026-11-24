# Model info sheet

This sheet follows the model info sheet of Kapoor and Narayanan ([reproducible.cs.princeton.edu](https://reproducible.cs.princeton.edu/)). It argues the three claims their taxonomy asks for, for the honest arm (`subject`) of this audit:

- clean separation between training and test data;
- legitimate features;
- a test distribution that matches the claim.

The leaky arms break the first claim on purpose, and the table in section 1 says how.

The claim the honest arm supports: **per-subject Cohen's κ on held-out subjects** from the Sleep-EDF Sleep Cassette cohort, with single-channel Fpz-Cz (plus horizontal EOG for the YASA models). It does not support any claim about other cohorts, other hardware or clinical use.

## 1. Clean train-test separation

| Kapoor–Narayanan leak class | Status in the honest arm | Evidence |
|---|---|---|
| **No test set** (L1.1) | Every epoch is predicted out of fold, exactly once per seed | `test_every_row_tested_once`, `test_oof_covers_all_rows` |
| **Pre-processing on train and test together** (L1.2) | Only per-recording steps happen before the split: label clipping, trimming, band power, context features, and YASA's per-night smoothing and scaling. None is fitted across recordings. The only fitted transform, `StandardScaler`, sits inside the pipeline and is fitted on training rows only. | `test_spy_scaler_fit_rows`; `subject_prefit` deliberately breaks this and measures the effect |
| **Feature selection on train and test together** (L1.3) | No feature selection; feature sets are fixed by prefix | `models.MODELS` |
| **Duplicates in train and test** (L1.4) | Each epoch occurs once. Each recording belongs to one subject (pairing on the first 7 characters of the file name) | `test_pair_scorer_codes_differ`, `test_sc78_shape` |
| **Model uses features not legitimate** (L2) | See section 2 | — |
| **Temporal leakage** (L3.1) | Not a forecasting task; no claim is made about future nights of the same person | — |
| **Non-independence between train and test** (L3.2) | Folds group by subject, so both nights of a person are on the same side, and so are all epochs of a night. Checked again on every full run, before training. | `test_subject_arm_no_overlap`, `test_one_night_subject_ok`, `test_unsorted_row_order_is_fine`; `pipeline_checks.honest_folds_subject_disjoint` in `meta.json` |
| **Sampling bias in the test distribution** (L3.3) | See section 3 | — |
| **Tuning on the test set** | No hyperparameters are tuned and no model uses early stopping: logistic regression is at scikit-learn defaults with balanced class weights, and LightGBM uses YASA's published settings. There is no inner validation loop to leak. | `test_factories_use_fixed_hyperparameters`, `test_lgbm_uses_fixed_hyperparameters_and_seed` |

How each leaky arm breaks separation, by design:

- `epoch`: neighbouring epochs, the same night and the same person are shared.
- `block`: the same night and person, but not epochs whose features overlap. The buffer is 8 kept epochs, wider than YASA's ±7-epoch smoothing window and the ±2-epoch context.
- `night`: the person's other night.
- `subject_prefit`: scaling statistics.

## 2. Each feature is legitimate

A feature is legitimate if it would be available, with the same meaning, when scoring a new person's night with no labels.

- **Band power (`bp_*`) and its neighbours (`bpctx_*`):** computed from that night's own EEG. A scorer has the whole night's signal, so neighbouring epochs are legitimate inputs. They are not labels.
- **YASA features (`yasa_*`):** computed from the night's own EEG and EOG, smoothed and robust-scaled within that night. Per-night normalisation uses no other person's data and no labels.
- **`time_hour` and `time_norm`:** position in the analysed segment. The segment is trimmed with the reference labels (±30 min of wake around the scored sleep period), the convention used by DeepSleepNet, TinySleepNet, AttnSleep and most Sleep-EDF papers. Position therefore carries some label information that a new, untrimmed night would not offer.
  - This is a limitation of the convention, stated in the pre-registration (B5).
  - It applies identically in every arm, so it raises absolute scores but not the leaky-minus-honest differences the audit measures.
- **Pretrained probabilities (`yasa_proba_*`):** come from a model that never saw Sleep-EDF (section 4). They are used only by `yasa_pretrained`, never as inputs to a trained model.
- **Never used:** subject ID, recording ID, night number, age and sex are never model inputs. Age is used only in the descriptive night-level analysis.

## 3. The test distribution matches the claim

- **Held-out subjects:** the claim is about subjects the model has not seen, drawn from the same SC cohort. The honest arm tests exactly that.
- **Random fold assignment:** folds are assigned at random over subjects (`GroupKFold(shuffle=True)`), never by ID range. This matters because ID ranges coincide with age (IDs 0–19 aged 25–34; IDs 20 and above aged 50–101).
- **Reporting unit:** subjects. Each subject's κ is averaged over seeds and the paired bootstrap resamples subjects, so the interval reflects which people were tested.
- **Out of scope:** no claim is made for people aged 35–49 (absent from SC), for patients with sleep disorders, for other montages or for AASM-scored data. No outside dataset is used, so every result describes held-out subjects from this one cohort.

## 4. Pretrained weights and contamination

"Contamination" means a model's published weights were trained on some of the same Sleep-EDF recordings the audit tests on. This leakage channel is separate from the split, and no choice of split can remove it.

| Model or checkpoint | Saw Sleep-EDF before the audit? | Decision |
|---|---|---|
| `majority`, `bandpower_lr`, `bandpower_ctx_lr`, `yasa_lgbm` | No; fitted from scratch inside each fold | Used |
| YASA 0.8.1 pretrained classifier (`yasa_pretrained`) | **No.** Trained on 3,163 nights from seven NSRR cohorts (CCSHS, CFS, CHAT, HomePAP, MESA, MrOS, SHHS); Sleep-EDF appears nowhere in the paper or the training repository. | Used, as a zero-shot reference and split-invariance control. It expects a central EEG, an outer-canthus EOG, chin EMG and AASM scoring. It gets Fpz-Cz, horizontal EOG, no EMG and R&K labels, so its score includes a domain shift. |
| U-Sleep v1 pretrained (webserver or API) | **Yes.** Perslev et al. (2021) placed SEDF-SC (78 subjects) and SEDF-ST in the train/test group, each split into at least 75% training, at most 10% validation and at most 15% testing. | **Excluded.** Its score would be contaminated; in addition its weights are not public and its API refused connections on 24 Nov 2026. |
| braindecode's Chambon2018 checkpoint on the Hugging Face Hub | **Yes**; trained on Sleep-EDF data | **Excluded** for the same reason |
| braindecode `SleepStagerChambon2018` / `USleep` trained from scratch | No | Out of scope for this registration; possible later addendum, trained inside the folds |

**Library note.** YASA's pretrained classifier was pickled with scikit-learn 0.24.2 and loads under 1.9.1 with an `InconsistentVersionWarning`. On the four-subject fixture it agrees with the hypnogram at 0.85–0.86 accuracy. Its per-subject κ is required to be identical in every arm and seed (`pipeline_checks.yasa_pretrained_split_invariant`), which also checks that the evaluation code cannot invent a gap.
