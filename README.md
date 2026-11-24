# Sleep staging leakage audit

Checks how much a sleep-staging AI score depends on a leaky evaluation, by comparing an honest subject-wise test against random and partly grouped splits on the same open data.

**The protocol is pre-registered:** [docs/preregistration.md](docs/preregistration.md), frozen at git tag `prereg-v1` before any full-data model run.

## Why this matters

A sleep-tracking product or research model can report a high accuracy number that was measured with the same person's nights in both training and test data. That number does not show how the model will work on a new person. This project measures the gap on open data so anyone reading a sleep-AI accuracy claim can ask whether it was tested on new people.

## Scope

- Not a medical tool. It makes no clinical claims of any kind.
- Uses only public, de-identified data. No recruitment of people.
- Compares evaluation methods, not a product.

## Quickstart

```bash
uv sync --locked
uv run python -m sleepaudit all --config configs/sc78.yaml   # primary: 78 subjects, 7.6 GB download
uv run python -m sleepaudit all --config configs/sc20.yaml   # secondary: subjects 0-19
uv run python -m sleepaudit all --config configs/ci.yaml     # 4-subject fixture (~400 MB), runs in CI
```

- **Data directory:** data go to `$SLEEPAUDIT_DATA` (default `~/sleepaudit_data`), outside the repository.
- **Results:** written to `results/<config name>/`.
- **Steps:** `all` runs `fetch-labels`, `fetch-psg`, `features`, `evaluate` and `report`. Each can also be run on its own.
- **Gate:** `sc20` and `sc78` refuse to run unless the `prereg-v1` tag is in the current history.
- **Tests:** `uv run pytest` runs the tests. Tests that need data are skipped when it is absent; set `SLEEPAUDIT_REQUIRE_DATA=1` to make them fail instead.

## Data

[Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) 1.0.0 (PhysioNet; ODC-By 1.0), **Sleep Cassette files only**. See the [data card](docs/data_card.md).

- **Primary:** `sc78`, all 78 SC subjects and 153 recordings, 10-fold.
- **Secondary:** `sc20`, subjects 0–19 and 39 recordings, 20-fold, which makes the subject-wise arm leave-one-subject-out.
- **Signal and labels:** single-channel `EEG Fpz-Cz` at 100 Hz (plus `EOG horizontal` for YASA) and 30-s epochs. R&K stages 3 and 4 are merged into N3, and movement and unscored epochs are dropped.
- **Wake trimming:** wake is kept for 30 minutes on each side of sleep. This reproduces the published 42,308 and 195,479 epochs exactly.
- **Downloads:** files are checked against PhysioNet's SHA256 sums, with an AWS mirror as fallback.

## Models

1. **`majority`**: always predicts the most common stage (κ = 0).
2. **`bandpower_lr`**: relative band power of the current epoch, then logistic regression.
3. **`bandpower_ctx_lr`**: the same, plus band power of the two epochs on each side.
4. **`yasa_lgbm`**: [YASA](https://github.com/raphaelvallat/yasa) 0.8.1 features (smoothed over several minutes), then LightGBM with YASA's published settings, retrained in each fold.
5. **`yasa_pretrained`**: YASA's own classifier, which never saw Sleep-EDF. It is never trained here, so its score must be identical under every split: a built-in check on the evaluation code.

All hyperparameters are fixed in advance; nothing is tuned. Pretrained U-Sleep is excluded because it was trained on at least 75% of Sleep-EDF. The [model info sheet](docs/model_info_sheet.md) explains that exclusion and the case for leak-free training in the honest arm.

## Evaluation protocol

The same models, features, seeds (0–4) and number of folds are used in every arm; only the split rule changes.

| Arm | Held out together | What still leaks |
|---|---|---|
| `epoch` | single 30-s epochs | neighbouring epochs, the night, the person |
| `block` | 30-minute blocks, with an 8-epoch buffer dropped around each | the night, the person |
| `night` | whole recordings | the person's other night |
| `subject` (honest) | all of a person's nights | nothing |
| `subject_prefit` | as `subject`, but the scaler is fitted on all data first | scaling statistics |

- **Primary outcome:** per-subject Cohen's κ. The headline is the paired per-subject difference (honest minus leaky), with a 95% subject-level bootstrap interval.
- **Secondary outcomes:** pooled metrics, per-class F1, confusion matrices, calibration error, accuracy by distance to the nearest stage change, and night-level fragmentation.
- **Hypotheses, decision rules, expected sizes and pipeline-failure criteria** are in the [pre-registration](docs/preregistration.md). The [REFORMS checklist](docs/reforms_checklist.md) maps each reporting item to where it is addressed.

## Decisions (formerly open questions)

- **Expected size of the gap:** modest. The one controlled measurement on Sleep-EDF (Köksal 2026, SC-78) found about 0.10 macro-F1 and 0.11 κ between random epoch-wise and subject-wise splits, far below the near-100% vs 55–85% gaps reported for EEG disease-classification tasks. The pre-registration commits to 0.07–0.12 κ for the feature models, and to 1–2 accuracy points for night-wise vs subject-wise.
- **Sleep-EDF-20 or -78:** both, from one download. SC-78 is primary: twice as many subjects narrow the interval, and it matches Köksal. SC-20 is secondary and literature-comparable.
- **U-Sleep:** not used. Its pretrained weights saw most of Sleep-EDF, are not public, and the API was unreachable. Training it from scratch is possible future work, as a separately pre-registered addendum.

## Deliverables

- A repository that runs with one command.
- A results table: each model under each arm, with the honest-minus-leaky gap and its interval (after the full runs).
- A GitHub Actions workflow that runs unit tests, the label-count checks on all 153 hypnograms, and the 4-subject fixture.
- The pre-registration, written and tagged before the final run.
- A limitations section (with the results).

## Background sources

- **Leakage taxonomy:** Kapoor & Narayanan, <https://arxiv.org/html/2207.07048v1>. REFORMS: <https://reforms.cs.princeton.edu/>.
- **The gap on Sleep-EDF:** Köksal (2026), <https://dergipark.org.tr/en/download/article-file/5192156>.
- **Leakage in EEG deep learning:** Del Pup et al., <https://arxiv.org/html/2505.13021v1>. The gap was large on disease-classification tasks.
- **Subject-wise protocols on Sleep-EDF:**
  - DeepSleepNet: <https://arxiv.org/abs/1703.04046>
  - XSleepNet: <https://arxiv.org/abs/2007.05492>
  - SleepTransformer: <https://arxiv.org/abs/2105.11043>
- **YASA's weakness on fragmented sleep,** flagged by a peer reviewer: <https://doi.org/10.7554/eLife.70092.sa2>.
- **Full research notes:** [reports/Sleep staging leakage audit setup.md](reports/Sleep%20staging%20leakage%20audit%20setup.md).

## Timeline (estimate, not measured)

2–4 weeks.
