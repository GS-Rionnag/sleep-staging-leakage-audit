# Sleep staging leakage audit

Checks how much a sleep-staging AI score depends on a leaky evaluation, by comparing an honest subject-wise test against a random split on the same open data.

## Why this matters

A sleep-tracking product or research model can report a high accuracy number that was measured with the same person's nights in both training and test data. That number does not show how the model will work on a new person. This project measures the gap on open data so anyone reading a sleep-AI accuracy claim can ask whether it was tested on new people.

## Scope

- Not a medical tool. No diagnosis, no screening claims.
- Uses only public, de-identified data. No recruitment of people.
- Compares evaluation methods, not a product.

## Data

- **Sleep-EDF Expanded** (PhysioNet): 8.1 GB, 197 whole-night polysomnography recordings. Open Data Commons Attribution license, no registration.
  - Source: https://physionet.org/content/sleep-edfx/1.0.0/
  - Load with MNE-Python. The WFDB library does not decode the EDF+ annotations.
  - Decide which subset to use (Sleep-EDF-20 or -78). Not yet decided.

## Models

1. **Majority-class baseline**: always predict the most common stage.
2. **Feature baseline**: band-power features plus logistic regression.
3. **YASA**: open-source Python sleep-staging library (BSD-3), https://github.com/raphaelvallat/yasa
4. **U-Sleep** (optional): deep-learning sleep-staging model with public code. Whether it runs on free compute is unverified.

## Evaluation protocol

- **Leaky split (comparison only):** random split by 30-second epoch. Chunks from the same night can land in train and test.
- **Subject-wise split (the honest test):** every epoch from one person goes entirely into train or entirely into test. Validation data sits inside the training folds.
- Check for other leaks: fitting normalization or feature selection before the split.
- Report several random seeds, intervals, and per-person results.
- Error analysis by sleep fragmentation (short wake interruptions), since a peer reviewer flagged YASA as weaker there:
  https://doi.org/10.7554/eLife.70092.sa2

## Deliverables

- A repo that runs with one command.
- A results table: leaky vs. subject-wise, per model.
- A GitHub Actions workflow that runs a small subset.
- A README section written before the final run that states the split rules and what would count as a failure.
- A limitations section.

## Background sources

- Leakage in EEG deep-learning studies: https://arxiv.org/html/2505.13021v1 (the gap was large on disease-detection tasks; the size for sleep staging is untested here)
- Subject-wise cross-validation protocol example for Sleep-EDF: https://arxiv.org/pdf/2611.08311

## Timeline (estimate, not measured)

2–4 weeks.

## Open questions

- Size of the leaky-vs-honest gap for sleep staging.
- Sleep-EDF-20 or Sleep-EDF-78.
- Whether U-Sleep runs on free compute.
