# Data card: Sleep-EDF Expanded, Sleep Cassette subset

## Source

- **Dataset:** Sleep-EDF Expanded, version 1.0.0, on PhysioNet: <https://physionet.org/content/sleep-edfx/1.0.0/>.
- **Primary download:** `https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/`.
- **Fallback mirror:** the anonymous AWS mirror, `https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/`.
- **Integrity:** every file is checked against PhysioNet's `SHA256SUMS.txt`, whose Sleep Cassette lines are committed in `data/manifests/sc_files.csv` (153 PSG and 153 hypnogram files). A partial or corrupted download is never renamed into place.
- **Location:** data live outside the repository, under `$SLEEPAUDIT_DATA` (default `~/sleepaudit_data`). EDF files are never committed or uploaded as CI artifacts.

## Licence and citation

The data are under the **Open Data Commons Attribution License v1.0 (ODC-By 1.0)**, which requires citing:

- B. Kemp, A. H. Zwinderman, B. Tuk, H. A. C. Kamphuisen and J. J. L. Oberyé. "Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG." *IEEE Transactions on Biomedical Engineering* 47(9):1185–1194, 2000.
- A. L. Goldberger et al. "PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals." *Circulation* 101(23):e215–e220, 2000.

## What is used

**Sleep Cassette (SC) only.** The SC study (1987–1991) looked at age effects on sleep in healthy people aged 25–101 who took no sleep-related medication. It recorded two consecutive day-night periods of about 20 hours each, at the subjects' homes. The median recording here is 23.0 hours.

**Sleep Telemetry (ST) is never downloaded or used.** Its subjects had mild difficulty falling asleep and were recorded in hospital after temazepam or placebo, on different hardware. Pooling ST would mix a domain shift into the leakage comparison.

| | `sc20` (secondary) | `sc78` (primary) |
|---|---|---|
| Subjects | 20 (IDs 0–19) | 78 (IDs 0–82; 39, 68, 69, 78, 79 do not exist) |
| Recordings | 39 | 153 |
| Two-night subjects | 19 (subject 13 has night 1 only) | 75 (13 has night 1 only; 36 and 52 have night 2 only) |
| Sex (female / male) | 10 / 10 | 41 / 37 |
| Age range (median) | 25–34 (28) | 25–101 (57) |
| Kept 30-s epochs | 42,308 | 195,479 |
| W / N1 / N2 / N3 / REM | 8,285 / 2,804 / 17,799 / 5,703 / 7,717 | 65,951 / 21,522 / 69,132 / 13,039 / 25,835 |
| Class mix (%) | 19.6 / 6.6 / 42.1 / 13.5 / 18.2 | 33.7 / 11.0 / 35.4 / 6.7 / 13.2 |
| Fewest epochs per subject / per recording | 1,028 / 672 | 824 / 672 |

Sources: epoch counts were computed by this pipeline, and they match AttnSleep and DeepSleepNet-Lite exactly. Age and sex come from MNE's `age_records.csv`.

**Signals.** The pipeline reads two signals:

- `EEG Fpz-Cz` at 100 Hz, for every model.
- `EOG horizontal` at 100 Hz, for the YASA features only.

`EMG submental` is never used: in SC files it is a 1-Hz envelope, not raw EMG.

## Labels (the reference standard)

- **Scoring:** one human scorer per night, using the Rechtschaffen & Kales (R&K) rules. The scorer code (the eighth character of the hypnogram name) can differ between a subject's two nights.
- **Conversion:**
  - R&K stages 3 and 4 → N3.
  - W, 1, 2 and R → W, N1, N2, REM.
  - `Sleep stage ?`, `Movement time`, and epochs no annotation covers → dropped.
- **Epoch grid:** 30-s epochs from the PSG start. PSG and hypnogram start times agree for all 153 recordings, and every hypnogram's first annotation is at onset 0.
- **Clipping:** 150 of 153 hypnograms end in a `Sleep stage ?` block that runs past the end of the signal. Labels are clipped to the PSG length, which is read from the EDF header.
- **Trimming:** the dropped epochs are removed first. Then 60 epochs (30 min) of wake are kept before the first and after the last sleep epoch. This removes about half of the 416,424 epochs on the `sc78` grid; untrimmed, SC recordings are mostly wake.
- **Dropped epochs inside the trimmed range:** 302 in `sc78` and 61 in `sc20`, forming 141 interior gaps in 48 recordings (mostly 2 epochs long, the longest 98).

## Age structure: a trap for split design

**All 20 subjects with ID below 20 are aged 25–34. Every subject with ID 20 or above is aged 50–101 (median 67). No SC subject is aged 35–49.**

Assigning folds by contiguous ID ranges would therefore turn a split into an age-shift experiment. Folds here are assigned at random over groups, and `test_folds_not_contiguous_ids` guards this.

The two subsets also differ in age: `sc20` is a young, uniform cohort, while `sc78` is mostly older. Results for one do not transfer to the other without comment.

## Known limitations

- Single-scorer R&K labels. Human scorers agree with a consensus on about 83% of epochs, and on about 63% for N1, so neither honest nor leaky scores can be read as accuracy against the truth.
- Two nights per person at most, all from one study, one site and one recording setup. No external dataset is used, so results describe held-out subjects from this cohort only.
- The age gap (35–49 missing) and the healthy-volunteer sampling limit which populations the numbers describe.
- Wake trimming uses the reference labels. This is the field's convention, and it raises absolute scores: XSleepNet reported 2.4–3.7 accuracy points between wake policies.
