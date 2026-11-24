# Sleep-EDF Expanded (PhysioNet sleep-edfx 1.0.0): structure, subsets, loaders, preprocessing conventions

Notes as of 2026-11-24. "Computed" findings come from this researcher's own parsing of the primary files on PhysioNet: the RECORDS / RECORDS-v1 lists, the HTTP directory listings with byte sizes, all 197 hypnogram EDF+ files, and the first 4 KB (EDF header) of all 197 PSG files. Each epoch count below is a count of 30-s epochs. The computations were cross-checked against published numbers and agree to within 0–10 epochs (see Q1 and Q4). One exact cross-check: the computed trimmed counts for SC4001 (841) and SC4011 (1103) match the epoch counts printed in MNE's official tutorial.

## Q1. What exactly are Sleep-EDF-20 and Sleep-EDF-78? Subject IDs, files, one-night subjects, recording and epoch counts, and how the two compare

### Takeaway
Sleep-EDF-20 is the Sleep Cassette (SC) part of the 2013 release: SC subjects 0–19, 39 recordings, 1.96 GB of PSG. Sleep-EDF-78 is all SC data in the 2018 release: 78 subjects with IDs 0–82 (39, 68, 69, 78 and 79 do not exist), 153 recordings, 7.60 GB. In both, every subject has two nights except subjects 13, 36 and 52. The 2018 release removed none of these subjects; the only one of them inside Sleep-EDF-20 is subject 13, which has night 1 only. With the standard trimming (30 min of wake kept on each side of sleep, M and ? dropped, N3+N4 merged), Sleep-EDF-20 gives about 42.3k epochs and Sleep-EDF-78 about 195.5k. Sleep-EDF-20 contains only young adults (ages 25–34). Every subject with ID 20 or higher is aged 50 to 101. Moving from 20 to 78 subjects therefore changes the age range as well as the sample size.

### Cited Findings
- **Version history.** PhysioNet says that in 2013 "the database was greatly expanded to contain 61 PSGs with accompanying hypnograms", and in March 2018 it was "greatly expanded to contain 197 PSGs with accompanying hypnograms". A "Version 0" subset from 2002 is mentioned as well. — [PhysioNet sleep-edfx 1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/)
- **RECORDS-v1 (the 2013 list) is still shipped** in the 1.0.0 root folder (1,037 bytes) next to RECORDS (6,348 bytes), SC-subjects.xls, ST-subjects.xls and SHA256SUMS.txt. — [PhysioNet 1.0.0 file index](https://physionet.org/files/sleep-edfx/1.0.0/)
- **What RECORDS-v1 contains (computed).** It has 61 PSG files:
  - 39 SC files, SC4001E0 through SC4192E0, which are subjects 0–19. Subject 13 has only SC4131E0.
  - 22 ST files, one night per ST subject (ST7011J0, ST7022J0, ST7041J0, …, ST7241J0).
  - All 22 of those ST nights are the **placebo** nights, according to MNE's ST metadata table.
  - Sources: [RECORDS-v1](https://physionet.org/files/sleep-edfx/1.0.0/RECORDS-v1); placebo labels from [MNE temazepam_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/temazepam_records.csv)
- **What RECORDS contains (computed).** It has 197 PSG files: 153 under `sleep-cassette/` and 44 under `sleep-telemetry/`.
  - The SC files cover 78 subjects with IDs 0–82. IDs 39, 68, 69, 78 and 79 are absent.
  - Subjects with one night only:
    - Subject 13: night 1 only (`SC4131E0-PSG.edf`).
    - Subject 36: night 2 only (`SC4362F0-PSG.edf`).
    - Subject 52: night 2 only (`SC4522E0-PSG.edf`).
  - The other 75 SC subjects have exactly two nights.
  - The ST files cover 22 subjects with original IDs {1, 2, 4–22, 24}; IDs 3 and 23 are absent. Every ST subject has exactly two nights.
  - Source: [RECORDS](https://physionet.org/files/sleep-edfx/1.0.0/RECORDS)
- **PhysioNet's explanation of the missing nights:** "The first nights of subjects 36 and 52, and the second night of subject 13, were lost due to a failing cassette or laserdisk." — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
- **MNE gives the same facts.**
  - Valid subjects are "0-82 (inclusive), however the following subjects are not available: 39, 68, 69, 78 and 79".
  - "recording 1 for subject 36 and 52, and recording 2 for subject 13" are not available.
  - The SC set is "153 recordings from 37 males and 41 females that were 25-101 years old".
  - Source: [MNE age.py (main; identical signature on maint/1.13)](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age.py)
- **How the papers define the subsets.**
  - AttnSleep: "Sleep-EDF-20 contains data files for 20 subjects, while Sleep-EDF-78 is an expanded version with 78 subjects"; it "used the single Fpz-Cz channel" from SC only.
  - The AttnSleep README links Sleep-EDF-20 to a gist of the 2013 file list and Sleep-EDF-78 to the 1.0.0 page.
  - Sources: [AttnSleep paper, IEEE TNSRE 2021 (author PDF)](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf); [AttnSleep README](https://github.com/emadeldeen24/AttnSleep)
- **DeepSleepNet-Lite's description** of the 2018 SC data: "78 subjects (37 males and 41 females) aged from 25 to 101 years. Except for the first nights of subjects 36 and 52, and for the second night of subject 13 … resulting in 153 PSG recordings." It adds that "In the older upload there were only 39 PSG recordings from 20 subjects." — [DeepSleepNet-Lite, arXiv 2108.10600](https://arxiv.org/pdf/2108.10600)
- **Published epoch counts (AttnSleep Table I, Fpz-Cz, trimmed ±30 min):**

  | Subset | W | N1 | N2 | N3 | REM | Total |
  |---|---|---|---|---|---|---|
  | Sleep-EDF-20 | 8,285 (19.6%) | 2,804 (6.6%) | 17,799 (42.1%) | 5,703 (13.5%) | 7,717 (18.2%) | 42,308 |
  | Sleep-EDF-78 | 65,951 (33.7%) | 21,522 (11.0%) | 69,132 (35.4%) | 13,039 (6.7%) | 25,835 (13.2%) | 195,479 |

  Source: [AttnSleep paper](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf). DeepSleepNet-Lite Table II reports the same totals (42,308 and 195,479): [arXiv 2108.10600](https://arxiv.org/pdf/2108.10600)
- **This researcher's own counts (computed)** use the same rules: M and ? removed, stages 3 and 4 merged, 60 epochs kept before the first and after the last non-wake epoch.
  - Sleep-EDF-20: W 8,284 / N1 2,804 / N2 17,799 / N3 5,703 / REM 7,717 = 42,307.
  - Sleep-EDF-78: W 65,941 / N1 21,522 / N2 69,132 / N3 13,039 / REM 25,835 = 195,469.
  - Every sleep-stage count matches the papers exactly. Wake differs by 1 (SC-20) and 10 (SC-78) epochs. The likely cause is a boundary detail: DeepSleepNet removes M/? epochs before trimming, while this count did so after. That cause is an inference.
  - Source: computed from the [sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)
- **Per-recording and per-subject sizes (computed).**
  - Trimmed epochs per SC recording range from 672 (SC4051) to 2,667 (SC4731).
  - Trimmed epochs per SC subject (both nights) have min 824, median 2,322, max 4,985.
  - The one-night recordings give SC4131: 1,028, SC4362: 824, SC4522: 997.
  - Source: computed from the [sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)
- **Older count.** DeepSleepNet (Supratak 2017) is listed as using "41950 Fpz-Cz 30s" epochs with "20 patients LOO-CV". That is a few hundred fewer than 42,308. The difference is unexplained; it may come from the 2013-era files. This is reported secondhand in [Humayun et al. 2019, arXiv 1904.10255, Table II](https://arxiv.org/pdf/1904.10255) and was not checked against the DeepSleepNet paper itself.
- **Demographics (computed** from the age and sex columns of MNE's age_records.csv, which MNE builds from PhysioNet's SC-subjects.xls; see [age_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age_records.csv)):
  - SC-20 (subjects 0–19): ages 25–34, median 28, 10 female and 10 male.
  - SC-78: ages 25–101, median 57, 41 female and 37 male.
  - Subjects 20–82: ages 50–101, median 67.
  - No SC subject is aged 35–49.
- **Standard CV protocols in the literature are subject-wise.**
  - AttnSleep "adopted a subject-wise 20-fold cross-validation by dividing the subjects in each dataset into 20 groups … on Sleep-EDF-20 dataset with 20 subjects is thus leave-one-subject-out". — [AttnSleep paper](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf)
  - In AttnSleep's code, `load_folds_data` groups the .npz files by `file_name[3:5]` (the subject ID), so both nights stay in the same fold. It then applies a fixed permutation (`r_permute_20.npy` / `r_permute_78.npy`) and calls `np.array_split`. — [AttnSleep utils/util.py](https://github.com/emadeldeen24/AttnSleep/blob/main/utils/util.py)
  - DeepSleepNet's loader picks the held-out files with the regex `"[a-zA-Z0-9]*0{fold}[1-9]E0\.npz$"`, which selects both nights of subject = fold_idx. — [deepsleepnet data_loader.py](https://github.com/akaraspt/deepsleepnet/blob/master/deepsleep/data_loader.py)
  - TinySleepNet's sleepedf config sets `"n_folds": 20, "n_subjects": 20`. — [tinysleepnet config/sleepedf.py](https://github.com/akaraspt/tinysleepnet/blob/main/config/sleepedf.py)
- **Accuracy under subject-wise CV (computed from AttnSleep's confusion matrices, Tables II and III):**
  - Sleep-EDF-20: diagonal 35,759 of 42,308 = **84.5%**.
  - Sleep-EDF-78: diagonal 158,930 of 195,479 = **81.3%**.
  - Source: [AttnSleep paper](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf)
- **DeepSleepNet-Lite's data split.** Its Table III lists 20 subjects with 20-fold CV (4 validation, 1 test) for v1-2013 and 78 subjects with 20-fold CV (7 validation, 7 test) for v2-2018. The body text instead says k is 20 for v1-2013 and 10 for v2-2018, so the paper contradicts itself. Its results are 84.0% accuracy on v1-2013 ±30 min (Table IV, base+LSu with MC dropout). — [arXiv 2108.10600](https://arxiv.org/pdf/2108.10600)

### Inferences
- For code, a reproducible definition of the subsets:
  - `SC20 = {SC files whose subject = int(name[3:5]) is < 20}`, which gives 39 recordings.
  - `SC78 = all 153 SC files`.
  - Neither subset needs the ST files.
  - Defining SC-20 by subject ID gives exactly the RECORDS-v1 SC list, so no separate 2013 download is needed. The 2013 and 2018 hypnograms for those 39 files agree: the computed 2018-file counts reproduce the 2013 Sleep-EDF-20 numbers.
- **Comparison to support a recommendation:**
  - **Sleep-EDF-20.** 1.96 GB and 39 recordings. LOSO-20 is the literature standard, so results are comparable with DeepSleepNet, TinySleepNet and AttnSleep. It downloads quickly and suits CI. Its weaknesses are only 20 subjects (wide confidence intervals for subject-level variance) and a homogeneous cohort of 25–34-year-olds. That homogeneity may make the leaky-vs-honest gap look smaller than it would be on a heterogeneous population.
  - **Sleep-EDF-78.** 7.60 GB and 153 recordings. It spans ages 25–101, has about 4.6x more trimmed epochs, more wake inside the sleep period, and harder N1/N3 balance. Published subject-wise accuracy is about 3 pp lower than on Sleep-EDF-20 (81.3 vs 84.5 for AttnSleep). Greater between-subject heterogeneity should make subject-identity leakage more visible.
  - A reasonable design: SC-20 as the primary, fast, comparable benchmark; SC-78 as the scale-up and robustness check; one or two SC-20 subjects as the CI fixture.
- Because age is confounded with subject ID (every ID of 20 or more is aged 50+), a split by contiguous ID ranges (for example, train on 0–59 and test on 60–82) would also be an age-shift split. Subject-wise folds should be randomized over subject IDs, or stratified by age.

### Gaps
- The original 2013 web listing and the AttnSleep Sleep-EDF-20 gist were not fetched. The claim that the 2013 SC files equal subjects 0–19 rests on RECORDS-v1, which is authoritative.
- The exact cause of the 1-epoch and 10-epoch wake differences from AttnSleep was not traced line by line.
- The DeepSleepNet paper's own epoch count (41,950) was not checked against the paper itself.

## Q2. File naming, mapping file names to subject and night, ST naming, and the two-nights-per-subject structure

### Takeaway
Take the subject from characters [3:5] and the night from character [5] of the file name. Pair each PSG with its hypnogram using the first 7 characters. Do not hard-code the "E0"/"EC" suffixes: the 7th character varies (E/F/G) and the 8th character of a hypnogram name is the scorer's code. Each SC subject normally contributes two consecutive day-night recordings. Each ST subject contributes a placebo night and a temazepam night, and the night number does **not** tell you which is which.

### Cited Findings
- **PhysioNet's naming description.**
  - SC files are "SC4ssNEO-PSG.edf" and ST files are "ST7ssNJ0-PSG.edf", where ss is the subject number and N is the night.
  - "*PSG.edf" files hold the signals and "*Hypnogram.edf" files hold the annotations.
  - The scorer is identified by "the eighth letter of the hypnogram filename".
  - Source: [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
  - The page writes "EO", but the real files use the digit zero ("E0").
- **SC naming details (computed** from the [sleep-cassette listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)):
  - Character 7 of a PSG name is E in 102 files, F in 31 and G in 20. Subjects 0–25 are all "E". The first F/G files are SC4261F0 and SC4281G0.
  - A hypnogram always shares the first 7 characters of its PSG. Examples: SC4362F0-PSG ↔ SC4362FC-Hypnogram; SC4011E0-PSG ↔ SC4011EH-Hypnogram.
  - SC hypnograms use 11 different 8th-letter scorer codes: C 61, J 17, A 17, V 14, M 14, P 9, W 7, U 4, G 4, H 3, Y 3. The two nights of one subject can carry different codes, for example SC4021EH and SC4022EJ.
- **SC-20 pairs (computed)**, written as PSG / hypnogram suffix:

  ```
  SC4001E0/EC SC4002E0/EC SC4011E0/EH SC4012E0/EC SC4021E0/EH SC4022E0/EJ SC4031E0/EC SC4032E0/EP
  SC4041E0/EC SC4042E0/EC SC4051E0/EC SC4052E0/EC SC4061E0/EC SC4062E0/EC SC4071E0/EC SC4072E0/EH
  SC4081E0/EC SC4082E0/EP SC4091E0/EC SC4092E0/EC SC4101E0/EC SC4102E0/EC SC4111E0/EC SC4112E0/EC
  SC4121E0/EC SC4122E0/EV SC4131E0/EC SC4141E0/EU SC4142E0/EU SC4151E0/EC SC4152E0/EC SC4161E0/EC
  SC4162E0/EC SC4171E0/EU SC4172E0/EC SC4181E0/EC SC4182E0/EC SC4191E0/EP SC4192E0/EV
  ```

  Source: [sleep-cassette listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)
- **ST naming (computed).** Every ST PSG uses "J0", for example ST7011J0-PSG.edf. ST hypnogram 8th letters are A 10, E 10, R 8, O 6, M 4, P 2, W 2, J 2. — [sleep-telemetry listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/)
- **SC recordings span two days.** "Two PSGs of about 20 hours each were recorded during two subsequent day-night periods at the subjects homes." — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
- **ST recordings are one drug night and one placebo night**: "one of which was after temazepam intake, and the other of which was after placebo intake". — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
- **The ST night number does not encode the condition (computed).** In MNE's ST metadata (built from ST-subjects.xls), the placebo night is night 1 for 12 subjects and night 2 for 10. For example, ST7011 is Placebo and ST7012 is Temazepam. — [MNE temazepam_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/temazepam_records.csv)
- **MNE re-numbers ST subjects 0–21** with `data["subject"] = data.index // 2` and keeps the original ID in `subject_orig`. — [MNE _utils.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/_utils.py)
- **Braindecode parses IDs the same way**: `subj_nb = int(basename[3:5])` and `sess_nb = int(basename[5])`. — [braindecode sleep_physionet.py](https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/sleep_physionet.py)
- **The EDF header also identifies the subject.** SC4001E0's patient field reads "X F X Female_33yr", and the start is 24.04.89 at 16.13.00 (fetched through a byte-range request). — [S3 mirror of SC4001E0-PSG.edf](https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf)

### Inferences
- For leakage, the grouping key must be the **subject** (`name[3:5]`, plus the SC/ST prefix if both studies are mixed), not the recording. Any record-wise or night-wise split puts the same person in both train and test for 75 of 78 SC subjects.
- A recommended parser: `re.match(r"(SC4|ST7)(\d{2})(\d)([A-Z])", fname)` gives the study, the subject, the night and the recording letter. Pair files by the first 7 characters.
- If ST is ever used, the drug condition must be joined from ST-subjects.xls or MNE's CSV.
- The scorer code varies between recordings, so a small part of label variance is tied to the scorer rather than the subject. Scorer identity is probably not a big leakage channel, but it could be reported as a covariate.

### Gaps
- PhysioNet does not document what the 7th character (E/F/G) of the PSG name means. It may be a cassette or batch code, but this is unverified.
- No mapping from scorer letter to person was found.

## Q3. Channels and sampling rates, and which channels the literature uses

### Takeaway
SC files have 7 signals: Fpz-Cz EEG, Pz-Oz EEG and horizontal EOG at 100 Hz; respiration, submental EMG (as an rms envelope), rectal temperature and the event marker at 1 Hz. ST files have 5 signals, with EEG, EOG and EMG all at 100 Hz. The dominant choice in the literature is single-channel Fpz-Cz at 100 Hz with 3,000-sample (30 s) epochs.

### Cited Findings
- **SC header (computed** from the first 4 KB of all 153 SC PSG files, which are identical in layout; see the [sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)):
  - Labels: `['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','Resp oro-nasal','EMG submental','Temp rectal','Event marker']`.
  - Sampling rates: `[100,100,100,1,1,1,1]` Hz.
  - Data-record length: 30 s.
  - Prefilter strings:
    - EEG and EOG: "HP:0.5Hz LP:100Hz [enhanced cassette BW]".
    - EMG: "HP:16Hz Rectification LP:0.7Hz".
    - Respiration: "HP:0.03Hz LP:0.9Hz".
- **ST header (computed** from all 44 ST files; see the [sleep-telemetry files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/)):
  - Labels: `['EEG Fpz-Cz','EEG Pz-Oz','EOG horizontal','EMG submental','Marker']`.
  - Sampling rates: `[100,100,100,100,10]` Hz.
  - Data-record length: 10 s.
  - Prefilter: "LP:800Hz HP:0.03Hz".
- **PhysioNet's channel description.**
  - SC: "EEG and EOG at 100 Hz". Submental EMG is "electronically highpass filtered, rectified and low-pass filtered" and then sampled at 1 Hz as an rms envelope. Airflow, temperature and the marker are at 1 Hz.
  - ST: EOG, EMG and EEG at 100 Hz.
  - Scoring followed "the 1968 Rechtschaffen and Kales manual" using "Fpz-Cz/Pz-Oz EEGs instead of C4-A1/C3-A2 EEGs".
  - Source: [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
  - The page states the ST event marker is at 1 Hz, but the headers show 10 Hz for the ST "Marker" channel.
- **Fpz-Cz is the usual default.**
  - DeepSleepNet `prepare_physionet.py` defaults to `--select_ch "EEG Fpz-Cz"`. — [deepsleepnet](https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py)
  - TinySleepNet `prepare_sleepedf.py` defaults to `"EEG Fpz-Cz"`. — [tinysleepnet](https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py)
  - AttnSleep "used the single Fpz-Cz channel". — [AttnSleep paper](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf)
  - DeepSleepNet-Lite "considered the EEG Fpz-Cz channel, with a sampling rate of 100 Hz and without any pre-processing". — [arXiv 2108.10600](https://arxiv.org/pdf/2108.10600)
  - Humayun et al. used Fpz-Cz. — [arXiv 1904.10255](https://arxiv.org/pdf/1904.10255)
- **Braindecode** keeps only the two EEG channels by default (`load_eeg_only=True` excludes EOG, resp, EMG, temperature and marker) and renames "EEG Fpz-Cz" to "Fpz-Cz". — [braindecode sleep_physionet.py](https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/sleep_physionet.py)
- **The MNE tutorial** builds its features from `picks="eeg"`, that is both EEG channels, and reads the files with `stim_channel="Event marker"` and `infer_types=True`. — [MNE tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)

### Inferences
- Single-channel Fpz-Cz is the most comparable choice. Fpz-Cz + Pz-Oz (+ EOG) at 100 Hz is easy to support because all three share a rate.
- SC EMG is a 1 Hz envelope, not a raw EMG signal. Pipelines that use it have to resample or treat it separately.
- SC and ST EMG/EOG differ in sampling and filtering, so mixing SC and ST is a domain shift, which Humayun et al. also observed (see Q8).

### Gaps
- None material.

## Q4. Hypnogram labels and standard conventions (N3+N4 merge, dropping M and ?, wake trimming), and how much trimming changes class balance and accuracy

### Takeaway
The hypnograms use R&K labels: "Sleep stage W/1/2/3/4/R/?" and "Movement time". The near-universal conventions are:
- map 3 and 4 to N3 (AASM, 5 classes);
- drop M and ? epochs;
- keep only 30 min of wake before the first and after the last sleep epoch.

Trimming matters a great deal for SC, because each SC file is about 20–24 h long:
- **SC-20:** wake falls from about 68% to about 20% of epochs, and the total from 106k to 42k epochs.
- **SC-78:** wake falls from 69% to 34%.
- The stricter "in-bed only" variant cuts wake further, to 15% (SC-20) or 26% (SC-78).

The only within-paper accuracy comparison found shows that keeping ±30 min instead of in-bed only raises accuracy by about 1.4–1.6 pp. Studies that kept all wake and used non-independent splits report 91–96% accuracy.

### Cited Findings
- **Label set.** "sleep stages W, R, 1, 2, 3, 4, M (Movement time) and ? (not scored)", scored by R&K 1968. — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Annotation strings and counts as they appear in the EDF+ files (computed).**

  | Annotation | SC count | ST count |
  |---|---|---|
  | 'Sleep stage W' | 3,698 | 763 |
  | 'Sleep stage 1' | 5,944 | 1,348 |
  | 'Sleep stage 2' | 6,378 | 1,793 |
  | 'Sleep stage 3' | 3,538 | 1,282 |
  | 'Sleep stage 4' | 1,015 | 447 |
  | 'Sleep stage R' | 1,560 | 393 |
  | 'Sleep stage ?' | 174 | none |
  | 'Movement time' | 123 | 73 |

  These are counts of annotation blocks. One block can span many 30-s epochs, and every sleep-stage block duration is a multiple of 30 s. Source: [sleep-cassette](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/) / [sleep-telemetry](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/) hypnograms
- **The DeepSleepNet convention, in code** ([prepare_physionet.py](https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py)):
  - `ann2label` maps "Sleep stage 4" to 3 and "Movement time" to UNKNOWN (5). Epochs labelled UNKNOWN are removed.
  - The trimming step is then:

    ```python
    w_edge_mins = 30
    nw_idx = np.where(y != stage_dict["W"])[0]
    start_idx = nw_idx[0] - (w_edge_mins * 2)
    end_idx = nw_idx[-1] + (w_edge_mins * 2)
    ```

  - Both ends are clipped to the array bounds.
- **TinySleepNet does the same.** It maps `"Sleep stage 3": 3, "Sleep stage 4": 3, # Follow AASM Manual`, uses `w_edge_mins = 30` with the same start/end rule, and has a "Remove movement and unknown" step. — [tinysleepnet prepare_sleepedf.py](https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py)
- **AttnSleep's preprocessing, in its own words:** "we excluded UNKNOWN stages … merged stages N3 and N4 into one stage (N3) according AASM standard … include only 30 minutes of wake periods before and after the sleep periods". — [AttnSleep paper](https://personal.ntu.edu.sg/ctguan/Publications/2021_Emad_IEEE_TNSRE.pdf)
- **Two trimming schemes in the literature.** DeepSleepNet-Lite: "In many recordings there were long wake periods before the patients went to sleep and after they woke up … two common ways these periods are trimmed in literature: 1) only in-bed parts are employed, i.e. from light-off time to light-on time; 2) 30 minutes of data before and after in-bed parts". Its Table II gives the counts:

  | Dataset and trimming | Epochs | W share |
  |---|---|---|
  | v1-2013 ±30 min | 42,308 | 19.6% |
  | v1-2013 in-bed | 38,961 | 15.2% |
  | v2-2018 ±30 min | 195,479 | 33.7% |
  | v2-2018 in-bed | 163,948 | 26.3% |

  Source: [DeepSleepNet-Lite, arXiv 2108.10600](https://arxiv.org/pdf/2108.10600)
- **Accuracy under each scheme.** DeepSleepNet-Lite reports "an overall accuracy equal to 86.1% on v1-2013 ±30mins (84.5% on in-bed only) and equal to 82.3% on v2-2018 ±30mins (80.9% on in-bed only)". These are its best numbers, given in the section on its uncertainty-based epoch-rejection procedure, so they may include rejection of uncertain epochs. Without rejection, its v1-2013 ±30 min accuracy is 84.0% (Table IV). — [arXiv 2108.10600](https://arxiv.org/pdf/2108.10600)
- **Untrimmed class balance (computed**, within the PSG duration, M and ? dropped, 3+4 merged; source: [sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)):

  | Subset | Untrimmed epochs (W share) | Trimmed ±30 min epochs (W share) |
  |---|---|---|
  | SC-20 | 106,414 (68.0%) | 42,307 (19.6%) |
  | SC-78 | 414,961 (68.8%) | 195,469 (33.7%) |
  | ST | 42,691 (10.4%) | 42,467 (9.9%) |

  Trimming barely changes ST, because ST is only the in-bed night.
- **Humayun et al.'s untrimmed counts** for SC subjects 0–19: S1 2,804, S2 17,799, S3 3,370, S4 2,333, REM 7,717, W 71,887, total 105,910 (W about 67.9%). — [Humayun et al. 2019, arXiv 1904.10255, Table I](https://arxiv.org/pdf/1904.10255)
- **Reported accuracy depends on splits and wake handling.** Humayun et al. Table II separates "Not independent" epoch splits from subject-independent ones on SC data:
  - Epoch splits: 80/20, 90/10 and 10-fold CV, reporting 5-stage accuracy of 95.5% (Aboalayon 2016, 23,806 10-s epochs) and 91.7% (Sharma 2018).
  - Subject-independent LOO-CV: 82.0% (DeepSleepNet), 82.5% (Phan 2018) and 93.0% (Jiang 2019).
  - The paper notes: "Evaluation metrics are susceptible to changes in class distributions, number of epochs used in the evaluation task and also the epoch selection scheme" and reports "a 8.8% reduction in the accuracy of [11]" when that method was re-run with more epochs and a patient-independent split.
  - Source: [arXiv 1904.10255](https://arxiv.org/pdf/1904.10255)
- **Wearable-sleep study on easy wake.** A 2026 PubMed-indexed study titled "What matters beyond model choice for wearable sleep staging? How personalization, evaluation choices, and easy-to-classify wake impact performance" was found by search. The search snippet says easy-to-classify wake epochs "can markedly inflate performance metrics". Only the title and snippet were seen; the abstract was not retrieved. — [PubMed 42333378](https://pubmed.ncbi.nlm.nih.gov/42333378/)
- **Imtiaz & Rodriguez-Villegas (EMBC 2015)** argued that Sleep-EDF recordings had been "used in an inconsistent fashion" and released a MATLAB toolbox for downloading, choosing segments, converting hypnograms and scoring. This comes from the search summary and the repository; the paper PDF returned 404. — [sleep-edfx-toolbox GitHub](https://github.com/anasimtiaz/sleep-edfx-toolbox); [Spiral record](https://spiral.imperial.ac.uk/handle/10044/1/29344)
- **Lights-off times exist for SC and ST, but no lights-on times were found.** For SC, MNE's age_records.csv carries a "lights off" column from SC-subjects.xls, for example SC4001 at 00:38:00 and SC4002 at 21:57:00. — [MNE age_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age_records.csv); [MNE _utils.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/_utils.py)

### Inferences
- In this audit, wake trimming is a confound to fix, not a free parameter. If trimming is left out, about 68% of SC epochs are daytime or evening wake. That makes accuracy look high under any split, and the trivial "always W" baseline already scores about 68%.
- Recommendation: use ±30 min trimming (the DeepSleepNet/AttnSleep convention, whose counts can be checked against 42,308 and 195,479). Report the majority-class baseline and macro-F1 or κ next to accuracy. Optionally, show trimming as a secondary ablation.
- Trim per recording **before** any split. The trimming rule uses only that recording's own labels, so it creates no cross-split leakage. It does use labels to choose test epochs, which all of the literature also does.

### Gaps
- No controlled study was found that compares accuracy on Sleep-EDF with no trimming against ±30 min trimming under the same model and split. Only class-balance numbers and the DeepSleepNet-Lite in-bed vs ±30 min comparison are available.
- How the "in-bed" variant obtains lights-on times is unverified. SC-subjects.xls appears to hold lights-off only, as seen through MNE's CSV.
- The SleepTransformer and L-SeqSleepNet (Phan) preprocessing details were not retrieved.

## Q5. MNE-Python: the `sleep_physionet` fetchers (current API) and the official sleep-staging tutorial

### Takeaway
MNE 1.13.2 (released 2026-09-11) is the latest release. `mne.datasets.sleep_physionet.age.fetch_data(subjects, recording=(1, 2), path=None, force_update=False, base_url=..., on_missing="raise", *, verbose=None)` downloads SC files per subject and night, checks each against a SHA1 hash, and returns `[[psg_path, hyp_path], ...]`. Files are cached in `~/mne_data/physionet-sleep-data/` unless overridden. `temazepam.fetch_data` returns **only the placebo night** of the 22 ST subjects. The tutorial trains on subject 0, night 1 and tests on subject 1, night 1: a strict one-subject-to-one-subject split, with no epoch-level leakage. It reaches 68.7% accuracy using relative band powers and a random forest.

### Cited Findings
- **Versions.** The latest MNE on PyPI is 1.13.2, uploaded 2026-09-11. Earlier releases: 1.13.0 (2026-09-09), 1.12.0 (2026-04-07), 1.11.0 (2025-11-21). The tutorial page header reads "MNE 1.13.2 documentation". — [PyPI mne](https://pypi.org/project/mne/); [MNE tutorial](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)
- **The `age.fetch_data` signature is identical on `main` and `maint/1.13`:**

  ```python
  fetch_data(subjects, recording=(1, 2), path=None, force_update=False,
             base_url=BASE_URL, on_missing="raise", *, verbose=None)
  ```

  - `BASE_URL = "https://physionet.org/physiobank/database/sleep-edfx/sleep-cassette/"`, which is a legacy path.
  - `subjects` may be 0–82 except 39, 68, 69, 78 and 79.
  - `recording` takes [1], [2] or [1, 2].
  - `on_missing` takes 'raise', 'warn' or 'ignore'. The docstring wrongly says "Default is 'error'"; the actual default is "raise".
  - The function returns a list of [PSG path, Hypnogram path] pairs.
  - Sources: [age.py main](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age.py); [age.py maint/1.13](https://github.com/mne-tools/mne-python/blob/maint/1.13/mne/datasets/sleep_physionet/age.py)
- **How downloading works.**
  - Files are downloaded with `pooch.retrieve(url=..., known_hash=f"sha1:{hashsum}")`. The SHA1 hashes and file names come from the bundled `age_records.csv` (columns: subject, night, record type, age, sex, lights off, sha, fname).
  - A file is skipped if it already exists, unless `force_update=True`.
  - The data path comes from `_get_path(path, "PHYSIONET_SLEEP_PATH", ...)` followed by `"physionet-sleep-data"`. The fallback order is the `PHYSIONET_SLEEP_PATH` config or environment variable, then `MNE_DATA`, then `~/mne_data`.
  - Sources: [_utils.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/_utils.py); [age.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age.py); [datasets/utils.py maint/1.13](https://github.com/mne-tools/mne-python/blob/maint/1.13/mne/datasets/utils.py)
- **`temazepam.fetch_data(subjects, path=None, force_update=False, base_url=BASE_URL, *, verbose=None)`.**
  - `subjects` runs 0–21 (re-indexed).
  - The base URL is `.../sleep-edfx/sleep-telemetry/`.
  - The loop keeps only rows where `records["record"][idx] == b"Placebo"`, so it **returns one placebo night per subject** and can never fetch the temazepam nights.
  - It has no `recording` or `on_missing` parameters.
  - Source: [temazepam.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/temazepam.py)
- **Download sizes per record** (computed; files are decimal MB):
  - SC PSG files: 37.2–52.5 MB, median 50.3 MB. Example: SC4001E0-PSG.edf is 48,338,048 bytes.
  - SC hypnograms: about 3.9–5 KB each.
  - ST PSG files: 22.1–31.6 MB.
  - Source: [sleep-cassette listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/); [sleep-telemetry listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/)
- **What the tutorial does** ([MNE 1.13.2 tutorial "Sleep stage classification from polysomnography (PSG) data"](https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html)):
  - **Data.** `ALICE, BOB = 0, 1` and `fetch_data(subjects=[ALICE, BOB], recording=[1])`.
  - **Split.** Alice (SC4001) is the training set and Bob (SC4011) is the test set. There is no within-subject split.
  - **Labels.** The mapping is `{"Sleep stage W":1, "Sleep stage 1":2, "Sleep stage 2":3, "Sleep stage 3":4, "Sleep stage 4":4, "Sleep stage R":5}`, so the final classes are W, 1, 2, 3/4 and R.
  - **Wake cropping.** `annot_train.crop(annot_train[1]["onset"] - 30 * 60, annot_train[-2]["onset"] + 30 * 60)`, described as keeping "the last 30-min wake events before sleep and first 30-min wake events after".
  - **Epochs.** 30-s epochs via `tmax = 30.0 - 1.0 / sfreq`. Alice gives 841 epochs and Bob gives 1,103.
  - **Features.** `epochs.compute_psd(picks="eeg", fmin=0.5, fmax=30.0)` (multitaper by default, per the page log), then normalised to relative power. Bands: delta 0.5–4.5, theta 4.5–8.5, alpha 8.5–11.5, sigma 11.5–15.5 and beta 15.5–30 Hz.
  - **Classifier.** `make_pipeline(FunctionTransformer(eeg_power_band), RandomForestClassifier(n_estimators=100, random_state=241))`.
  - **Results.** Accuracy 0.6872. Per-class F1: W 0.64, N1 0.24, N2 0.86, N3/4 0.86, R 0.26. Macro-F1 0.57.
  - **Exercise.** Readers are invited to fetch 50 subjects and run 5-fold CV, "leaving 10 subjects out" each time.
- **This researcher's trimming reproduces the tutorial's counts:** 841 epochs for SC4001 and 1,103 for SC4011. — computed from the [sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)

### Inferences
- The tutorial's `annot[1]` / `annot[-2]` cropping works only because SC hypnograms start with a single W block and end with a trailing "?" block (see Q8). It is brittle as a general trimming rule. A label-array rule, as in DeepSleepNet, is safer.
- The tutorial is a useful "honest" reference point: 1 train subject, 1 test subject, 68.7% accuracy. Its suggested exercise (subject-held-out 5-fold CV) matches the honest split this project plans.
- For CI, `fetch_data(subjects=[0, 1], recording=[1])` downloads about 99.5 MB (48.3 + 51.1 MB) and is cached across runs if `PHYSIONET_SLEEP_PATH` / `MNE_DATA` points to a cached directory.

### Gaps
- MNE's own release notes were not checked for 2026 changes to sleep_physionet. The source at `maint/1.13` and `main` matches what is quoted here.

## Q6. Braindecode `SleepPhysionet`: what it wraps, its parameters, and splitting in its examples

### Takeaway
`braindecode.datasets.SleepPhysionet` is a thin wrapper around MNE's `age.fetch_data` that covers SC only. Its parameters are `subject_ids`, `recording_ids`, `preload`, `load_eeg_only=True`, `crop_wake_mins=30` and `crop`. It stores `subject` and `recording` in each dataset's description, which makes grouped splits easy. Its sleep-staging example splits **by subject**.

### Cited Findings
- **Versions.** The latest stable braindecode on PyPI is 2.1.0 (2026-11-23); 2.0.0 and 1.8.2 were both released 2026-11-02. The online "stable" docs still render as "Braindecode 2.0.0 documentation". The source quoted here is from the `master` branch. — [PyPI braindecode](https://pypi.org/project/braindecode/); [braindecode example page](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)
- **Signature and behaviour** (source: [braindecode/datasets/sleep_physionet.py (master)](https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/sleep_physionet.py)):
  - The signature is `SleepPhysionet(subject_ids=None, recording_ids=None, preload=False, load_eeg_only=True, crop_wake_mins=30, crop=None)`.
  - `subject_ids=None` becomes `range(83)`.
  - It calls `fetch_data(subject_ids, recording=recording_ids, on_missing="warn")` and retries up to 5 times, 10 s apart, on `requests` errors.
  - It reads files with `mne.io.read_raw_edf(..., exclude=[EOG, Resp, EMG, Temp, Event marker])` when `load_eeg_only` is set, and attaches the hypnogram with `raw.set_annotations(annots, emit_warning=False)`.
- **`crop_wake_mins` in braindecode:**
  - It finds the first and last annotations whose description ends in 1, 2, 3, 4 or R.
  - It then calls `raw.crop(tmin=first_onset - crop_wake_mins*60, tmax=last_onset + last_duration + crop_wake_mins*60)`, clipped to the recording.
  - Its docstring says the purpose is "to reduce the imbalance in this dataset".
- **Descriptions and docstring errors.** Each recording gets `desc = pd.Series({"subject": subj_nb, "recording": sess_nb})`. The docstring says "each subject except 13 has two recordings"; that is incomplete, because 36 and 52 also have one.
- **Example usage** (source: [Chambon 2018 sleep-staging example, docs 2.0.0](https://braindecode.org/stable/auto_examples/applied_examples/plot_sleep_staging_chambon2018.html)):
  - `SleepPhysionet(subject_ids=[0, 1], recording_ids=[2], crop_wake_mins=30)`.
  - Scaling ×1e6 to µV and a 30 Hz low-pass filter.
  - `create_windows_from_events` with 3,000-sample windows, and stages 3 and 4 merged ("following AASM standards").
  - The split is `split_ids = dict(train=subject_ids[::2], valid=subject_ids[1::2])`, described as "taking every other subject as train or valid".
  - The example has no separate test set and does not discuss leakage.

### Inferences
- Braindecode's `ds.split("subject")` or a `GroupKFold` on `ds.description["subject"]` gives the honest split directly. A random split over windows across `BaseConcatDataset` would reproduce the leaky condition.
- Braindecode crops wake at the annotation-block level, not at the epoch level. It also crops before M/? removal, which happens implicitly through the window mapping. Its epoch counts can therefore differ slightly from DeepSleepNet's. A project should pick one trimming implementation and test it against the 42,308 / 195,479 counts.
- Braindecode depends on MNE's fetcher, so it inherits the same subject coverage, cache location and SHA1 checks.

### Gaps
- Other braindecode sleep examples (U-Sleep, Eldele 2021) could not be retrieved; raw GitHub example paths returned nothing. Only the Chambon example's split was verified.

## Q7. Download options and sizes (full, SC-only, CI subsets), mirrors and terms

### Takeaway
The full download is 8.1 GB. SC PSG files total 7.60 GB; the SC-20 PSG files total 1.96 GB; ST totals 1.12 GB. All hypnograms together are under 1 MB. Individual files can be fetched over HTTPS from physionet.org; the server answers with a 302 redirect, which curl follows with -L. They can also be fetched anonymously from the AWS Open Data bucket `s3://physionet-open/sleep-edfx/1.0.0/`, which supports HTTP range requests. This makes small CI subsets cheap: one SC recording is about 50 MB.

### Cited Findings
- **Official download options:**
  - ZIP of 8.1 GB.
  - `physionet download sleep-edfx --version 1.0.0`.
  - `wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/`.
  - `aws s3 sync --no-sign-request s3://physionet-open/sleep-edfx/1.0.0/ DESTINATION`.
  - Licence: "Open Data Commons Attribution License v1.0".
  - Required citations: Kemp et al., *IEEE-BME* 47(9):1185–1194 (2000), plus the standard PhysioNet citation (the page now lists Pollard et al. 2026, *Nature Health*, doi:10.1038/s44360-026-00096-z). Dataset DOI: 10.13026/C2X676.
  - Source: [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
- **The AWS Open Data registry lists Sleep-EDF.** — [registry.opendata.aws/sleep-edfx](https://registry.opendata.aws/sleep-edfx/)
- **The S3 mirror works without credentials** (verified 2026-11-24 by this researcher). `https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/RECORDS` returned HTTP 200 (6,348 bytes), and a byte-range GET on `.../sleep-cassette/SC4001E0-PSG.edf` returned HTTP 206. — [S3 RECORDS](https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/RECORDS)
- **Direct HTTPS file URLs** follow the pattern `https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/<FILE>`. This researcher downloaded all 197 hypnograms and 197 header byte-ranges from that pattern without errors. A HEAD request returns `302 Found`, so use `curl -L`. — [sleep-cassette listing](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/)
- **Sizes (computed** from the directory listings, decimal units; sources: [sleep-cassette](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/), [sleep-telemetry](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/)):
  - SC PSG files: 153 files, 7.596 GB.
  - SC hypnograms: 153 files, 0.678 MB.
  - SC-20 PSG files: 39 files, 1.959 GB (hypnograms 0.164 MB).
  - ST PSG files: 44 files, 1.118 GB.
  - Per-file sizes: SC PSG 37.2–52.5 MB; ST PSG 22.1–31.6 MB.
  - All files are dated 05-Apr-2018, except SHA256SUMS.txt (20-Feb-2019).
- **Integrity checks.** SHA256SUMS.txt (39,799 bytes) has entries such as `sleep-cassette/SC4512EW-Hypnogram.edf`. MNE uses its own bundled SHA1 list. — [SHA256SUMS.txt](https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt); [MNE _utils.py](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/_utils.py)
- **Access terms.** "Anyone can access the files, as long as they conform to the terms of the specified license." — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)

### Inferences
- A sensible CI fixture size:
  - Subjects 0 and 1, night 1 only: SC4001E0 + SC4011E0 plus 2 hypnograms, about 99.5 MB.
  - Or, for a grouped-split smoke test where every subject has two nights, subjects 0–3 with both nights: about 400 MB.
  - Hypnogram-only tests, such as label parsing, trimming and epoch-count assertions against 42,308 and 195,469/195,479, can run on the full set of 153 SC hypnograms, which total 0.68 MB, without downloading any PSG.
- Prefer the S3 mirror, or cache `~/mne_data` with `actions/cache`, in CI to avoid repeated hits on physionet.org. Pin files by SHA256 from SHA256SUMS.txt.
- Header-only range requests of about 4 KB are enough to check channel names and sampling rates without full downloads.

### Gaps
- No published PhysioNet rate limit or bandwidth policy was found. MNE and braindecode download one file at a time, and braindecode adds retries.
- The ZIP's exact byte size was not checked; the page says 8.1 GB, which is consistent with the computed 7.6 + 1.1 GB.

## Q8. Known data-quality issues, errata and label quirks

### Takeaway
PhysioNet lists no formal errata beyond the three lost nights. The files nevertheless have several structural quirks that loading code must handle:
- SC recordings run about 17–24 h and start in the afternoon, with sleep beginning a median of 7.5 h after recording start.
- 150 of 153 SC hypnograms end in a "Sleep stage ?" block that runs past the end of the PSG.
- ST hypnograms often start after the PSG starts and end before it ends.
- There are 24 "?" blocks in mid-recording positions (not the final annotation) across all files.
- ST data differ from SC in hardware, filtering, EMG sampling, setting and drug condition.

### Cited Findings
- **PhysioNet notes only the lost recordings** (SC subjects 36 and 52 night 1, subject 13 night 2), and that "?" means "not scored". — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/)
- **SC recording length and timing (computed).**
  - SC PSG durations are 17.0–24.0 h (median 22.97 h); SC-20 durations are 21.4–23.9 h.
  - The first sleep epoch comes 0.38–11.26 h after PSG start (median 7.49 h).
  - SC4001E0 starts at 16:13:00 according to its header.
  - Source: [sleep-cassette files](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/); [S3 SC4001E0 header](https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf)
- **Hypnogram vs PSG extent (computed).**
  - All 153 SC hypnograms start at onset 0 with "Sleep stage W". 150 of them end with "Sleep stage ?", which extends beyond the PSG end by as much as 420 min (SC4592). The other 3 end with W, and 1 W block also extends past the end.
  - In ST, 26 of 44 hypnograms have a first annotation onset greater than 0, so scoring starts after the recording starts. 8 of them begin with Stage 1, not W.
  - In ST, the hypnogram ends up to 82 min before the PSG ends (ST7192), leaving 2,672 unscored ST epochs. 5 ST W blocks extend past the PSG end.
  - Source: [sleep-cassette](https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/) / [sleep-telemetry](https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry/) hypnograms
- **The start times of each PSG and its hypnogram match** in all 197 pairs (EDF header startdate/starttime compared). No sleep-stage annotation had a duration that is not a multiple of 30 s. — computed from both directories
- **Movement and unscored epochs within the PSG duration (computed).** SC-20 has 61 M epochs and 905 unscored ("?" or uncovered) epochs. SC-78 has 128 M and 1,335 unscored. ST has 83 M and 2,672 unscored. There are 24 "?" blocks that are not the final annotation. — computed from the hypnograms
- **DeepSleepNet trims excess labels at the tail.** Its prep code has a "Remove extra index" step for when labels outrun the signal: "if len(label_idx) > len(select_idx) … Trim the tail". This shows that the tail mismatch is a known issue. — [deepsleepnet prepare_physionet.py](https://github.com/akaraspt/deepsleepnet/blob/master/prepare_physionet.py)
- **Braindecode suppresses MNE's warning** about annotations outside the data range (`set_annotations(annots, emit_warning=False)`). — [braindecode sleep_physionet.py](https://github.com/braindecode/braindecode/blob/master/braindecode/datasets/sleep_physionet.py)
- **SC and ST are heterogeneous.** Humayun et al. found accuracy drops on ST recordings when SC and ST are pooled. In a one-way ANOVA on Fpz-Cz features, all features except γ-band EnergySis "exhibit a significant difference in the distribution with a p ≪ 0.001", which "strengthens our argument regarding heterogeneity between the subsets". — [arXiv 1904.10255](https://arxiv.org/pdf/1904.10255)
- **ST study design.** ST was a hospital study of temazepam in subjects with "mild difficulty falling asleep". Per MNE's metadata, ST subjects are aged 18–79 (15 female, 7 male). Hardware and filters differ from SC (see Q3). — [PhysioNet sleep-edfx](https://physionet.org/content/sleep-edfx/1.0.0/); [MNE temazepam_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/temazepam_records.csv)
- **Lights-off times.** SC lights-off times are in SC-subjects.xls, and in MNE's CSV as "lights off", e.g. 00:38:00 for SC4001. They are not stored in the EDF files. ST lights-off times are per night in ST-subjects.xls. — [MNE age_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/age_records.csv); [MNE temazepam_records.csv](https://github.com/mne-tools/mne-python/blob/main/mne/datasets/sleep_physionet/temazepam_records.csv)

### Inferences
- **Loader checklist:**
  1. Clip labels to the PSG length, or let MNE crop the annotations.
  2. Drop "Sleep stage ?" and "Movement time" epochs.
  3. Map 3 and 4 to N3.
  4. Trim ±30 min around the first and last non-W epoch.
  5. Do not assume the hypnogram starts at t = 0 (it does not in ST).
  6. Assert the per-subset totals: SC-20 = 42,307/42,308, SC-78 = 195,469/195,479.
- Keep ST out of the core audit, or treat it as a separate cohort. It differs in domain (hardware, EMG, filtering, drug condition), and pooling it with SC would mix a domain shift into the leakage comparison.

### Gaps
- Kemp et al. (2000) was not retrieved; any errata or known misalignment reported there are unverified.
- No independent report of misaligned annotations in specific Sleep-EDF recordings was found. Checks here were limited to start-time agreement, epoch-multiple durations and tail extents, and found no misalignment.
- Lights-on times were not found in any file examined.
