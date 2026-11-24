# Sleep Staging Leakage Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-command, CI-tested Python repo that measures how much a sleep-staging score on Sleep-EDF (Sleep Cassette) is inflated by leaky splits, comparing four split arms plus a pipeline-leak condition across five models, with per-subject paired bootstrap intervals and a pre-registered protocol.

**Architecture:** A staged pipeline: manifest → download (checksummed) → labels (clip, drop, trim) → per-recording features (band power + YASA) cached as parquet → splits (the only place folds are made) → out-of-fold predictions → per-subject metrics → paired bootstrap + fragmentation strata → CSVs and figures. Every stage is a CLI subcommand of `python -m sleepaudit`; `all` chains them. Labels and splits are tested without any signal data; signal-dependent stages are tested on a 4-subject fixture.

**Tech Stack:** Python 3.12, uv (locked), MNE 1.13, YASA 0.8.1 (+ LightGBM), scikit-learn ≥1.6, numpy, scipy, pandas, pyarrow, pyyaml, requests, matplotlib, pytest, ruff, GitHub Actions.

**Spec:** `reports/Sleep staging leakage audit setup.md` (primary; its "Ten steps" table and section headings are the source of every value below) and `README.md` (scope and deliverables). Supporting notes: `research_notes/Sleep staging leakage audit setup/`.

## Global Constraints

- Python `>=3.11` (YASA 0.8.1 requirement); develop on 3.12.
- `scikit-learn>=1.6` (needed for `GroupKFold(shuffle=True)`), `mne>=1.13,<1.14`, `yasa[full]>=0.8.1,<0.9`.
- Data: Sleep-EDF Expanded 1.0.0, **Sleep Cassette (SC) only**. ST files are never downloaded or used.
- Subsets: `sc20` = SC subjects 0–19 (20 subjects, 39 recordings); `sc78` = all SC (78 subjects, 153 recordings; IDs 39, 68, 69, 78, 79 do not exist).
- Signal: channel `"EEG Fpz-Cz"`, 100 Hz, 30-s epochs (3,000 samples). Never use `"EMG submental"`.
- Label codes: `W=0, N1=1, N2=2, N3=3, REM=4`; stages 3 and 4 → N3; `"Movement time"`, `"Sleep stage ?"` and epochs not covered by any annotation → `-1` and dropped.
- Trimming (DeepSleepNet order): clip labels to PSG length → drop `-1` epochs → keep from 60 epochs before the first non-W to 60 epochs after the last non-W (inclusive, clipped to bounds).
- Folds: `sc78` K=10, `sc20` K=20; same K for every arm; seeds `[0, 1, 2, 3, 4]`. Never assign folds by contiguous subject-ID ranges.
- Primary metric: per-subject Cohen's κ (labels `[0,1,2,3,4]`). Bootstrap: 10,000 resamples of subjects, 95% percentile interval, paired.
- Hyperparameters are fixed (no tuning): logistic regression `class_weight="balanced"`, `max_iter=1000`, otherwise defaults; LightGBM `n_estimators=500, max_depth=5, num_leaves=90, colsample_bytree=0.6`, class weights `{N1: 2.2, N2: 1, N3: 1.2, REM: 1.4, W: 1}`.
- Data directory from env var `SLEEPAUDIT_DATA` (default `~/sleepaudit_data`), always outside the repo. EDF files are never committed or uploaded as CI artifacts.
- Download URLs: primary `https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/<name>`, fallback `https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/sleep-cassette/<name>`.
- Wording: not a medical tool; never "diagnose", "detect", "screen"; say "held-out subjects", never "external validation".
- The pre-registration section is committed and tagged `prereg-v1` before any full-data (`sc20`/`sc78`) model run.
- All paths via `pathlib`; every module importable on Windows and Linux.
- Compute: develop and run the 4-subject fixture on the Windows laptop. Full `sc20`/`sc78` runs and any GPU work run on the user's server `server` (`ssh server`; Ubuntu 24.04, RTX 3090 24 GB, 32 cores, 62 GB RAM, ~38 GB free disk as of 2026-11-24). Code reaches the server through a private bare git remote on it (`server:~/repos/sleep-staging-leakage-audit.git`), never GitHub. Long runs go inside `tmux` so an SSH drop doesn't kill them. Check `df -h ~` before downloading; stop if under 15 GB free.

## Review Focus

1. **Interrupted or corrupted download**: a partial file must never be mistaken for a good one. The downloader writes `<name>.part` and renames only after SHA256 matches (test in Task 3).
2. **Stale cache after a config change**: changing a preprocessing parameter must invalidate cached features, not silently reuse them (test in Task 5).
3. **A recording with no sleep epochs after dropping `-1`**: trimming must raise `ValueError` naming the recording, not crash on an empty index or return everything (test in Task 4).
4. **One-night subjects (13, 36, 52)** must work in night-wise and subject-wise arms, and K larger than the number of groups must raise a clear error instead of producing empty folds (tests in Task 6).
5. **Degenerate per-subject κ** (a subject where both truth and prediction are constant gives NaN): metrics keep NaN, the bootstrap drops those subjects pairwise and reports how many it dropped (tests in Tasks 7 and 9).

---

## File Structure

```
pyproject.toml, uv.lock, .gitattributes, .gitignore, .python-version
configs/ci.yaml, configs/sc20.yaml, configs/sc78.yaml
data/manifests/sc_files.csv          # committed: filename, kind, subject, night, sha256, size
src/sleepaudit/__init__.py
src/sleepaudit/__main__.py           # CLI dispatch
src/sleepaudit/config.py             # Config dataclass + YAML loader + data dir
src/sleepaudit/records.py            # filename parsing, pairing, subset selection, manifest build
src/sleepaudit/download.py           # checksummed fetch with fallback, EDF header duration
src/sleepaudit/labels.py             # annotations -> epoch labels, clip/drop/trim
src/sleepaudit/features.py           # band power, context features, per-recording feature cache
src/sleepaudit/yasa_model.py         # YASA features, LightGBM factory, pretrained predictions
src/sleepaudit/splits.py             # the only place folds are made
src/sleepaudit/models.py             # model registry (sklearn estimators)
src/sleepaudit/evaluate.py           # out-of-fold predictions, per-subject + pooled metrics, ECE
src/sleepaudit/stats.py              # paired subject bootstrap, Wilcoxon
src/sleepaudit/fragmentation.py      # transition distance, strata, night-level summaries
src/sleepaudit/report.py             # summary tables + figures
src/sleepaudit/runmeta.py            # meta.json
tests/conftest.py + one test file per module
docs/preregistration.md, docs/model_info_sheet.md, docs/reforms_checklist.md, docs/data_card.md
.github/workflows/ci.yml
```

Shared data shape used across tasks — **the epoch table** (`pandas.DataFrame`, one row per kept epoch):
`recording: str` (7-char stem, e.g. `"SC4001E"`), `subject: int`, `night: int`, `epoch_no: int` (index on the recording's original 30-s grid), `label: int8`, then feature columns with prefixes `bp_` (band power), `bpctx_` (context), `yasa_` (YASA features), `yasa_proba_` (pretrained probabilities, order W,N1,N2,N3,REM).

---

### Task 1: Repo skeleton, config and CLI

**Files:**
- Create: `pyproject.toml`, `.python-version` (`3.12`), `.gitattributes` (`*.edf binary`, `*.txt text eol=lf`), `.gitignore` (`.venv/`, `results/*/cache/`, `*.part`), `configs/ci.yaml`, `configs/sc20.yaml`, `configs/sc78.yaml`, `src/sleepaudit/__init__.py`, `src/sleepaudit/__main__.py`, `src/sleepaudit/config.py`, `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Config`: `name: str`, `subset: Literal["sc20","sc78"] | list[int]` (explicit subject IDs for fixtures), `k: int`, `seeds: list[int]`, `arms: list[str]`, `models: list[str]`, `n_boot: int = 10000`, `block_epochs: int = 60`, `buffer_epochs: int = 8`, `trim_epochs: int = 60`, `results_dir: Path`.
  - `load_config(path: Path) -> Config`; `data_dir() -> Path` (reads `SLEEPAUDIT_DATA`, creates it).
  - CLI: `python -m sleepaudit <command> --config <path>` with commands `build-manifest`, `fetch-labels`, `fetch-psg`, `features`, `evaluate`, `report`, `all` (stubs that raise `NotImplementedError` until their task lands).
  - Pytest markers `needs_labels` and `needs_psg`. `conftest.py` skips them when the data is absent, unless env `SLEEPAUDIT_REQUIRE_DATA=1`, in which case missing data fails the test.

Config values: `ci.yaml` → `subset: [0, 1, 2, 3]`, `k: 2`, `seeds: [0]`, `n_boot: 1000`, all five arms, all five models. `sc20.yaml` → `subset: sc20`, `k: 20`. `sc78.yaml` → `subset: sc78`, `k: 10`. Both full configs: seeds `[0,1,2,3,4]`, arms `[epoch, block, night, subject, subject_prefit]`, models `[majority, bandpower_lr, bandpower_ctx_lr, yasa_lgbm, yasa_pretrained]`.

- [x] **Step 1:** `git init`, create the files above, `uv init --package --name sleepaudit` adjusted to the `src/` layout, then `uv add` the Tech Stack packages with the Global Constraints bounds and `uv add --dev pytest ruff`. Commit `uv.lock`.
  - *Note (executed 2026-11-24): `uv init` defaults to `requires-python >=3.12`; set to `>=3.11` per Global Constraints. uv was missing and was installed with `winget install astral-sh.uv` (uv 0.13.9).*
- [x] **Step 2: Write failing tests** in `tests/test_config.py`:
  - `test_load_ci_config`: `load_config(Path("configs/ci.yaml"))` has `subset == [0,1,2,3]`, `k == 2`, `"subject" in arms`.
  - `test_full_configs_pin_k`: `sc78.yaml` → `k == 10`; `sc20.yaml` → `k == 20`; both `seeds == [0,1,2,3,4]`.
  - `test_unknown_arm_rejected`: a YAML with `arms: [random]` raises `ValueError` mentioning `"random"`.
  - `test_data_dir_env`: with `SLEEPAUDIT_DATA` set to `tmp_path / "dir with space"`, `data_dir()` returns that path and it exists.
- [x] **Step 3:** Run `uv run pytest tests/test_config.py -v`. Expected: FAIL (`ImportError`).
- [x] **Step 4:** Implement `config.py` and the CLI dispatcher (argparse).
  - *Note (executed 2026-11-24): `--config` is optional for `build-manifest` only (Task 2 runs it without a config). `VALID_ARMS`/`VALID_MODELS` live in `config.py`; `splits.ARMS` (Task 6) must equal `VALID_ARMS`.*
- [x] **Step 5:** Run `uv run pytest -v && uv run ruff check . && uv run python -m sleepaudit --help`. Expected: tests pass, ruff clean, help lists all seven commands.
- [x] **Step 6:** Commit: `chore: repo skeleton, config loader and CLI`.

---

### Task 2: Record names, subsets and the committed manifest

**Files:**
- Create: `src/sleepaudit/records.py`, `data/manifests/sc_files.csv`
- Modify: `src/sleepaudit/__main__.py` (wire `build-manifest`)
- Test: `tests/test_records.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Recording`: `stem: str` (first 7 chars), `subject: int`, `night: int`, `psg_file: str`, `hyp_file: str`.
  - `parse_name(filename: str) -> tuple[int, int]`: subject from `filename[3:5]`, night from `filename[5]`. Raises `ValueError` for names not starting with `SC4`.
  - `pair_files(filenames: Iterable[str]) -> list[Recording]`: pairs `*-PSG.edf` with `*-Hypnogram.edf` on the first 7 characters; raises if any PSG lacks a hypnogram.
  - `load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame` (columns `filename, kind, subject, night, sha256, size`).
  - `select_recordings(subset: str | list[int], manifest: pd.DataFrame | None = None) -> list[Recording]`.
  - `build_manifest(sha256sums_text: str, records_text: str) -> pd.DataFrame` (CLI fetches `SHA256SUMS.txt` and `RECORDS` from PhysioNet, falling back to the S3 mirror, keeps only `sleep-cassette/` entries, writes the CSV). `size` may be filled from HTTP `Content-Length` via a HEAD request.

- [x] **Step 1: Write failing tests:**
  - `test_parse_name`: `parse_name("SC4001E0-PSG.edf") == (0, 1)`; `parse_name("SC4022EJ-Hypnogram.edf") == (2, 2)`; `parse_name("ST7011J0-PSG.edf")` raises `ValueError`.
  - `test_pair_scorer_codes_differ`: `pair_files(["SC4021E0-PSG.edf","SC4021EH-Hypnogram.edf","SC4022E0-PSG.edf","SC4022EJ-Hypnogram.edf"])` yields stems `["SC4021E","SC4022E"]`.
  - `test_pair_missing_hypnogram_raises`.
  - `test_sc20_shape` (reads committed manifest): 39 recordings, 20 subjects, subject 13 has nights `{1}`.
  - `test_sc78_shape`: 153 recordings, 78 subjects, missing IDs `{39,68,69,78,79}`, subjects 36 and 52 have nights `{2}`, exactly 75 subjects have two nights.
  - `test_explicit_subset`: `select_recordings([0,1,2,3])` → 8 recordings.
  - `test_build_manifest_filters_st`: a fake `SHA256SUMS` text with one `sleep-cassette/` and one `sleep-telemetry/` line yields one row.
- [x] **Step 2:** Run `uv run pytest tests/test_records.py -v`. Expected: FAIL.
- [x] **Step 3:** Implement `records.py`; run `uv run python -m sleepaudit build-manifest` once and commit the generated `data/manifests/sc_files.csv`.
  - *Note (executed 2026-11-24): `build_manifest` gained an optional `sizes` argument; the CLI fills `size` from HEAD requests (306 files, none missing). Manifest holds 153 PSG + 153 hypnogram rows.*
- [x] **Step 4:** Run tests. Expected: PASS.
- [x] **Step 5:** Commit: `feat: record parsing, subsets and SC manifest`.

---

### Task 3: Checksummed downloader and EDF header duration

**Files:**
- Create: `src/sleepaudit/download.py`
- Modify: `src/sleepaudit/__main__.py` (wire `fetch-labels`, `fetch-psg`)
- Test: `tests/test_download.py`

**Interfaces:**
- Consumes: `Recording`, `load_manifest`, `select_recordings`, `data_dir()`.
- Produces:
  - `Getter = Callable[[str, dict[str, str]], bytes]` (url, headers → body; raises `requests.HTTPError` on non-2xx). Default implementation uses `requests` with 3 retries and a timeout.
  - `fetch_file(name: str, sha256: str, dest_dir: Path, get: Getter = http_get) -> Path`: skips if the existing file's hash matches; else tries primary then fallback URL; writes `<name>.part`, verifies SHA256, renames. On mismatch deletes the `.part` and raises `ChecksumError`.
  - `fetch_header(name: str, dest_dir: Path, get: Getter = http_get) -> Path`: `Range: bytes=0-4095`, saved as `<name>.header` (no checksum; documented).
  - `edf_duration_seconds(header: bytes) -> float`: `int(header[236:244]) * float(header[244:252])`.
  - CLI `fetch-labels`: all hypnograms + PSG headers for the config subset into `data_dir()/sleep-cassette/`. `fetch-psg`: full PSG files.

- [x] **Step 1: Write failing tests** (all use a fake `Getter`, no network):
  - `test_checksum_mismatch_leaves_no_file`: fake returns `b"bad"`; `ChecksumError` raised; neither `<name>` nor `<name>.part` exists.
  - `test_fallback_used_on_primary_error`: fake raises `HTTPError` for physionet.org URLs and returns good bytes for the S3 URL; file written, both URLs were called in that order.
  - `test_existing_good_file_skipped`: second call makes zero getter calls.
  - `test_edf_duration`: synthetic 256-byte header with `"84      "` at 236:244 and `"30      "` at 244:252 → `2520.0`.
- [x] **Step 2:** Run `uv run pytest tests/test_download.py -v`. Expected: FAIL.
- [x] **Step 3:** Implement `download.py`.
- [x] **Step 4:** Run tests (PASS), then `uv run python -m sleepaudit fetch-labels --config configs/sc78.yaml` and confirm 153 hypnograms + 153 headers (~1 MB) land in the data dir.
- [x] **Step 5:** Commit: `feat: checksummed downloader with S3 fallback`.

---

### Task 4: Epoch labels, clipping and trimming

**Files:**
- Create: `src/sleepaudit/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `edf_duration_seconds`, `Recording`.
- Produces:
  - `LABEL_MAP: dict[str, int]` (`"Sleep stage W"→0, "1"→1, "2"→2, "3"→3, "4"→3, "R"→4, "?"→-1, "Movement time"→-1`).
  - `epoch_labels(onsets: np.ndarray, durations: np.ndarray, descriptions: Sequence[str], psg_seconds: float) -> np.ndarray` (int8, length `floor(psg_seconds/30)`, `-1` where uncovered; annotations past the PSG end are clipped; onsets need not start at 0).
  - `trim(labels: np.ndarray, edge: int = 60, name: str = "") -> np.ndarray`: returns the **original-grid indices** of kept epochs: drop `-1` first, then on the remaining sequence keep `[first_nonW - edge, last_nonW + edge]` inclusive, clipped. Raises `ValueError(f"{name}: no sleep epochs")` if no non-W epoch.
  - `recording_labels(rec: Recording, root: Path, edge: int = 60) -> pd.DataFrame` with columns `recording, subject, night, epoch_no, label` (reads the hypnogram via `mne.read_annotations`, duration from the `.header` file or the full PSG).

- [x] **Step 1: Write failing tests:**
  - `test_epoch_labels_merges_n4_and_clips`: annotations W 0–60 s, "4" 60–120 s, "?" 120–600 s with `psg_seconds=150` → `[0,0,3,3,-1]`.
  - `test_epoch_labels_nonzero_onset`: first annotation at 60 s → first two epochs are `-1`.
  - `test_trim_drops_unknown_before_trimming`: a label array where a `-1` run sits inside the 60-epoch wake margin; the kept indices skip it and still include 60 valid W epochs before the first non-W.
  - `test_trim_all_wake_raises`: `ValueError` containing the recording name.
  - `@needs_labels test_sc20_counts`: concatenated labels for `sc20` → per-class counts N1 2,804; N2 17,799; N3 5,703; REM 7,717 exactly, W within 1 of 8,285; total within 1 of 42,308.
  - `@needs_labels test_sc78_counts`: N1 21,522; N2 69,132; N3 13,039; REM 25,835 exactly; W within 10 of 65,951; total within 10 of 195,479.
  - `@needs_labels test_tutorial_recordings`: `SC4001E` → 841 epochs, `SC4011E` → 1,103.
- [x] **Step 2:** Run `uv run pytest tests/test_labels.py -v`. Expected: FAIL.
- [x] **Step 3:** Implement `labels.py`.
- [x] **Step 4:** Run with data present and `SLEEPAUDIT_REQUIRE_DATA=1`. Expected: PASS. If a count test is off by more than its tolerance, stop and investigate the trimming order before going on; every later number depends on it.
  - *Note (executed 2026-11-24): All counts matched exactly, including W and totals, on the first run; no trimming-order investigation was needed.*
- [x] **Step 5:** Commit: `feat: label parsing, clipping and ±30 min wake trimming`.

---

### Task 5: Band-power and context features with a per-recording cache

**Files:**
- Create: `src/sleepaudit/features.py`
- Modify: `src/sleepaudit/__main__.py` (wire `features`)
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: `recording_labels`, `select_recordings`, `data_dir()`, `Config`.
- Produces:
  - `BANDS = {"delta": (0.5, 4.5), "theta": (4.5, 8.5), "alpha": (8.5, 11.5), "sigma": (11.5, 15.5), "beta": (15.5, 30.0)}` (MNE tutorial bands).
  - `band_power(epochs: np.ndarray, sfreq: float = 100.0) -> np.ndarray` shape `(n, 5)`, relative power (rows sum to 1), Welch PSD per epoch.
  - `context_features(df: pd.DataFrame, cols: list[str], width: int = 2) -> pd.DataFrame`: for each row adds the `cols` of the epochs at offsets `-width..+width` (excluding 0) **within the same recording by `epoch_no` order**, edge-padded by repeating the nearest epoch; columns named `bpctx_{col}_{offset:+d}`.
  - `load_signal_epochs(rec: Recording, root: Path, epoch_nos: np.ndarray) -> np.ndarray` (`(n, 3000)` float32 from `"EEG Fpz-Cz"` via `mne.io.read_raw_edf(include=[...])`).
  - `cache_key(rec: Recording, sha256: str, params: dict) -> str` (hash of PSG checksum + feature parameters + package version).
  - `build_feature_table(cfg: Config) -> pd.DataFrame`: the epoch table with `bp_*` and `bpctx_*` columns (plus `yasa_*` once Task 8 registers its extractor); per-recording parquet cached at `data_dir()/cache/<cache_key>.parquet`; the combined table written to `cfg.results_dir / "features.parquet"`.

- [x] **Step 1: Write failing tests:**
  - `test_band_power_alpha_sine`: 10 Hz sine epochs → argmax column is alpha; rows sum to 1 (atol 1e-6).
  - `test_context_does_not_cross_recordings`: two recordings of 3 epochs each; the `-1` offset of recording B's first epoch equals B's own first epoch, not A's last.
  - `test_cache_key_changes_with_params`: changing `trim_epochs` or `BANDS` changes the key.
  - `@needs_psg test_feature_table_rows_match_labels`: for subject 0, `build_feature_table` row count equals the `recording_labels` row count and `epoch_no` values match.
- [x] **Step 2:** Run `uv run pytest tests/test_features.py -v`. Expected: FAIL.
- [x] **Step 3:** Implement `features.py`.
- [x] **Step 4:** `uv run python -m sleepaudit fetch-psg --config configs/ci.yaml && uv run pytest tests/test_features.py -v` with `SLEEPAUDIT_REQUIRE_DATA=1`. Expected: PASS.
- [x] **Step 5:** Commit: `feat: band-power and context features with cache`.

---

### Task 6: Splits for the four arms

**Files:**
- Create: `src/sleepaudit/splits.py`
- Test: `tests/test_splits.py`

**Interfaces:**
- Consumes: the epoch table columns `recording, subject, night, epoch_no`.
- Produces:
  - `ARMS = ("epoch", "block", "night", "subject", "subject_prefit")`.
  - `Fold = NamedTuple("Fold", [("train", np.ndarray), ("test", np.ndarray)])` (positional row indices).
  - `make_folds(index: pd.DataFrame, arm: str, k: int, seed: int, block_epochs: int = 60, buffer_epochs: int = 8) -> list[Fold]`:
    - `epoch`: `KFold(k, shuffle=True, random_state=seed)` over rows.
    - `block`: group = `(recording, epoch_no // block_epochs)`; `GroupKFold(k, shuffle=True, random_state=seed)`; then drop from `train` every row of the same recording whose `epoch_no` is within `buffer_epochs` of any test row's `epoch_no`.
    - `night`: `GroupKFold(..., shuffle=True)` with `groups=recording`.
    - `subject` and `subject_prefit`: `GroupKFold(..., shuffle=True)` with `groups=subject` (identical folds; the arms differ only in preprocessing, see Task 7).
    - Raises `ValueError` if `k` exceeds the number of groups for the arm.

- [x] **Step 1: Write failing tests** on a synthetic index (subjects 0–5 with two nights each plus subject 13 with one night, 200 epochs per night):
  - `test_subject_arm_no_overlap`: for every fold, `set(subject[train]) & set(subject[test]) == set()`.
  - `test_night_arm_shares_subjects_not_recordings`: recordings disjoint, at least one fold shares a subject.
  - `test_epoch_arm_leaks`: at least one fold shares a subject.
  - `test_every_row_tested_once`: for each arm, concatenated `test` is a permutation of `range(n)`.
  - `test_block_buffer`: for every fold, no train row is within 8 epochs of a test row in the same recording.
  - `test_same_seed_same_folds` and `test_different_seed_different_folds`.
  - `test_one_night_subject_ok`: subject 13 appears in exactly one fold's test set for `night` and `subject`.
  - `test_k_too_large_raises`: `k=20` on the 7-subject index with `arm="subject"` raises `ValueError`.
  - `test_folds_not_contiguous_ids`: with 20 subjects and k=4, at least one test fold's subject IDs are not a contiguous range.
- [x] **Step 2:** Run `uv run pytest tests/test_splits.py -v`. Expected: FAIL.
- [x] **Step 3:** Implement `splits.py`.
- [x] **Step 4:** Run tests. Expected: PASS.
  - *Note (executed 2026-11-24): "within `buffer_epochs`" implemented as distance `<= buffer_epochs` (8 epochs = 4 min dropped each side; the spec requires the buffer to exceed YASA's 7.5-epoch window). Beyond the plan's list the file also tests: every arm raises for k > groups, subject_prefit folds identical to subject, block tests whole blocks, buffer=0 keeps all, unsorted row order, input not mutated, `ARMS == VALID_ARMS`. Mutation-checked: grouping subject by recording, off-by-one buffer, no buffer, no shuffle and block key without recording each fail at least one test.*
- [x] **Step 5:** Commit: `feat: leaky, blocked, night-wise and subject-wise splits`.

---

### Task 7: Models, out-of-fold predictions and per-subject metrics

**Files:**
- Create: `src/sleepaudit/models.py`, `src/sleepaudit/evaluate.py`
- Test: `tests/test_models.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `make_folds`, `Fold`, the epoch table.
- Produces:
  - `ModelSpec` dataclass: `name: str`, `columns: Callable[[pd.DataFrame], list[str]]` (selects feature columns by prefix), `factory: Callable[[int], BaseEstimator]` (seed → unfitted estimator), `has_scaler: bool`.
  - `MODELS: dict[str, ModelSpec]` with `majority` (`DummyClassifier("most_frequent")`), `bandpower_lr` (`bp_*` → `Pipeline([StandardScaler(), LogisticRegression(...)])`), `bandpower_ctx_lr` (`bp_*` + `bpctx_*`, same pipeline). Task 8 adds `yasa_lgbm` and `yasa_pretrained`.
  - `run_oof(spec: ModelSpec, table: pd.DataFrame, folds: list[Fold], seed: int, prefit_scaler: bool = False) -> pd.DataFrame` with columns `row, y_pred, proba_0..proba_4`, one row per epoch. With `prefit_scaler=True` (arm `subject_prefit`), a `StandardScaler` is fitted on **all** rows before the loop and the pipeline's own scaler is replaced by `"passthrough"`; only allowed when `spec.has_scaler`.
  - `per_subject_metrics(oof: pd.DataFrame) -> pd.DataFrame` (input has `subject, y_true, y_pred, proba_*`): one row per subject with `kappa, accuracy, macro_f1, ece, n_epochs`; κ uses `labels=[0,1,2,3,4]` and stays NaN when undefined.
  - `pooled_metrics(oof) -> dict[str, float]` (accuracy, κ, macro-F1, per-class F1 `f1_W..f1_REM`).
  - `expected_calibration_error(y_true, proba, n_bins=15) -> float` (max-probability, equal-width bins).

- [x] **Step 1: Write failing tests:**
  - `test_spy_scaler_fit_rows` (`test_evaluate.py`): a spy transformer recording fitted row indices; under arm `subject` it never sees a test row; under `subject_prefit` it was fitted once on all rows.
  - `test_prefit_rejected_without_scaler`: `run_oof(MODELS["majority"], ..., prefit_scaler=True)` raises `ValueError`.
  - `test_oof_covers_all_rows`: `len(oof) == len(table)` and `oof.row` is unique.
  - `test_majority_kappa_zero`: majority model gives κ = 0 for subjects with varied labels.
  - `test_kappa_nan_when_degenerate`: a subject with all-W truth and all-W predictions → κ is NaN, accuracy 1.0.
  - `test_ece_perfect_is_zero`: one-hot correct probabilities → ECE 0.
  - `test_bandpower_lr_learns` (`test_models.py`): on synthetic separable features, subject-wise accuracy > 0.9.
- [x] **Step 2:** Run `uv run pytest tests/test_models.py tests/test_evaluate.py -v`. Expected: FAIL.
- [x] **Step 3:** Implement `models.py` and `evaluate.py`.
- [x] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: model registry, out-of-fold evaluation and per-subject metrics`.

---

### Task 8: YASA features, retrained LightGBM and pretrained control

**Files:**
- Create: `src/sleepaudit/yasa_model.py`
- Modify: `src/sleepaudit/features.py` (call the YASA extractor in `build_feature_table`), `src/sleepaudit/models.py` (register two models)
- Test: `tests/test_yasa_model.py`

**Interfaces:**
- Consumes: `Recording`, `recording_labels`, `MODELS`, `ModelSpec`.
- Produces:
  - `yasa_features(rec: Recording, root: Path, epoch_nos: np.ndarray) -> pd.DataFrame`: crop the raw PSG to `[min(epoch_nos), max(epoch_nos)+1) * 30 s` (the trimmed segment, unfiltered and not z-scored), run `yasa.SleepStaging(raw, eeg_name="EEG Fpz-Cz", eog_name="EOG horizontal")`, take `get_features()` rows at `epoch_nos - min(epoch_nos)` (prefix columns `yasa_`), and add `yasa_proba_{W,N1,N2,N3,REM}` from `predict()` → `hyp.proba`.
  - `YASA_TO_CODE = {"WAKE": 0, "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4}`.
  - `PrecomputedClassifier(BaseEstimator, ClassifierMixin)`: `fit` is a no-op that sets `classes_ = [0..4]`; `predict_proba(X)` returns `X` (the five `yasa_proba_*` columns); `predict` returns argmax.
  - `MODELS["yasa_lgbm"]` (`yasa_*` minus `yasa_proba_*` → `LGBMClassifier` with the Global Constraints settings and `random_state=seed`, `has_scaler=False`); `MODELS["yasa_pretrained"]` (`yasa_proba_*` → `PrecomputedClassifier`, `has_scaler=False`).

- [ ] **Step 1: Write failing tests:**
  - `test_yasa_code_map`: every YASA stage string maps into `0..4`.
  - `test_precomputed_ignores_training`: fitting on any data then predicting a fixed probability matrix gives the argmax of that matrix.
  - `test_pretrained_split_invariant`: on a synthetic table with `yasa_proba_*` columns, per-subject κ from `run_oof` is identical (exact equality) for every arm in `ARMS`.
  - `@needs_psg test_yasa_rows_align`: for `SC4001E`, `yasa_features` returns one row per kept epoch with no NaN in `yasa_proba_*`.
- [ ] **Step 2:** Run `uv run pytest tests/test_yasa_model.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement; verify the exact feature-column names and `Hypnogram.proba` column labels against the installed YASA 0.8.1 before writing the mapping.
- [ ] **Step 4:** Run tests with `SLEEPAUDIT_REQUIRE_DATA=1`. Expected: PASS. Also run `uv run python -m sleepaudit features --config configs/ci.yaml` and record the per-night extraction time in the PR description.
- [ ] **Step 5:** Commit: `feat: YASA features, LightGBM model and pretrained control`.

---

### Task 9: Paired subject bootstrap and fragmentation strata

**Files:**
- Create: `src/sleepaudit/stats.py`, `src/sleepaudit/fragmentation.py`
- Test: `tests/test_stats.py`, `tests/test_fragmentation.py`

**Interfaces:**
- Consumes: per-subject metric frames from Task 7 (columns `subject, seed, kappa, ...`).
- Produces:
  - `@dataclass BootResult`: `mean_diff, lo, hi, frac_honest_lower, wilcoxon_p, n_subjects, n_dropped`.
  - `paired_bootstrap(honest: pd.DataFrame, leaky: pd.DataFrame, metric: str = "kappa", n_boot: int = 10000, seed: int = 0) -> BootResult`: average over seeds within subject, inner-join on subject, drop subjects NaN in either arm (count in `n_dropped`), `d = honest - leaky`, resample subjects with replacement, 2.5/97.5 percentiles; `wilcoxon_p` from `scipy.stats.wilcoxon(d)` (NaN if fewer than 6 subjects).
  - `transition_distance(labels: np.ndarray) -> np.ndarray`: per epoch, distance in epochs to the nearest index where `labels[i] != labels[i-1]`; computed per recording on the kept-epoch sequence.
  - `STRATA = {"0-1": (0, 1), "2-3": (2, 3), "4": (4, 4), "stable": (5, inf)}` and `stratum(dist: np.ndarray) -> np.ndarray[str]`.
  - `night_summary(table: pd.DataFrame) -> pd.DataFrame` per recording: `transition_share`, `n1_share`, `n_awakenings` (W runs after first sleep and before last sleep), `waso_min`, `age` (from MNE's `mne/datasets/sleep_physionet/age_records.csv`).

- [ ] **Step 1: Write failing tests:**
  - `test_bootstrap_known_shift`: 30 subjects, honest = leaky − 0.1 exactly → `mean_diff == -0.1`, `lo == hi == -0.1` (atol 1e-12), `frac_honest_lower == 1.0`.
  - `test_bootstrap_drops_nan_pairwise`: one subject NaN in leaky only → `n_dropped == 1`, `n_subjects` reduced by one.
  - `test_bootstrap_seed_averaging`: two seeds per subject with values 0.5 and 0.7 → the subject enters as 0.6.
  - `test_bootstrap_reproducible`: same seed → identical `lo, hi`.
  - `test_transition_distance`: `[0,0,0,2,2,2,2,2]` → `[3,2,1,0,1,2,3,4]` (the epoch where the change lands counts as distance 0; its predecessor 1).
  - `test_strata_bins`: distances `[0,1,2,3,4,5,9]` → `["0-1","0-1","2-3","2-3","4","stable","stable"]`.
  - `test_awakenings_count`: synthetic night with two W runs inside sleep → 2.
- [ ] **Step 2:** Run `uv run pytest tests/test_stats.py tests/test_fragmentation.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement both modules.
- [ ] **Step 4:** Run tests. Expected: PASS.
- [ ] **Step 5:** Commit: `feat: paired subject bootstrap and fragmentation strata`.

---

### Task 10: Orchestration, results files, figures and run metadata

**Files:**
- Create: `src/sleepaudit/report.py`, `src/sleepaudit/runmeta.py`
- Modify: `src/sleepaudit/__main__.py` (wire `evaluate`, `report`, `all`)
- Test: `tests/test_report.py`, `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces (under `cfg.results_dir`, e.g. `results/sc78/`):
  - `per_subject.csv`: `dataset, model, arm, seed, subject, kappa, accuracy, macro_f1, ece, n_epochs`.
  - `oof.parquet`: `recording, subject, epoch_no, y_true, y_pred, proba_0..4, model, arm, seed, trans_dist, stratum`.
  - `pooled.csv`: one row per `model, arm, seed` with `pooled_metrics` fields.
  - `summary.csv`: one row per `model` × comparison (`subject` vs each of `epoch, block, night, subject_prefit`) with `BootResult` fields.
  - `strata.csv`: accuracy per `model, arm, stratum`.
  - `nights.csv`: `night_summary` joined with per-recording κ per `model, arm`, plus a `tercile` column for `transition_share` and `n1_share`, so the lowest and highest thirds can be compared.
  - `figures/slopegraph_<model>.png` (per-subject κ, leaky arm vs `subject`, faint lines, bold mean ± CI, dashed majority baseline) and `figures/forest.png` (paired gaps with CIs per model and comparison).
  - `meta.json`: `commit, dirty, uv_lock_sha256, manifest_sha256, packages{name: version}, config, seeds, started, finished, host, platform, cpu_count, gpu` (`gpu` from `nvidia-smi --query-gpu=name --format=csv,noheader` or `null`).
  - `write_meta(cfg: Config, out: Path) -> dict`; `render_report(results_dir: Path) -> None`.
  - `all` = `fetch-labels → fetch-psg → features → evaluate → report`; skips `subject_prefit` for models without a scaler and logs that it did.
  - `evaluate` refuses to run a config whose subset is `sc20` or `sc78` unless the git tag `prereg-v1` exists in the current history (escape hatch: `--allow-without-prereg`, recorded in `meta.json`).

- [ ] **Step 1: Write failing tests:**
  - `test_summary_columns` (`test_report.py`): `render_report` on a synthetic `per_subject.csv` writes `summary.csv` with the columns above and both figure files.
  - `test_meta_fields`: `write_meta` output contains every key listed, and `dirty` is a bool.
  - `test_prereg_gate`: `evaluate` with an `sc78` config in a repo without the tag exits non-zero with a message naming `prereg-v1`.
  - `@needs_psg test_ci_smoke` (`test_cli_smoke.py`): `python -m sleepaudit all --config configs/ci.yaml` exits 0 and `summary.csv` has a row for each of the five models.
- [ ] **Step 2:** Run `uv run pytest tests/test_report.py tests/test_cli_smoke.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full suite with `SLEEPAUDIT_REQUIRE_DATA=1`. Expected: PASS; the smoke run completes in under 30 minutes on the laptop.
- [ ] **Step 5:** Commit: `feat: orchestration, results tables, figures and run metadata`.

---

### Task 11: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: pytest markers, `configs/ci.yaml`, `data/manifests/sc_files.csv`.

Jobs:
- `unit` (matrix `ubuntu-latest`, `windows-latest`): `astral-sh/setup-uv` pinned to a commit SHA with `enable-cache: true`; `uv sync --locked`; `uv run ruff check .`; `uv run pytest -m "not needs_labels and not needs_psg"`.
- `labels` (ubuntu): restore `actions/cache` keyed `labels-${{ hashFiles('data/manifests/sc_files.csv') }}` at `~/sleepaudit_data`; `fetch-labels --config configs/sc78.yaml`; `SLEEPAUDIT_REQUIRE_DATA=1 uv run pytest -m needs_labels`.
- `smoke` (ubuntu, `timeout-minutes: 30`, needs `unit`): cache keyed `fixture-${{ hashFiles('data/manifests/sc_files.csv', 'configs/ci.yaml') }}`; `uv run python -m sleepaudit all --config configs/ci.yaml`; `SLEEPAUDIT_REQUIRE_DATA=1 uv run pytest -m needs_psg`; upload `results/ci/` (CSVs, PNGs, `meta.json` only) with `retention-days: 14`.
- Triggers: `push`, `pull_request`, `workflow_dispatch`, and `schedule: cron "0 6 * * 1"` (weekly, keeps the cache from 7-day eviction).

- [ ] **Step 1:** Write the workflow.
- [ ] **Step 2:** Validate locally with `uvx actionlint` (or `uvx check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/ci.yml`). Expected: no errors.
- [ ] **Step 3:** Commit: `ci: unit, label-count and fixture smoke jobs`.
- [ ] **Step 4:** Ask the user before creating the GitHub repository and pushing (outward-facing). After the push, confirm all three jobs pass; the first `smoke` run downloads ~400 MB.

---

### Task 12: Pre-registration and reporting documents, then tag

**Files:**
- Create: `docs/preregistration.md`, `docs/model_info_sheet.md`, `docs/reforms_checklist.md`, `docs/data_card.md`
- Modify: `README.md` (link the pre-registration; replace "Open questions" with the decisions from the spec; add Quickstart: `uv sync --locked`, `uv run python -m sleepaudit all --config configs/sc78.yaml`)

Content pinned by the spec (section "Subjects are the unit of evidence"):
- **Phase A (before training):** question, outcome (per-subject κ), features, dataset (`sc78` primary K=10, `sc20` secondary K=20), split rules for all five arms, fixed hyperparameters, seeds, baselines.
- **Phase B (before testing):** methods, test-access statement ("development used only subjects 0–3 plus label-only checks on all 153 hypnograms"), secondary analyses (pooled metrics, per-class F1, confusion matrices, ECE, strata, night-level fragmentation, `sc20` vs `sc78`), deviations log (empty table).
- **Hypotheses:** H1 every trained model scores lower in `subject` than `epoch`; H2 gap ordering `epoch ≥ block ≥ night ≥ subject`; H3 the gap is larger for `bandpower_ctx_lr` and `yasa_lgbm` than for `bandpower_lr`; H4 the leaky gain concentrates in the `stable` stratum.
- **Expected sizes:** feature models `epoch` vs `subject` 5–10 accuracy points, 0.07–0.12 κ; `night` vs `subject` 1–2 points.
- **Success criterion:** H1 holds for a model if its 95% interval for mean `d` excludes 0; the headline claim stands if that holds for at least 2 of the 3 trained core models (`bandpower_lr`, `bandpower_ctx_lr`, `yasa_lgbm`). If every interval includes 0, report a null result.
- **Pipeline-failure criteria:** any honest fold shares a subject; `yasa_pretrained` κ differs across arms; epoch counts outside the Task 4 tolerances.
- **Model info sheet:** Kapoor–Narayanan argument for train-test separation, legitimate features, matching test distribution; plus the contamination table (U-Sleep excluded because its weights were trained on ≥75% of Sleep-EDF; braindecode's Chambon checkpoint excluded for the same reason).
- **REFORMS:** one row per item (32), each "addressed in <file/section>" or "N/A because …".
- **Data card:** source, licence (ODC-By 1.0, cite Kemp et al. 2000 and PhysioNet), SC-only, label conversions, age structure (IDs 0–19 aged 25–34, ≥20 aged 50–101).

- [ ] **Step 1:** Write the four documents and the README changes.
- [ ] **Step 2:** Self-check: every hypothesis names an arm and model that exist in `ARMS`/`MODELS`; every failure criterion maps to an existing test.
- [ ] **Step 3:** Commit: `docs: pre-registration, model info sheet, REFORMS and data card`.
- [ ] **Step 4:** `git tag -a prereg-v1 -m "Pre-registration frozen before any full-data run"`. Push the tag only with the user's go-ahead.

---

### Task 13: Full runs and write-up

**Files:**
- Modify: `README.md` (results table, figures, limitations), `docs/preregistration.md` (deviations log only)
- Create: `results/sc78/*`, `results/sc20/*` (CSVs, PNGs, `meta.json`; no parquet over 50 MB, no EDF)

- [ ] **Step 1:** On `server` (inside `tmux`, after `git pull` from the private remote and `uv sync --locked`): `uv run python -m sleepaudit all --config configs/sc78.yaml`, then `--config configs/sc20.yaml`. Expected: `meta.json` shows `dirty: false`, `host` = the server, and a commit descending from `prereg-v1`. Copy `results/` back to the laptop with `scp -r` (or commit on the server and pull) before writing up.
- [ ] **Step 2:** Check the pipeline-failure criteria from Task 12 against the outputs before reading any headline number. If one fires, fix, log the deviation, rerun.
- [ ] **Step 3:** Fill the README results table (per model: κ and accuracy under each arm, the `subject`-minus-`epoch` gap with its 95% CI), embed the forest plot and one slopegraph, report H1–H4 as supported / not supported exactly per the pre-registered criteria, and write the limitations section (single scorer per night, R&K labels, SC-only age gap 35–49, YASA domain shift, intervals reflect subject sampling only, no external dataset).
- [ ] **Step 4:** Commit: `results: sc78 and sc20 audit runs`. Push only with the user's go-ahead.

---

## Out of scope for this plan

Deep-model stretch work (braindecode `SleepStagerChambon2018` and `USleep`, both trained from scratch inside the folds) and the optional SLEEPYLAND reference. With the RTX 3090 on `server` these are now practical on SC-78, not just SC-20. Plan them separately after Task 13 (see Chat 6 in `chat-prompts.md`), reusing `splits.py`, `evaluate.py` and the results schema, and pre-register them as a deviation-free addendum before running.
