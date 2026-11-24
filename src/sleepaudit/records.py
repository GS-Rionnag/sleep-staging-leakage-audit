"""Recording names, pairing, subset selection and the committed file manifest."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PRIMARY_BASE = "https://physionet.org/files/sleep-edfx/1.0.0"
FALLBACK_BASE = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0"
MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "manifests" / "sc_files.csv"
MANIFEST_COLUMNS = ["filename", "kind", "subject", "night", "sha256", "size"]
_KINDS = {"-PSG.edf": "psg", "-Hypnogram.edf": "hypnogram"}


@dataclass(frozen=True)
class Recording:
    stem: str
    subject: int
    night: int
    psg_file: str
    hyp_file: str


def parse_name(filename: str) -> tuple[int, int]:
    """Return (subject, night) from a Sleep Cassette file name."""
    if not filename.startswith("SC4"):
        raise ValueError(f"not a Sleep Cassette file name: {filename!r}")
    return int(filename[3:5]), int(filename[5])


def pair_files(filenames: Iterable[str]) -> list[Recording]:
    """Pair PSG and hypnogram files on their first 7 characters (scorer codes differ)."""
    psg: dict[str, str] = {}
    hyp: dict[str, str] = {}
    for name in filenames:
        if name.endswith("-PSG.edf"):
            psg[name[:7]] = name
        elif name.endswith("-Hypnogram.edf"):
            hyp[name[:7]] = name
    missing = sorted(set(psg) - set(hyp))
    if missing:
        raise ValueError(f"PSG without hypnogram: {missing}")
    orphans = sorted(set(hyp) - set(psg))
    if orphans:
        raise ValueError(f"hypnogram without PSG: {orphans}")
    recs = []
    for stem in sorted(psg):
        subject, night = parse_name(stem)
        recs.append(Recording(stem, subject, night, psg[stem], hyp[stem]))
    return recs


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sha256": str})


def select_recordings(
    subset: str | list[int], manifest: pd.DataFrame | None = None
) -> list[Recording]:
    manifest = load_manifest() if manifest is None else manifest
    if isinstance(subset, list):
        keep = manifest[manifest["subject"].isin(subset)]
    elif subset == "sc78":
        keep = manifest
    elif subset == "sc20":
        keep = manifest[manifest["subject"] < 20]
    else:
        raise ValueError(f"unknown subset {subset!r}")
    return pair_files(keep["filename"])


def build_manifest(
    sha256sums_text: str, records_text: str, sizes: dict[str, int] | None = None
) -> pd.DataFrame:
    """Manifest of Sleep Cassette PSG and hypnogram files (sleep-telemetry is ignored).

    Every PSG listed in RECORDS must also appear in SHA256SUMS.
    """
    prefix = "sleep-cassette/"
    rows = []
    for line in sha256sums_text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip().startswith(prefix):
            continue
        name = parts[1].strip()[len(prefix):]
        kind = next((k for suffix, k in _KINDS.items() if name.endswith(suffix)), None)
        if kind is None:
            continue
        subject, night = parse_name(name)
        size = (sizes or {}).get(name)
        rows.append((name, kind, subject, night, parts[0].lower(), size))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df = df.sort_values(["subject", "night", "kind"], ignore_index=True)
    listed = {
        Path(r.strip()).name for r in records_text.splitlines() if r.strip().startswith(prefix)
    }
    absent = sorted(listed - set(df["filename"]))
    if absent:
        raise ValueError(f"in RECORDS but not in SHA256SUMS: {absent}")
    return df


def _get_text(rel: str) -> str:
    import requests

    last: Exception | None = None
    for base in (PRIMARY_BASE, FALLBACK_BASE):
        try:
            resp = requests.get(f"{base}/{rel}", timeout=60)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:  # try the mirror
            last = exc
    raise RuntimeError(f"could not fetch {rel}") from last


def _head_size(name: str) -> int | None:
    import requests

    for base in (PRIMARY_BASE, FALLBACK_BASE):
        try:
            resp = requests.head(f"{base}/sleep-cassette/{name}", timeout=60, allow_redirects=True)
            resp.raise_for_status()
            return int(resp.headers["Content-Length"])
        except (requests.RequestException, KeyError, ValueError):
            continue
    return None


def write_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    sums, records = _get_text("SHA256SUMS.txt"), _get_text("RECORDS")
    df = build_manifest(sums, records)
    df["size"] = [_head_size(n) for n in df["filename"]]
    df["size"] = df["size"].astype("Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return df
