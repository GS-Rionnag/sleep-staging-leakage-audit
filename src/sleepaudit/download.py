"""Checksummed downloads with a mirror fallback, and EDF header duration."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

import requests

from sleepaudit.config import Config, data_dir
from sleepaudit.records import FALLBACK_BASE, PRIMARY_BASE, load_manifest, select_recordings

Getter = Callable[[str, dict[str, str]], bytes]
HEADER_BYTES = 4096


class ChecksumError(RuntimeError):
    """Downloaded bytes did not match the manifest SHA256."""


def http_get(url: str, headers: dict[str, str]) -> bytes:
    """GET with 3 retries and a timeout; raises requests.HTTPError on non-2xx."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last = exc
            if attempt < 2:
                time.sleep(2**attempt)
    assert last is not None
    raise last


def _urls(name: str) -> list[str]:
    return [f"{base}/sleep-cassette/{name}" for base in (PRIMARY_BASE, FALLBACK_BASE)]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_first(urls: list[str], headers: dict[str, str], get: Getter) -> bytes:
    last: Exception | None = None
    for url in urls:
        try:
            return get(url, headers)
        except requests.RequestException as exc:
            last = exc
    assert last is not None
    raise last


def fetch_file(name: str, sha256: str, dest_dir: Path, get: Getter = http_get) -> Path:
    """Download `name` into `dest_dir`; the final path only ever holds verified bytes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / name
    if final.exists() and _sha256_file(final) == sha256:
        return final
    final.unlink(missing_ok=True)
    part = dest_dir / f"{name}.part"
    part.unlink(missing_ok=True)
    body = _get_first(_urls(name), {}, get)
    part.write_bytes(body)
    if _sha256_file(part) != sha256:
        part.unlink()
        raise ChecksumError(f"{name}: SHA256 mismatch")
    part.replace(final)
    return final


def fetch_header(name: str, dest_dir: Path, get: Getter = http_get) -> Path:
    """First 4 KiB of an EDF as `<name>.header`. No checksum: a ranged read cannot be verified
    against the whole-file hash, and only the duration fields are used."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / f"{name}.header"
    if final.exists() and final.stat().st_size >= 256:
        return final
    body = _get_first(_urls(name), {"Range": f"bytes=0-{HEADER_BYTES - 1}"}, get)
    part = dest_dir / f"{name}.header.part"
    part.write_bytes(body[:HEADER_BYTES])  # a server that ignores Range sends the whole file
    part.replace(final)
    return final


def edf_duration_seconds(header: bytes) -> float:
    """Recording length: number of data records times record duration (EDF header 236:252)."""
    return int(header[236:244]) * float(header[244:252])


def _root() -> Path:
    return data_dir() / "sleep-cassette"


def fetch_labels(cfg: Config) -> None:
    manifest = load_manifest()
    sha = dict(zip(manifest["filename"], manifest["sha256"], strict=True))
    recs = select_recordings(cfg.subset, manifest)
    for i, rec in enumerate(recs, 1):
        fetch_file(rec.hyp_file, sha[rec.hyp_file], _root())
        fetch_header(rec.psg_file, _root())
        print(f"[{i}/{len(recs)}] {rec.stem}")


def fetch_psg(cfg: Config) -> None:
    manifest = load_manifest()
    sha = dict(zip(manifest["filename"], manifest["sha256"], strict=True))
    recs = select_recordings(cfg.subset, manifest)
    for i, rec in enumerate(recs, 1):
        fetch_file(rec.psg_file, sha[rec.psg_file], _root())
        print(f"[{i}/{len(recs)}] {rec.stem}")
