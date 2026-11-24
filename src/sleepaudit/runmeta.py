"""Run metadata (meta.json) and the pre-registration gate."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from sleepaudit.config import Config
from sleepaudit.records import MANIFEST_PATH

PREREG_TAG = "prereg-v1"


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def prereg_tag_in_history(cwd: Path | None = None) -> bool:
    """True if the tag ``prereg-v1`` exists and is an ancestor of (or equal to) HEAD."""
    try:
        if _git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}", cwd=cwd).returncode:
            return False
        return _git("merge-base", "--is-ancestor", PREREG_TAG, "HEAD", cwd=cwd).returncode == 0
    except FileNotFoundError:  # git not installed
        return False


def check_prereg(cfg: Config, allow: bool, cwd: Path | None = None) -> None:
    """Refuse full-data configs (``sc20``/``sc78``) unless the pre-registration tag is in history."""
    if allow or isinstance(cfg.subset, list):
        return
    if not prereg_tag_in_history(cwd):
        raise SystemExit(
            f"refusing to run subset {cfg.subset!r}: git tag {PREREG_TAG} is not in the current "
            f"history. Freeze the pre-registration first, or pass --allow-without-prereg "
            f"(recorded in meta.json)."
        )


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _gpu() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    names = proc.stdout.strip().splitlines()
    return "; ".join(names) if proc.returncode == 0 and names else None


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_meta(
    cfg: Config,
    out: Path,
    started: datetime | None = None,
    finished: datetime | None = None,
    extra: dict | None = None,
) -> dict:
    """Write ``meta.json`` and return it. ``dirty`` ignores untracked files (results are written)."""
    now = datetime.now(UTC)
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    root = Path.cwd()
    meta = {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else False,
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "packages": {
            d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]
        },
        "config": dataclasses.asdict(cfg),
        "seeds": list(cfg.seeds),
        "started": (started or now).isoformat(),
        "finished": (finished or now).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "gpu": _gpu(),
    }
    meta.update(extra or {})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, default=_json_safe), encoding="utf-8")
    return meta
