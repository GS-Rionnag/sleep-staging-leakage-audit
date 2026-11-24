"""Run configuration, YAML loader and the data directory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

VALID_ARMS = ("epoch", "block", "night", "subject", "subject_prefit")
VALID_MODELS = (
    "majority",
    "bandpower_lr",
    "bandpower_ctx_lr",
    "yasa_lgbm",
    "yasa_pretrained",
)


@dataclass(frozen=True)
class Config:
    name: str
    subset: Literal["sc20", "sc78"] | list[int]
    k: int
    seeds: list[int]
    arms: list[str]
    models: list[str]
    results_dir: Path
    n_boot: int = 10000
    block_epochs: int = 60
    buffer_epochs: int = 8
    trim_epochs: int = 60
    extra: dict = field(default_factory=dict, compare=False)


def _check(values: list[str], valid: tuple[str, ...], what: str) -> None:
    unknown = [v for v in values if v not in valid]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; valid: {list(valid)}")


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw["results_dir"] = Path(raw["results_dir"])
    _check(raw["arms"], VALID_ARMS, "arm")
    _check(raw["models"], VALID_MODELS, "model")
    subset = raw["subset"]
    if not isinstance(subset, list) and subset not in ("sc20", "sc78"):
        raise ValueError(f"unknown subset {subset!r}")
    return Config(**raw)


def data_dir() -> Path:
    """Data directory from SLEEPAUDIT_DATA (default ~/sleepaudit_data), created if missing."""
    path = Path(os.environ.get("SLEEPAUDIT_DATA", Path.home() / "sleepaudit_data"))
    path.mkdir(parents=True, exist_ok=True)
    return path
