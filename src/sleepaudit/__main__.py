"""CLI dispatch: python -m sleepaudit <command> --config <path>."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from sleepaudit.config import Config, load_config

COMMANDS = (
    "build-manifest",
    "fetch-labels",
    "fetch-psg",
    "features",
    "evaluate",
    "report",
    "all",
)


def _stub(name: str) -> Callable[[Config | None], None]:
    def run(cfg: Config | None) -> None:
        raise NotImplementedError(f"{name} is not implemented yet")

    return run


def _build_manifest(cfg: Config | None) -> None:
    from sleepaudit.records import write_manifest

    df = write_manifest()
    print(f"wrote manifest with {len(df)} files")


def _fetch_labels(cfg: Config | None) -> None:
    from sleepaudit.download import fetch_labels

    fetch_labels(cfg)


def _fetch_psg(cfg: Config | None) -> None:
    from sleepaudit.download import fetch_psg

    fetch_psg(cfg)


HANDLERS: dict[str, Callable[[Config | None], None]] = {c: _stub(c) for c in COMMANDS}
HANDLERS["build-manifest"] = _build_manifest
HANDLERS["fetch-labels"] = _fetch_labels
HANDLERS["fetch-psg"] = _fetch_psg
NO_CONFIG = {"build-manifest"}  # works on the manifest only; --config is optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sleepaudit", description="Sleep-staging leakage audit pipeline."
    )
    parser.add_argument("command", choices=COMMANDS, metavar="command", help=" | ".join(COMMANDS))
    parser.add_argument("--config", type=Path, default=None, help="path to a YAML config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is None and args.command not in NO_CONFIG:
        parser.error(f"--config is required for {args.command}")
    cfg = load_config(args.config) if args.config is not None else None
    HANDLERS[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
