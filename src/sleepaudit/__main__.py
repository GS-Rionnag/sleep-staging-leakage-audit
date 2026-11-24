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


HANDLERS: dict[str, Callable[[Config | None], None]] = {c: _stub(c) for c in COMMANDS}
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
