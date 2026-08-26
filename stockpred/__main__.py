"""Command-line entry point: ``python -m stockpred <command>``.

Subcommands
-----------
``run``   -- full pipeline: fetch -> features -> walk-forward -> CQR ->
             GARCH -> production refit -> assemble -> backtest ->
             write_artifacts (see :func:`stockpred.pipeline.run`).
``fetch`` -- data fetch/cache only (see :func:`stockpred.pipeline.fetch`).
"""

from __future__ import annotations

import argparse
import logging
import sys

from stockpred import pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stockpred")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full forecasting pipeline.")
    run_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    run_parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip the data fetch stage and use the existing parquet cache.",
    )

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and cache data only.")
    fetch_parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        pipeline.run(config_path=args.config, skip_fetch=args.skip_fetch)
    elif args.command == "fetch":
        pipeline.fetch(config_path=args.config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
