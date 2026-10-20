"""Command-line interface for the reproducible study pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .errors import LidcBiasError
from .figures import create_publication_figures
from .ingestion import run_ingestion
from .rq1 import run_rq1
from .rq2 import run_rq2
from .rq3 import run_rq3
from .rq4 import run_rq4
from .rq5 import run_rq5
from .rq6 import run_rq6
from .rq7 import run_rq7
from .validation import (
    raise_for_failed_report,
    validate_processed_files,
    write_validation_report,
)

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidc-bias",
        description="Reproducible LIDC-IDRI reference-label misclassification analysis",
    )
    parser.add_argument(
        "command",
        choices=(
            "ingest",
            "validate",
            "rq1",
            "rq2",
            "rq3",
            "rq4",
            "rq5",
            "rq6",
            "rq7",
            "figures",
            "run",
        ),
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/analysis.toml"),
        help="Path to the TOML analysis configuration",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _run_validation(config_path: Path) -> None:
    config = load_config(config_path)
    report = validate_processed_files(config)
    output = config.paths.processed_dir / "validation_report.csv"
    write_validation_report(report, output)
    raise_for_failed_report(report)
    LOGGER.info("All %d validation checks passed", len(report))


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute one command and return a process exit code."""
    options = _parser().parse_args(arguments)
    logging.basicConfig(
        level=getattr(logging, options.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(options.config)
        if options.command == "ingest":
            run_ingestion(config)
        elif options.command == "validate":
            _run_validation(options.config)
        elif options.command == "rq1":
            run_rq1(config)
        elif options.command == "rq2":
            run_rq2(config)
        elif options.command == "rq3":
            run_rq3(config)
        elif options.command == "rq4":
            run_rq4(config)
        elif options.command == "rq5":
            run_rq5(config)
        elif options.command == "rq6":
            run_rq6(config)
        elif options.command == "rq7":
            run_rq7(config)
        elif options.command == "figures":
            create_publication_figures(config)
        elif options.command == "run":
            run_ingestion(config)
            _run_validation(options.config)
            run_rq1(config)
            run_rq6(config)
            run_rq2(config)
            run_rq3(config)
            run_rq4(config)
            run_rq5(config)
            run_rq7(config)
            create_publication_figures(config)
    except LidcBiasError as error:
        LOGGER.error("%s", error)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
