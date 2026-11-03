"""Command-line interface for the focused reproducible study pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import load_config
from .errors import LidcBiasError
from .ingestion import run_ingestion
from .label_case_effects import run_label_case_effects
from .publication import create_publication_outputs
from .shared_assessment import run_shared_assessment
from .validation import raise_for_failed_report, validate_processed_files, write_validation_report

LOGGER = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lidc-bias",
        description="Controlled analysis of annotation effects in LIDC-IDRI",
    )
    parser.add_argument(
        "command",
        choices=(
            "ingest",
            "validate",
            "label-case-effects",
            "shared-assessment",
            "publication",
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
        elif options.command == "label-case-effects":
            run_label_case_effects(config)
        elif options.command == "shared-assessment":
            run_shared_assessment(config)
        elif options.command == "publication":
            create_publication_outputs(config)
        elif options.command == "run":
            run_ingestion(config)
            _run_validation(options.config)
            run_label_case_effects(config)
            run_shared_assessment(config)
            create_publication_outputs(config)
    except LidcBiasError as error:
        LOGGER.error("%s", error)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
