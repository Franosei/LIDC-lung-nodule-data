"""End-to-end ingestion of immutable source files into canonical tables."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AnalysisConfig
from .diagnosis import read_diagnoses
from .dicom_sr import read_archive
from .provenance import base_manifest, sha256_file, write_json
from .tables import build_nodule_table
from .validation import (
    processed_paths,
    raise_for_failed_report,
    validate_tables,
    write_validation_report,
)

LOGGER = logging.getLogger(__name__)
DICOM_CACHE_SCHEMA_VERSION = 1


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _read_or_build_dicom_cache(
    config: AnalysisConfig, archive_sha256: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = config.paths.processed_dir / ".staging"
    annotation_path = cache_dir / "annotations.parquet"
    segmentation_path = cache_dir / "segmentations.parquet"
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file() and annotation_path.is_file() and segmentation_path.is_file():
        try:
            cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache_manifest = {}
        cache_matches = (
            cache_manifest.get("archive_sha256") == archive_sha256
            and cache_manifest.get("schema_version") == DICOM_CACHE_SCHEMA_VERSION
        )
        if cache_matches:
            LOGGER.info("Using source-matched DICOM staging cache")
            return pd.read_parquet(annotation_path), pd.read_parquet(segmentation_path)

    annotations, segmentations = read_archive(
        config.paths.dicom_archive, config.progress_every
    )
    _write_parquet_atomic(annotations, annotation_path)
    _write_parquet_atomic(segmentations, segmentation_path)
    write_json(
        manifest_path,
        {
            "archive_sha256": archive_sha256,
            "schema_version": DICOM_CACHE_SCHEMA_VERSION,
            "annotation_rows": len(annotations),
            "segmentation_rows": len(segmentations),
        },
    )
    return annotations, segmentations


def run_ingestion(config: AnalysisConfig) -> dict[str, Any]:
    """Read, validate and persist all canonical source tables."""
    LOGGER.info("Starting source ingestion")
    archive_sha256 = sha256_file(config.paths.dicom_archive)
    annotations, segmentations = _read_or_build_dicom_cache(
        config, archive_sha256
    )
    diagnosis_patients, diagnosis_nodules = read_diagnoses(config.paths.diagnosis_workbook)
    nodules = build_nodule_table(annotations)

    report = validate_tables(
        annotations,
        segmentations,
        nodules,
        diagnosis_patients,
        diagnosis_nodules,
        config,
    )
    validation_path = config.paths.processed_dir / "validation_report.csv"
    write_validation_report(report, validation_path)
    raise_for_failed_report(report)

    frames = {
        "annotations": annotations,
        "segmentations": segmentations,
        "nodules": nodules,
        "diagnosis_patients": diagnosis_patients,
        "diagnosis_nodules": diagnosis_nodules,
    }
    paths = processed_paths(config)
    for name, frame in frames.items():
        LOGGER.info("Writing %s with %d rows", paths[name], len(frame))
        _write_parquet_atomic(frame, paths[name])

    manifest = base_manifest(config.project_root, config.config_path, "ingestion")
    manifest.update(
        {
            "inputs": {
                "dicom_archive": {
                    "path": str(config.paths.dicom_archive.relative_to(config.project_root)),
                    "sha256": archive_sha256,
                },
                "diagnosis_workbook": {
                    "path": str(config.paths.diagnosis_workbook.relative_to(config.project_root)),
                    "sha256": sha256_file(config.paths.diagnosis_workbook),
                },
            },
            "outputs": {
                name: {
                    "path": str(path.relative_to(config.project_root)),
                    "rows": len(frames[name]),
                    "sha256": sha256_file(path),
                }
                for name, path in paths.items()
            },
            "validation": {
                "path": str(validation_path.relative_to(config.project_root)),
                "checks": len(report),
                "failed": int((report["status"] == "fail").sum()),
            },
        }
    )
    manifest_path = config.paths.processed_dir / "ingestion_manifest.json"
    write_json(manifest_path, manifest)
    LOGGER.info("Ingestion completed successfully")
    return manifest
