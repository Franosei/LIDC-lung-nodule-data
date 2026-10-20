"""Explicit source and canonical-table validation rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AnalysisConfig
from .errors import SourceDataError, ValidationError
from .tables import ALL_MODEL_FEATURES


class ValidationReport:
    """Collect validation outcomes and raise one informative failure."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(
        self,
        check: str,
        passed: bool,
        observed: Any,
        expected: Any,
        detail: str = "",
    ) -> None:
        self._rows.append(
            {
                "check": check,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._rows)

    def raise_for_failures(self) -> None:
        failed = [row for row in self._rows if row["status"] == "fail"]
        if failed:
            names = ", ".join(row["check"] for row in failed)
            raise ValidationError(f"{len(failed)} validation checks failed: {names}")


def _count_check(
    report: ValidationReport, name: str, observed: int, expected: int, strict: bool
) -> None:
    passed = observed == expected
    report.add(
        name,
        passed or not strict,
        observed,
        expected,
        "strict" if strict else "informational",
    )


def validate_tables(
    annotations: pd.DataFrame,
    segmentations: pd.DataFrame,
    nodules: pd.DataFrame,
    diagnosis_patients: pd.DataFrame,
    diagnosis_nodules: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    """Validate source counts, referential integrity, ranges and model inputs."""
    report = ValidationReport()
    strict = config.strict_source_counts
    expected = config.expected

    _count_check(report, "annotation_count", len(annotations), expected.annotations, strict)
    _count_check(report, "segmentation_count", len(segmentations), expected.segmentations, strict)
    _count_check(
        report,
        "patient_count",
        annotations["patient_id"].nunique(),
        expected.patients,
        strict,
    )
    _count_check(report, "nodule_count", len(nodules), expected.nodules, strict)
    _count_check(
        report,
        "diagnosis_patient_count",
        len(diagnosis_patients),
        expected.diagnosis_patients,
        strict,
    )

    reader_counts = nodules["reader_count"]
    count_1_2 = int(reader_counts.between(1, 2).sum())
    count_3_4 = int(reader_counts.between(3, 4).sum())
    count_more_4 = int((reader_counts > 4).sum())
    _count_check(
        report,
        "nodules_with_1_or_2_readers",
        count_1_2,
        expected.nodules_with_1_or_2_readers,
        strict,
    )
    _count_check(
        report,
        "nodules_with_3_or_4_readers",
        count_3_4,
        expected.nodules_with_3_or_4_readers,
        strict,
    )
    _count_check(
        report,
        "nodules_with_more_than_4_readers",
        count_more_4,
        expected.nodules_with_more_than_4_readers,
        strict,
    )

    uniqueness_checks = {
        "unique_annotation_ids": not annotations["annotation_id"].duplicated().any(),
        "unique_segmentation_ids": not segmentations["segmentation_sop_uid"].duplicated().any(),
        "unique_nodule_ids": not nodules["nodule_id"].duplicated().any(),
        "unique_diagnosis_patient_ids": not diagnosis_patients["patient_id"].duplicated().any(),
    }
    for name, passed in uniqueness_checks.items():
        report.add(name, passed, passed, True)

    tracking_to_nodule = annotations.groupby("tracking_unique_identifier")["nodule_id"].nunique()
    tracking_conflicts = int((tracking_to_nodule != 1).sum())
    report.add(
        "tracking_uid_identifies_one_nodule",
        tracking_conflicts == 0,
        tracking_conflicts,
        0,
    )
    nodule_to_tracking = annotations.groupby("nodule_id")["tracking_unique_identifier"].nunique()
    nodule_tracking_conflicts = int((nodule_to_tracking != 1).sum())
    report.add(
        "nodule_has_one_tracking_uid",
        nodule_tracking_conflicts == 0,
        nodule_tracking_conflicts,
        0,
    )

    referenced = set(annotations["referenced_segmentation_sop_uid"].astype(str))
    available = set(segmentations["segmentation_sop_uid"].astype(str))
    missing_segments = sorted(referenced - available)
    unreferenced_segments = sorted(available - referenced)
    report.add(
        "all_referenced_segmentations_exist",
        not missing_segments,
        len(missing_segments),
        0,
        json.dumps(missing_segments[:10]),
    )
    report.add(
        "all_segmentations_are_referenced",
        not unreferenced_segments,
        len(unreferenced_segments),
        0,
        json.dumps(unreferenced_segments[:10]),
    )

    linked = annotations[["referenced_segmentation_sop_uid", "patient_id"]].merge(
        segmentations[["segmentation_sop_uid", "patient_id"]],
        left_on="referenced_segmentation_sop_uid",
        right_on="segmentation_sop_uid",
        how="left",
        suffixes=("_sr", "_seg"),
    )
    mismatched_patients = int((linked["patient_id_sr"] != linked["patient_id_seg"]).sum())
    report.add("sr_seg_patient_match", mismatched_patients == 0, mismatched_patients, 0)

    rating_ranges = {
        "subtlety": (1, 5),
        "internal_structure": (1, 4),
        "calcification": (1, 6),
        "sphericity": (1, 5),
        "margin": (1, 5),
        "lobulation": (1, 5),
        "spiculation": (1, 5),
        "texture": (1, 5),
        "malignancy": (1, 5),
    }
    for column, (minimum, maximum) in rating_ranges.items():
        present = annotations[column].dropna()
        invalid = int((~present.between(minimum, maximum)).sum())
        report.add(
            f"{column}_range",
            invalid == 0,
            invalid,
            0,
            f"valid range {minimum} to {maximum}",
        )
        missing = int(annotations[column].isna().sum())
        report.add(
            f"{column}_missing_values",
            missing == 0 if column == "malignancy" else True,
            missing,
            0 if column == "malignancy" else "reported; fold-local imputation",
        )

    geometry_columns = ["volume_mm3", "diameter_mm", "surface_area_mm2", "compactness"]
    for column in geometry_columns:
        values = annotations[column].astype(float).to_numpy()
        invalid = int((~np.isfinite(values) | (values <= 0.0)).sum())
        report.add(f"positive_finite_{column}", invalid == 0, invalid, 0)

    missing_model_values = int(nodules[ALL_MODEL_FEATURES].isna().sum().sum())
    report.add(
        "nodule_model_feature_missing_values",
        True,
        missing_model_values,
        "reported; fold-local imputation",
    )

    unknown_diagnosis_patients = sorted(
        set(diagnosis_patients["patient_id"]) - set(annotations["patient_id"])
    )
    report.add(
        "diagnosis_patients_without_standardised_annotations",
        True,
        len(unknown_diagnosis_patients),
        "reported; unavailable for nodule linkage",
        json.dumps(unknown_diagnosis_patients),
    )

    orphan_nodule_diagnoses = sorted(
        set(diagnosis_nodules["patient_id"]) - set(diagnosis_patients["patient_id"])
    )
    report.add(
        "nodule_diagnoses_have_patient_record",
        not orphan_nodule_diagnoses,
        len(orphan_nodule_diagnoses),
        0,
        json.dumps(orphan_nodule_diagnoses[:10]),
    )
    return report.frame()


def validate_processed_files(config: AnalysisConfig) -> pd.DataFrame:
    """Load cached canonical tables and rerun all validation rules."""
    paths = processed_paths(config)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SourceDataError(f"Processed files are missing. Run ingest first: {missing}")
    tables = {name: pd.read_parquet(path) for name, path in paths.items()}
    return validate_tables(
        tables["annotations"],
        tables["segmentations"],
        tables["nodules"],
        tables["diagnosis_patients"],
        tables["diagnosis_nodules"],
        config,
    )


def processed_paths(config: AnalysisConfig) -> dict[str, Path]:
    base = config.paths.processed_dir
    return {
        "annotations": base / "annotations.parquet",
        "segmentations": base / "segmentations.parquet",
        "nodules": base / "nodules.parquet",
        "diagnosis_patients": base / "diagnosis_patients.parquet",
        "diagnosis_nodules": base / "diagnosis_nodules.parquet",
    }


def write_validation_report(report: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, lineterminator="\n")


def raise_for_failed_report(report: pd.DataFrame) -> None:
    failed = report.loc[report["status"] == "fail", "check"].tolist()
    if failed:
        raise ValidationError(f"{len(failed)} validation checks failed: {', '.join(failed)}")
