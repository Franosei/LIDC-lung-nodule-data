"""Typed loading and validation of the focused analysis configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SourceDataError


@dataclass(frozen=True)
class PathsConfig:
    dicom_archive: Path
    processed_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class ExpectedCounts:
    patients: int
    annotations: int
    segmentations: int
    nodules: int
    nodules_with_1_or_2_readers: int
    nodules_with_3_or_4_readers: int
    nodules_with_more_than_4_readers: int


@dataclass(frozen=True)
class LogisticConfig:
    regularisation_c: float
    maximum_iterations: int
    class_weight: str | None


@dataclass(frozen=True)
class LabelCaseEffectsConfig:
    minimum_readers: int
    maximum_readers: int
    cross_validation_folds: int
    cross_validation_repetitions: int
    bootstrap_repetitions: int
    confidence_level: float
    random_reader_seed_offset: int
    partition_seed_offset: int
    training_reference: str
    model: str
    primary_references: tuple[str, ...]
    sensitivity_references: tuple[str, ...]
    logistic_regression: LogisticConfig


@dataclass(frozen=True)
class SharedAssessmentConfig:
    minimum_readers: int
    maximum_readers: int
    assignment_repetitions: int
    assignment_seed_offset: int
    cross_validation_folds: int
    cross_validation_repetitions: int
    partition_seed_offset: int
    bootstrap_repetitions: int
    confidence_level: float
    model: str


@dataclass(frozen=True)
class PublicationConfig:
    dpi: int
    formats: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisConfig:
    config_path: Path
    project_root: Path
    paths: PathsConfig
    master_seed: int
    progress_every: int
    strict_source_counts: bool
    expected: ExpectedCounts
    label_case_effects: LabelCaseEffectsConfig
    shared_assessment: SharedAssessmentConfig
    publication: PublicationConfig


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SourceDataError(f"Missing configuration key: {context}.{key}")
    return mapping[key]


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _check_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise SourceDataError(f"{name} must be between 0 and 1")


def load_config(path: str | Path) -> AnalysisConfig:
    """Load TOML settings and resolve paths against the repository root."""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise SourceDataError(f"Configuration file does not exist: {config_path}")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project_root = config_path.parent.parent.resolve()
    paths_raw = _required(raw, "paths", "root")
    reproducibility = _required(raw, "reproducibility", "root")
    ingestion = _required(raw, "ingestion", "root")
    validation = _required(raw, "validation", "root")
    label_raw = _required(raw, "label_case_effects", "root")
    shared_raw = _required(raw, "shared_assessment", "root")
    publication_raw = _required(raw, "publication", "root")

    paths = PathsConfig(
        dicom_archive=_resolve(project_root, _required(paths_raw, "dicom_archive", "paths")),
        processed_dir=_resolve(project_root, _required(paths_raw, "processed_dir", "paths")),
        results_dir=_resolve(project_root, _required(paths_raw, "results_dir", "paths")),
    )
    logistic = LogisticConfig(
        **_required(label_raw, "logistic_regression", "label_case_effects")
    )
    label_values = {
        key: value for key, value in label_raw.items() if key != "logistic_regression"
    }
    label = LabelCaseEffectsConfig(
        **{
            **label_values,
            "primary_references": tuple(label_values["primary_references"]),
            "sensitivity_references": tuple(label_values["sensitivity_references"]),
        },
        logistic_regression=logistic,
    )
    shared = SharedAssessmentConfig(**shared_raw)
    publication = PublicationConfig(
        **{**publication_raw, "formats": tuple(publication_raw["formats"])}
    )

    for name, folds, repetitions in (
        (
            "label_case_effects",
            label.cross_validation_folds,
            label.cross_validation_repetitions,
        ),
        (
            "shared_assessment",
            shared.cross_validation_folds,
            shared.cross_validation_repetitions,
        ),
    ):
        if folds < 2 or repetitions < 1:
            raise SourceDataError(f"{name} requires at least two folds and one repetition")
    if shared.assignment_repetitions < 1:
        raise SourceDataError("shared_assessment.assignment_repetitions must be positive")
    _check_probability(label.confidence_level, "label_case_effects.confidence_level")
    _check_probability(shared.confidence_level, "shared_assessment.confidence_level")
    if label.training_reference != "median_exclude_3":
        raise SourceDataError(
            "The fixed-prediction design requires training_reference='median_exclude_3'"
        )
    if label.model != "combined_logistic" or shared.model != "combined_logistic":
        raise SourceDataError("The proposal fixes both analyses to combined_logistic")
    expected_primary = ("median_3_benign", "median_3_malignant")
    if label.primary_references != expected_primary:
        raise SourceDataError(
            "primary_references must be median_3_benign and median_3_malignant"
        )

    return AnalysisConfig(
        config_path=config_path,
        project_root=project_root,
        paths=paths,
        master_seed=int(_required(reproducibility, "master_seed", "reproducibility")),
        progress_every=int(_required(ingestion, "progress_every", "ingestion")),
        strict_source_counts=bool(_required(ingestion, "strict_source_counts", "ingestion")),
        expected=ExpectedCounts(**_required(validation, "expected", "validation")),
        label_case_effects=label,
        shared_assessment=shared,
        publication=publication,
    )
