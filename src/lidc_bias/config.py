"""Typed loading and validation of the analysis configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SourceDataError


@dataclass(frozen=True)
class PathsConfig:
    dicom_archive: Path
    diagnosis_workbook: Path
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
    diagnosis_patients: int


@dataclass(frozen=True)
class Rq1Config:
    minimum_readers: int
    maximum_readers: int
    bootstrap_repetitions: int
    confidence_level: float


@dataclass(frozen=True)
class LogisticConfig:
    regularisation_c: float
    maximum_iterations: int
    class_weight: str | None


@dataclass(frozen=True)
class RandomForestConfig:
    trees: int
    minimum_leaf_size: int
    maximum_features: str | float
    class_weight: str | None
    n_jobs: int


@dataclass(frozen=True)
class Rq2Config:
    minimum_readers: int
    maximum_readers: int
    cross_validation_folds: int
    cross_validation_repetitions: int
    bootstrap_repetitions: int
    confidence_level: float
    random_reader_seed_offset: int
    fixed_prediction_reference: str
    fixed_prediction_model: str
    excluded_case_reference: str
    logistic_regression: LogisticConfig
    random_forest: RandomForestConfig


@dataclass(frozen=True)
class Rq3Config:
    minimum_readers: int
    maximum_readers: int
    cross_validation_folds: int
    cross_validation_repetitions: int
    bootstrap_repetitions: int
    confidence_level: float
    reader_assignment_seed_offset: int


@dataclass(frozen=True)
class Rq4Config:
    minimum_readers: int
    maximum_readers: int
    relabelling_percentages: tuple[int, ...]
    bootstrap_repetitions: int
    confidence_level: float
    reader_assignment_seed_offset: int
    model: str


@dataclass(frozen=True)
class Rq5Config:
    addition_fractions: tuple[float, ...]
    cross_validation_folds: int
    repetitions: int
    bootstrap_repetitions: int
    confidence_level: float
    models: tuple[str, ...]
    consensus_reference: str
    noisy_reference: str


@dataclass(frozen=True)
class Rq6Config:
    minimum_readers: int
    maximum_readers: int
    minimum_classes: int
    maximum_classes: int
    random_starts: int
    maximum_iterations: int
    tolerance: float
    posterior_threshold: float


@dataclass(frozen=True)
class Rq7Config:
    bootstrap_repetitions: int
    confidence_level: float
    training_reference: str
    model: str


@dataclass(frozen=True)
class FiguresConfig:
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
    rq1: Rq1Config
    rq2: Rq2Config
    rq3: Rq3Config
    rq4: Rq4Config
    rq5: Rq5Config
    rq6: Rq6Config
    rq7: Rq7Config
    figures: FiguresConfig


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise SourceDataError(f"Missing configuration key: {context}.{key}")
    return mapping[key]


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> AnalysisConfig:
    """Load a TOML configuration and resolve paths against the project root."""
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
    rq1_raw = _required(raw, "rq1", "root")
    rq2_raw = _required(raw, "rq2", "root")
    rq3_raw = _required(raw, "rq3", "root")
    rq4_raw = _required(raw, "rq4", "root")
    rq5_raw = _required(raw, "rq5", "root")
    rq6_raw = _required(raw, "rq6", "root")
    rq7_raw = _required(raw, "rq7", "root")
    figures_raw = _required(raw, "figures", "root")

    paths = PathsConfig(
        dicom_archive=_resolve(project_root, _required(paths_raw, "dicom_archive", "paths")),
        diagnosis_workbook=_resolve(
            project_root, _required(paths_raw, "diagnosis_workbook", "paths")
        ),
        processed_dir=_resolve(project_root, _required(paths_raw, "processed_dir", "paths")),
        results_dir=_resolve(project_root, _required(paths_raw, "results_dir", "paths")),
    )
    expected = ExpectedCounts(**_required(validation, "expected", "validation"))
    rq1 = Rq1Config(**rq1_raw)
    logistic = LogisticConfig(**_required(rq2_raw, "logistic_regression", "rq2"))
    forest = RandomForestConfig(**_required(rq2_raw, "random_forest", "rq2"))
    rq2_values = {
        key: value
        for key, value in rq2_raw.items()
        if key not in {"logistic_regression", "random_forest"}
    }
    rq2 = Rq2Config(**rq2_values, logistic_regression=logistic, random_forest=forest)
    rq3 = Rq3Config(**rq3_raw)
    rq4 = Rq4Config(
        **{
            **rq4_raw,
            "relabelling_percentages": tuple(rq4_raw["relabelling_percentages"]),
        }
    )
    rq5 = Rq5Config(
        **{
            **rq5_raw,
            "addition_fractions": tuple(rq5_raw["addition_fractions"]),
            "models": tuple(rq5_raw["models"]),
        }
    )
    rq6 = Rq6Config(**rq6_raw)
    rq7 = Rq7Config(**rq7_raw)
    figures = FiguresConfig(**{**figures_raw, "formats": tuple(figures_raw["formats"])})

    if not 0.0 < rq1.confidence_level < 1.0:
        raise SourceDataError("rq1.confidence_level must be between 0 and 1")
    if not 0.0 < rq2.confidence_level < 1.0:
        raise SourceDataError("rq2.confidence_level must be between 0 and 1")
    if rq2.cross_validation_folds < 2 or rq2.cross_validation_repetitions < 1:
        raise SourceDataError(
            "RQ2 cross-validation settings must be positive and use at least two folds"
        )
    for name, level in (
        ("rq3", rq3.confidence_level),
        ("rq4", rq4.confidence_level),
        ("rq5", rq5.confidence_level),
        ("rq7", rq7.confidence_level),
    ):
        if not 0.0 < level < 1.0:
            raise SourceDataError(f"{name}.confidence_level must be between 0 and 1")
    if tuple(sorted(set(rq4.relabelling_percentages))) != rq4.relabelling_percentages:
        raise SourceDataError("rq4.relabelling_percentages must be sorted and unique")
    if rq4.relabelling_percentages[0] != 0 or rq4.relabelling_percentages[-1] != 100:
        raise SourceDataError("RQ4 relabelling percentages must span 0 to 100")
    if any(not 0.0 <= fraction <= 1.0 for fraction in rq5.addition_fractions):
        raise SourceDataError("RQ5 addition fractions must be between 0 and 1")
    if not 0.0 < rq6.posterior_threshold < 1.0:
        raise SourceDataError("rq6.posterior_threshold must be between 0 and 1")

    return AnalysisConfig(
        config_path=config_path,
        project_root=project_root,
        paths=paths,
        master_seed=int(_required(reproducibility, "master_seed", "reproducibility")),
        progress_every=int(_required(ingestion, "progress_every", "ingestion")),
        strict_source_counts=bool(
            _required(ingestion, "strict_source_counts", "ingestion")
        ),
        expected=expected,
        rq1=rq1,
        rq2=rq2,
        rq3=rq3,
        rq4=rq4,
        rq5=rq5,
        rq6=rq6,
        rq7=rq7,
        figures=figures,
    )
