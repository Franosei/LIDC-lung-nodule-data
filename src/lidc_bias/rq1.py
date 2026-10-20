"""RQ1 analysis of disagreement in malignancy ratings."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import krippendorff
import numpy as np
import pandas as pd

from .bootstrap import percentile_interval
from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .provenance import base_manifest, sha256_file, write_json
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)

METRIC_LABELS = {
    "ordinal_alpha": "Krippendorff alpha, ordinal malignancy rating",
    "binary_alpha": "Krippendorff alpha, benign 1 to 2 versus malignant 4 to 5",
    "pairwise_exact_agreement": "Pairwise exact agreement on the 1 to 5 rating",
    "pairwise_binary_agreement": "Pairwise agreement after excluding rating 3",
    "all_readers_agree": "Nodules with identical malignancy ratings",
    "benign_malignant_discordance": "Nodules rated 1 to 2 and 4 to 5",
}


def _rating_matrix(annotations: pd.DataFrame, binary: bool) -> pd.DataFrame:
    ratings = annotations[["nodule_id", "reader_index", "malignancy"]].copy()
    if binary:
        ratings["malignancy"] = ratings["malignancy"].map(
            lambda value: 0.0 if value <= 2 else (1.0 if value >= 4 else np.nan)
        )
    matrix = ratings.pivot(index="nodule_id", columns="reader_index", values="malignancy")
    return matrix.astype(float)


def _safe_alpha(matrix: np.ndarray, level: str) -> float:
    try:
        return float(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
        )
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return float("nan")


def _pairwise_agreement(annotations: pd.DataFrame, binary: bool) -> float:
    agreements = 0
    comparisons = 0
    for ratings in annotations.groupby("nodule_id", sort=False)["malignancy"]:
        values = ratings[1].to_numpy(dtype=int)
        if binary:
            values = values[values != 3]
            values = (values >= 4).astype(int)
        for left, right in combinations(values, 2):
            comparisons += 1
            agreements += int(left == right)
    return agreements / comparisons if comparisons else float("nan")


def _pairwise_counts(matrix: np.ndarray, binary: bool) -> tuple[np.ndarray, np.ndarray]:
    agreements = np.zeros(matrix.shape[0], dtype=int)
    comparisons = np.zeros(matrix.shape[0], dtype=int)
    for left_index, right_index in combinations(range(matrix.shape[1]), 2):
        left = matrix[:, left_index]
        right = matrix[:, right_index]
        valid = np.isfinite(left) & np.isfinite(right)
        if binary:
            valid &= (left != 3) & (right != 3)
            left = left >= 4
            right = right >= 4
        comparisons += valid.astype(int)
        agreements += (valid & (left == right)).astype(int)
    return agreements, comparisons


def _prepare_bootstrap_statistics(annotations: pd.DataFrame) -> dict[str, Any]:
    matrix_frame = _rating_matrix(annotations, binary=False)
    matrix = matrix_frame.to_numpy(dtype=float)
    patient_by_nodule = annotations.groupby("nodule_id", sort=True)["patient_id"].first()
    patient_by_nodule = patient_by_nodule.reindex(matrix_frame.index)
    patients, patient_codes = np.unique(patient_by_nodule.astype(str), return_inverse=True)
    exact_agreements, exact_comparisons = _pairwise_counts(matrix, binary=False)
    binary_agreements, binary_comparisons = _pairwise_counts(matrix, binary=True)
    minima = np.nanmin(matrix, axis=1)
    maxima = np.nanmax(matrix, axis=1)
    return {
        "matrix": matrix,
        "patient_count": len(patients),
        "patient_codes": patient_codes,
        "exact_agreements": exact_agreements,
        "exact_comparisons": exact_comparisons,
        "binary_agreements": binary_agreements,
        "binary_comparisons": binary_comparisons,
        "all_agree": (minima == maxima).astype(float),
        "discordant": ((minima <= 2) & (maxima >= 4)).astype(float),
    }


def _weighted_metrics(statistics: dict[str, Any], weights: np.ndarray) -> dict[str, float]:
    matrix = np.repeat(statistics["matrix"], weights, axis=0)
    binary_matrix = np.where(matrix <= 2, 0.0, np.where(matrix >= 4, 1.0, np.nan))

    exact_comparisons = float(np.dot(weights, statistics["exact_comparisons"]))
    binary_comparisons = float(np.dot(weights, statistics["binary_comparisons"]))
    total_weight = float(weights.sum())
    return {
        "ordinal_alpha": _safe_alpha(matrix.T, "ordinal"),
        "binary_alpha": _safe_alpha(binary_matrix.T, "nominal"),
        "pairwise_exact_agreement": (
            float(np.dot(weights, statistics["exact_agreements"])) / exact_comparisons
        ),
        "pairwise_binary_agreement": (
            float(np.dot(weights, statistics["binary_agreements"])) / binary_comparisons
        ),
        "all_readers_agree": float(np.dot(weights, statistics["all_agree"])) / total_weight,
        "benign_malignant_discordance": (
            float(np.dot(weights, statistics["discordant"])) / total_weight
        ),
    }


def disagreement_metrics(annotations: pd.DataFrame) -> dict[str, float]:
    """Calculate all prespecified RQ1 disagreement metrics."""
    statistics = _prepare_bootstrap_statistics(annotations)
    weights = np.ones(statistics["matrix"].shape[0], dtype=int)
    return _weighted_metrics(statistics, weights)


def _analysis_population(
    annotations: pd.DataFrame, minimum_readers: int, maximum_readers: int
) -> pd.DataFrame:
    counts = annotations.groupby("nodule_id").size()
    eligible_ids = counts.index[counts.between(minimum_readers, maximum_readers)]
    population = annotations.loc[annotations["nodule_id"].isin(eligible_ids)].copy()
    if population.empty:
        raise AnalysisError("RQ1 analysis population is empty")
    return population


def run_rq1(config: AnalysisConfig) -> dict[str, Any]:
    """Run RQ1 with a patient-level non-parametric bootstrap."""
    annotation_path = processed_paths(config)["annotations"]
    if not annotation_path.is_file():
        raise SourceDataError("Processed annotations are missing. Run ingest first")
    annotations = pd.read_parquet(annotation_path)
    population = _analysis_population(
        annotations, config.rq1.minimum_readers, config.rq1.maximum_readers
    )
    statistics = _prepare_bootstrap_statistics(population)
    unit_count = statistics["matrix"].shape[0]
    estimates = _weighted_metrics(statistics, np.ones(unit_count, dtype=int))

    rng = np.random.default_rng(config.master_seed + 101)
    bootstrap_rows: list[dict[str, float | int]] = []
    for repetition in range(config.rq1.bootstrap_repetitions):
        patient_weights = rng.multinomial(
            statistics["patient_count"],
            np.full(statistics["patient_count"], 1.0 / statistics["patient_count"]),
        )
        unit_weights = patient_weights[statistics["patient_codes"]]
        row: dict[str, float | int] = {"bootstrap_repetition": repetition}
        row.update(_weighted_metrics(statistics, unit_weights))
        bootstrap_rows.append(row)
        if config.progress_every > 0 and (repetition + 1) % config.progress_every == 0:
            LOGGER.info(
                "Completed %d of %d RQ1 bootstrap repetitions",
                repetition + 1,
                config.rq1.bootstrap_repetitions,
            )

    bootstrap = pd.DataFrame.from_records(bootstrap_rows)
    summary_rows: list[dict[str, Any]] = []
    for metric, estimate in estimates.items():
        lower, upper, valid = percentile_interval(
            bootstrap[metric].to_numpy(), config.rq1.confidence_level
        )
        summary_rows.append(
            {
                "metric": metric,
                "description": METRIC_LABELS[metric],
                "estimate": estimate,
                "confidence_level": config.rq1.confidence_level,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstrap_repetitions": valid,
            }
        )
    summary = pd.DataFrame.from_records(summary_rows)

    output_dir = config.paths.results_dir / "rq1"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    bootstrap_path = output_dir / "bootstrap_replicates.parquet"
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    bootstrap.to_parquet(bootstrap_path, index=False, compression="zstd")

    manifest = base_manifest(config.project_root, config.config_path, "rq1")
    manifest.update(
        {
            "population": {
                "patients": int(population["patient_id"].nunique()),
                "nodules": int(population["nodule_id"].nunique()),
                "annotations": len(population),
                "minimum_readers": config.rq1.minimum_readers,
                "maximum_readers": config.rq1.maximum_readers,
            },
            "method": {
                "binary_mapping": {"benign": [1, 2], "missing": [3], "malignant": [4, 5]},
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": config.rq1.bootstrap_repetitions,
                "confidence_level": config.rq1.confidence_level,
                "seed": config.master_seed + 101,
            },
            "input": {
                "path": str(annotation_path.relative_to(config.project_root)),
                "sha256": sha256_file(annotation_path),
            },
            "outputs": {
                "summary": {
                    "path": str(summary_path.relative_to(config.project_root)),
                    "sha256": sha256_file(summary_path),
                },
                "bootstrap_replicates": {
                    "path": str(bootstrap_path.relative_to(config.project_root)),
                    "sha256": sha256_file(bootstrap_path),
                },
            },
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
