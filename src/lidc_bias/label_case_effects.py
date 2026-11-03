"""Controlled fixed-prediction comparisons of label assignment and case exclusion."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .bootstrap import percentile_interval
from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .modelling import average_predictions, cross_validated_predictions, make_patient_partitions
from .provenance import base_manifest, sha256_file, write_json
from .references import REFERENCE_DESCRIPTIONS, build_reference_labels, reference_summary
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)
FOCUSED_METRICS = ("auc", "brier_score")


def _metric_values(frame: pd.DataFrame) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    if np.unique(labels).size != 2:
        return {"auc": float("nan"), "brier_score": float("nan")}
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }


def _patient_indices(frame: pd.DataFrame, patients: np.ndarray) -> dict[str, np.ndarray]:
    values = frame["patient_id"].astype(str).to_numpy()
    return {patient: np.flatnonzero(values == patient) for patient in patients}


def _paired_metric_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    identifiers: dict[str, Any],
    first_name: str,
    second_name: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare two paired conditions by resampling patients once per replicate."""
    patients = np.asarray(sorted(set(first["patient_id"].astype(str))))
    if set(second["patient_id"].astype(str)) - set(patients):
        raise AnalysisError("The second comparison condition contains unmatched patients")
    first_indices = _patient_indices(first, patients)
    second_indices = _patient_indices(second, patients)
    rng = np.random.default_rng(seed)
    replicate_rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        sampled = rng.choice(patients, len(patients), replace=True)
        first_draw = np.concatenate([first_indices[patient] for patient in sampled])
        second_parts = [
            second_indices[patient]
            for patient in sampled
            if second_indices[patient].size
        ]
        second_draw = np.concatenate(second_parts) if second_parts else np.array([], dtype=int)
        first_values = _metric_values(first.iloc[first_draw])
        second_values = (
            _metric_values(second.iloc[second_draw])
            if second_draw.size
            else {metric: float("nan") for metric in FOCUSED_METRICS}
        )
        for metric in FOCUSED_METRICS:
            replicate_rows.append(
                {
                    **identifiers,
                    "bootstrap_repetition": repetition,
                    "metric": metric,
                    "first": first_values[metric],
                    "second": second_values[metric],
                    "difference_second_minus_first": second_values[metric]
                    - first_values[metric],
                }
            )
    replicates = pd.DataFrame.from_records(replicate_rows)
    first_estimates = _metric_values(first)
    second_estimates = _metric_values(second)
    performance_rows = []
    effect_rows = []
    for metric in FOCUSED_METRICS:
        metric_replicates = replicates.loc[replicates["metric"] == metric]
        for condition, name, estimate in (
            ("first", first_name, first_estimates[metric]),
            ("second", second_name, second_estimates[metric]),
        ):
            lower, upper, valid = percentile_interval(
                metric_replicates[condition].to_numpy(), confidence_level
            )
            performance_rows.append(
                {
                    **identifiers,
                    "condition": name,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": confidence_level,
                    "valid_bootstrap_repetitions": valid,
                }
            )
        differences = metric_replicates["difference_second_minus_first"].to_numpy()
        lower, upper, valid = percentile_interval(differences, confidence_level)
        effect_rows.append(
            {
                **identifiers,
                "metric": metric,
                "effect": f"{second_name}_minus_{first_name}",
                "estimate": second_estimates[metric] - first_estimates[metric],
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "valid_bootstrap_repetitions": valid,
            }
        )
    return (
        pd.DataFrame.from_records(performance_rows),
        pd.DataFrame.from_records(effect_rows),
        replicates,
    )


def _evaluation_frame(
    predictions: pd.DataFrame, labels: pd.DataFrame, reference: str, restricted: bool
) -> pd.DataFrame:
    frame = predictions.merge(
        labels[["nodule_id", reference, "median_exclude_3"]],
        on="nodule_id",
        validate="one_to_one",
    )
    if restricted:
        frame = frame.loc[frame["median_exclude_3"].notna()]
    frame = frame.loc[frame[reference].notna()].copy()
    frame["label"] = frame[reference].astype(int)
    return frame[["nodule_id", "patient_id", "label", "probability"]]


def run_label_case_effects(config: AnalysisConfig) -> dict[str, Any]:
    """Run the proposal's label-assignment and case-exclusion comparisons."""
    settings = config.label_case_effects
    paths = processed_paths(config)
    required = [paths["annotations"], paths["nodules"]]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise SourceDataError(f"Processed inputs are missing. Run ingest first: {missing}")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    population = nodules.loc[
        nodules["reader_count"].between(settings.minimum_readers, settings.maximum_readers)
    ].copy()
    if population.empty:
        raise AnalysisError("The label/case analysis population is empty")
    population_annotations = annotations.loc[
        annotations["nodule_id"].isin(population["nodule_id"])
    ]
    labels = build_reference_labels(
        population,
        population_annotations,
        config.master_seed + settings.random_reader_seed_offset,
    )
    analysis = population.merge(labels, on="nodule_id", validate="one_to_one")
    analysis["partition_stratum"] = analysis["median_3_benign"].astype(int)
    partitions = make_patient_partitions(
        analysis,
        "partition_stratum",
        settings.cross_validation_folds,
        settings.cross_validation_repetitions,
        config.master_seed + settings.partition_seed_offset,
    )
    LOGGER.info("Fitting the fixed combined logistic model on recorded partitions")
    raw_predictions = cross_validated_predictions(
        analysis,
        settings.training_reference,
        settings.model,
        settings,
        config.master_seed + 10000,
        partitions,
    )
    fixed_predictions = average_predictions(raw_predictions)

    performance_frames = []
    effect_frames = []
    bootstrap_frames = []
    first_reference, second_reference = settings.primary_references
    first = _evaluation_frame(fixed_predictions, labels, first_reference, restricted=False)
    second = _evaluation_frame(fixed_predictions, labels, second_reference, restricted=False)
    performance, effects, bootstrap = _paired_metric_comparison(
        first,
        second,
        {"comparison": "label_assignment"},
        first_reference,
        second_reference,
        settings.bootstrap_repetitions,
        settings.confidence_level,
        config.master_seed + 20000,
    )
    performance_frames.append(performance)
    effect_frames.append(effects)
    bootstrap_frames.append(bootstrap)

    for index, reference in enumerate(settings.primary_references):
        full = _evaluation_frame(fixed_predictions, labels, reference, restricted=False)
        restricted = _evaluation_frame(fixed_predictions, labels, reference, restricted=True)
        performance, effects, bootstrap = _paired_metric_comparison(
            full,
            restricted,
            {"comparison": "case_exclusion", "reference": reference},
            "all_eligible_nodules",
            "exclude_median_3",
            settings.bootstrap_repetitions,
            settings.confidence_level,
            config.master_seed + 21000 + index,
        )
        performance_frames.append(performance)
        effect_frames.append(effects)
        bootstrap_frames.append(bootstrap)

    for index, reference in enumerate(settings.sensitivity_references):
        sensitivity = _evaluation_frame(fixed_predictions, labels, reference, restricted=False)
        performance, _, bootstrap = _paired_metric_comparison(
            sensitivity,
            sensitivity,
            {"comparison": "sensitivity", "reference": reference},
            reference,
            reference,
            settings.bootstrap_repetitions,
            settings.confidence_level,
            config.master_seed + 22000 + index,
        )
        performance_frames.append(performance.iloc[::2].reset_index(drop=True))
        bootstrap_frames.append(bootstrap)

    case_list = analysis[
        [
            "nodule_id",
            "patient_id",
            "reader_count",
            "malignancy_median",
            *labels.columns.drop("nodule_id").tolist(),
        ]
    ].copy()
    references = reference_summary(labels)
    references["description"] = references["reference"].map(REFERENCE_DESCRIPTIONS)
    retained_mask = labels["median_exclude_3"].notna()
    retained_counts = []
    retained_malignant = []
    for reference in references["reference"]:
        retained = labels.loc[retained_mask, reference].dropna().astype(int)
        retained_counts.append(len(retained))
        retained_malignant.append(int(retained.sum()))
    references["retained_after_excluding_median_3"] = retained_counts
    references["malignant_after_excluding_median_3"] = retained_malignant
    outputs: dict[str, tuple[pd.DataFrame, str]] = {
        "reference_definitions": (references, "csv"),
        "case_list": (case_list, "csv"),
        "partitions": (partitions, "csv"),
        "predictions": (raw_predictions, "parquet"),
        "performance": (pd.concat(performance_frames, ignore_index=True), "csv"),
        "effects": (pd.concat(effect_frames, ignore_index=True), "csv"),
        "bootstrap_replicates": (pd.concat(bootstrap_frames, ignore_index=True), "parquet"),
    }
    output_dir = config.paths.results_dir / "label_case_effects"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, (frame, file_type) in outputs.items():
        path = output_dir / f"{name}.{file_type}"
        if file_type == "csv":
            frame.to_csv(path, index=False, lineterminator="\n")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        output_paths[name] = path

    manifest = base_manifest(config.project_root, config.config_path, "label_case_effects")
    manifest.update(
        {
            "population": {
                "patients": int(population["patient_id"].nunique()),
                "nodules": len(population),
                "annotations": len(population_annotations),
            },
            "method": {
                "fixed_prediction_training_reference": settings.training_reference,
                "fixed_model": settings.model,
                "primary_references": list(settings.primary_references),
                "sensitivity_references": list(settings.sensitivity_references),
                "cross_validation_unit": "patient",
                "cross_validation_folds": settings.cross_validation_folds,
                "cross_validation_repetitions": settings.cross_validation_repetitions,
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": settings.bootstrap_repetitions,
                "case_exclusion_rule": "median_malignancy_rating_equals_3",
            },
            "inputs": {name: sha256_file(paths[name]) for name in ("annotations", "nodules")},
            "outputs": {
                name: {
                    "path": str(path.relative_to(config.project_root)),
                    "sha256": sha256_file(path),
                }
                for name, path in output_paths.items()
            },
        }
    )
    write_json(output_dir / "manifest.json", manifest)
    return manifest
