"""RQ2 evaluation under alternative radiologist-derived references."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .metrics import classification_metrics, long_metric_summary
from .modelling import MODEL_FEATURES, cross_validated_predictions
from .provenance import base_manifest, sha256_file, write_json
from .references import (
    IMPLEMENTED_REFERENCES,
    REFERENCE_DESCRIPTIONS,
    build_reference_labels,
    reference_summary,
)
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)


def _population(nodules: pd.DataFrame, minimum: int, maximum: int) -> pd.DataFrame:
    population = nodules.loc[nodules["reader_count"].between(minimum, maximum)].copy()
    if population.empty:
        raise AnalysisError("RQ2 analysis population is empty")
    return population.reset_index(drop=True)


def _aggregate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities to one value per nodule."""
    identity = predictions.groupby("nodule_id", sort=True).agg(
        patient_id=("patient_id", "first"),
        label=("label", "first"),
        probability=("probability", "mean"),
        prediction_repetitions=("probability", "size"),
    )
    return identity.reset_index()


def _per_repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repetition, group in predictions.groupby("repetition", sort=True):
        evaluable = group.loc[group["label"].notna()]
        metrics = classification_metrics(
            evaluable["label"].to_numpy(dtype=int),
            evaluable["probability"].to_numpy(dtype=float),
        )
        rows.extend(
            {
                "reference": str(group["reference"].iloc[0]),
                "model": str(group["model"].iloc[0]),
                "repetition": int(repetition),
                "metric": metric,
                "value": value,
                "nodules": len(evaluable),
                "patients": int(evaluable["patient_id"].nunique()),
            }
            for metric, value in metrics.items()
        )
    return pd.DataFrame.from_records(rows)


def _fixed_reference_comparisons(
    fixed_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    replicates: list[pd.DataFrame] = []
    for reference_index, reference in enumerate(IMPLEMENTED_REFERENCES):
        comparison = fixed_predictions.drop(columns="label").merge(
            labels[["nodule_id", reference]], on="nodule_id", how="left", validate="one_to_one"
        )
        comparison = comparison.rename(columns={reference: "label"})
        comparison = comparison.loc[comparison["label"].notna()].copy()
        comparison["label"] = comparison["label"].astype(int)
        summary, bootstrap = long_metric_summary(
            {
                "prediction_reference": config.rq2.fixed_prediction_reference,
                "prediction_model": config.rq2.fixed_prediction_model,
                "evaluation_reference": reference,
                "status": "implemented",
            },
            comparison,
            config.rq2.bootstrap_repetitions,
            config.rq2.confidence_level,
            config.master_seed + 3000 + reference_index,
        )
        summaries.append(summary)
        replicates.append(bootstrap)

    return pd.concat(summaries, ignore_index=True), pd.concat(replicates, ignore_index=True)


def _excluded_case_analysis(
    fixed_predictions: pd.DataFrame,
    labels: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = config.rq2.excluded_case_reference
    if reference not in IMPLEMENTED_REFERENCES:
        raise AnalysisError(f"Unsupported excluded-case reference: {reference}")
    merged = fixed_predictions.drop(columns="label").merge(
        labels[["nodule_id", "median_exclude_3", reference]],
        on="nodule_id",
        how="left",
        validate="one_to_one",
    )
    excluded = merged.loc[merged["median_exclude_3"].isna() & merged[reference].notna()].copy()
    excluded = excluded.rename(columns={reference: "label"})
    excluded["label"] = excluded["label"].astype(int)
    if excluded["label"].nunique() != 2:
        raise AnalysisError(
            f"Excluded-case reference {reference!r} does not contain both classes"
        )
    summary, bootstrap = long_metric_summary(
        {
            "training_reference": config.rq2.fixed_prediction_reference,
            "model": config.rq2.fixed_prediction_model,
            "evaluation_reference": reference,
            "population": "median_rating_3_excluded_from_training",
            "status": "implemented",
        },
        excluded,
        config.rq2.bootstrap_repetitions,
        config.rq2.confidence_level,
        config.master_seed + 4000,
    )
    return summary, bootstrap, excluded


def run_rq2(config: AnalysisConfig) -> dict[str, Any]:
    """Train and evaluate all prespecified RQ2 models and references."""
    paths = processed_paths(config)
    required = [paths["annotations"], paths["nodules"]]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise SourceDataError(f"Processed inputs are missing. Run ingest first: {missing}")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    latent_path = config.paths.results_dir / "rq6" / "posterior.parquet"
    if not latent_path.is_file():
        raise SourceDataError("RQ6 posterior is missing. Run rq6 before rq2")
    latent_posterior = pd.read_parquet(latent_path)
    population = _population(
        nodules, config.rq2.minimum_readers, config.rq2.maximum_readers
    )
    population_annotations = annotations.loc[
        annotations["nodule_id"].isin(population["nodule_id"])
    ].copy()
    labels = build_reference_labels(
        population,
        population_annotations,
        config.master_seed + config.rq2.random_reader_seed_offset,
        latent_posterior,
    )
    analysis = population.merge(labels, on="nodule_id", how="left", validate="one_to_one")

    prediction_frames: list[pd.DataFrame] = []
    per_repeat_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    models = list(MODEL_FEATURES)

    comparison_index = 0
    for reference in IMPLEMENTED_REFERENCES:
        for model in models:
            LOGGER.info("Running RQ2 reference=%s model=%s", reference, model)
            predictions = cross_validated_predictions(
                analysis,
                reference,
                model,
                config.rq2,
                config.master_seed + 10000 * comparison_index,
            )
            prediction_frames.append(predictions)
            per_repeat_frames.append(_per_repeat_metrics(predictions))
            aggregated = _aggregate_predictions(predictions)
            evaluable = aggregated.loc[aggregated["label"].notna()].copy()
            evaluable["label"] = evaluable["label"].astype(int)
            summary, bootstrap = long_metric_summary(
                {"reference": reference, "model": model, "status": "implemented"},
                evaluable,
                config.rq2.bootstrap_repetitions,
                config.rq2.confidence_level,
                config.master_seed + 20000 + comparison_index,
            )
            summary_frames.append(summary)
            bootstrap_frames.append(bootstrap)
            comparison_index += 1

    predictions = pd.concat(prediction_frames, ignore_index=True)
    per_repeat = pd.concat(per_repeat_frames, ignore_index=True)
    performance = pd.concat(summary_frames, ignore_index=True)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)

    fixed_mask = (
        (predictions["reference"] == config.rq2.fixed_prediction_reference)
        & (predictions["model"] == config.rq2.fixed_prediction_model)
    )
    if not fixed_mask.any():
        raise AnalysisError("Configured fixed prediction model and reference were not produced")
    fixed_predictions = _aggregate_predictions(predictions.loc[fixed_mask])
    fixed_summary, fixed_bootstrap = _fixed_reference_comparisons(
        fixed_predictions, labels, config
    )
    excluded_summary, excluded_bootstrap, excluded_predictions = _excluded_case_analysis(
        fixed_predictions, labels, config
    )
    references = reference_summary(labels)

    output_dir = config.paths.results_dir / "rq2"
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, tuple[pd.DataFrame, str]] = {
        "reference_summary": (references, "csv"),
        "performance": (performance, "csv"),
        "per_repeat_metrics": (per_repeat, "parquet"),
        "predictions": (predictions, "parquet"),
        "bootstrap_replicates": (bootstrap, "parquet"),
        "fixed_reference_comparison": (fixed_summary, "csv"),
        "fixed_reference_bootstrap": (fixed_bootstrap, "parquet"),
        "excluded_case_summary": (excluded_summary, "csv"),
        "excluded_case_bootstrap": (excluded_bootstrap, "parquet"),
        "excluded_case_predictions": (excluded_predictions, "parquet"),
    }
    output_paths = {}
    for name, (frame, file_type) in files.items():
        path = output_dir / f"{name}.{file_type}"
        if file_type == "csv":
            frame.to_csv(path, index=False, lineterminator="\n")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        output_paths[name] = path

    manifest = base_manifest(config.project_root, config.config_path, "rq2")
    manifest.update(
        {
            "population": {
                "patients": int(population["patient_id"].nunique()),
                "nodules": len(population),
                "annotations": len(population_annotations),
                "minimum_readers": config.rq2.minimum_readers,
                "maximum_readers": config.rq2.maximum_readers,
            },
            "method": {
                "references": {
                    name: REFERENCE_DESCRIPTIONS[name]
                    for name in (*IMPLEMENTED_REFERENCES, "latent_class")
                },
                "latent_class_status": "implemented",
                "cross_validation_unit": "patient",
                "cross_validation_folds": config.rq2.cross_validation_folds,
                "cross_validation_repetitions": config.rq2.cross_validation_repetitions,
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": config.rq2.bootstrap_repetitions,
                "fixed_prediction_reference": config.rq2.fixed_prediction_reference,
                "fixed_prediction_model": config.rq2.fixed_prediction_model,
                "excluded_case_reference": config.rq2.excluded_case_reference,
            },
            "inputs": {
                "annotations": sha256_file(paths["annotations"]),
                "nodules": sha256_file(paths["nodules"]),
                "rq6_posterior": sha256_file(latent_path),
            },
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
