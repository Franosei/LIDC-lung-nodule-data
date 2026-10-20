"""RQ3 shared-rater error analysis."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bootstrap import percentile_interval
from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .metrics import long_metric_summary
from .modelling import cross_validated_predictions
from .provenance import base_manifest, sha256_file, write_json
from .tables import GEOMETRY_FEATURES, SEMANTIC_FEATURES
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)


def assign_readers(
    annotations: pd.DataFrame, nodule_ids: set[str], seed: int
) -> pd.DataFrame:
    """Assign distinct A, B and C annotations within each eligible nodule."""
    working = annotations.loc[annotations["nodule_id"].isin(nodule_ids)].copy()
    working["log_volume"] = np.log(working["volume_mm3"].astype(float))
    working["log_surface_area"] = np.log(working["surface_area_mm2"].astype(float))
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    features = GEOMETRY_FEATURES + SEMANTIC_FEATURES
    for nodule_id, group in working.sort_values(
        ["nodule_id", "annotation_id"]
    ).groupby("nodule_id", sort=True):
        if len(group) < 3:
            raise AnalysisError(f"RQ3 nodule has fewer than three readers: {nodule_id}")
        selected = group.iloc[rng.permutation(len(group))[:3]]
        record: dict[str, Any] = {
            "nodule_id": nodule_id,
            "patient_id": selected.iloc[0]["patient_id"],
            "reader_a_annotation_id": selected.iloc[0]["annotation_id"],
            "reader_b_annotation_id": selected.iloc[1]["annotation_id"],
            "reader_c_annotation_id": selected.iloc[2]["annotation_id"],
            "label_a": int(selected.iloc[0]["malignancy"] >= 4),
            "label_c": int(selected.iloc[2]["malignancy"] >= 4),
        }
        for source, position in (("a", 0), ("b", 1)):
            for feature in features:
                record[f"{source}_{feature}"] = selected.iloc[position][feature]
        records.append(record)
    return pd.DataFrame.from_records(records)


def _source_frame(assignments: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = assignments[
        ["nodule_id", "patient_id", "label_a", "label_c"]
        + [f"{source}_{feature}" for feature in GEOMETRY_FEATURES + SEMANTIC_FEATURES]
    ].copy()
    return frame.rename(
        columns={
            f"{source}_{feature}": feature
            for feature in GEOMETRY_FEATURES + SEMANTIC_FEATURES
        }
    )


def _aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("nodule_id", sort=True)
        .agg(
            patient_id=("patient_id", "first"),
            probability=("probability", "mean"),
        )
        .reset_index()
    )


def _paired_auc_difference(
    frame: pd.DataFrame,
    left: str,
    right: str,
    label: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float, int, pd.DataFrame]:
    def difference(indices: np.ndarray) -> float:
        labels = frame[label].to_numpy(dtype=int)[indices]
        if np.unique(labels).size < 2:
            return np.nan
        return float(
            roc_auc_score(labels, frame[left].to_numpy()[indices])
            - roc_auc_score(labels, frame[right].to_numpy()[indices])
        )

    patients = np.asarray(sorted(frame["patient_id"].astype(str).unique()))
    patient_values = frame["patient_id"].astype(str).to_numpy()
    indices = {patient: np.flatnonzero(patient_values == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    values = []
    for _repetition in range(repetitions):
        sampled = rng.choice(patients, len(patients), replace=True)
        draw = np.concatenate([indices[patient] for patient in sampled])
        values.append(difference(draw))
    lower, upper, valid = percentile_interval(values, confidence_level)
    estimate = difference(np.arange(len(frame)))
    replicates = pd.DataFrame(
        {"bootstrap_repetition": np.arange(repetitions), "auc_difference": values}
    )
    return estimate, lower, upper, valid, replicates


def run_rq3(config: AnalysisConfig) -> dict[str, Any]:
    """Evaluate same-reader inflation and feature-family rank reversal."""
    paths = processed_paths(config)
    if not paths["annotations"].is_file() or not paths["nodules"].is_file():
        raise SourceDataError("Processed inputs are missing. Run ingest first")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    population = nodules.loc[
        nodules["reader_count"].between(config.rq3.minimum_readers, config.rq3.maximum_readers)
    ]
    assignments = assign_readers(
        annotations,
        set(population["nodule_id"]),
        config.master_seed + config.rq3.reader_assignment_seed_offset,
    )
    cv_config = replace(
        config.rq2,
        cross_validation_folds=config.rq3.cross_validation_folds,
        cross_validation_repetitions=config.rq3.cross_validation_repetitions,
    )

    raw_predictions = []
    aggregate_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for source, source_name in (("a", "same_reader"), ("b", "other_reader")):
        frame = _source_frame(assignments, source)
        for model_index, model in enumerate(("geometry_logistic", "semantic_logistic")):
            LOGGER.info("Running RQ3 source=%s model=%s", source_name, model)
            predictions = cross_validated_predictions(
                frame,
                "label_a",
                model,
                cv_config,
                config.master_seed + 30000 + model_index * 1000 + (0 if source == "a" else 5000),
            )
            predictions["feature_source"] = source_name
            raw_predictions.append(predictions)
            aggregate_lookup[(source_name, model)] = _aggregate(predictions)

    summaries = []
    bootstrap = []
    evaluation_frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    analysis_labels = assignments[["nodule_id", "label_a", "label_c"]]
    comparison_index = 0
    for (source, model), aggregate in aggregate_lookup.items():
        joined = aggregate.merge(analysis_labels, on="nodule_id", validate="one_to_one")
        for evaluation, label_column in (("reader_a", "label_a"), ("reader_c", "label_c")):
            metric_frame = joined[["nodule_id", "patient_id", "probability", label_column]].rename(
                columns={label_column: "label"}
            )
            summary, replicates = long_metric_summary(
                {
                    "feature_source": source,
                    "model": model,
                    "evaluation_reader": evaluation,
                },
                metric_frame,
                config.rq3.bootstrap_repetitions,
                config.rq3.confidence_level,
                config.master_seed + 31000 + comparison_index,
            )
            summaries.append(summary)
            bootstrap.append(replicates)
            evaluation_frames[(source, model, evaluation)] = metric_frame
            comparison_index += 1

    comparison_rows = []
    comparison_bootstraps = []
    comparison_specs = []
    for model in ("geometry_logistic", "semantic_logistic"):
        comparison_specs.append(
            ("same_reader_inflation", model, "reader_a", "same_reader", "other_reader")
        )
    for source in ("same_reader", "other_reader"):
        for evaluation in ("reader_a", "reader_c"):
            comparison_specs.append(
                (
                    "semantic_minus_geometry",
                    source,
                    evaluation,
                    "semantic_logistic",
                    "geometry_logistic",
                )
            )

    for index, (comparison, stratum, evaluation, left_key, right_key) in enumerate(
        comparison_specs
    ):
        if comparison == "same_reader_inflation":
            left_frame = evaluation_frames[(left_key, stratum, evaluation)]
            right_frame = evaluation_frames[(right_key, stratum, evaluation)]
        else:
            left_frame = evaluation_frames[(stratum, left_key, evaluation)]
            right_frame = evaluation_frames[(stratum, right_key, evaluation)]
        paired = left_frame.merge(
            right_frame[["nodule_id", "probability"]],
            on="nodule_id",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        ).rename(columns={"probability_left": "left", "probability_right": "right"})
        estimate, lower, upper, valid, replicates = _paired_auc_difference(
            paired,
            "left",
            "right",
            "label",
            config.rq3.bootstrap_repetitions,
            config.rq3.confidence_level,
            config.master_seed + 32000 + index,
        )
        comparison_rows.append(
            {
                "comparison": comparison,
                "stratum": stratum,
                "evaluation_reader": evaluation,
                "estimate": estimate,
                "confidence_level": config.rq3.confidence_level,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstrap_repetitions": valid,
            }
        )
        replicates["comparison"] = comparison
        replicates["stratum"] = stratum
        replicates["evaluation_reader"] = evaluation
        comparison_bootstraps.append(replicates)

    outputs = {
        "reader_assignments": assignments,
        "predictions": pd.concat(raw_predictions, ignore_index=True),
        "performance": pd.concat(summaries, ignore_index=True),
        "metric_bootstrap": pd.concat(bootstrap, ignore_index=True),
        "auc_comparisons": pd.DataFrame.from_records(comparison_rows),
        "comparison_bootstrap": pd.concat(comparison_bootstraps, ignore_index=True),
    }
    output_dir = config.paths.results_dir / "rq3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, frame in outputs.items():
        suffix = "csv" if name in {"performance", "auc_comparisons"} else "parquet"
        path = output_dir / f"{name}.{suffix}"
        if suffix == "csv":
            frame.to_csv(path, index=False, lineterminator="\n")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        output_paths[name] = path

    manifest = base_manifest(config.project_root, config.config_path, "rq3")
    manifest.update(
        {
            "population": {
                "patients": int(assignments["patient_id"].nunique()),
                "nodules": len(assignments),
            },
            "method": {
                "reader_assignment": "deterministic_seeded_without_replacement_within_nodule",
                "cross_validation_unit": "patient",
                "cross_validation_folds": config.rq3.cross_validation_folds,
                "cross_validation_repetitions": config.rq3.cross_validation_repetitions,
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": config.rq3.bootstrap_repetitions,
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
