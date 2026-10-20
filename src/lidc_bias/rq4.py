"""RQ4 frozen-classifier evaluation under panel relabelling."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .metrics import long_metric_summary
from .modelling import cross_validated_predictions
from .provenance import base_manifest, sha256_file, write_json
from .references import build_reference_labels
from .tables import GEOMETRY_FEATURES, SEMANTIC_FEATURES
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)


def _reader_a_frame(annotations: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    working = annotations.copy()
    working["log_volume"] = np.log(working["volume_mm3"].astype(float))
    working["log_surface_area"] = np.log(working["surface_area_mm2"].astype(float))
    columns = [
        "annotation_id",
        "nodule_id",
        "patient_id",
        "malignancy",
        *GEOMETRY_FEATURES,
        *SEMANTIC_FEATURES,
    ]
    frame = assignments[["nodule_id", "reader_a_annotation_id"]].merge(
        working[columns],
        left_on=["nodule_id", "reader_a_annotation_id"],
        right_on=["nodule_id", "annotation_id"],
        validate="one_to_one",
    )
    frame["local_label"] = np.where(
        frame["malignancy"] == 3,
        np.nan,
        (frame["malignancy"] >= 4).astype(float),
    )
    return frame


def _aggregate(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("nodule_id", sort=True)
        .agg(
            patient_id=("patient_id", "first"),
            probability=("probability", "mean"),
            prediction_repetitions=("probability", "size"),
        )
        .reset_index()
    )


def _diameter_boundary(frame: pd.DataFrame, label: str) -> float:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    diameter = frame[["diameter_mm"]].to_numpy(dtype=float)
    model.fit(diameter, frame[label].astype(int))
    coefficient = float(model.coef_[0, 0])
    if abs(coefficient) < 1e-12:
        return np.nan
    return float(-model.intercept_[0] / coefficient)


def run_rq4(config: AnalysisConfig) -> dict[str, Any]:
    """Freeze local-reader predictions and vary only panel labels."""
    paths = processed_paths(config)
    required = ("annotations", "nodules")
    if any(not paths[name].is_file() for name in required):
        raise SourceDataError("Processed inputs are missing. Run ingest first")
    assignment_path = config.paths.results_dir / "rq3" / "reader_assignments.parquet"
    latent_path = config.paths.results_dir / "rq6" / "posterior.parquet"
    if not assignment_path.is_file() or not latent_path.is_file():
        raise SourceDataError("RQ3 reader assignments and RQ6 posterior are required for RQ4")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    assignments = pd.read_parquet(assignment_path)
    latent = pd.read_parquet(latent_path)
    population = nodules.loc[
        nodules["reader_count"].between(config.rq4.minimum_readers, config.rq4.maximum_readers)
    ]
    local = _reader_a_frame(annotations, assignments)
    if set(local["nodule_id"]) != set(population["nodule_id"]):
        raise AnalysisError("RQ4 reader assignment does not match its analysis population")

    cv_config = replace(
        config.rq2,
        cross_validation_folds=config.rq3.cross_validation_folds,
        cross_validation_repetitions=config.rq3.cross_validation_repetitions,
    )
    LOGGER.info("Fitting frozen RQ4 classifier")
    raw_predictions = cross_validated_predictions(
        local,
        "local_label",
        config.rq4.model,
        cv_config,
        config.master_seed + 40000,
    )
    frozen = _aggregate(raw_predictions).merge(
        local[["nodule_id", "malignancy", "diameter_mm"]],
        on="nodule_id",
        validate="one_to_one",
    )

    population_annotations = annotations.loc[annotations["nodule_id"].isin(population["nodule_id"])]
    references = build_reference_labels(
        population,
        population_annotations,
        config.master_seed + config.rq2.random_reader_seed_offset,
        latent,
    )
    frozen = frozen.merge(
        references[["nodule_id", "any_reader_4_or_5", "latent_class"]],
        on="nodule_id",
        validate="one_to_one",
    )

    indeterminate = frozen.loc[frozen["malignancy"] == 3, "nodule_id"].sort_values().to_numpy()
    rng = np.random.default_rng(config.master_seed + config.rq4.reader_assignment_seed_offset)
    nested_order = indeterminate[rng.permutation(len(indeterminate))]
    summary_frames = []
    bootstrap_frames = []
    panel_rows = []
    panel_labels = []
    for index, percentage in enumerate(config.rq4.relabelling_percentages):
        count = round(len(indeterminate) * percentage / 100.0)
        relabelled = set(nested_order[:count])
        evaluation = frozen.copy()
        evaluation["label"] = (evaluation["malignancy"] >= 4).astype(int)
        evaluation.loc[evaluation["nodule_id"].isin(relabelled), "label"] = 1
        summary, bootstrap = long_metric_summary(
            {
                "panel": "simulated_cautious_panel",
                "relabelled_percentage": percentage,
            },
            evaluation,
            config.rq4.bootstrap_repetitions,
            config.rq4.confidence_level,
            config.master_seed + 41000 + index,
        )
        summary_frames.append(summary)
        bootstrap_frames.append(bootstrap)
        boundary = _diameter_boundary(evaluation, "label")
        panel_rows.append(
            {
                "relabelled_percentage": percentage,
                "indeterminate_nodules": len(indeterminate),
                "relabelled_nodules": count,
                "malignant_nodules": int(evaluation["label"].sum()),
                "prevalence": float(evaluation["label"].mean()),
                "panel_implied_diameter_boundary_mm": boundary,
            }
        )
        panel_labels.append(
            evaluation[["nodule_id", "label"]].assign(relabelled_percentage=percentage)
        )

    for panel_name, reference in (
        ("real_any_reader_panel", "any_reader_4_or_5"),
        ("latent_reference", "latent_class"),
    ):
        evaluation = frozen.rename(columns={reference: "label"})
        summary, bootstrap = long_metric_summary(
            {"panel": panel_name, "relabelled_percentage": np.nan},
            evaluation,
            config.rq4.bootstrap_repetitions,
            config.rq4.confidence_level,
            config.master_seed + (42000 if panel_name.startswith("real") else 43000),
        )
        summary_frames.append(summary)
        bootstrap_frames.append(bootstrap)

    output_dir = config.paths.results_dir / "rq4"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "raw_predictions": raw_predictions,
        "frozen_predictions": frozen,
        "panel_labels": pd.concat(panel_labels, ignore_index=True),
        "panel_summary": pd.DataFrame.from_records(panel_rows),
        "performance": pd.concat(summary_frames, ignore_index=True),
        "bootstrap_replicates": pd.concat(bootstrap_frames, ignore_index=True),
    }
    output_paths = {}
    for name, frame in outputs.items():
        suffix = "csv" if name in {"panel_summary", "performance"} else "parquet"
        path = output_dir / f"{name}.{suffix}"
        if suffix == "csv":
            frame.to_csv(path, index=False, lineterminator="\n")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        output_paths[name] = path

    manifest = base_manifest(config.project_root, config.config_path, "rq4")
    manifest.update(
        {
            "population": {
                "patients": int(frozen["patient_id"].nunique()),
                "nodules": len(frozen),
                "local_reader_indeterminate_nodules": len(indeterminate),
            },
            "method": {
                "frozen_model": config.rq4.model,
                "local_reader": "reader_a_from_rq3",
                "local_rating_3_excluded_from_training": True,
                "relabel_order": "fixed_seeded_nested_random_order",
                "diameter_boundary": "univariable_logistic_panel_probability_0.5",
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": config.rq4.bootstrap_repetitions,
            },
            "inputs": {
                **{name: sha256_file(paths[name]) for name in required},
                "rq3_reader_assignments": sha256_file(assignment_path),
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
