"""Reciprocal matched analysis of shared annotation assessments."""

from __future__ import annotations

import json
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
from .tables import GEOMETRY_FEATURES, SEMANTIC_FEATURES
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)
EFFECT_METRICS = ("auc", "brier_score")


def assign_annotation_roles(
    annotations: pd.DataFrame,
    nodules: pd.DataFrame,
    seed: int,
    assignment_repetition: int,
) -> pd.DataFrame:
    """Assign three distinct annotation records to roles A, B and C per nodule."""
    working = annotations.loc[annotations["nodule_id"].isin(nodules["nodule_id"])].copy()
    working["log_volume"] = np.log(working["volume_mm3"].astype(float))
    working["log_surface_area"] = np.log(working["surface_area_mm2"].astype(float))
    fixed_geometry = nodules.set_index("nodule_id")[GEOMETRY_FEATURES]
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for nodule_id, group in working.sort_values(
        ["nodule_id", "annotation_id"]
    ).groupby("nodule_id", sort=True):
        if len(group) < 3:
            raise AnalysisError(
                f"Shared-assessment nodule has fewer than three records: {nodule_id}"
            )
        selected = group.iloc[rng.permutation(len(group))[:3]]
        record: dict[str, Any] = {
            "assignment_repetition": assignment_repetition,
            "assignment_seed": seed,
            "nodule_id": nodule_id,
            "patient_id": selected.iloc[0]["patient_id"],
        }
        for role, position in (("a", 0), ("b", 1), ("c", 2)):
            annotation = selected.iloc[position]
            record[f"annotation_{role}_id"] = annotation["annotation_id"]
            record[f"rating_{role}"] = int(annotation["malignancy"])
            record[f"label_{role}"] = int(annotation["malignancy"] >= 4)
        for feature in GEOMETRY_FEATURES:
            record[f"fixed_{feature}"] = fixed_geometry.loc[nodule_id, feature]
        for source, position in (("a", 0), ("b", 1)):
            for feature in GEOMETRY_FEATURES + SEMANTIC_FEATURES:
                record[f"{source}_{feature}"] = selected.iloc[position][feature]
        records.append(record)
    assignments = pd.DataFrame.from_records(records)
    role_ids = assignments[["annotation_a_id", "annotation_b_id", "annotation_c_id"]]
    if not role_ids.nunique(axis=1).eq(3).all():
        raise AnalysisError("Annotation roles are not distinct within every nodule")
    return assignments


def _model_frame(
    assignments: pd.DataFrame,
    source: str,
    geometry: str,
    training_label: str,
) -> pd.DataFrame:
    frame = assignments[["nodule_id", "patient_id", training_label]].copy()
    for feature in GEOMETRY_FEATURES:
        column = f"fixed_{feature}" if geometry == "fixed" else f"{source}_{feature}"
        frame[feature] = assignments[column]
    for feature in SEMANTIC_FEATURES:
        frame[feature] = assignments[f"{source}_{feature}"]
    return frame


def _effect(frame: pd.DataFrame, label_column: str) -> dict[str, float]:
    labels = frame[label_column].to_numpy(dtype=int)
    if np.unique(labels).size != 2:
        return {metric: float("nan") for metric in EFFECT_METRICS}
    same = frame["probability_same"].to_numpy(dtype=float)
    other = frame["probability_other"].to_numpy(dtype=float)
    return {
        "auc": float(roc_auc_score(labels, same) - roc_auc_score(labels, other)),
        # Positive values consistently mean better same-assessment performance.
        "brier_score": float(
            brier_score_loss(labels, other) - brier_score_loss(labels, same)
        ),
    }


def _set_effect_definitions(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    same_assessment = summary["evaluation"].eq("training_reader")
    summary["effect_definition"] = np.select(
        [
            same_assessment & summary["metric"].eq("auc"),
            same_assessment & summary["metric"].eq("brier_score"),
            ~same_assessment & summary["metric"].eq("auc"),
        ],
        [
            "AUC(same-assessment source) - AUC(other source)",
            "Brier(other source) - Brier(same-assessment source)",
            "AUC(training-label source) - AUC(other source), evaluated against C",
        ],
        default="Brier(other source) - Brier(training-label source), evaluated against C",
    )
    return summary


def _pair_sources(
    predictions: dict[str, pd.DataFrame],
    target_role: str,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    same_source = target_role
    other_source = "b" if target_role == "a" else "a"
    paired = predictions[same_source].merge(
        predictions[other_source][["nodule_id", "probability"]],
        on="nodule_id",
        suffixes=("_same", "_other"),
        validate="one_to_one",
    )
    return paired.merge(
        labels[["nodule_id", f"label_{target_role}", "label_c"]],
        on="nodule_id",
        validate="one_to_one",
    )


def _bootstrap_effects(
    paired_frames: dict[tuple[int, str, str], pd.DataFrame],
    geometry: str,
    evaluation: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Bootstrap patients while retaining every assignment and reciprocal contrast."""
    example = next(iter(paired_frames.values()))
    patients = np.asarray(sorted(example["patient_id"].astype(str).unique()))
    index_lookup = {
        key: {
            patient: np.flatnonzero(frame["patient_id"].astype(str).to_numpy() == patient)
            for patient in patients
        }
        for key, frame in paired_frames.items()
        if key[1] == geometry
    }
    label_column = "label_c" if evaluation == "reader_c" else None
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        sampled = rng.choice(patients, len(patients), replace=True)
        assignment_values = {metric: [] for metric in EFFECT_METRICS}
        for key, lookup in index_lookup.items():
            _, _, target_role = key
            frame = paired_frames[key]
            draw = np.concatenate([lookup[patient] for patient in sampled])
            values = _effect(
                frame.iloc[draw],
                label_column or f"label_{target_role}",
            )
            for metric in EFFECT_METRICS:
                assignment_values[metric].append(values[metric])
        for metric in EFFECT_METRICS:
            rows.append(
                {
                    "geometry": geometry,
                    "evaluation": evaluation,
                    "metric": metric,
                    "bootstrap_repetition": repetition,
                    "effect": float(np.nanmean(assignment_values[metric])),
                }
            )
    replicates = pd.DataFrame.from_records(rows)
    summaries = []
    for metric in EFFECT_METRICS:
        values = replicates.loc[replicates["metric"] == metric, "effect"].to_numpy()
        lower, upper, valid = percentile_interval(values, confidence_level)
        summaries.append(
            {
                "geometry": geometry,
                "evaluation": evaluation,
                "metric": metric,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": confidence_level,
                "valid_bootstrap_repetitions": valid,
            }
        )
    return replicates, summaries


def run_shared_assessment(config: AnalysisConfig) -> dict[str, Any]:
    """Run reciprocal A/B comparisons over 20 recorded assignments and fixed folds."""
    settings = config.shared_assessment
    model_settings = config.label_case_effects
    paths = processed_paths(config)
    if not paths["annotations"].is_file() or not paths["nodules"].is_file():
        raise SourceDataError("Processed inputs are missing. Run ingest first")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    population = nodules.loc[
        nodules["reader_count"].between(settings.minimum_readers, settings.maximum_readers)
    ].copy()
    population["partition_stratum"] = (population["malignancy_median"] > 3).astype(int)
    partitions = make_patient_partitions(
        population,
        "partition_stratum",
        settings.cross_validation_folds,
        settings.cross_validation_repetitions,
        config.master_seed + settings.partition_seed_offset,
    )

    assignments_frames = []
    raw_prediction_frames = []
    aggregate_prediction_frames = []
    assignment_effect_rows = []
    partition_effect_rows = []
    paired_frames: dict[tuple[int, str, str], pd.DataFrame] = {}
    for assignment in range(settings.assignment_repetitions):
        assignment_seed = config.master_seed + settings.assignment_seed_offset + assignment
        LOGGER.info(
            "Shared-assessment assignment %d/%d (seed=%d)",
            assignment + 1,
            settings.assignment_repetitions,
            assignment_seed,
        )
        assigned = assign_annotation_roles(
            annotations, population, assignment_seed, assignment
        )
        assignments_frames.append(assigned)
        for geometry in ("fixed", "reader_specific"):
            for target_role in ("a", "b"):
                training_label = f"label_{target_role}"
                source_predictions: dict[str, pd.DataFrame] = {}
                raw_by_source: dict[str, pd.DataFrame] = {}
                for source in ("a", "b"):
                    frame = _model_frame(assigned, source, geometry, training_label)
                    raw = cross_validated_predictions(
                        frame,
                        training_label,
                        settings.model,
                        model_settings,
                        config.master_seed + 50000,
                        partitions,
                    )
                    raw["assignment_repetition"] = assignment
                    raw["assignment_seed"] = assignment_seed
                    raw["geometry"] = geometry
                    raw["target_role"] = target_role
                    raw["feature_source"] = source
                    raw_prediction_frames.append(raw)
                    raw_by_source[source] = raw
                    aggregate = average_predictions(raw)
                    aggregate["assignment_repetition"] = assignment
                    aggregate["assignment_seed"] = assignment_seed
                    aggregate["geometry"] = geometry
                    aggregate["target_role"] = target_role
                    aggregate["feature_source"] = source
                    aggregate_prediction_frames.append(aggregate)
                    source_predictions[source] = aggregate

                paired = _pair_sources(source_predictions, target_role, assigned)
                paired_frames[(assignment, geometry, target_role)] = paired
                for evaluation, label_column in (
                    ("training_reader", training_label),
                    ("reader_c", "label_c"),
                ):
                    values = _effect(paired, label_column)
                    for metric, value in values.items():
                        assignment_effect_rows.append(
                            {
                                "assignment_repetition": assignment,
                                "assignment_seed": assignment_seed,
                                "geometry": geometry,
                                "target_role": target_role,
                                "evaluation": evaluation,
                                "metric": metric,
                                "effect": value,
                            }
                        )
                    for repetition in sorted(partitions["repetition"].unique()):
                        per_repeat = {
                            source: raw.loc[raw["repetition"] == repetition]
                            .rename(columns={"probability": f"probability_{source}"})
                            for source, raw in raw_by_source.items()
                        }
                        same_source = target_role
                        other_source = "b" if target_role == "a" else "a"
                        repeat_pair = per_repeat[same_source].merge(
                            per_repeat[other_source][["nodule_id", f"probability_{other_source}"]],
                            on="nodule_id",
                            validate="one_to_one",
                        ).rename(
                            columns={
                                f"probability_{same_source}": "probability_same",
                                f"probability_{other_source}": "probability_other",
                            }
                        )
                        repeat_pair = repeat_pair.merge(
                            assigned[["nodule_id", training_label, "label_c"]],
                            on="nodule_id",
                            validate="one_to_one",
                        )
                        for evaluation, label_column in (
                            ("training_reader", training_label),
                            ("reader_c", "label_c"),
                        ):
                            for metric, value in _effect(repeat_pair, label_column).items():
                                partition_effect_rows.append(
                                    {
                                        "assignment_repetition": assignment,
                                        "geometry": geometry,
                                        "target_role": target_role,
                                        "evaluation": evaluation,
                                        "partition_repetition": int(repetition),
                                        "metric": metric,
                                        "effect": value,
                                    }
                                )

    assignment_effects = pd.DataFrame.from_records(assignment_effect_rows)
    partition_effects = pd.DataFrame.from_records(partition_effect_rows)
    reciprocal_assignment = (
        assignment_effects.groupby(
            ["assignment_repetition", "assignment_seed", "geometry", "evaluation", "metric"],
            as_index=False,
        )["effect"]
        .mean()
        .rename(columns={"effect": "reciprocal_mean_effect"})
    )
    reciprocal_partition = (
        partition_effects.groupby(
            ["partition_repetition", "geometry", "evaluation", "metric"],
            as_index=False,
        )["effect"]
        .mean()
        .rename(columns={"effect": "mean_effect_across_assignments_and_directions"})
    )

    bootstrap_frames = []
    interval_rows = []
    comparison_index = 0
    for geometry in ("fixed", "reader_specific"):
        for evaluation in ("training_reader", "reader_c"):
            bootstrap, summaries = _bootstrap_effects(
                paired_frames,
                geometry,
                evaluation,
                settings.bootstrap_repetitions,
                settings.confidence_level,
                config.master_seed + 60000 + comparison_index,
            )
            bootstrap_frames.append(bootstrap)
            interval_rows.extend(summaries)
            comparison_index += 1
    intervals = pd.DataFrame.from_records(interval_rows)
    summary = (
        reciprocal_assignment.groupby(["geometry", "evaluation", "metric"])[
            "reciprocal_mean_effect"
        ]
        .agg(
            mean_effect="mean",
            assignment_sd="std",
            assignment_min="min",
            assignment_max="max",
        )
        .reset_index()
    )
    partition_variation = (
        reciprocal_partition.groupby(["geometry", "evaluation", "metric"])[
            "mean_effect_across_assignments_and_directions"
        ]
        .std()
        .rename("partition_sd")
        .reset_index()
    )
    summary = summary.merge(partition_variation, on=["geometry", "evaluation", "metric"])
    summary = summary.merge(intervals, on=["geometry", "evaluation", "metric"])
    summary = _set_effect_definitions(summary)

    outputs: dict[str, tuple[pd.DataFrame, str]] = {
        "assignments": (pd.concat(assignments_frames, ignore_index=True), "parquet"),
        "partitions": (partitions, "csv"),
        "predictions": (pd.concat(raw_prediction_frames, ignore_index=True), "parquet"),
        "averaged_predictions": (
            pd.concat(aggregate_prediction_frames, ignore_index=True),
            "parquet",
        ),
        "directional_assignment_effects": (assignment_effects, "csv"),
        "assignment_effects": (reciprocal_assignment, "csv"),
        "partition_effects": (reciprocal_partition, "csv"),
        "effect_summary": (summary, "csv"),
        "bootstrap_replicates": (pd.concat(bootstrap_frames, ignore_index=True), "parquet"),
    }
    output_dir = config.paths.results_dir / "shared_assessment"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, (frame, file_type) in outputs.items():
        path = output_dir / f"{name}.{file_type}"
        if file_type == "csv":
            frame.to_csv(path, index=False, lineterminator="\n")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        output_paths[name] = path

    manifest = base_manifest(config.project_root, config.config_path, "shared_assessment")
    manifest.update(
        {
            "population": {
                "patients": int(population["patient_id"].nunique()),
                "nodules": len(population),
            },
            "method": {
                "assignment_repetitions": settings.assignment_repetitions,
                "assignment_seeds": sorted(
                    pd.concat(assignments_frames)["assignment_seed"].unique().tolist()
                ),
                "roles": "three distinct annotation records within each nodule",
                "reciprocal_targets": ["reader_a", "reader_b"],
                "independent_check": "reader_c",
                "geometry_analyses": ["fixed_nodule_average", "reader_specific"],
                "model": settings.model,
                "cross_validation_unit": "patient",
                "partitions_reused_for_every_assignment_source_and_geometry": True,
                "cross_validation_folds": settings.cross_validation_folds,
                "cross_validation_repetitions": settings.cross_validation_repetitions,
                "bootstrap_unit": "patient",
                "bootstrap_preserves_all_assignment_results": True,
                "bootstrap_repetitions": settings.bootstrap_repetitions,
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


def refresh_shared_assessment_metadata(config: AnalysisConfig) -> dict[str, Any]:
    """Refresh descriptive effect labels and provenance without refitting models."""
    output_dir = config.paths.results_dir / "shared_assessment"
    summary_path = output_dir / "effect_summary.csv"
    manifest_path = output_dir / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise SourceDataError("Shared-assessment outputs are missing; run the analysis first")
    summary = _set_effect_definitions(pd.read_csv(summary_path))
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(base_manifest(config.project_root, config.config_path, "shared_assessment"))
    for output in manifest["outputs"].values():
        path = config.project_root / output["path"]
        output["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    return manifest
