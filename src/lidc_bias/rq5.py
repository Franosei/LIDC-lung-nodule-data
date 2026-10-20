"""RQ5 training-set size versus label-quality experiment."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .modelling import build_model
from .provenance import base_manifest, sha256_file, write_json
from .references import build_reference_labels
from .rq3 import _paired_auc_difference
from .rq7 import _auc_interval, link_confirmed_diagnoses
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)


def _sample_noisy(
    frame: pd.DataFrame,
    excluded_patients: set[str],
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    available = frame.loc[~frame["patient_id"].isin(excluded_patients)].sort_values(
        "nodule_id"
    )
    target = round(fraction * len(available))
    if target == 0:
        return available.iloc[0:0].copy()
    if target == len(available):
        return available.copy()
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(available), size=target, replace=False))
    return available.iloc[positions].copy()


def _aggregate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    identity = frame.groupby("nodule_id", sort=True).agg(
        patient_id=("patient_id", "first"),
        probability=("probability", "mean"),
        prediction_repetitions=("probability", "size"),
    )
    return identity.reset_index()


def _write_auc_changes(
    config: AnalysisConfig, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Write paired AUC changes versus baseline from persisted predictions."""
    output_dir = config.paths.results_dir / "rq5"
    main = pd.read_parquet(output_dir / "main_predictions.parquet")
    diagnosis = pd.read_parquet(output_dir / "diagnosis_predictions.parquet")
    rows = []
    replicate_frames = []
    comparison_index = 0
    for model_name in config.rq5.models:
        for evaluation, raw, label_column in (
            ("consensus", main, "consensus_label"),
            ("latent_class", main, "latent_class"),
            ("confirmed_diagnosis", diagnosis, "diagnosis"),
        ):
            model_frame = raw.loc[raw["model"] == model_name]

            def aggregate_fraction(
                fraction: float,
                source: pd.DataFrame = model_frame,
                selected_label: str = label_column,
            ) -> pd.DataFrame:
                selected = source.loc[source["noisy_fraction"] == fraction]
                return (
                    selected.groupby("nodule_id", sort=True)
                    .agg(
                        patient_id=("patient_id", "first"),
                        label=(selected_label, "first"),
                        probability=("probability", "mean"),
                    )
                    .reset_index()
                )

            baseline = aggregate_fraction(0.0)
            for fraction in config.rq5.addition_fractions[1:]:
                comparison = aggregate_fraction(fraction).merge(
                    baseline[["nodule_id", "probability"]],
                    on="nodule_id",
                    suffixes=("_fraction", "_baseline"),
                    validate="one_to_one",
                )
                estimate, lower, upper, valid, replicates = _paired_auc_difference(
                    comparison,
                    "probability_fraction",
                    "probability_baseline",
                    "label",
                    config.rq5.bootstrap_repetitions,
                    config.rq5.confidence_level,
                    config.master_seed + 56000 + comparison_index,
                )
                rows.append(
                    {
                        "model": model_name,
                        "noisy_fraction": fraction,
                        "baseline_fraction": 0.0,
                        "evaluation_reference": evaluation,
                        "auc_change": estimate,
                        "confidence_level": config.rq5.confidence_level,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "valid_bootstrap_repetitions": valid,
                        "nodules": len(comparison),
                        "patients": int(comparison["patient_id"].nunique()),
                    }
                )
                replicates["model"] = model_name
                replicates["noisy_fraction"] = fraction
                replicates["evaluation_reference"] = evaluation
                replicate_frames.append(replicates)
                comparison_index += 1

    paths = {
        "auc_changes": output_dir / "auc_changes.csv",
        "auc_change_bootstrap": output_dir / "auc_change_bootstrap.parquet",
    }
    pd.DataFrame.from_records(rows).to_csv(
        paths["auc_changes"], index=False, lineterminator="\n"
    )
    pd.concat(replicate_frames, ignore_index=True).to_parquet(
        paths["auc_change_bootstrap"], index=False, compression="zstd"
    )
    manifest["method"]["paired_auc_change_baseline_fraction"] = 0.0
    for name, path in paths.items():
        manifest["outputs"][name] = {
            "path": str(path.relative_to(config.project_root)),
            "sha256": sha256_file(path),
        }
    return manifest


def refresh_rq5_auc_changes(config: AnalysisConfig) -> dict[str, Any]:
    """Add paired AUC changes to an existing complete RQ5 run."""
    manifest_path = config.paths.results_dir / "rq5" / "manifest.json"
    if not manifest_path.is_file():
        raise SourceDataError("RQ5 manifest is missing. Run rq5 first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _write_auc_changes(config, manifest)
    write_json(manifest_path, manifest)
    return manifest


def run_rq5(config: AnalysisConfig) -> dict[str, Any]:
    """Add progressively more single-reader labels to fixed consensus test folds."""
    paths = processed_paths(config)
    required = ("annotations", "nodules", "diagnosis_nodules")
    if any(not paths[name].is_file() for name in required):
        raise SourceDataError("Processed inputs are missing. Run ingest first")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    diagnosis_nodules = pd.read_parquet(paths["diagnosis_nodules"])
    latent_path = config.paths.results_dir / "rq6" / "posterior.parquet"
    if not latent_path.is_file():
        raise SourceDataError("RQ6 posterior is missing. Run rq6 before rq5")
    latent = pd.read_parquet(latent_path)

    consensus = nodules.loc[nodules["reader_count"].between(3, 4)].copy()
    noisy = nodules.loc[nodules["reader_count"].between(1, 2)].copy()
    consensus_annotations = annotations.loc[annotations["nodule_id"].isin(consensus["nodule_id"])]
    noisy_annotations = annotations.loc[annotations["nodule_id"].isin(noisy["nodule_id"])]
    consensus_labels = build_reference_labels(
        consensus,
        consensus_annotations,
        config.master_seed + config.rq2.random_reader_seed_offset,
        latent,
    )
    noisy_labels = build_reference_labels(
        noisy,
        noisy_annotations,
        config.master_seed + config.rq2.random_reader_seed_offset,
    )
    consensus = consensus.merge(consensus_labels, on="nodule_id", validate="one_to_one")
    noisy = noisy.merge(noisy_labels, on="nodule_id", validate="one_to_one")
    consensus["training_label"] = consensus[config.rq5.consensus_reference].astype(int)
    noisy["training_label"] = noisy[config.rq5.noisy_reference].astype(int)

    linked = link_confirmed_diagnoses(nodules, diagnosis_nodules)
    diagnosis_patients = set(linked["patient_id"])
    linked["diagnosis"] = (linked["diagnosis_code"] >= 2).astype(int)

    main_predictions = []
    diagnosis_predictions = []
    training_rows = []
    per_repeat_rows = []
    groups = consensus["patient_id"].astype(str)
    strata = consensus["training_label"].astype(int)

    for model_index, model_name in enumerate(config.rq5.models):
        for fraction_index, fraction in enumerate(config.rq5.addition_fractions):
            LOGGER.info("Running RQ5 model=%s noisy_fraction=%.2f", model_name, fraction)
            for repetition in range(config.rq5.repetitions):
                splitter = StratifiedGroupKFold(
                    n_splits=config.rq5.cross_validation_folds,
                    shuffle=True,
                    random_state=config.master_seed + 50000 + repetition,
                )
                repeat_predictions = []
                for fold, (train_index, test_index) in enumerate(
                    splitter.split(consensus, strata, groups), start=1
                ):
                    high_train = consensus.iloc[train_index]
                    test = consensus.iloc[test_index]
                    test_patients = set(test["patient_id"])
                    low_train = _sample_noisy(
                        noisy,
                        test_patients,
                        fraction,
                        config.master_seed
                        + 51000
                        + fraction_index * 10000
                        + repetition * 100
                        + fold,
                    )
                    training = pd.concat([high_train, low_train], ignore_index=True)
                    if training["training_label"].nunique() != 2:
                        raise AnalysisError("RQ5 training fold does not contain both classes")
                    estimator = build_model(
                        model_name,
                        config.rq2,
                        config.master_seed
                        + 52000
                        + model_index * 100000
                        + fraction_index * 10000
                        + repetition * 100
                        + fold,
                    )
                    estimator.fit(training, training["training_label"].astype(int))
                    probabilities = estimator.predict_proba(test)[:, 1]
                    predicted = pd.DataFrame(
                        {
                            "nodule_id": test["nodule_id"].to_numpy(),
                            "patient_id": test["patient_id"].to_numpy(),
                            "model": model_name,
                            "noisy_fraction": fraction,
                            "repetition": repetition,
                            "fold": fold,
                            "probability": probabilities,
                            "consensus_label": test[config.rq5.consensus_reference].to_numpy(),
                            "latent_class": test["latent_class"].to_numpy(),
                        }
                    )
                    main_predictions.append(predicted)
                    repeat_predictions.append(predicted)
                    training_rows.append(
                        {
                            "model": model_name,
                            "noisy_fraction": fraction,
                            "repetition": repetition,
                            "fold": fold,
                            "consensus_training_nodules": len(high_train),
                            "noisy_training_nodules": len(low_train),
                            "training_patients": int(training["patient_id"].nunique()),
                            "test_nodules": len(test),
                            "test_patients": int(test["patient_id"].nunique()),
                        }
                    )
                repeat_frame = pd.concat(repeat_predictions, ignore_index=True)
                for reference in ("consensus_label", "latent_class"):
                    per_repeat_rows.append(
                        {
                            "model": model_name,
                            "noisy_fraction": fraction,
                            "repetition": repetition,
                            "reference": reference,
                            "auc": float(
                                roc_auc_score(
                                    repeat_frame[reference], repeat_frame["probability"]
                                )
                            ),
                        }
                    )

                high_diagnosis_train = consensus.loc[
                    ~consensus["patient_id"].isin(diagnosis_patients)
                ]
                low_diagnosis_train = _sample_noisy(
                    noisy,
                    diagnosis_patients,
                    fraction,
                    config.master_seed + 53000 + fraction_index * 1000 + repetition,
                )
                diagnosis_training = pd.concat(
                    [high_diagnosis_train, low_diagnosis_train], ignore_index=True
                )
                diagnosis_model = build_model(
                    model_name,
                    config.rq2,
                    config.master_seed
                    + 54000
                    + model_index * 100000
                    + fraction_index * 1000
                    + repetition,
                )
                diagnosis_model.fit(
                    diagnosis_training, diagnosis_training["training_label"].astype(int)
                )
                diagnosis_probabilities = diagnosis_model.predict_proba(linked)[:, 1]
                diagnosis_predictions.append(
                    pd.DataFrame(
                        {
                            "nodule_id": linked["nodule_id"].to_numpy(),
                            "patient_id": linked["patient_id"].to_numpy(),
                            "model": model_name,
                            "noisy_fraction": fraction,
                            "repetition": repetition,
                            "diagnosis": linked["diagnosis"].to_numpy(),
                            "probability": diagnosis_probabilities,
                        }
                    )
                )

    main_predictions_frame = pd.concat(main_predictions, ignore_index=True)
    diagnosis_predictions_frame = pd.concat(diagnosis_predictions, ignore_index=True)
    summary_rows = []
    bootstrap_frames = []
    comparison_index = 0
    for model_name in config.rq5.models:
        for fraction in config.rq5.addition_fractions:
            main_group = main_predictions_frame.loc[
                (main_predictions_frame["model"] == model_name)
                & (main_predictions_frame["noisy_fraction"] == fraction)
            ]
            aggregate = _aggregate_predictions(main_group).merge(
                consensus[["nodule_id", config.rq5.consensus_reference, "latent_class"]],
                on="nodule_id",
                validate="one_to_one",
            )
            evaluation_specs = (
                ("consensus", config.rq5.consensus_reference, aggregate),
                ("latent_class", "latent_class", aggregate),
            )
            diagnosis_group = diagnosis_predictions_frame.loc[
                (diagnosis_predictions_frame["model"] == model_name)
                & (diagnosis_predictions_frame["noisy_fraction"] == fraction)
            ]
            diagnosis_aggregate = _aggregate_predictions(diagnosis_group).merge(
                linked[["nodule_id", "diagnosis"]], on="nodule_id", validate="one_to_one"
            )
            evaluation_specs += (("confirmed_diagnosis", "diagnosis", diagnosis_aggregate),)
            for evaluation, label_column, evaluation_frame in evaluation_specs:
                estimate, lower, upper, valid, bootstrap = _auc_interval(
                    evaluation_frame,
                    label_column,
                    "probability",
                    config.rq5.bootstrap_repetitions,
                    config.rq5.confidence_level,
                    config.master_seed + 55000 + comparison_index,
                )
                summary_rows.append(
                    {
                        "model": model_name,
                        "noisy_fraction": fraction,
                        "evaluation_reference": evaluation,
                        "auc": estimate,
                        "confidence_level": config.rq5.confidence_level,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "valid_bootstrap_repetitions": valid,
                        "nodules": len(evaluation_frame),
                        "patients": int(evaluation_frame["patient_id"].nunique()),
                    }
                )
                bootstrap["model"] = model_name
                bootstrap["noisy_fraction"] = fraction
                bootstrap["evaluation_reference"] = evaluation
                bootstrap_frames.append(bootstrap)
                comparison_index += 1

    output_dir = config.paths.results_dir / "rq5"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_predictions": main_predictions_frame,
        "diagnosis_predictions": diagnosis_predictions_frame,
        "training_counts": pd.DataFrame.from_records(training_rows),
        "per_repeat_auc": pd.DataFrame.from_records(per_repeat_rows),
        "auc_summary": pd.DataFrame.from_records(summary_rows),
        "bootstrap_replicates": pd.concat(bootstrap_frames, ignore_index=True),
    }
    output_paths = {}
    for name, frame in outputs.items():
        suffix = "csv" if name == "auc_summary" else "parquet"
        path = output_dir / f"{name}.{suffix}"
        if suffix == "csv":
            frame.to_csv(path, index=False, lineterminator="\n")
        else:
            frame.to_parquet(path, index=False, compression="zstd")
        output_paths[name] = path

    manifest = base_manifest(config.project_root, config.config_path, "rq5")
    manifest.update(
        {
            "population": {
                "consensus_nodules": len(consensus),
                "noisy_label_nodules": len(noisy),
                "diagnosis_nodules": len(linked),
            },
            "method": {
                "consensus_reference": config.rq5.consensus_reference,
                "noisy_reference": config.rq5.noisy_reference,
                "addition_fractions": list(config.rq5.addition_fractions),
                "models": list(config.rq5.models),
                "cross_validation_unit": "patient",
                "fixed_test_folds_across_fractions": True,
                "diagnosis_patients_excluded_from_training": True,
                "repetitions": config.rq5.repetitions,
            },
            "inputs": {
                **{name: sha256_file(paths[name]) for name in required},
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
    manifest = _write_auc_changes(config, manifest)
    write_json(output_dir / "manifest.json", manifest)
    return manifest
