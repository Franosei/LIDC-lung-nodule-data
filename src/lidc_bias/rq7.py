"""RQ7 evaluation against linked confirmed diagnoses."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .bootstrap import percentile_interval
from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .modelling import build_model
from .provenance import base_manifest, sha256_file, write_json
from .references import IMPLEMENTED_REFERENCES, build_reference_labels
from .validation import processed_paths


def link_confirmed_diagnoses(
    nodules: pd.DataFrame, diagnosis_nodules: pd.DataFrame
) -> pd.DataFrame:
    """Apply the prespecified one-known-diagnosis and one-main-nodule linkage."""
    main = nodules.loc[nodules["reader_count"].between(3, 4)].copy()
    known = diagnosis_nodules.loc[diagnosis_nodules["diagnosis_code"] != 0].copy()
    one_diagnosis = known.groupby("patient_id").size().loc[lambda values: values == 1].index
    one_main = main.groupby("patient_id").size().loc[lambda values: values == 1].index
    eligible = set(one_diagnosis) & set(one_main)
    linked = main.loc[main["patient_id"].isin(eligible)].merge(
        known.loc[known["patient_id"].isin(eligible)],
        on="patient_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_diagnosis"),
    )
    linked["diagnosis"] = (linked["diagnosis_code"] >= 2).astype(int)
    if len(linked) != 38 or int(linked["diagnosis"].sum()) != 24:
        raise AnalysisError(
            "Strict diagnosis linkage did not reproduce 38 nodules with 24 malignancies"
        )
    return linked


def _auc_interval(
    frame: pd.DataFrame,
    label: str,
    score: str,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float, int, pd.DataFrame]:
    labels = frame[label].to_numpy(dtype=int)
    scores = frame[score].to_numpy(dtype=float)
    estimate = float(roc_auc_score(labels, scores))
    patients = np.asarray(sorted(frame["patient_id"].astype(str).unique()))
    patient_array = frame["patient_id"].astype(str).to_numpy()
    indices = {patient: np.flatnonzero(patient_array == patient) for patient in patients}
    rng = np.random.default_rng(seed)
    values = []
    for _repetition in range(repetitions):
        sampled = rng.choice(patients, len(patients), replace=True)
        draw = np.concatenate([indices[patient] for patient in sampled])
        sampled_labels = labels[draw]
        values.append(
            float(roc_auc_score(sampled_labels, scores[draw]))
            if np.unique(sampled_labels).size == 2
            else np.nan
        )
    lower, upper, valid = percentile_interval(values, confidence_level)
    return (
        estimate,
        lower,
        upper,
        valid,
        pd.DataFrame(
            {"bootstrap_repetition": np.arange(repetitions), "auc": values}
        ),
    )


def run_rq7(config: AnalysisConfig) -> dict[str, Any]:
    """Train away from and evaluate on the strict diagnosis-linked subset."""
    paths = processed_paths(config)
    required = ("annotations", "nodules", "diagnosis_nodules")
    if any(not paths[name].is_file() for name in required):
        raise SourceDataError("Processed inputs are missing. Run ingest first")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    diagnosis_nodules = pd.read_parquet(paths["diagnosis_nodules"])
    latent_path = config.paths.results_dir / "rq6" / "posterior.parquet"
    if not latent_path.is_file():
        raise SourceDataError("RQ6 posterior is missing. Run rq6 before rq7")
    latent = pd.read_parquet(latent_path)

    linked = link_confirmed_diagnoses(nodules, diagnosis_nodules)
    population = nodules.loc[nodules["reader_count"].between(3, 4)].copy()
    population_annotations = annotations.loc[annotations["nodule_id"].isin(population["nodule_id"])]
    labels = build_reference_labels(
        population,
        population_annotations,
        config.master_seed + config.rq2.random_reader_seed_offset,
        latent,
    )
    analysis = population.merge(labels, on="nodule_id", validate="one_to_one")
    train = analysis.loc[~analysis["patient_id"].isin(linked["patient_id"])].copy()
    eligible = train[config.rq7.training_reference].notna()
    training_labels = train.loc[eligible, config.rq7.training_reference].astype(int)
    model = build_model(config.rq7.model, config.rq2, config.master_seed + 7000)
    model.fit(train.loc[eligible], training_labels)
    linked["model_probability"] = model.predict_proba(linked)[:, 1]

    linked_annotations = annotations.loc[annotations["nodule_id"].isin(linked["nodule_id"])]
    mean_rating = linked_annotations.groupby("nodule_id")["malignancy"].mean()
    linked["radiologist_mean_rating"] = linked["nodule_id"].map(mean_rating)
    linked["radiologist_mean_score"] = (linked["radiologist_mean_rating"] - 1.0) / 4.0
    linked = linked.merge(labels, on="nodule_id", validate="one_to_one")

    all_nodule_counts = nodules.groupby("patient_id").size()
    linked["strictest_linkage"] = linked["patient_id"].map(all_nodule_counts).eq(1)
    linked["pathology_confirmed"] = linked["method_label"].isin(
        ["biopsy", "surgical_resection"]
    )
    subsets = {
        "all_confirmed": np.ones(len(linked), dtype=bool),
        "pathology_confirmed": linked["pathology_confirmed"].to_numpy(),
        "primary_vs_benign": linked["diagnosis_label"].ne("metastatic_malignancy").to_numpy(),
        "strictest_linkage": linked["strictest_linkage"].to_numpy(),
    }

    summary_rows = []
    bootstrap_frames = []
    comparison_index = 0
    for subset_name, mask in subsets.items():
        subset = linked.loc[mask].copy()
        for score_name, score_column in (
            (config.rq7.model, "model_probability"),
            ("radiologist_mean_rating", "radiologist_mean_score"),
        ):
            if subset["diagnosis"].nunique() < 2:
                continue
            estimate, lower, upper, valid, bootstrap = _auc_interval(
                subset,
                "diagnosis",
                score_column,
                config.rq7.bootstrap_repetitions,
                config.rq7.confidence_level,
                config.master_seed + 7100 + comparison_index,
            )
            summary_rows.append(
                {
                    "analysis": "confirmed_diagnosis",
                    "subset": subset_name,
                    "score": score_name,
                    "reference": "diagnosis",
                    "auc": estimate,
                    "confidence_level": config.rq7.confidence_level,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "valid_bootstrap_repetitions": valid,
                    "nodules": len(subset),
                    "malignant_nodules": int(subset["diagnosis"].sum()),
                }
            )
            bootstrap["analysis"] = "confirmed_diagnosis"
            bootstrap["subset"] = subset_name
            bootstrap["score"] = score_name
            bootstrap_frames.append(bootstrap)
            comparison_index += 1

    for reference in IMPLEMENTED_REFERENCES:
        subset = linked.loc[linked[reference].notna()].copy()
        estimate, lower, upper, valid, bootstrap = _auc_interval(
            subset,
            reference,
            "model_probability",
            config.rq7.bootstrap_repetitions,
            config.rq7.confidence_level,
            config.master_seed + 7200 + comparison_index,
        )
        summary_rows.append(
            {
                "analysis": "radiologist_reference",
                "subset": "all_linked",
                "score": config.rq7.model,
                "reference": reference,
                "auc": estimate,
                "confidence_level": config.rq7.confidence_level,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstrap_repetitions": valid,
                "nodules": len(subset),
                "malignant_nodules": int(subset[reference].sum()),
            }
        )
        bootstrap["analysis"] = "radiologist_reference"
        bootstrap["subset"] = "all_linked"
        bootstrap["score"] = config.rq7.model
        bootstrap["reference"] = reference
        bootstrap_frames.append(bootstrap)
        comparison_index += 1

    output_dir = config.paths.results_dir / "rq7"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame.from_records(summary_rows)
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    output_paths = {
        "linked_predictions": output_dir / "linked_predictions.parquet",
        "auc_summary": output_dir / "auc_summary.csv",
        "bootstrap_replicates": output_dir / "bootstrap_replicates.parquet",
    }
    linked.to_parquet(output_paths["linked_predictions"], index=False, compression="zstd")
    summary.to_csv(output_paths["auc_summary"], index=False, lineterminator="\n")
    bootstrap.to_parquet(output_paths["bootstrap_replicates"], index=False, compression="zstd")

    manifest = base_manifest(config.project_root, config.config_path, "rq7")
    manifest.update(
        {
            "population": {
                "nodules": len(linked),
                "benign": int((linked["diagnosis"] == 0).sum()),
                "primary_lung_cancer": int(
                    (linked["diagnosis_label"] == "primary_lung_cancer").sum()
                ),
                "metastatic_malignancy": int(
                    (linked["diagnosis_label"] == "metastatic_malignancy").sum()
                ),
            },
            "method": {
                "training_reference": config.rq7.training_reference,
                "model": config.rq7.model,
                "diagnosis_patients_excluded_from_training": True,
                "linkage": "one_known_nodule_diagnosis_and_one_three_to_four_reader_nodule",
                "bootstrap_unit": "patient",
                "bootstrap_repetitions": config.rq7.bootstrap_repetitions,
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
    write_json(output_dir / "manifest.json", manifest)
    return manifest
