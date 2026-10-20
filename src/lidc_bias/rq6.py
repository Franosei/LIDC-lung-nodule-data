"""RQ6 exchangeable-rater latent class analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from .config import AnalysisConfig
from .errors import AnalysisError, SourceDataError
from .provenance import base_manifest, sha256_file, write_json
from .validation import processed_paths

LOGGER = logging.getLogger(__name__)
RATING_LEVELS = np.arange(1, 6)


@dataclass(frozen=True)
class LatentFit:
    classes: int
    log_likelihood: float
    mixing: np.ndarray
    rating_probabilities: np.ndarray
    posterior: np.ndarray
    iterations: int
    start: int


def _rating_counts(annotations: pd.DataFrame, nodule_ids: pd.Series) -> np.ndarray:
    counts = (
        annotations.loc[annotations["nodule_id"].isin(nodule_ids)]
        .groupby(["nodule_id", "malignancy"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=nodule_ids, columns=RATING_LEVELS, fill_value=0)
    )
    return counts.to_numpy(dtype=float)


def _fit_once(
    counts: np.ndarray,
    classes: int,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
    start: int,
) -> LatentFit:
    rng = np.random.default_rng(seed)
    mixing = rng.dirichlet(np.ones(classes))
    rating_probabilities = rng.dirichlet(np.ones(5), size=classes)
    previous = -np.inf
    posterior = np.empty((len(counts), classes), dtype=float)
    for iteration in range(1, maximum_iterations + 1):
        log_joint = np.log(mixing)[None, :] + counts @ np.log(rating_probabilities).T
        normaliser = logsumexp(log_joint, axis=1)
        log_likelihood = float(normaliser.sum())
        posterior = np.exp(log_joint - normaliser[:, None])

        class_mass = posterior.sum(axis=0)
        if np.any(class_mass < 1e-8):
            return LatentFit(
                classes, -np.inf, mixing, rating_probabilities, posterior, iteration, start
            )
        mixing = class_mass / len(counts)
        weighted_counts = posterior.T @ counts
        rating_probabilities = weighted_counts / weighted_counts.sum(axis=1, keepdims=True)
        rating_probabilities = np.clip(rating_probabilities, 1e-12, None)
        rating_probabilities /= rating_probabilities.sum(axis=1, keepdims=True)

        if np.isfinite(previous) and abs(log_likelihood - previous) <= tolerance * (
            1.0 + abs(previous)
        ):
            break
        previous = log_likelihood
    else:
        raise AnalysisError(
            f"Latent class EM did not converge: classes={classes}, start={start}"
        )

    expected_rating = rating_probabilities @ RATING_LEVELS
    order = np.argsort(expected_rating)
    mixing = mixing[order]
    rating_probabilities = rating_probabilities[order]
    posterior = posterior[:, order]
    return LatentFit(
        classes,
        log_likelihood,
        mixing,
        rating_probabilities,
        posterior,
        iteration,
        start,
    )


def fit_latent_classes(
    counts: np.ndarray,
    classes: int,
    starts: int,
    seed: int,
    maximum_iterations: int,
    tolerance: float,
) -> LatentFit:
    """Return the highest-likelihood converged EM fit across random starts."""
    fits = [
        _fit_once(
            counts,
            classes,
            seed + start,
            maximum_iterations,
            tolerance,
            start,
        )
        for start in range(starts)
    ]
    finite = [fit for fit in fits if np.isfinite(fit.log_likelihood)]
    if not finite:
        raise AnalysisError(f"No finite latent class fit for {classes} classes")
    return max(finite, key=lambda fit: fit.log_likelihood)


def _model_selection_row(fit: LatentFit, nodules: int) -> dict[str, Any]:
    parameters = (fit.classes - 1) + fit.classes * 4
    bic = -2.0 * fit.log_likelihood + parameters * np.log(nodules)
    return {
        "classes": fit.classes,
        "log_likelihood": fit.log_likelihood,
        "parameters": parameters,
        "bic": float(bic),
        "best_start": fit.start,
        "iterations": fit.iterations,
    }


def _summaries(fit: LatentFit) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = fit.rating_probabilities @ RATING_LEVELS
    suspicious = fit.classes - 1
    if fit.classes == 1:
        specificity = np.nan
    else:
        other_weights = fit.mixing[:-1] / fit.mixing[:-1].sum()
        specificity = float(other_weights @ fit.rating_probabilities[:-1, :3].sum(axis=1))
    class_rows = []
    rating_rows = []
    for class_index in range(fit.classes):
        class_rows.append(
            {
                "class": class_index + 1,
                "class_label": (
                    "suspicious" if class_index == suspicious else f"class_{class_index + 1}"
                ),
                "mixing_proportion": float(fit.mixing[class_index]),
                "expected_rating": float(expected[class_index]),
                "map_nodules": int((fit.posterior.argmax(axis=1) == class_index).sum()),
                "single_reader_sensitivity": (
                    float(fit.rating_probabilities[class_index, 3:].sum())
                    if class_index == suspicious
                    else np.nan
                ),
                "single_reader_specificity": specificity if class_index == suspicious else np.nan,
            }
        )
        for rating_index, rating in enumerate(RATING_LEVELS):
            rating_rows.append(
                {
                    "class": class_index + 1,
                    "class_label": class_rows[-1]["class_label"],
                    "rating": int(rating),
                    "probability": float(fit.rating_probabilities[class_index, rating_index]),
                }
            )
    return pd.DataFrame(class_rows), pd.DataFrame(rating_rows)


def run_rq6(config: AnalysisConfig) -> dict[str, Any]:
    """Select, fit and export the exchangeable-rater latent class model."""
    paths = processed_paths(config)
    if not paths["annotations"].is_file() or not paths["nodules"].is_file():
        raise SourceDataError("Processed inputs are missing. Run ingest first")
    annotations = pd.read_parquet(paths["annotations"])
    nodules = pd.read_parquet(paths["nodules"])
    population = nodules.loc[
        nodules["reader_count"].between(config.rq6.minimum_readers, config.rq6.maximum_readers)
    ].sort_values("nodule_id")
    counts = _rating_counts(annotations, population["nodule_id"])
    if not np.array_equal(counts.sum(axis=1), population["reader_count"].to_numpy()):
        raise AnalysisError("Latent class rating counts do not match nodule reader counts")

    fits: list[LatentFit] = []
    rows = []
    for classes in range(config.rq6.minimum_classes, config.rq6.maximum_classes + 1):
        LOGGER.info("Fitting RQ6 latent model with %d classes", classes)
        fit = fit_latent_classes(
            counts,
            classes,
            config.rq6.random_starts,
            config.master_seed + 6000 + classes * 1000,
            config.rq6.maximum_iterations,
            config.rq6.tolerance,
        )
        fits.append(fit)
        rows.append(_model_selection_row(fit, len(population)))
    model_selection = pd.DataFrame(rows)
    selected_index = int(model_selection["bic"].idxmin())
    selected = fits[selected_index]
    model_selection["selected"] = model_selection.index == selected_index

    class_summary, rating_probabilities = _summaries(selected)
    suspicious_posterior = selected.posterior[:, -1]
    posterior = population[["nodule_id", "patient_id"]].copy()
    for class_index in range(selected.classes):
        posterior[f"class_{class_index + 1}_posterior"] = selected.posterior[:, class_index]
    posterior["suspicious_posterior"] = suspicious_posterior
    posterior["latent_class"] = (
        suspicious_posterior >= config.rq6.posterior_threshold
    ).astype(int)
    posterior["map_class"] = selected.posterior.argmax(axis=1) + 1

    output_dir = config.paths.results_dir / "rq6"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "model_selection": output_dir / "model_selection.csv",
        "class_summary": output_dir / "class_summary.csv",
        "rating_probabilities": output_dir / "rating_probabilities.csv",
        "posterior": output_dir / "posterior.parquet",
    }
    model_selection.to_csv(output_paths["model_selection"], index=False, lineterminator="\n")
    class_summary.to_csv(output_paths["class_summary"], index=False, lineterminator="\n")
    rating_probabilities.to_csv(
        output_paths["rating_probabilities"], index=False, lineterminator="\n"
    )
    posterior.to_parquet(output_paths["posterior"], index=False, compression="zstd")

    manifest = base_manifest(config.project_root, config.config_path, "rq6")
    manifest.update(
        {
            "population": {
                "patients": int(population["patient_id"].nunique()),
                "nodules": len(population),
                "ratings": int(counts.sum()),
            },
            "method": {
                "random_starts": config.rq6.random_starts,
                "candidate_classes": list(
                    range(config.rq6.minimum_classes, config.rq6.maximum_classes + 1)
                ),
                "selection": "minimum_bic",
                "selected_classes": selected.classes,
                "posterior_threshold": config.rq6.posterior_threshold,
                "rater_assumption": "exchangeable_and_conditionally_independent",
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
