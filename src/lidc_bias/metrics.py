"""Classification metrics and patient-level confidence intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from .bootstrap import percentile_interval

METRIC_NAMES = (
    "auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "prevalence",
    "brier_score",
    "calibration_intercept",
    "calibration_slope",
)


def _calibration_parameters(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Fit an unpenalised calibration model, failing safely under separation."""
    labels = np.asarray(labels, dtype=float)
    if np.unique(labels).size != 2:
        return float("nan"), float("nan")
    clipped = np.clip(probabilities.astype(float), 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped))
    design = np.column_stack([np.ones_like(logits), logits])
    parameters = np.array([0.0, 1.0], dtype=float)

    def negative_log_likelihood(candidate: np.ndarray) -> float:
        linear_predictor = design @ candidate
        return float(np.sum(np.logaddexp(0.0, linear_predictor) - labels * linear_predictor))

    objective = negative_log_likelihood(parameters)
    for _ in range(100):
        fitted = expit(design @ parameters)
        weights = np.clip(fitted * (1.0 - fitted), 1e-10, None)
        information = design.T @ (weights[:, None] * design)
        score = design.T @ (labels - fitted)
        try:
            step = np.linalg.solve(information, score)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if not np.all(np.isfinite(step)):
            return float("nan"), float("nan")

        step_fraction = 1.0
        accepted = False
        while step_fraction >= 2.0**-25:
            candidate = parameters + step_fraction * step
            candidate_objective = negative_log_likelihood(candidate)
            if np.isfinite(candidate_objective) and candidate_objective <= objective:
                accepted = True
                break
            step_fraction /= 2.0
        if not accepted:
            return float("nan"), float("nan")

        parameters = candidate
        objective = candidate_objective
        if np.max(np.abs(step_fraction * step)) < 1e-8:
            if not np.all(np.isfinite(parameters)):
                return float("nan"), float("nan")
            return float(parameters[0]), float(parameters[1])

    # Complete or quasi-complete separation has no finite maximum-likelihood
    # estimate. Returning the last iterate would create arbitrary huge values.
    if not np.all(np.isfinite(parameters)):
        return float("nan"), float("nan")
    return float("nan"), float("nan")


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Calculate discrimination, classification and calibration metrics."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= 0.5).astype(int)
    positives = labels == 1
    negatives = labels == 0
    sensitivity = float((predictions[positives] == 1).mean()) if positives.any() else float("nan")
    specificity = float((predictions[negatives] == 0).mean()) if negatives.any() else float("nan")
    auc = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else float("nan")
    )
    intercept, slope = _calibration_parameters(labels, probabilities)
    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "prevalence": float(labels.mean()),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def patient_bootstrap_metrics(
    frame: pd.DataFrame,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, tuple[float, float, int]], pd.DataFrame]:
    """Bootstrap all metrics by resampling patients with replacement."""
    required = {"patient_id", "label", "probability"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Metric frame lacks required columns: {sorted(missing)}")
    patients = np.asarray(sorted(frame["patient_id"].astype(str).unique()))
    row_indices = {
        patient: np.flatnonzero(frame["patient_id"].astype(str).to_numpy() == patient)
        for patient in patients
    }
    labels = frame["label"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for repetition in range(repetitions):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        indices = np.concatenate([row_indices[patient] for patient in sampled])
        values: dict[str, float | int] = {"bootstrap_repetition": repetition}
        values.update(classification_metrics(labels[indices], probabilities[indices]))
        rows.append(values)
    replicates = pd.DataFrame.from_records(rows)
    intervals = {
        metric: percentile_interval(replicates[metric].to_numpy(), confidence_level)
        for metric in METRIC_NAMES
    }
    return intervals, replicates


def long_metric_summary(
    identifiers: dict[str, Any],
    frame: pd.DataFrame,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-form estimates and bootstrap replicates for one comparison."""
    estimates = classification_metrics(
        frame["label"].to_numpy(dtype=int), frame["probability"].to_numpy(dtype=float)
    )
    intervals, replicates = patient_bootstrap_metrics(
        frame, repetitions, confidence_level, seed
    )
    summary_rows: list[dict[str, Any]] = []
    for metric, estimate in estimates.items():
        lower, upper, valid = intervals[metric]
        summary_rows.append(
            {
                **identifiers,
                "metric": metric,
                "estimate": estimate,
                "confidence_level": confidence_level,
                "ci_lower": lower,
                "ci_upper": upper,
                "valid_bootstrap_repetitions": valid,
                "nodules": len(frame),
                "patients": int(frame["patient_id"].nunique()),
            }
        )
    for key, value in identifiers.items():
        replicates[key] = value
    return pd.DataFrame.from_records(summary_rows), replicates
