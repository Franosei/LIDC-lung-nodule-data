"""Exploratory exchangeable-rater latent-class model.

This model is retained for reproducibility but is deliberately not used by the
focused analysis because conditional independence is a strong, unresolved
assumption. It does not define labels for the primary or sensitivity analyses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ..errors import AnalysisError

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
                classes,
                -np.inf,
                mixing,
                rating_probabilities,
                posterior,
                iteration,
                start,
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

    order = np.argsort(rating_probabilities @ RATING_LEVELS)
    return LatentFit(
        classes,
        log_likelihood,
        mixing[order],
        rating_probabilities[order],
        posterior[:, order],
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
    if counts.ndim != 2 or counts.shape[1] != 5:
        raise AnalysisError("Latent-class counts must have five rating columns")
    if np.any(counts < 0) or np.any(counts.sum(axis=1) == 0):
        raise AnalysisError("Latent-class counts must be non-negative with no empty nodule")
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
