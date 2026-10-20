"""Radiologist-derived reference-standard definitions for RQ2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .errors import AnalysisError

IMPLEMENTED_REFERENCES = (
    "median_exclude_3",
    "median_3_benign",
    "median_3_malignant",
    "any_reader_4_or_5",
    "random_reader",
    "latent_class",
)

REFERENCE_DESCRIPTIONS = {
    "median_exclude_3": "Median rating above 3 is malignant; median 3 is excluded",
    "median_3_benign": "Median rating above 3 is malignant; median 3 is benign",
    "median_3_malignant": "Median rating at least 3 is malignant",
    "any_reader_4_or_5": "Malignant if any reader assigns rating 4 or 5",
    "random_reader": "One deterministic random reader; rating 4 or 5 is malignant",
    "latent_class": "Suspicious-class posterior at the configured RQ6 threshold",
}


def build_reference_labels(
    nodules: pd.DataFrame,
    annotations: pd.DataFrame,
    seed: int,
    latent_posterior: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the six prespecified RQ2 reference definitions."""
    required_nodule = {"nodule_id", "malignancy_median"}
    required_annotation = {"nodule_id", "annotation_id", "malignancy"}
    if missing := required_nodule - set(nodules.columns):
        raise AnalysisError(f"Nodule table lacks reference fields: {sorted(missing)}")
    if missing := required_annotation - set(annotations.columns):
        raise AnalysisError(f"Annotation table lacks reference fields: {sorted(missing)}")

    labels = nodules[["nodule_id", "malignancy_median"]].copy()
    median = labels["malignancy_median"].astype(float)
    labels["median_exclude_3"] = np.where(
        median == 3.0, np.nan, (median > 3.0).astype(float)
    )
    labels["median_3_benign"] = (median > 3.0).astype(float)
    labels["median_3_malignant"] = (median >= 3.0).astype(float)

    maximum = annotations.groupby("nodule_id", sort=True)["malignancy"].max()
    labels["any_reader_4_or_5"] = labels["nodule_id"].map((maximum >= 4).astype(float))

    rng = np.random.default_rng(seed)
    selected: dict[str, float] = {}
    ordered = annotations.sort_values(["nodule_id", "annotation_id"])
    for nodule_id, group in ordered.groupby("nodule_id", sort=True):
        ratings = group["malignancy"].to_numpy(dtype=int)
        selected[str(nodule_id)] = float(ratings[rng.integers(0, len(ratings))] >= 4)
    labels["random_reader"] = labels["nodule_id"].map(selected)
    if latent_posterior is None:
        labels["latent_class"] = np.nan
    else:
        required_latent = {"nodule_id", "latent_class"}
        if missing := required_latent - set(latent_posterior.columns):
            raise AnalysisError(f"RQ6 posterior lacks fields: {sorted(missing)}")
        latent = latent_posterior.set_index("nodule_id")["latent_class"]
        labels["latent_class"] = labels["nodule_id"].map(latent)
        if labels["latent_class"].isna().any():
            raise AnalysisError("RQ6 posterior does not cover every RQ2 nodule")
    return labels.drop(columns="malignancy_median")


def reference_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Summarise availability and prevalence for every protocol reference."""
    rows: list[dict[str, Any]] = []
    for reference in IMPLEMENTED_REFERENCES:
        values = labels[reference].dropna().astype(int)
        rows.append(
            {
                "reference": reference,
                "description": REFERENCE_DESCRIPTIONS[reference],
                "status": "implemented",
                "available_nodules": len(values),
                "malignant_nodules": int(values.sum()),
                "prevalence": float(values.mean()),
            }
        )
    return pd.DataFrame.from_records(rows)
