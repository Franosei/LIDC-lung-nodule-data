"""Construction of canonical analysis tables from annotation-level records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ValidationError

GEOMETRY_FEATURES = ["log_volume", "diameter_mm", "log_surface_area", "compactness"]
SEMANTIC_FEATURES = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]
ALL_MODEL_FEATURES = GEOMETRY_FEATURES + SEMANTIC_FEATURES


def build_nodule_table(annotations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reader measurements into one analysis row per nodule."""
    required = {
        "nodule_id",
        "patient_id",
        "study_instance_uid",
        "source_series_instance_uid",
        "tracking_identifier",
        "volume_mm3",
        "diameter_mm",
        "surface_area_mm2",
        "compactness",
        "malignancy",
        *SEMANTIC_FEATURES,
    }
    missing = required - set(annotations.columns)
    if missing:
        raise ValidationError(f"Cannot build nodule table. Missing columns: {sorted(missing)}")

    working = annotations.copy()
    working["log_volume"] = np.log(working["volume_mm3"].astype(float))
    working["log_surface_area"] = np.log(working["surface_area_mm2"].astype(float))

    identity_columns = [
        "patient_id",
        "study_instance_uid",
        "source_series_instance_uid",
        "tracking_identifier",
    ]
    identity_counts = working.groupby("nodule_id")[identity_columns].nunique(dropna=False)
    if (identity_counts > 1).any(axis=None):
        bad_ids = identity_counts.index[(identity_counts > 1).any(axis=1)].tolist()[:10]
        raise ValidationError(f"Nodule identifiers map to conflicting source identities: {bad_ids}")

    feature_columns = ALL_MODEL_FEATURES
    features = working.groupby("nodule_id", sort=True)[feature_columns].mean()
    identity = working.groupby("nodule_id", sort=True)[identity_columns].first()
    ratings = working.groupby("nodule_id", sort=True)["malignancy"].agg(
        reader_count="size",
        malignancy_min="min",
        malignancy_max="max",
        malignancy_median="median",
    )
    ratings["all_readers_agree"] = ratings["malignancy_min"] == ratings["malignancy_max"]
    ratings["benign_malignant_discordance"] = (
        (ratings["malignancy_min"] <= 2) & (ratings["malignancy_max"] >= 4)
    )

    nodules = identity.join(features).join(ratings).reset_index()
    return nodules.sort_values(
        ["patient_id", "source_series_instance_uid", "nodule_id"]
    ).reset_index(
        drop=True,
    )
