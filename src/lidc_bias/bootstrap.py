"""Patient-level bootstrap utilities shared by the research questions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def percentile_interval(
    values: np.ndarray | list[float], confidence_level: float
) -> tuple[float, float, int]:
    """Return a percentile interval after removing non-finite replicates."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), 0
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper), int(finite.size)


def resample_patients(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    patient_column: str = "patient_id",
    unit_column: str = "nodule_id",
) -> pd.DataFrame:
    """Resample patient clusters and make duplicated analytical units distinct."""
    patient_ids = np.asarray(sorted(frame[patient_column].astype(str).unique()))
    sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
    blocks: list[pd.DataFrame] = []
    for draw_index, patient_id in enumerate(sampled):
        block = frame.loc[frame[patient_column].astype(str) == patient_id].copy()
        block["bootstrap_patient_id"] = f"{draw_index}:{patient_id}"
        prefix = f"{draw_index}:"
        block[unit_column] = prefix + block[unit_column].astype(str)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)
