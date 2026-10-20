"""Explicit mappings from DICOM coded concepts to original LIDC ratings."""

from __future__ import annotations

import re

from .errors import SourceDataError

DIRECT_MEANING_MAPPINGS: dict[str, dict[str, int]] = {
    "Internal structure": {
        "Soft tissue": 1,
        "Fluid": 2,
        "Adipose tissue": 3,
        "Air": 4,
    },
    "Calcification": {
        "Popcorn calcification sign": 1,
        "Laminated appearance": 2,
        "Solid appearance": 3,
        "Non-central appearance": 4,
        "Central calcification": 5,
        "Absent": 6,
    },
    "Sphericity": {
        "linear": 1,
        "ovoid": 3,
        "round": 5,
    },
    "Margin": {
        "Indistinct margin": 1,
        "Circumscribed margin": 5,
    },
    "Texture": {
        "non-solid pulmonary nodule": 1,
        "part-solid pulmonary nodule": 3,
        "solid pulmonary nodule": 5,
    },
}

RATING_CONCEPTS = {
    "Subtlety score": "subtlety",
    "Internal structure": "internal_structure",
    "Calcification": "calcification",
    "Sphericity": "sphericity",
    "Margin": "margin",
    "Lobular Pattern": "lobulation",
    "Spiculation": "spiculation",
    "Texture": "texture",
    "Malignancy": "malignancy",
}


def decode_rating(concept: str, code_value: str, code_meaning: str) -> int:
    """Decode one coded LIDC characteristic without heuristic folder parsing."""
    match = re.match(r"^([1-6]) out of [4-6](?:\b| )", code_meaning)
    if match:
        return int(match.group(1))

    mapping = DIRECT_MEANING_MAPPINGS.get(concept, {})
    if code_meaning in mapping:
        return mapping[code_meaning]

    raise SourceDataError(
        f"Unsupported coded rating for {concept}: value={code_value!r}, meaning={code_meaning!r}"
    )
