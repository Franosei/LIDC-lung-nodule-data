"""Read and normalise the TCIA diagnosis workbook without changing the source."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import SourceDataError

SHEET_NAME = "Diagnosis Truth"
EXPECTED_COLUMN_COUNT = 14

DIAGNOSIS_LABELS = {
    0: "unknown",
    1: "benign_or_non_malignant",
    2: "primary_lung_cancer",
    3: "metastatic_malignancy",
}

METHOD_LABELS = {
    0: "unknown",
    1: "two_year_radiological_stability",
    2: "biopsy",
    3: "surgical_resection",
    4: "progression_or_response",
}


def _integer_code(value: object, allowed: set[int], field: str) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise SourceDataError(f"Unsupported {field} code: {value!r}") from error
    if not numeric.is_integer() or int(numeric) not in allowed:
        raise SourceDataError(f"Unsupported {field} code: {value!r}")
    return int(numeric)


def read_diagnoses(workbook_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return patient-level and long-form nodule-level diagnosis tables."""
    if not workbook_path.is_file():
        raise SourceDataError(f"Diagnosis workbook does not exist: {workbook_path}")
    try:
        source = pd.read_excel(workbook_path, sheet_name=SHEET_NAME, engine="xlrd")
    except Exception as error:
        raise SourceDataError(f"Could not read diagnosis workbook: {workbook_path}") from error
    if source.shape[1] != EXPECTED_COLUMN_COUNT:
        raise SourceDataError(
            f"Diagnosis workbook has {source.shape[1]} columns, expected {EXPECTED_COLUMN_COUNT}"
        )

    patient_rows: list[dict[str, object]] = []
    nodule_rows: list[dict[str, object]] = []
    for values in source.itertuples(index=False, name=None):
        patient_id = str(values[0]).strip()
        patient_diagnosis = _integer_code(values[1], set(DIAGNOSIS_LABELS), "diagnosis")
        patient_method = _integer_code(values[2], set(METHOD_LABELS), "diagnosis method")
        patient_rows.append(
            {
                "patient_id": patient_id,
                "diagnosis_code": patient_diagnosis,
                "diagnosis_label": DIAGNOSIS_LABELS[patient_diagnosis],
                "method_code": patient_method,
                "method_label": METHOD_LABELS[patient_method],
                "primary_tumour_site": None if pd.isna(values[3]) else str(values[3]).strip(),
            }
        )
        for nodule_number in range(1, 6):
            diagnosis_index = 4 + (nodule_number - 1) * 2
            method_index = diagnosis_index + 1
            diagnosis_code = _integer_code(
                values[diagnosis_index], set(DIAGNOSIS_LABELS), "nodule diagnosis"
            )
            method_code = _integer_code(
                values[method_index], set(METHOD_LABELS), "nodule diagnosis method"
            )
            if diagnosis_code is None and method_code is None:
                continue
            nodule_rows.append(
                {
                    "patient_id": patient_id,
                    "diagnosis_nodule_number": nodule_number,
                    "diagnosis_code": diagnosis_code,
                    "diagnosis_label": (
                        None if diagnosis_code is None else DIAGNOSIS_LABELS[diagnosis_code]
                    ),
                    "method_code": method_code,
                    "method_label": None if method_code is None else METHOD_LABELS[method_code],
                }
            )

    patients = (
        pd.DataFrame.from_records(patient_rows).sort_values("patient_id").reset_index(drop=True)
    )
    nodules = pd.DataFrame.from_records(nodule_rows).sort_values(
        ["patient_id", "diagnosis_nodule_number"]
    )
    return patients.reset_index(drop=True), nodules.reset_index(drop=True)
