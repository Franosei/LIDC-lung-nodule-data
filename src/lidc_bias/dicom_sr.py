"""Parsing of the TCIA LIDC-IDRI DICOM Structured Reports."""

from __future__ import annotations

import hashlib
import io
import logging
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from pydicom.dataset import Dataset

from .errors import SourceDataError
from .provenance import stable_identifier
from .ratings import RATING_CONCEPTS, decode_rating

LOGGER = logging.getLogger(__name__)

ENHANCED_SR_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.88.22"
SEGMENTATION_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.66.4"

MEASUREMENT_CONCEPTS = {
    "Volume": ("volume_mm3", "mm3"),
    "Diameter": ("diameter_mm", "mm"),
    "Surface area of mesh": ("surface_area_mm2", "mm2"),
}

ANNOTATION_COLUMNS = [
    "annotation_id",
    "nodule_id",
    "patient_id",
    "study_instance_uid",
    "source_series_instance_uid",
    "tracking_identifier",
    "tracking_unique_identifier",
    "referenced_segmentation_sop_uid",
    "reader_index",
    "volume_mm3",
    "diameter_mm",
    "surface_area_mm2",
    "compactness",
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
    "source_member",
    "source_member_sha256",
]


def _walk_content(items: Sequence[Dataset]) -> Iterator[Dataset]:
    for item in items:
        yield item
        if hasattr(item, "ContentSequence"):
            yield from _walk_content(item.ContentSequence)


def _concept_name(item: Dataset) -> str | None:
    if not hasattr(item, "ConceptNameCodeSequence"):
        return None
    sequence = item.ConceptNameCodeSequence
    return str(sequence[0].CodeMeaning) if sequence else None


def _find_one(nodes: Iterable[Dataset], concept_name: str) -> Dataset:
    matches = [node for node in nodes if _concept_name(node) == concept_name]
    if len(matches) != 1:
        raise SourceDataError(
            f"Expected one {concept_name!r} content item, found {len(matches)}"
        )
    return matches[0]


def _find_optional(nodes: Iterable[Dataset], concept_name: str) -> Dataset | None:
    matches = [node for node in nodes if _concept_name(node) == concept_name]
    if len(matches) > 1:
        raise SourceDataError(
            f"Expected at most one {concept_name!r} content item, found {len(matches)}"
        )
    return matches[0] if matches else None


def _numeric_value(node: Dataset, expected_unit: str) -> float:
    try:
        measured = node.MeasuredValueSequence[0]
        value = float(measured.NumericValue)
        unit = str(measured.MeasurementUnitsCodeSequence[0].CodeValue).strip()
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise SourceDataError(f"Malformed numeric content item: {_concept_name(node)!r}") from error
    if unit != expected_unit:
        raise SourceDataError(
            f"Unexpected unit for {_concept_name(node)!r}: "
            f"expected {expected_unit!r}, found {unit!r}"
        )
    return value


def _coded_value(node: Dataset) -> tuple[str, str]:
    try:
        coded = node.ConceptCodeSequence[0]
        return str(coded.CodeValue).strip(), str(coded.CodeMeaning).strip()
    except (AttributeError, IndexError) as error:
        raise SourceDataError(f"Malformed coded content item: {_concept_name(node)!r}") from error


def _referenced_sop_uid(node: Dataset) -> str:
    try:
        return str(node.ReferencedSOPSequence[0].ReferencedSOPInstanceUID)
    except (AttributeError, IndexError) as error:
        raise SourceDataError("Referenced Segment does not contain a SOP Instance UID") from error


def _annotation_label(member_name: str) -> str | None:
    match = re.search(r"Annotation (.+?) evaluations-", member_name)
    return match.group(1) if match else None


def parse_structured_report(data: bytes, member_name: str) -> dict[str, Any]:
    """Parse one Enhanced SR annotation into a canonical row."""
    try:
        dataset = pydicom.dcmread(io.BytesIO(data))
    except Exception as error:
        raise SourceDataError(f"Could not parse DICOM SR member: {member_name}") from error

    if str(dataset.SOPClassUID) != ENHANCED_SR_SOP_CLASS_UID:
        raise SourceDataError(
            f"Unexpected SOP Class UID in SR member {member_name}: {dataset.SOPClassUID}"
        )

    nodes = list(_walk_content(dataset.ContentSequence))
    patient_id = str(dataset.PatientID)
    study_uid = str(dataset.StudyInstanceUID)
    source_series_uid = str(_find_one(nodes, "Source series for segmentation").UID)
    tracking_identifier = str(_find_one(nodes, "Tracking Identifier").TextValue)
    tracking_uid = str(_find_one(nodes, "Tracking Unique Identifier").UID)
    referenced_segmentation_uid = _referenced_sop_uid(_find_one(nodes, "Referenced Segment"))
    nodule_id = stable_identifier(patient_id, source_series_uid, tracking_identifier)

    row: dict[str, Any] = {
        "annotation_id": str(dataset.SOPInstanceUID),
        "nodule_id": nodule_id,
        "patient_id": patient_id,
        "study_instance_uid": study_uid,
        "source_series_instance_uid": source_series_uid,
        "tracking_identifier": tracking_identifier,
        "tracking_unique_identifier": tracking_uid,
        "referenced_segmentation_sop_uid": referenced_segmentation_uid,
        "annotation_label": _annotation_label(member_name),
        "source_member": member_name,
        "source_member_sha256": hashlib.sha256(data).hexdigest(),
    }

    for concept_name, (column_name, expected_unit) in MEASUREMENT_CONCEPTS.items():
        row[column_name] = _numeric_value(_find_one(nodes, concept_name), expected_unit)

    for concept_name, column_name in RATING_CONCEPTS.items():
        item = _find_one(nodes, concept_name) if concept_name == "Malignancy" else _find_optional(
            nodes, concept_name
        )
        if item is None:
            row[column_name] = None
            row[f"{column_name}_code"] = None
            row[f"{column_name}_meaning"] = None
            continue
        code_value, code_meaning = _coded_value(item)
        row[column_name] = decode_rating(concept_name, code_value, code_meaning)
        row[f"{column_name}_code"] = code_value
        row[f"{column_name}_meaning"] = code_meaning

    volume = float(row["volume_mm3"])
    surface_area = float(row["surface_area_mm2"])
    if volume <= 0.0 or surface_area <= 0.0:
        raise SourceDataError(
            f"Non-positive geometry in {member_name}: volume={volume}, surface={surface_area}"
        )
    row["compactness"] = 36.0 * 3.141592653589793 * volume**2 / surface_area**3
    return row


def parse_segmentation_header(data: bytes, member_name: str) -> dict[str, str]:
    """Read identifiers from one DICOM Segmentation object."""
    try:
        dataset = pydicom.dcmread(
            io.BytesIO(data),
            stop_before_pixels=True,
            specific_tags=[
                "SOPClassUID",
                "SOPInstanceUID",
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
            ],
        )
    except Exception as error:
        raise SourceDataError(f"Could not parse DICOM SEG member: {member_name}") from error
    if str(dataset.SOPClassUID) != SEGMENTATION_SOP_CLASS_UID:
        raise SourceDataError(
            f"Unexpected SOP Class UID in SEG member {member_name}: {dataset.SOPClassUID}"
        )
    return {
        "segmentation_sop_uid": str(dataset.SOPInstanceUID),
        "patient_id": str(dataset.PatientID),
        "study_instance_uid": str(dataset.StudyInstanceUID),
        "segmentation_series_instance_uid": str(dataset.SeriesInstanceUID),
        "source_member": member_name,
        "source_member_sha256": hashlib.sha256(data).hexdigest(),
    }


def archive_members(archive: zipfile.ZipFile) -> tuple[list[str], list[str]]:
    """Return sorted SR and SEG member names from the standard TCIA layout."""
    dicom_members = sorted(name for name in archive.namelist() if name.lower().endswith(".dcm"))
    reports = [name for name in dicom_members if " evaluations-" in name]
    segmentations = [name for name in dicom_members if "-Segmentation of Nodule " in name]
    unclassified = set(dicom_members) - set(reports) - set(segmentations)
    if unclassified:
        sample = sorted(unclassified)[:5]
        raise SourceDataError(f"Unclassified DICOM archive members: {sample}")
    return reports, segmentations


def read_archive(archive_path: Path, progress_every: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read all source annotations and segmentation headers from the ZIP archive."""
    if not archive_path.is_file():
        raise SourceDataError(f"DICOM archive does not exist: {archive_path}")

    annotation_rows: list[dict[str, Any]] = []
    segmentation_rows: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="lidc-bias-") as temporary_directory:
        local_archive = Path(temporary_directory) / archive_path.name
        LOGGER.info("Copying source archive to local temporary storage")
        shutil.copyfile(archive_path, local_archive)
        with zipfile.ZipFile(local_archive) as archive:
            bad_members = archive.testzip()
            if bad_members is not None:
                raise SourceDataError(f"ZIP integrity check failed at member: {bad_members}")
            report_members, segmentation_members = archive_members(archive)
            LOGGER.info("Reading %d Structured Reports", len(report_members))
            for index, member_name in enumerate(report_members, start=1):
                annotation_rows.append(
                    parse_structured_report(archive.read(member_name), member_name)
                )
                if progress_every > 0 and index % progress_every == 0:
                    LOGGER.info("Read %d of %d Structured Reports", index, len(report_members))

            LOGGER.info("Reading %d segmentation headers", len(segmentation_members))
            for index, member_name in enumerate(segmentation_members, start=1):
                segmentation_rows.append(
                    parse_segmentation_header(archive.read(member_name), member_name)
                )
                if progress_every > 0 and index % progress_every == 0:
                    LOGGER.info(
                        "Read %d of %d segmentation headers", index, len(segmentation_members)
                    )

    annotations = pd.DataFrame.from_records(annotation_rows)
    annotations = annotations.sort_values(["nodule_id", "annotation_id"]).reset_index(drop=True)
    annotations["reader_index"] = annotations.groupby("nodule_id").cumcount() + 1
    segmentations = pd.DataFrame.from_records(segmentation_rows).sort_values(
        "segmentation_sop_uid"
    )
    segmentations = segmentations.reset_index(drop=True)
    return annotations, segmentations
