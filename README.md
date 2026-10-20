# Misclassification bias in clinical AI

This repository contains the reproducible Python pipeline for the study
"Misclassification bias in the evaluation of clinical AI: a study using the
LIDC-IDRI lung nodule annotations".

The current implementation covers:

- immutable ingestion of the TCIA DICOM Structured Reports and diagnosis workbook;
- provenance manifests and explicit data-quality validation;
- canonical annotation, nodule and diagnosis tables;
- RQ1 inter-reader disagreement;
- RQ2 model evaluation under six alternative reference standards;
- RQ3 shared-rater error;
- RQ4 panel relabelling;
- RQ5 data quantity versus label quality;
- RQ6 exchangeable-rater latent class correction;
- RQ7 evaluation against confirmed diagnosis;
- Figures in vector and 600 dpi raster formats.


## Data

Place these public source files in the repository root:

- `LIDC-IDRI-StandardizedRepresentation-March2020-manifest.zip`
- `tcia-diagnosis-data-2012-04-20.xls`

The pipeline reads the archive without extracting it and never modifies either
source file. Generated data and results are ignored by Git.

## Installation

Python 3.11 or later is required.

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run order

```powershell
lidc-bias ingest --config config/analysis.toml
lidc-bias validate --config config/analysis.toml
lidc-bias rq1 --config config/analysis.toml
lidc-bias rq6 --config config/analysis.toml
lidc-bias rq2 --config config/analysis.toml
lidc-bias rq3 --config config/analysis.toml
lidc-bias rq4 --config config/analysis.toml
lidc-bias rq5 --config config/analysis.toml
lidc-bias rq7 --config config/analysis.toml
lidc-bias figures --config config/analysis.toml
```

Run the complete implemented workflow with:

```powershell
lidc-bias run --config config/analysis.toml
```

Ingestion is the slow step. It reads every DICOM object once and writes cached
Parquet tables. Later stages use those tables and do not reopen the archive.

## Reproducibility

All statistical and engineering settings are held in `config/analysis.toml`.
Each stage writes a JSON manifest containing input checksums, the configuration
checksum, package versions and the Git revision. Random operations use the
configured master seed and derived deterministic seeds.

RQ6 runs before analyses that consume its suspicious-class posterior. The
pipeline fails with an explicit dependency error if a required upstream result
is absent. Figure source data and every stage's outputs are checksummed.

## Quality checks

```powershell
py -m pytest
py -m ruff check .
```
