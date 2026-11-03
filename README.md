# Annotation Data Quality and Bias in Clinical AI Evaluation

Reproducible code for a controlled LIDC-IDRI study of three annotation-related
effects:

1. **Label assignment:** score unchanged predictions against two treatments of
   median malignancy rating 3 on the same nodules.
2. **Case exclusion:** under each unchanged full-cohort median rule, compare the
   full cohort with the subset obtained by excluding median rating 3.
3. **Shared assessment:** compare predictor characteristics and labels from the
   same annotation record with characteristics from another record, using a
   reciprocal A/B design and an additional C-label check.

The study does not combine these effects into a data-quality score and does not
claim diagnostic validity. Its contribution is a controlled evaluation
procedure for separating effects that are otherwise easily confounded.

## Data

Place `LIDC-IDRI-StandardizedRepresentation-March2020-manifest.zip` in the
repository root. The pipeline reads the archive without modifying it and
creates canonical annotation, segmentation and nodule tables.

## Installation

Python 3.11 or later is required.

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Run the study

```powershell
lidc-bias ingest --config config/analysis.toml
lidc-bias validate --config config/analysis.toml
lidc-bias label-case-effects --config config/analysis.toml
lidc-bias shared-assessment --config config/analysis.toml
lidc-bias publication --config config/analysis.toml
```

Run all stages with:

```powershell
lidc-bias run --config config/analysis.toml
```

The output includes case lists, patient partitions, 20 annotation-assignment
seeds, fitted-model predictions, bootstrap replicates, compact result tables,
input/output checksums, package versions and the Git revision. The two primary
metrics are AUC and Brier score.

## Verification

```powershell
py -m pytest
py -m ruff check .
```

The exact operational definitions and effect directions are in
[`docs/methods.md`](docs/methods.md).
