# Implemented analysis specification

## Objective and scope

The analysis determines how reference-label construction, exclusion of
indeterminate cases and overlap between predictor and label assessments affect
reported model performance while controlling cases, predictions and patient
partitions. It is a strengthened analysis of an existing study, not an
untouched confirmatory experiment.

Only three comparisons are in the active pipeline. Earlier analyses of reader
agreement, progressive panel relabelling, data quantity, latent classes and
confirmed diagnosis are not part of this study.
The earlier latent-class fitting routine remains under
`src/lidc_bias/exploratory/` for reproducibility, but its output is not consumed
by any active stage because the exchangeability and conditional-independence
assumptions do not resolve the study's main question.

## Source identity and population

Annotations are identified by Structured Report SOP Instance UID. A nodule is
identified by patient ID, source CT Series Instance UID and DICOM measurement-
group Tracking Identifier. Validation checks identifier uniqueness,
SR-to-segmentation linkage, source-member checksums, missingness, rating ranges,
geometry validity, reader counts, malignancy disagreement and the number of
nodules with median rating 3.

The analysis population contains nodules with three or four annotation records.
The records are anonymous assessments: reader identity cannot be followed
across patients, so A, B and C are within-nodule annotation roles rather than
named radiologists.

## Predictors and preprocessing

The fixed model is regularised combined logistic regression. Geometry predictors
are log volume, diameter, log surface area and compactness. Ordinal semantic
predictors are subtlety, sphericity, margin, lobulation, spiculation and texture.
Internal structure and calcification are nominal and are one-hot encoded; their
integer codes are not averaged or treated as an ordered numerical scale.

Missing numeric values are median-imputed within each training fold. Missing
nominal values are most-frequent-imputed within the training fold. Numeric
predictors are standardised within the training fold. All model settings are
fixed in `config/analysis.toml`.

## Recorded patient partitions

Both analyses use 5-fold patient-grouped cross-validation repeated 20 times.
For each analysis, one partition table is generated before model fitting and is
reused for every comparison in that analysis. Partitions are approximately
stratified by the fixed full-cohort rule `median_3_benign`; this avoids changing
folds when the training label or annotation roles change. Every partition row,
repetition and fold is released.

## Comparison 1: label assignment

One combined logistic model is trained under `median_exclude_3`: median rating 3
is absent from training, ratings below 3 are benign and ratings above 3 are
malignant. Each nodule receives an out-of-fold probability in every partition
repetition, including median-3 nodules, and the 20 probabilities are averaged.

Those unchanged probabilities are scored on the same full cohort under:

- `median_3_benign`: malignant when the median rating is greater than 3;
- `median_3_malignant`: malignant when the median rating is at least 3.

The reported paired effect is the second rule minus the first rule for both AUC
and Brier score. Any-reader-positive (at least one rating 4 or 5) and one seeded
random-reader rule (rating 4 or 5) are sensitivity references, not primary
comparisons.

## Comparison 2: case exclusion

For each of the two full-cohort median rules separately, the same probabilities
and unchanged labels are evaluated on:

- all eligible three-to-four-record nodules;
- the subset after removing nodules whose median rating equals 3.

The effect is restricted population minus full population. Because the label
rule is unchanged within a comparison, this effect measures population
selection rather than mixing selection with relabelling. Sample retention and
class counts are available from the released case list and reference table.

## Comparison 3: shared assessment

For each of 20 recorded assignment seeds, three distinct annotation records in
each nodule are assigned to A, B and C. For target A, models using A-source and
B-source predictors are trained on A labels and evaluated against A labels.
The analysis then reverses the roles: B-source and A-source models are trained
on B labels and evaluated against B labels. Both directional comparisons are
also evaluated against C labels. The A and B directional effects are averaged
within each assignment so neither selected record is privileged.

Binary annotation labels are 1 for malignancy ratings 4 or 5 and 0 for ratings
1, 2 or 3. Every source comparison uses the same cases, training labels,
preprocessing, regularisation, model type and patient partitions.

Two geometry analyses are reported:

- **Fixed geometry (primary):** nodule-level geometry averaged across all
  available annotations; semantic characteristics come from A or B.
- **Reader-specific geometry (sensitivity):** geometry and semantic
  characteristics both come from A or B.

For AUC, the same-assessment advantage is `AUC(same) - AUC(other)`. For Brier
score, it is `Brier(other) - Brier(same)`. Positive values therefore always mean
better apparent performance when predictors and labels share an assessment.
Those directions describe evaluation against A or B training labels. Against C,
the reported contrast is training-label-source predictors minus other-source
predictors for AUC (and the reversed subtraction for Brier score); neither
predictor source shares C's assessment. The A/B result is described as a
same-assessment advantage, not proof that one annotation is independent
clinical truth.

## Uncertainty and reporting

AUC is primary and Brier score is supporting. Patients are sampled with
replacement while all their nodules are retained. Label and exclusion effects
use the same patient draw for both conditions. Shared-assessment bootstrap draws
preserve all 20 assignment results and both reciprocal directions, then average
the paired effects. These intervals are conditional on the fitted models.

Assignment variability is the standard deviation of reciprocal effects across
the 20 assignments. Partition variability is reported separately as the
standard deviation across the 20 cross-validation repetitions after averaging
assignments and reciprocal directions. No metric is selected after inspecting
which one produces the strongest result.

## Paper outputs

The publication stage creates only:

- Table 1: reference definitions and retained-case summaries;
- Figure 1: separate label-assignment and case-exclusion effects;
- Table 2: matched shared-assessment effects with assignment and partition
  variability.

Manifests record the configuration checksum, input/output checksums, package
versions, random seeds and Git revision.
