# Implemented analysis specification

## Scope

The pipeline implements source ingestion, provenance, validation, all seven
research questions and the publication figure. Every operational choice that
is not fixed by the study protocol is exposed in `config/analysis.toml`.

## Source identity and grouping

The DICOM Structured Reports are read directly from the TCIA ZIP archive.
Each annotation is identified by its SR SOP Instance UID. Each nodule is
identified by the combination of:

1. TCIA patient identifier;
2. source CT Series Instance UID;
3. DICOM measurement-group Tracking Identifier.

The composite is stored as a stable SHA-256-derived identifier. Folder names
are retained for provenance but are not used as the analytical key. The source
Series Instance UID and Tracking Identifier produce the published total of
2,651 nodules and the expected reader-count distribution.

The Tracking Unique Identifier is shared by the reader annotations grouped as
one nodule. Validation therefore requires each Tracking Unique Identifier to
map to exactly one canonical nodule and each canonical nodule to have exactly
one Tracking Unique Identifier. It does not incorrectly require one value per
annotation.

This grouping follows the DICOM measurement-report model, in which tracking
identifiers identify a finding and must correspond between a Structured Report
and its referenced Segmentation object. See the
[DICOM structured-report templates](https://dicom.nema.org/medical/dicom/current/output/html/part16.html).

## Rating conversion

Every semantic characteristic is read from its coded DICOM content item.
Mappings are explicit in `src/lidc_bias/ratings.py`. Unknown code and meaning
combinations stop ingestion. They are never converted from folder names or
silently treated as missing.

One source annotation lacks the `Internal structure` content item. Its value is
stored as missing and counted in the validation report. Reader-level feature
means use the available ratings. Any remaining model-level missing value is
imputed from the training fold only. Malignancy ratings are required and
missing malignancy stops validation.

The affected annotation is the only annotation for its nodule. The resulting
nodule-level missing value is outside the three-to-four-reader RQ1 and RQ2
population. It remains in the canonical table.

## Geometry

Volume, diameter and surface area are read with their UCUM units and required
to be positive. Logarithms are natural logarithms. Compactness is the
dimensionless isoperimetric quotient:

`36 * pi * volume^2 / surface_area^3`

A sphere has compactness 1. Compactness is calculated for each reader's
segmentation and then averaged across readers, as required by the study
protocol.

## Analysis population

RQ1 to RQ4 and RQ6 use nodules with three or four annotations. RQ5 additionally
uses the 1,259 nodules with one or two annotations. RQ7 applies the strict
diagnosis linkage described below. The 14 nodules with more than four
annotations remain in the canonical tables but are not analysed.

## RQ1

Ordinal Krippendorff alpha uses the five-level malignancy rating. Binary alpha
maps ratings 1 and 2 to benign, ratings 4 and 5 to malignant, and rating 3 to
missing. Pairwise binary agreement applies the same rule. Other outcomes are
exact pairwise agreement, unanimous ratings, and the proportion of nodules
with at least one benign rating and at least one malignant rating.

Confidence intervals use a non-parametric patient-level percentile bootstrap.
When a patient is sampled more than once, its nodules receive distinct
bootstrap identifiers.

## RQ2 references

The implemented references are:

- `median_exclude_3`: malignant when the median is above 3, with median 3 excluded;
- `median_3_benign`: malignant when the median is above 3;
- `median_3_malignant`: malignant when the median is at least 3;
- `any_reader_4_or_5`: malignant when any reader gives rating 4 or 5;
- `random_reader`: one seeded reader per nodule, malignant for rating 4 or 5;
- `latent_class`: suspicious-class posterior at the configured probability threshold.

The code handles half-integer medians directly. The random-reader selection is
made once from annotations sorted by stable source identifiers and is recorded
through the configured seed.

## Models and validation

The four models are geometry logistic regression, semantic logistic
regression, combined logistic regression and combined random forest. Every
model is a scikit-learn pipeline. Median imputation and logistic scaling are
fitted within each training fold.

Validation uses five patient-grouped, approximately stratified folds repeated
20 times by default. A patient cannot occur in both training and test data.
Each nodule receives one out-of-fold prediction per repetition. Repeated
probabilities are averaged before the main metric and patient-bootstrap
calculations.

The reported metrics are AUC, accuracy, sensitivity, specificity, prevalence,
Brier score, calibration intercept and calibration slope. Classification uses
a probability threshold of 0.5. Calibration parameters are fitted by maximum
likelihood using the logit of the out-of-fold probability.
The fit uses step-halved Newton updates and accepts only a finite converged
maximum-likelihood estimate. Bootstrap samples with complete separation or a
non-convergent calibration fit are recorded as missing and excluded from the
percentile interval. Every summary reports the number of valid bootstrap
replicates.

## Fixed predictions and excluded cases

Configuration identifies one fixed prediction source. The default is combined
logistic regression trained under `median_exclude_3`. Those unchanged
predictions are evaluated against every implemented reference.

The same model predicts median-3 nodules out of sample because fold assignment
includes all eligible-population patients while model fitting uses only cases
labelled by the training reference. The excluded-case analysis evaluates these
predictions against the explicitly configured `any_reader_4_or_5` reference.

## RQ3 shared-rater error

Three distinct annotations are selected without replacement within every
nodule using a fixed seed. Reader A supplies the binary label. Reader A or
reader B supplies the individual annotation features, and reader C supplies
the independent evaluation label. Geometry and semantic logistic models use
identical patient-grouped folds. Same-reader inflation is the paired AUC
difference between reader-A and reader-B feature models when both are scored
against reader A. Model-family rank differences are also estimated against
the independent reader C. Confidence intervals use paired patient bootstrap
samples.

## RQ4 panel relabelling

The local reader is reader A from RQ3. Ratings 1 and 2 are benign, ratings 4
and 5 are malignant, and rating 3 is excluded from model fitting. The geometry
classifier is frozen after repeated patient-grouped out-of-fold prediction.
Locally indeterminate nodules are placed in one fixed seeded random order.
Nested panels relabel the first 0%, 10%, and so on through 100% as malignant.
The order is independent of model predictions and nodule features.

Frozen-model performance is reported at the fixed probability threshold. The
moving diameter result is a separate panel-implied quantity: at each step, an
unpenalised univariable logistic model relates the panel outcome to diameter,
and the reported boundary is the diameter with fitted probability 0.5. The
real any-reader panel and the RQ6 latent reference are evaluated separately.

## RQ5 data quantity and label quality

The consensus set comprises all 1,378 three-to-four-reader nodules, labelled
by `median_3_benign`. The additional set comprises 1,259 one-to-two-reader
nodules, each assigned one seeded reader label. Additions are 0%, 25%, 50%,
75% and 100% of the available additional set. Test folds are identical across
addition levels. No additional nodule from a held-out test patient is allowed
into that fold's training data.

All four model specifications are evaluated against the consensus label, the
latent-class label and the fixed 38-nodule diagnosis subset. Diagnosis-linked
patients are excluded from every model used for the diagnosis evaluation.
Every addition level is compared directly with the 0% baseline using a paired
patient-level bootstrap of the AUC difference.

## RQ6 latent class model

The exchangeable-rater model treats each nodule as a five-category rating-count
vector. Conditional on latent class, ratings are independent draws from a
class-specific categorical distribution. Models with one through five classes
are fitted by expectation-maximisation using 30 random starts. BIC uses one
nodule as the independent observational unit and counts `5K - 1` parameters
for K classes. Classes are ordered by expected malignancy rating. The most
suspicious class supplies the soft posterior and the configured posterior
threshold supplies the binary downstream reference.

Implied single-reader sensitivity is the suspicious class probability of a
rating of 4 or 5. Specificity is the mixing-proportion-weighted probability of
a rating of 1 to 3 among all other classes. The conditional-independence
assumption is a limitation, not a claim about radiologist behaviour.

## RQ7 confirmed diagnosis

A patient is linked only when the workbook contains exactly one non-unknown
nodule diagnosis and the main population contains exactly one nodule for that
patient. This yields 38 nodules: 14 benign, 12 primary lung cancers and 12
metastases. The diagnosis-linked patients are excluded from model training.

The combined logistic model is scored against confirmed diagnosis and all six
radiologist-derived references. The radiologists' mean malignancy rating is
also scored against diagnosis. Robustness subsets are pathology confirmation,
primary lung cancer versus benign nodules, and patients with exactly one
canonical nodule in the entire dataset.

## Publication figure

The main seven-panel figure is generated only from versioned result tables.
It is exported as vector PDF and SVG and as 600 dpi PNG and LZW-compressed
TIFF. A source-data CSV, caption and checksummed manifest accompany the image.

## Provenance

Each stage records source or processed-input SHA-256 checksums, configuration
checksum, package versions, Git revision, random seeds and output checksums.
The source archive and workbook remain unchanged.

The diagnosis workbook contains whitespace-only cells in otherwise unused
nodule slots. These cells are normalised to missing values. They are not
converted to diagnosis code zero.

Fifteen patients in the diagnosis workbook have no nodule annotation in the
875-patient Standardized Representation. They remain in the diagnosis table
and are reported by validation, but cannot be linked to a canonical nodule.

A DICOM staging cache is keyed by the complete archive SHA-256 digest and an
explicit cache-schema version. A workbook or downstream validation failure can
therefore be corrected without reopening every DICOM object. A changed archive
or parser schema invalidates the cache.

The standardised DICOM representation was produced using pylidc for XML
parsing, clustering and geometry, and dcmqi for DICOM encoding. See the
[TCIA analysis-result description](https://www.cancerimagingarchive.net/analysis-result/dicom-lidc-idri-nodules/).
