"""Leakage-resistant modelling with reusable patient-level partitions."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LabelCaseEffectsConfig
from .errors import AnalysisError
from .tables import (
    GEOMETRY_FEATURES,
    NOMINAL_SEMANTIC_FEATURES,
    ORDINAL_SEMANTIC_FEATURES,
)

MODEL_FEATURES = {
    "combined_logistic": (
        GEOMETRY_FEATURES + ORDINAL_SEMANTIC_FEATURES + NOMINAL_SEMANTIC_FEATURES
    )
}


def _class_weight(value: str | None) -> str | None:
    return None if value in {None, "none", "None"} else value


def build_model(model_name: str, config: LabelCaseEffectsConfig, seed: int) -> Pipeline:
    """Build the fixed combined logistic model used in both comparisons."""
    if model_name != "combined_logistic":
        raise AnalysisError(f"Unsupported model in the focused protocol: {model_name}")
    numeric_features = GEOMETRY_FEATURES + ORDINAL_SEMANTIC_FEATURES
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "nominal",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                NOMINAL_SEMANTIC_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = LogisticRegression(
        C=config.logistic_regression.regularisation_c,
        max_iter=config.logistic_regression.maximum_iterations,
        class_weight=_class_weight(config.logistic_regression.class_weight),
        solver="lbfgs",
        random_state=seed,
    )
    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def make_patient_partitions(
    frame: pd.DataFrame,
    strata_column: str,
    n_splits: int,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Create one recorded partition plan reusable by every fitted comparison."""
    required = {"nodule_id", "patient_id", strata_column}
    if missing := required - set(frame.columns):
        raise AnalysisError(f"Partition frame lacks fields: {sorted(missing)}")
    if frame["nodule_id"].duplicated().any():
        raise AnalysisError("Partition frame must contain one row per nodule")
    groups = frame["patient_id"].astype(str)
    strata = frame[strata_column].astype(int)
    records: list[pd.DataFrame] = []
    for repetition in range(repetitions):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + repetition,
        )
        for fold, (_, test_indices) in enumerate(
            splitter.split(frame, strata, groups), start=1
        ):
            test = frame.iloc[test_indices]
            records.append(
                pd.DataFrame(
                    {
                        "nodule_id": test["nodule_id"].to_numpy(),
                        "patient_id": test["patient_id"].to_numpy(),
                        "repetition": repetition,
                        "fold": fold,
                    }
                )
            )
    partitions = pd.concat(records, ignore_index=True)
    expected = len(frame) * repetitions
    if len(partitions) != expected:
        raise AnalysisError(f"Expected {expected} partition rows, produced {len(partitions)}")
    if partitions.duplicated(["nodule_id", "repetition"]).any():
        raise AnalysisError("A nodule occurs in multiple folds within one repetition")
    patient_folds = partitions.groupby(["patient_id", "repetition"])["fold"].nunique()
    if (patient_folds != 1).any():
        raise AnalysisError("A patient occurs in multiple folds within one repetition")
    return partitions.sort_values(["repetition", "fold", "patient_id", "nodule_id"]).reset_index(
        drop=True
    )


def cross_validated_predictions(
    frame: pd.DataFrame,
    label_column: str,
    model_name: str,
    config: LabelCaseEffectsConfig,
    master_seed: int,
    partitions: pd.DataFrame,
) -> pd.DataFrame:
    """Fit on recorded folds and predict every nodule once per repetition."""
    if frame["nodule_id"].duplicated().any():
        raise AnalysisError("Model frame must contain one row per nodule")
    expected_ids = set(frame["nodule_id"])
    partition_ids = set(partitions["nodule_id"])
    if expected_ids != partition_ids:
        raise AnalysisError("Model frame and partition plan contain different nodules")
    records: list[pd.DataFrame] = []
    repetitions = sorted(partitions["repetition"].unique())
    for repetition in repetitions:
        repeat = partitions.loc[partitions["repetition"] == repetition]
        fold_by_id = repeat.set_index("nodule_id")["fold"]
        folds = frame["nodule_id"].map(fold_by_id)
        if folds.isna().any():
            raise AnalysisError("Partition plan does not cover every model row")
        for fold in sorted(repeat["fold"].unique()):
            test_mask = folds.eq(fold)
            train = frame.loc[~test_mask]
            test = frame.loc[test_mask]
            if set(train["patient_id"].astype(str)) & set(test["patient_id"].astype(str)):
                raise AnalysisError("Patient leakage detected between training and test folds")
            eligible_train = train[label_column].notna()
            training_labels = train.loc[eligible_train, label_column].astype(int)
            if training_labels.nunique() != 2:
                raise AnalysisError(
                    f"Training fold lacks both classes: label={label_column}, "
                    f"repetition={repetition}, fold={fold}"
                )
            model = build_model(
                model_name,
                config,
                master_seed + int(repetition) * 100 + int(fold),
            )
            model.fit(train.loc[eligible_train], training_labels)
            records.append(
                pd.DataFrame(
                    {
                        "nodule_id": test["nodule_id"].to_numpy(),
                        "patient_id": test["patient_id"].to_numpy(),
                        "reference": label_column,
                        "model": model_name,
                        "repetition": int(repetition),
                        "fold": int(fold),
                        "label": test[label_column].to_numpy(),
                        "probability": model.predict_proba(test)[:, 1],
                    }
                )
            )
    predictions = pd.concat(records, ignore_index=True)
    expected = len(frame) * len(repetitions)
    if len(predictions) != expected:
        raise AnalysisError(f"Expected {expected} predictions, produced {len(predictions)}")
    if predictions.duplicated(["nodule_id", "repetition"]).any():
        raise AnalysisError("A nodule received multiple predictions in one repetition")
    return predictions


def average_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities to one value per nodule."""
    return (
        predictions.groupby("nodule_id", sort=True)
        .agg(
            patient_id=("patient_id", "first"),
            probability=("probability", "mean"),
            prediction_repetitions=("probability", "size"),
        )
        .reset_index()
    )
